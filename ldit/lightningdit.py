# Adapted for PVD from ldit/lightningdit.py distributed with FACM.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import PatchEmbed, Mlp
from ldit.swiglu_ffn import SwiGLUFFN
from ldit.pos_embed import VisionRotaryEmbeddingFast
from ldit.rmsnorm import RMSNorm
from typing import Optional, List, Union


def make_overlapping_intervals(
    n_segments: int,
    overlap: Union[float, List[float]] = 0.0,
    segment_lengths: Optional[Union[List[float], torch.Tensor]] = None,
    device: str = "cpu",
):
    "Construct overlapping time intervals [start, end].\n\n    Args:\n        n_segments: Number of intervals.\n        overlap: Global overlap fraction or one value per adjacent interval pair.\n        segment_lengths: Relative interval lengths, normalized to sum to one.\n            None selects equally sized intervals.\n        device: Device for the returned tensor.\n    Returns:\n        Tensor of shape (n_segments, 2) containing interval boundaries.\n"
    if isinstance(overlap, float):
        assert 0 <= overlap < 1, "Overlap must be in [0, 1)"
        overlap = [overlap] * (n_segments - 1)
    else:
        assert (
            isinstance(overlap, list) and len(overlap) == n_segments - 1
        ), f"Overlap list length must be {n_segments - 1} (n_segments-1)"
        assert all((0 <= o < 1 for o in overlap)), "All overlaps must be in [0, 1)"
    if segment_lengths is None:
        base_len = 1.0 / (n_segments - sum(overlap) * (1.0 / n_segments) * n_segments)
        seg_lengths = [base_len] * n_segments
    else:
        seg_lengths = torch.tensor(
            segment_lengths, dtype=torch.float32, device=device
        ).flatten()
        assert (
            len(seg_lengths) == n_segments
        ), f"Segment lengths must have {n_segments} elements"
        assert torch.all(seg_lengths > 0), "Segment lengths must be positive"
        seg_lengths = (seg_lengths / seg_lengths.sum()).tolist()
    intervals = []
    start = 0.0
    for i in range(n_segments):
        seg_len = seg_lengths[i]
        end = start + seg_len
        intervals.append((start, end))
        if i < n_segments - 1:
            start = end - seg_len * overlap[i]
    intervals[-1] = (intervals[-1][0], 1.0)
    return torch.tensor(intervals, dtype=torch.float32, device=device)


def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """

    def __init__(
        self, num_classes, hidden_size, dropout_prob, disable_label_dropout=False
    ):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        if disable_label_dropout:
            self.dropout_prob = 0

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if train and use_dropout or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = fused_attn
        if use_rmsnorm:
            norm_layer = RMSNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)
        q, k = (self.q_norm(q), self.k_norm(k))
        if rope is not None:
            q = rope(q)
            k = rope(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
                4, dim=1
            )
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-06
            )
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT_PVD(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        auxiliary_time_cond=False,
        disable_label_dropout=False,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if auxiliary_time_cond:
            self.r_embedder = TimestepEmbedder(hidden_size)
        else:
            self.r_embedder = None
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, disable_label_dropout
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=hw_seq_len
            )
        else:
            self.feat_rope = None
        self.blocks1 = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for _ in range(depth // 2)
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for _ in range(depth // 2)
            ]
        )
        self.div_blocks = [self.blocks1, self.blocks2]
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.r_embedder is not None:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.constant_(self.r_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.r_embedder.mlp[2].bias, 0)
        for blocks in self.div_blocks:
            for b in blocks:
                nn.init.constant_(b.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(b.adaLN_modulation[-1].bias, 0)
        if self.final_layer is not None:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def enable_fused_attn(self):
        for blocks in self.div_blocks:
            for b in blocks:
                b.attn.fused_attn = True

    def disable_fused_attn(self):
        for blocks in self.div_blocks:
            for b in blocks:
                b.attn.fused_attn = False

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)).contiguous()
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p)).contiguous()
        return imgs

    def block_forward(
        self, index, x, x_latent, c, c_pre, blocks, final_layer, use_checkpoint
    ):
        x = self.x_embedder(x) + self.pos_embed
        for b in blocks:
            if use_checkpoint:
                x = checkpoint(b, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = b(x, c, self.feat_rope)
        x_latent = x
        x = final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return (x, x_latent)

    def forward(
        self,
        x,
        t=None,
        r=None,
        y=None,
        x_latent=None,
        c_pre=None,
        use_checkpoint=False,
        index=None,
        intervals=None,
    ):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, 1, 1, 1) tensor of diffusion timesteps
        r: (N, 1, 1, 1) tensor of bridge timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        if not self.training:
            return self.inference(x, y, intervals, use_checkpoint)
        use_checkpoint = self.use_checkpoint or use_checkpoint
        y_emb = self.y_embedder(y, self.training)
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r - t) if self.r_embedder is not None else 0
        c = t_emb + y_emb + r_emb
        x, x_latent = self.block_forward(
            index,
            x,
            x_latent,
            c,
            c_pre,
            self.div_blocks[index],
            self.final_layer,
            use_checkpoint,
        )
        return (x, x_latent, c)

    def inference(self, x, y=None, intervals=None, use_checkpoint=False):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of bridge timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        b, _, _, _ = x.shape
        device = x.device
        use_checkpoint = self.use_checkpoint or use_checkpoint
        num_steps = len(self.div_blocks)
        if intervals == None:
            intervals = [0.4, 0.6]
        time_interval = make_overlapping_intervals(
            num_steps, segment_lengths=intervals, device=device
        )
        x_latent = None
        c_pre = None
        x_next = x
        y_emb = self.y_embedder(y, self.training)
        for index in range(num_steps):
            t, r = (time_interval[index][0], time_interval[index][1])
            t_emb = self.t_embedder(t.expand(b))
            r_emb = (
                self.r_embedder((r - t).expand(b)) if self.r_embedder is not None else 0
            )
            c = t_emb + y_emb + r_emb
            v, x_latent = self.block_forward(
                index,
                x_next,
                x_latent,
                c,
                c_pre,
                self.div_blocks[index],
                self.final_layer,
                use_checkpoint,
            )
            x_t = x_next + v * (r - t)
            x_next = x_t
            c_pre = c
        return x_t


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        auxiliary_time_cond=False,
        disable_label_dropout=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if auxiliary_time_cond:
            self.r_embedder = TimestepEmbedder(hidden_size)
        else:
            self.r_embedder = None
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, disable_label_dropout
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=hw_seq_len
            )
        else:
            self.feat_rope = None
        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.r_embedder is not None:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.constant_(self.r_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.r_embedder.mlp[2].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def enable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = True

    def disable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = False

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p)).contiguous()
        return imgs

    def forward(self, x, t=None, r=None, y=None, use_checkpoint=False):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of bridge timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        use_checkpoint = self.use_checkpoint or use_checkpoint
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        r = self.r_embedder(r) if self.r_embedder is not None else 0
        y = self.y_embedder(y, self.training)
        c = t + y + r
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def LightningDiT_XL_1_PVD(**kwargs):
    return LightningDiT_PVD(
        depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs
    )


def LightningDiT_XL_1(**kwargs):
    return LightningDiT(
        depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs
    )


LightningDiT_models = {
    "LightningDiT-XL/1-PVD": LightningDiT_XL_1_PVD,
    "LightningDiT-XL/1": LightningDiT_XL_1,
}
