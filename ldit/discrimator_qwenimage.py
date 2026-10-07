from typing import Any, Callable, Dict
from math import prod
import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import SpectralNorm
import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from ldit.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.utils import is_torch_version


class ResidualBlock(nn.Module):

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        virtual_bs: int = 8,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):

    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim
        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9)),
        )
        if self.c_dim > 0:
            self.cmapper = nn.Linear(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            h = self.main(x)
            out = self.cls(h)
            if self.c_dim > 0:
                cmap = self.cmapper(c).unsqueeze(-1)
                out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            return out


def transformer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
    retrival_index: list[int] = None,
    return_feature: bool = False,
) -> torch.Tensor | Transformer2DModelOutput:
    """
    The [`QwenImageTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask for the encoder hidden states. Expected to have 1.0 for valid tokens and 0.0 for padding tokens.
                Used in the attention processor to prevent attending to padding tokens. The mask can have any pattern
                (not just contiguous valid tokens followed by padding) since it's applied element-wise in attention.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                Image shapes for RoPE computation.
            txt_seq_lens (`list[int]`, *optional*, **Deprecated**):
                Deprecated parameter. Use `encoder_hidden_states_mask` instead. If provided, the maximum value will be
                used to compute RoPE sequence length.
            guidance (`torch.Tensor`, *optional*):
                Guidance tensor for conditional generation.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_block_samples (*optional*):
                ControlNet block samples to add to the transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
    """
    retrival_features = []
    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [
                [0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]])
                for sample in img_shapes
            ],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )
    image_rotary_emb = self.pos_embed(
        img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
    )
    block_attention_kwargs = (
        attention_kwargs.copy() if attention_kwargs is not None else {}
    )
    if encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        image_mask = torch.ones(
            (batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device
        )
        joint_attention_mask = torch.cat(
            [encoder_hidden_states_mask, image_mask], dim=1
        )
        block_attention_kwargs["attention_mask"] = joint_attention_mask
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                None,
                temb,
                image_rotary_emb,
                block_attention_kwargs,
                modulate_index,
                **ckpt_kwargs
            )
            if return_feature and index_block in retrival_index:
                retrival_features.append(hidden_states)
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )
            if return_feature and index_block in retrival_index:
                retrival_features.append(hidden_states)
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_block_samples[index_block // interval_control]
            )
    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if return_feature:
        return (output, retrival_features)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class DiTDiscriminator(nn.Module):

    def __init__(self, transformer, adv_index):
        super().__init__()
        self.transformer = transformer
        self.transformer.forward = transformer_forward
        self.adv_index = adv_index
        self.heads = nn.ModuleList(
            [DiscHead(self.transformer.inner_dim, 0, 0) for _ in range(len(adv_index))]
        )

    @property
    def model(self):
        return self.transformer

    def forward(self, return_discriminator=False, **kwargs):
        if not return_discriminator:
            return self.transformer.forward(self.transformer, **kwargs)
        else:
            kwargs["retrival_index"] = self.adv_index
            kwargs["return_feature"] = True
            output, retrival_features = self.transformer.forward(
                self.transformer, **kwargs
            )
            res_list = []
            for feat, head in zip(retrival_features, self.heads):
                res_list.append(
                    head(feat.transpose(1, 2), None).reshape(feat.shape[0], -1)
                )
            concat_res = torch.cat(res_list, dim=1)
            return concat_res
