from typing import Any, Callable, Dict, Optional
import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import SpectralNorm
import numpy as np
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
        with torch.cuda.amp.autocast(enabled=False):
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
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    return_dict: bool = False,
    return_feature: bool = False,
    retrival_index: Optional[list] = None,
    **kwargs
):
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    if return_feature:
        retrival_features = []
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
                temb,
                image_rotary_emb,
                **ckpt_kwargs
            )
            if return_feature and index_block in retrival_index:
                retrival_features.append(hidden_states)
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            if return_feature and index_block in retrival_index:
                retrival_features.append(hidden_states)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for index_block, block in enumerate(self.single_transformer_blocks):
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
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs
            )
            if (
                return_feature
                and index_block + len(self.transformer_blocks) in retrival_index
            ):
                retrival_features.append(
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            if (
                return_feature
                and index_block + len(self.transformer_blocks) in retrival_index
            ):
                retrival_features.append(
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                )
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if return_feature:
        return retrival_features
    if not return_dict:
        return (output,)
    return output


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
            retrival_features = self.transformer.forward(self.transformer, **kwargs)
            res_list = []
            for feat, head in zip(retrival_features, self.heads):
                res_list.append(
                    head(feat.transpose(1, 2), None).reshape(feat.shape[0], -1)
                )
            concat_res = torch.cat(res_list, dim=1)
            return concat_res
