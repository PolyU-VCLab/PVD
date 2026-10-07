# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor, encoder_hidden_states_mask: torch.Tensor | None
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Compute text sequence length without assuming contiguous masks. Returns length for RoPE and a normalized bool mask.
    """
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return (text_seq_len, None, None)
    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match (batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )
    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
    position_ids = torch.arange(
        text_seq_len, device=encoder_hidden_states.device, dtype=torch.long
    )
    active_positions = torch.where(
        encoder_hidden_states_mask, position_ids, position_ids.new_zeros(())
    )
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return (text_seq_len, per_sample_len, encoder_hidden_states_mask)
