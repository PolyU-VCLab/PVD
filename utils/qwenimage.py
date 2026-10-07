"""Initialize QwenImage adapters without retaining optimizer or experiment metadata."""

from pathlib import Path
from typing import List, Union
import torch
from utils.weight_io import load_checkpoint


def load_init_lora_checkpoint(args, model, freezed_teacher, accelerator):
    if getattr(args, "ckpt_path", None):
        return
    path = Path(args.init_ckpt)
    if path.is_dir():
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        state = load_file(str(path / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model, state)
        missing = [
            k for k in result.missing_keys if "lora_" in k or "modules_to_save" in k
        ]
        if missing or result.unexpected_keys:
            raise ValueError("Adapter configuration does not match the training model")
        return
    checkpoint = load_checkpoint(path, map_location="cpu", mmap=True)
    state = checkpoint.get("model", checkpoint)
    state = {
        k.removeprefix("module."): v
        for (k, v) in state.items()
        if "lora_" in k or "modules_to_save" in k
    }
    if not state:
        raise ValueError("Initialization checkpoint contains no adapter tensors")
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if "lora_" in k or "modules_to_save" in k]
    if missing or result.unexpected_keys:
        raise ValueError(
            "Initialization checkpoint does not match the training adapter"
        )
    if freezed_teacher is not None and checkpoint.get("teacher_heads") is not None:
        teacher = accelerator.unwrap_model(freezed_teacher)
        teacher.heads.load_state_dict(checkpoint["teacher_heads"], strict=True)


QWENIMAGE_TOKENIZER_MAX_LENGTH = 1024

QWENIMAGE_PROMPT_TEMPLATE_ENCODE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

QWENIMAGE_PROMPT_TEMPLATE_ENCODE_START_IDX = 34


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def _get_qwenimage_prompt_embeds(
    prompt: Union[str, List[str]],
    device: torch.device,
    tokenizer,
    text_encoder,
    dtype: torch.dtype = torch.bfloat16,
):
    """Encode training captions using the QwenImage prompt template."""
    prompt = [prompt] if isinstance(prompt, str) else list(prompt)
    txt = [QWENIMAGE_PROMPT_TEMPLATE_ENCODE.format(e) for e in prompt]
    drop_idx = QWENIMAGE_PROMPT_TEMPLATE_ENCODE_START_IDX
    txt_tokens = tokenizer(
        txt,
        max_length=QWENIMAGE_TOKENIZER_MAX_LENGTH + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    encoder_out = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_out.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(
        hidden_states, txt_tokens.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [
        torch.ones(e.size(0), dtype=torch.long, device=e.device)
        for e in split_hidden_states
    ]
    max_seq_len = max((e.size(0) for e in split_hidden_states))
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    return (prompt_embeds.to(dtype=dtype, device=device), encoder_attention_mask)


@torch.no_grad()
def encode_prompt(
    prompt: Union[str, List[str]],
    device: torch.device,
    tokenizer,
    text_encoder,
    max_sequence_length: int = 1024,
):
    prompt_embeds, prompt_embeds_mask = _get_qwenimage_prompt_embeds(
        prompt, device=device, tokenizer=tokenizer, text_encoder=text_encoder
    )
    prompt_embeds = prompt_embeds[:, :max_sequence_length]
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
    return (prompt_embeds, prompt_embeds_mask)
