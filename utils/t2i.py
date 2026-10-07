from pathlib import Path
from transformers import (
    BaseImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PreTrainedModel,
    T5EncoderModel,
    T5TokenizerFast,
)
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.transforms import InterpolationMode, transforms
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict


def load_clip_tokenizer(pretrained_model_name_or_path: str, **kwargs) -> CLIPTokenizer:
    """Load CLIP BPE tokenizer without WordPiece-style space cleanup warnings."""
    kwargs.setdefault("clean_up_tokenization_spaces", False)
    return CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


def _dataloader_batch_size_per_process(
    global_batch_size: int, num_processes: int
) -> int:
    """Per-process batch size for ``DataLoader`` under Accelerate data parallelism."""
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    q, r = divmod(global_batch_size, num_processes)
    if q == 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be >= num_processes ({num_processes}), otherwise each GPU's batch_size is 0. Increase --global-batch-size (e.g. set it to at least the total GPU count, or BATCH_SIZE_PER_GPU * num_processes in your launch script)."
        )
    if r != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by num_processes ({num_processes}); remainder is {r}."
        )
    return q


def collate_qwenimage_text_embeds(
    batch: List[Tuple[torch.Tensor, Dict]],
    seq_key: str = "encoder_hidden_states",
    max_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List]]]:
    """Collate image + QwenImage-style ragged text embeddings to fixed ``(B, Lmax, D)``.

    Each sample's ``.pt`` dict should contain ``encoder_hidden_states`` (or ``seq_key``)
    with shape ``[L, D]`` (e.g. ``[30, 3584]`` and ``[42, 3584]``). Pads sequence dim
    with zeros to ``max(L)`` in the batch (or ``max_seq_len`` if set, with truncation).

    Returns
    -------
    images : Tensor, shape ``(B, C, H, W)``
    labels : dict with
        - ``encoder_hidden_states`` : ``(B, Lmax, D)``, dtype same as inputs
        - ``text_attention_mask`` : ``(B, Lmax)``, ``torch.long``, 1 = valid token, 0 = pad
        - ``label`` : list of optional string labels when present in each dict
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    dicts = [item[1] for item in batch]
    seq_list = []
    for d in dicts:
        if seq_key not in d:
            raise KeyError(
                f"Expected key '{seq_key}' in embed dict, got {set(d.keys())}"
            )
        t = d[seq_key]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        if t.dim() != 2:
            raise ValueError(
                f"Expected {seq_key} of shape [L, D], got {tuple(t.shape)}"
            )
        if max_seq_len is not None and t.shape[0] > max_seq_len:
            t = t[:max_seq_len].contiguous()
        seq_list.append(t)
    lengths = [t.shape[0] for t in seq_list]
    Lmax = max(lengths)
    if max_seq_len is not None:
        Lmax = min(Lmax, max_seq_len)
    D = seq_list[0].shape[1]
    dtype = seq_list[0].dtype
    device = seq_list[0].device
    padded = torch.zeros(len(seq_list), Lmax, D, dtype=dtype, device=device)
    text_attention_mask = torch.zeros(
        len(seq_list), Lmax, dtype=torch.long, device=device
    )
    for i, t in enumerate(seq_list):
        Li = min(t.shape[0], Lmax)
        padded[i, :Li].copy_(t[:Li])
        text_attention_mask[i, :Li] = 1
    out: Dict[str, Union[torch.Tensor, List]] = {
        "encoder_hidden_states": padded,
        "encoder_hidden_states_mask": text_attention_mask,
    }
    if all(("label" in d for d in dicts)):
        out["label"] = [d["label"] for d in dicts]
    return (images, out)


@torch.no_grad()
def decode_latent(latents, vae, output_type="pil"):
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    vae_scale_factor = (
        2 ** (len(vae.config.block_out_channels) - 1)
        if hasattr(vae, "config") and vae.config is not None
        else 8
    )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.postprocess(image, output_type=output_type)
    return image


@torch.no_grad()
def encode_latent(vae, image):
    latents = vae.encode(image, return_dict=False)[0].sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


@torch.no_grad()
def _get_t5_prompt_embeds(
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 256,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    tokenizer_max_length: Optional[int] = None,
    tokenizer_3: Optional[T5TokenizerFast] = None,
    text_encoder_3: Optional[T5EncoderModel] = None,
):
    dtype = dtype or text_encoder_3.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder_3(text_input_ids.to(device))[0]
    dtype = text_encoder_3.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


@torch.no_grad()
def _get_clip_prompt_embeds(
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    clip_skip: Optional[int] = None,
    clip_model_index: int = 0,
    tokenizer_max_length: Optional[int] = None,
    tokenizer: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[CLIPTextModelWithProjection] = None,
    tokenizer_2: Optional[CLIPTokenizer] = None,
    text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
):
    clip_tokenizers = [tokenizer, tokenizer_2]
    clip_text_encoders = [text_encoder, text_encoder_2]
    tokenizer = clip_tokenizers[clip_model_index]
    text_encoder = clip_text_encoders[clip_model_index]
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, -1
    )
    return (prompt_embeds, pooled_prompt_embeds)


@torch.no_grad()
def encode_prompt(
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    prompt_3: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    clip_skip: Optional[int] = None,
    max_sequence_length: int = 256,
    lora_scale: Optional[float] = None,
    tokenizer: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[CLIPTextModelWithProjection] = None,
    tokenizer_2: Optional[CLIPTokenizer] = None,
    text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
    tokenizer_3: Optional[BaseImageProcessor] = None,
    text_encoder_3: Optional[PreTrainedModel] = None,
):
    """
    Encode a batch of prompts into model conditioning embeddings.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    tokenizer_max_length = tokenizer.model_max_length
    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        prompt_3 = prompt_3 or prompt
        prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3
        prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=0,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
        )
        prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=1,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        t5_prompt_embed = _get_t5_prompt_embeds(
            prompt=prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_3=tokenizer_3,
            text_encoder_3=text_encoder_3,
        )
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embed, pooled_prompt_2_embed], dim=-1
        )
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt
        negative_prompt_3 = negative_prompt_3 or negative_prompt
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )
        negative_prompt_2 = (
            batch_size * [negative_prompt_2]
            if isinstance(negative_prompt_2, str)
            else negative_prompt_2
        )
        negative_prompt_3 = (
            batch_size * [negative_prompt_3]
            if isinstance(negative_prompt_3, str)
            else negative_prompt_3
        )
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
            )
        negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
            negative_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=None,
            clip_model_index=0,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
        )
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = (
            _get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
                tokenizer_max_length=tokenizer_max_length,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                tokenizer_2=tokenizer_2,
                text_encoder_2=text_encoder_2,
            )
        )
        negative_clip_prompt_embeds = torch.cat(
            [negative_prompt_embed, negative_prompt_2_embed], dim=-1
        )
        t5_negative_prompt_embed = _get_t5_prompt_embeds(
            prompt=negative_prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_3=tokenizer_3,
            text_encoder_3=text_encoder_3,
        )
        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (
                0,
                t5_negative_prompt_embed.shape[-1]
                - negative_clip_prompt_embeds.shape[-1],
            ),
        )
        negative_prompt_embeds = torch.cat(
            [negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2
        )
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )
    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


class BLIP3oDataset(Dataset):
    """BLIP3o JSONL records with image_path and caption fields."""

    def __init__(self, data_file, image_root, image_size=1024, text_embed_root=None):
        self.records = load_dataset("json", data_files=data_file, split="train")
        if not len(self.records):
            raise ValueError("The BLIP3o JSONL file contains no samples.")
        missing = {"image_path", "caption"} - set(self.records.column_names)
        if missing:
            raise ValueError(f"BLIP3o JSONL is missing fields: {sorted(missing)}")
        self.image_root = Path(image_root)
        self.text_embed_root = Path(text_embed_root) if text_embed_root else None
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if not isinstance(row["image_path"], str) or not row["image_path"].strip():
            raise ValueError(
                f"BLIP3o record {index}: image_path must be a nonempty string."
            )
        if not isinstance(row["caption"], str) or not row["caption"].strip():
            raise ValueError(
                f"BLIP3o record {index}: caption must be a nonempty string."
            )
        relative_path = Path(row["image_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"BLIP3o record {index}: image_path must be relative to image_root."
            )
        with Image.open(self.image_root / relative_path) as image:
            image = self.transform(image.convert("RGB"))
        if self.text_embed_root is not None:
            embedding_path = self.text_embed_root / relative_path.with_suffix(".pt")
            return image, torch.load(
                embedding_path, weights_only=True, map_location="cpu"
            )
        return image, row["caption"].strip()


def get_blip3o_dataset(
    args,
    data_file,
    image_root,
    accelerator,
    text_embed_root=None,
    qwenimage_text_embed_root=None,
):
    """Build the shared BLIP3o loader for SD3.5, FLUX and QwenImage training."""
    if text_embed_root is not None and qwenimage_text_embed_root is not None:
        raise ValueError("Specify only one text embedding root.")
    dataset = BLIP3oDataset(
        data_file,
        image_root,
        args.image_size,
        text_embed_root=text_embed_root or qwenimage_text_embed_root,
    )
    batch_size = _dataloader_batch_size_per_process(
        args.global_batch_size, accelerator.num_processes
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_qwenimage_text_embeds if qwenimage_text_embed_root else None,
    )
    return loader, dataset, len(loader)
