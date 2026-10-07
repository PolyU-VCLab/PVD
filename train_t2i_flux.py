# PVD FLUX training.
# PVD text-to-image extension; training setup, resume and logging helpers are adapted from FACM train.py (https://github.com/ali-vilab/FACM)

from utils.weight_io import load_checkpoint
import os
import argparse
import time
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from ema_pytorch import PostHocEMA
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL
import torchvision
import gc
from ldit.discrimator_flux import DiTDiscriminator
from ldit.transformer_flux_distillation import FluxTransformer2DModel_Distill
from ldit.transformer_flux import FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from losses.flux import DiscriminatorLossFLUXT2I, PVDLossFLUX
from utils import log, RandomStateManager, load_ckpt
from typing import List, Optional, Union
from transformers import CLIPTextModel, T5TokenizerFast, T5EncoderModel
from utils.t2i import get_blip3o_dataset, load_clip_tokenizer
from peft import PeftModel


def cleanup_memory():
    """Run occasional host/GPU memory cleanup after heavy ops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_ckpt(
    args,
    model,
    emas,
    opt,
    epoch,
    step,
    checkpoint_dir,
    accelerator,
    scheduler=None,
    freezed_teacher=None,
):
    """Save checkpoint, including scheduler state"""
    if accelerator.is_main_process:
        base_model = accelerator.unwrap_model(model)
        checkpoint = {
            "model": base_model.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "epoch": epoch,
            "step": step,
        }
        if emas is not None:
            checkpoint["emas"] = emas.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if freezed_teacher is not None:
            teacher_model = accelerator.unwrap_model(freezed_teacher)
            checkpoint["teacher_heads"] = teacher_model.heads.state_dict()
        checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
        accelerator.save(checkpoint, checkpoint_path)
        accelerator.save(checkpoint, f"{checkpoint_dir}/latest.pt")
        log(f"Checkpoint saved to {checkpoint_path}", accelerator)
        if emas is not None:
            emas.checkpoint()
    accelerator.wait_for_everyone()


def load_teacher_heads_ckpt(args, freezed_teacher, accelerator):
    """Load discriminator heads from checkpoint when available."""
    if freezed_teacher is None:
        return
    ckpt_path = (
        getattr(args, "teacher_heads_ckpt", None)
        or args.ckpt_path
        or getattr(args, "init_ckpt", None)
    )
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        log(f"Teacher-head load skipped (ckpt not found): {ckpt_path}", accelerator)
        return
    ckpt = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    teacher_heads_state = ckpt.get("teacher_heads", None)
    if teacher_heads_state is None:
        log(
            "Teacher-head state not found in checkpoint; using current initialization.",
            accelerator,
        )
        return
    teacher_model = accelerator.unwrap_model(freezed_teacher)
    normalized_state = {}
    for key, value in teacher_heads_state.items():
        if key.startswith("module."):
            normalized_state[key[len("module.") :]] = value
        else:
            normalized_state[key] = value
    missing_keys, unexpected_keys = teacher_model.heads.load_state_dict(
        normalized_state, strict=False
    )
    log(f"Loaded teacher heads from {ckpt_path}", accelerator)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        log(f"Teacher heads missing keys: {missing_keys}", accelerator)
        log(f"Teacher heads unexpected keys: {unexpected_keys}", accelerator)


def _prepare_latent_image_ids(height, width):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids


@torch.no_grad()
def encode_prompt(
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
    tokenizer=None,
    text_encoder=None,
    tokenizer_2=None,
    text_encoder_2=None,
):
    """
    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in all text-encoders
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        pooled_prompt_embeds = _get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        prompt_embeds = _get_t5_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
        )
    dtype = text_encoder.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return (prompt_embeds, pooled_prompt_embeds, text_ids)


def _get_t5_prompt_embeds(
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    tokenizer_2=None,
    text_encoder_2=None,
):
    dtype = text_encoder_2.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder_2(
        text_input_ids.to(device), output_hidden_states=False
    )[0]
    dtype = text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _get_clip_prompt_embeds(
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    tokenizer=None,
    text_encoder=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def _pack_latents(
    latents, batch_size, num_channels_latents, height, width, vae_scale_factor=8
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, height // 2 * (width // 2), num_channels_latents * 4
    )
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor=8):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


@torch.no_grad()
def decode_latent(vae, latents, height, width, vae_scale_factor=8, output_type="pil"):
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    image = image_processor.postprocess(image, output_type=output_type)
    return image


@torch.no_grad()
def encode_image(vae, image, height, width, vae_scale_factor=8):
    latents = vae.encode(image, return_dict=False)[0].sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = _pack_latents(latents, latents.shape[0], latents.shape[1], height, width)
    return latents


def setup_experiment(args, accelerator):
    """Setup experiment directories and logging"""
    checkpoint_dir = f"{args.results_dir}/checkpoint"
    visualize_dir = f"{args.results_dir}/visualize"
    ema_checkpoint_dir = f"{checkpoint_dir}/ema_checkpoints"
    if accelerator.is_main_process:
        log_file = f"{args.results_dir}/training_log.txt"
        log.log_file = log_file
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        if args.use_ema:
            os.makedirs(ema_checkpoint_dir, exist_ok=True)
        log(f"Created experiment directory at {args.results_dir}", accelerator)
    return (checkpoint_dir, visualize_dir, ema_checkpoint_dir)


def auto_resume_checkpoint(args, accelerator):
    """Auto-resume from latest checkpoint if exists"""
    latest_ckpt = os.path.join(args.results_dir, "checkpoint", "latest.pt")
    if os.path.exists(latest_ckpt):
        original_ckpt_path = args.ckpt_path
        args.ckpt_path = latest_ckpt
        log(f">>>>>> Auto-resuming from {args.ckpt_path} <<<<<<", accelerator)
        return original_ckpt_path
    return args.ckpt_path


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Normalize DDP / compile prefixes from saved checkpoints."""
    out = {}
    for key, value in state_dict.items():
        k = key
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = value
    return out


def _load_dit_init_state_dict(ckpt_path: str) -> dict:
    """
    Load DiT weights for ``create_model_and_ema``.

    Supports training checkpoints (``latest.pt`` with ``model``) or raw state dict.
    """
    raw = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw:
        state_dict = raw["model"]
    elif isinstance(raw, dict) and "ema" in raw:
        state_dict = raw["ema"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise TypeError(f"Unsupported checkpoint type at {ckpt_path}: {type(raw)}")
    return _strip_state_dict_prefixes(state_dict)


def _default_peft_adapter_path(adapter_name: str) -> str:
    """Default PEFT adapter dir under weights/pvd_flux/."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "weights", "pvd_flux", adapter_name)


def _resolve_peft_adapter_path(args) -> tuple[str, str]:
    adapter_name = (
        getattr(args, "peft_adapter_name", None) or "part1"
    ).strip() or "part1"
    adapter_path = getattr(args, "peft_adapter_path", None)
    if not adapter_path:
        adapter_path = _default_peft_adapter_path(adapter_name)
    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"PEFT adapter directory not found: {adapter_path}. Pass --peft-adapter-path or ensure weights/pvd_flux/{{name}} exists."
        )
    return (adapter_name, adapter_path)


def create_model(args, accelerator):
    model = FluxTransformer2DModel_Distill(
        patch_size=1,
        in_channels=64,
        num_layers=9,
        num_single_layers=19,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        enable_end_timestep=True,
    )
    model.enable_fused_attn()
    from utils.weight_io import load_weights

    model.load_state_dict(
        load_weights(
            os.environ.get("FLUX_BACKBONE", "weights/pvd_flux/backbone.safetensors")
        ),
        strict=True,
    )
    adapter_name, adapter_path = _resolve_peft_adapter_path(args)
    model = PeftModel.from_pretrained(
        model, adapter_path, adapter_name=adapter_name, is_trainable=True
    )
    model.set_adapter(adapter_name)
    init_ckpt = (getattr(args, "init_ckpt", None) or "").strip() or None
    if init_ckpt:
        if not os.path.isabs(init_ckpt):
            init_ckpt = os.path.abspath(init_ckpt)
        if not os.path.isfile(init_ckpt):
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        state_dict = _load_dit_init_state_dict(init_ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        log(f"Loaded init weights from {init_ckpt}", accelerator)
        if missing_keys:
            log(
                f"Init missing keys ({len(missing_keys)}): {missing_keys[:8]}...",
                accelerator,
            )
        if unexpected_keys:
            log(
                f"Init unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:8]}...",
                accelerator,
            )
    else:
        log(
            f"No --init-ckpt; student uses PEFT adapter weights from {adapter_path} only.",
            accelerator,
        )
    trainable, total = model.get_nb_trainable_parameters()
    log(
        f"PEFT LoRA trainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.4f}% trainable)",
        accelerator,
    )
    log(
        f"DiT Trainable Parameters: {sum((p.numel() for p in model.parameters() if p.requires_grad)):,}",
        accelerator,
    )
    return model


def setup_training_components(model, freezed_teacher, args):
    """Setup optimizer and scheduler"""
    student_params = [p for p in model.parameters() if p.requires_grad]
    disc_lr = getattr(args, "disc_lr", 4e-06)
    if freezed_teacher is not None:
        disc_params = list(freezed_teacher.heads.parameters())
        param_groups = [
            {"params": student_params, "lr": args.lr},
            {"params": disc_params, "lr": disc_lr},
        ]
    else:
        param_groups = [{"params": student_params, "lr": args.lr}]
    opt = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta),
        eps=args.eps,
    )
    scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=opt,
        num_warmup_steps=20000,
        num_training_steps=args.max_steps,
    )
    facm_loss = PVDLossFLUX()
    disc_teacher_loss = DiscriminatorLossFLUXT2I()
    return (opt, scheduler, facm_loss, disc_teacher_loss)


def setup_teacher_model(args, accelerator):
    """Setup teacher model for distillation"""
    if not args.distill:
        return None
    model_root = getattr(
        args, "flux_model_root", os.environ.get("FLUX_MODEL_ROOT", "models/FLUX.1-dev")
    )
    freezed_teacher = DiTDiscriminator(
        FluxTransformer2DModel.from_pretrained(
            model_root, subfolder="transformer", torch_dtype=torch.bfloat16
        ),
        adv_index=[2, 10, 18, 21, 39, 56],
    )
    for param in freezed_teacher.parameters():
        param.requires_grad = False
    freezed_teacher.eval()
    freezed_teacher.heads.to(torch.bfloat16)
    freezed_teacher.heads.requires_grad_(True)
    freezed_teacher.heads.train()
    return freezed_teacher


@torch.no_grad()
def evaluate(
    args, model, vae, accelerator, train_steps, visualize_dir, noise, cond, cfgw
):
    """Evaluate the model"""
    log(f"Evaluating at step {train_steps}...", accelerator)
    model.eval()
    batch_size = noise.shape[0]
    intervals = [float(i) for i in args.intervals.split("_")]
    timesteps = torch.tensor(intervals, device=noise.device)
    t, r = (timesteps[0], timesteps[1])
    input_kwargs = {
        "hidden_states": noise,
        "encoder_hidden_states": cond["encoder_hidden_states"],
        "pooled_projections": cond["pooled_projections"],
        "timestep": t.repeat(batch_size),
        "end_timestep": (t - r).expand(batch_size),
        "img_ids": cond["img_ids"],
        "txt_ids": cond["txt_ids"],
        "guidance": cond["guidance"],
        "return_dict": False,
    }
    pred_v = model(**input_kwargs)[0]
    sampled_x = noise + pred_v * (r - t).view(-1, 1, 1)
    _x_hat = (
        decode_latent(
            vae,
            sampled_x.bfloat16(),
            args.image_size,
            args.image_size,
            output_type="latent",
        )
        + 1
    ) / 2
    x_hat_gathered = accelerator.gather(_x_hat)
    if accelerator.is_main_process:
        x_hat_gathered = x_hat_gathered.clamp(0, 1)
        img = torchvision.utils.make_grid(x_hat_gathered[:64], nrow=8)
        img = torchvision.transforms.functional.to_pil_image(img.cpu().float())
        img.save(f"{visualize_dir}/cfgw{cfgw}_steps{train_steps}.png")
    log(
        f"(steps={train_steps}) images saved to {visualize_dir}/cfgw{cfgw}_steps{train_steps}.png",
        accelerator,
    )
    return 0


def perform_evaluation_and_checkpointing(
    args,
    model,
    emas,
    vae,
    accelerator,
    train_steps,
    visualize_dir,
    eval_noise,
    eval_cond,
    start_step,
    checkpoint_dir,
    opt,
    epoch,
    scheduler,
    freezed_teacher=None,
):
    """Handle evaluation and checkpointing during training"""
    cfgw = args.cfgw
    did_heavy_work = False
    if train_steps % args.ckpt_every == 0 and train_steps > start_step:
        save_ckpt(
            args,
            model,
            emas,
            opt,
            epoch,
            train_steps,
            checkpoint_dir,
            accelerator,
            scheduler=scheduler,
            freezed_teacher=freezed_teacher,
        )
        did_heavy_work = True
    if args.eval_every != -1 and train_steps % args.eval_every == 0:
        eval_model = emas.ema_models[0] if args.use_ema and emas is not None else model
        if args.use_ema and emas is not None:
            eval_model.to(accelerator.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with RandomStateManager(
                eval_seed=args.global_seed + accelerator.process_index
            ):
                evaluate(
                    args,
                    eval_model,
                    vae,
                    accelerator,
                    train_steps,
                    visualize_dir,
                    noise=eval_noise,
                    cond=eval_cond,
                    cfgw=cfgw,
                )
        if args.use_ema and emas is not None:
            eval_model.to("cpu")
        did_heavy_work = True
    if did_heavy_work:
        cleanup_memory()


def training_loop(
    args,
    model,
    emas,
    emas_model,
    opt,
    scheduler,
    facm_loss,
    loader_train,
    accelerator,
    start_epoch,
    start_step,
    device,
    eval_noise,
    eval_cond,
    vae,
    visualize_dir,
    checkpoint_dir,
    freezed_teacher,
    disc_teacher_loss,
):
    """Main training loop"""
    use_precomputed_text = getattr(args, "text_embed_root", None) is not None
    tokenizer = tokenizer_2 = None
    text_encoder = text_encoder_2 = None
    if not use_precomputed_text:
        model_root = getattr(
            args,
            "flux_model_root",
            os.environ.get("FLUX_MODEL_ROOT", "models/FLUX.1-dev"),
        )
        tokenizer = load_clip_tokenizer(f"{model_root}/tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            f"{model_root}/text_encoder", torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer_2 = T5TokenizerFast.from_pretrained(f"{model_root}/tokenizer_2")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            f"{model_root}/text_encoder_2", torch_dtype=torch.bfloat16
        ).to(device)
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        log(
            f"Online FLUX text encoding (CLIP+T5) from {model_root}; max_seq_len={getattr(args, 'max_seq_len', 512)}",
            accelerator,
        )
    else:
        log(
            f"Using precomputed text embeddings under {args.text_embed_root}",
            accelerator,
        )
    img_ids = _prepare_latent_image_ids(
        args.image_size // 8 // 2, args.image_size // 8 // 2
    ).to(device)
    if eval_cond is not None:
        eval_cond.update({"img_ids": img_ids})
    generator_update_interval = args.generator_update_interval
    discriminator_update_interval = args.discriminator_update_interval
    has_discriminator = freezed_teacher is not None
    epoch, train_steps = (start_epoch, start_step)
    teacher_train_steps = 0
    train_generator = True
    student_log_steps, teacher_log_steps, running_loss = (0, 0, 0)
    teacher_running_loss, teacher_running_fake_loss, teacher_running_real_loss = (
        0,
        0,
        0,
    )
    running_cm_loss, running_fm_loss = (0, 0)
    running_layer_losses = []
    student_start_time, teacher_start_time = (time.time(), time.time())
    student_grad_norm, teacher_grad_norm = (0.0, 0.0)
    intervals = [float(i) for i in args.intervals.split("_")]
    ema_reference_model = (
        emas.ema_models[1] if args.use_ema and emas is not None else model
    )
    while train_steps < args.max_steps:
        log(f"Starting epoch {epoch}...", accelerator)
        for data in loader_train:
            if train_steps >= args.max_steps:
                break
            with torch.no_grad():
                images, labels = data
                x = encode_image(
                    vae, images.bfloat16(), args.image_size, args.image_size
                )
                if isinstance(labels, dict):
                    encoder_hidden_states = labels["prompt_embeds"].to(
                        device=device, dtype=torch.bfloat16
                    )
                    pooled_projections = labels["pooled_embeds"].to(
                        device=device, dtype=torch.bfloat16
                    )
                else:
                    captions = labels
                    if isinstance(captions, str):
                        captions = [captions]
                    elif not isinstance(captions, (list, tuple)):
                        captions = list(captions)
                    max_seq = getattr(args, "max_seq_len", 512)
                    prompt_embeds, pooled_embeds, _ = encode_prompt(
                        prompt=list(captions),
                        prompt_2=list(captions),
                        device=device,
                        max_sequence_length=max_seq,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        tokenizer_2=tokenizer_2,
                        text_encoder_2=text_encoder_2,
                    )
                    encoder_hidden_states = prompt_embeds.to(dtype=torch.bfloat16)
                    pooled_projections = pooled_embeds.to(dtype=torch.bfloat16)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                    device=device
                )
                guidance = torch.tensor([args.cfgw] * images.shape[0]).to(device)
                y = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                    "txt_ids": txt_ids,
                    "img_ids": img_ids,
                    "guidance": guidance,
                }
                un_y = {}
                model_kwargs = dict(y=y, un_y=un_y)
            if train_generator or not has_discriminator:
                if has_discriminator:
                    freezed_teacher.heads.eval()
                    freezed_teacher.heads.requires_grad_(False)
                model.train()
                with accelerator.accumulate(model):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, cm_loss, fm_loss, layer_losses = facm_loss(
                            accelerator,
                            model,
                            model,
                            ema_reference_model,
                            x,
                            train_steps,
                            model_kwargs=model_kwargs,
                            args=args,
                            freezed_teacher=freezed_teacher,
                            epoch=epoch,
                            time_interval=intervals,
                        )
                        loss = torch.nan_to_num(loss, nan=0.0)
                    accelerator.backward(loss)
                    running_loss += loss.item()
                    running_cm_loss += cm_loss.item()
                    running_fm_loss += fm_loss.item()
                    running_layer_losses.append(layer_losses)
                    student_log_steps += 1
                    if accelerator.sync_gradients:
                        student_grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    opt.step()
                    opt.zero_grad()
                    if accelerator.sync_gradients:
                        scheduler.step()
                        if (
                            args.use_ema
                            and emas is not None
                            and (emas_model is not None)
                        ):
                            emas.update()
                        train_steps += 1
                        perform_evaluation_and_checkpointing(
                            args,
                            model,
                            emas,
                            vae,
                            accelerator,
                            train_steps,
                            visualize_dir,
                            eval_noise,
                            eval_cond,
                            start_step,
                            checkpoint_dir,
                            opt,
                            epoch,
                            scheduler,
                            freezed_teacher=freezed_teacher,
                        )
                        if train_steps % args.log_every == 0:
                            _log_training_stats(
                                accelerator,
                                device,
                                args,
                                running_loss,
                                running_cm_loss,
                                running_fm_loss,
                                running_layer_losses,
                                student_log_steps,
                                student_start_time,
                                student_grad_norm,
                                train_steps,
                                scheduler,
                            )
                            running_loss, running_cm_loss, running_fm_loss = (0, 0, 0)
                            running_layer_losses = []
                            student_log_steps = 0
                            student_start_time = time.time()
                        if (
                            has_discriminator
                            and train_steps % generator_update_interval == 0
                        ):
                            train_generator = False
                            freezed_teacher.heads.requires_grad_(True)
                            freezed_teacher.heads.train()
            else:
                with accelerator.accumulate(freezed_teacher.heads):
                    model.eval()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        disc_loss, loss_real, loss_gen = disc_teacher_loss(
                            accelerator,
                            model,
                            model,
                            ema_reference_model,
                            x,
                            teacher_train_steps,
                            model_kwargs=model_kwargs,
                            args=args,
                            freezed_teacher=freezed_teacher,
                            epoch=epoch,
                            time_interval=intervals,
                        )
                    disc_loss = torch.nan_to_num(disc_loss, nan=0.0)
                    accelerator.backward(disc_loss)
                    if accelerator.sync_gradients:
                        teacher_grad_norm = accelerator.clip_grad_norm_(
                            freezed_teacher.heads.parameters(), args.max_grad_norm
                        )
                    opt.step()
                    opt.zero_grad()
                    teacher_log_steps += 1
                    teacher_running_fake_loss += loss_gen.item()
                    teacher_running_real_loss += loss_real.item()
                    teacher_running_loss += disc_loss.item()
                    if accelerator.sync_gradients:
                        teacher_train_steps += 1
                        if teacher_train_steps % args.log_every == 0:
                            end_time = time.time()
                            steps_per_sec = teacher_log_steps / (
                                end_time - teacher_start_time
                            )
                            log(
                                f"(step={teacher_train_steps}) loss_disc: {teacher_running_loss / teacher_log_steps:.4f}, loss_real: {teacher_running_real_loss / teacher_log_steps:.4f}, loss_gen: {teacher_running_fake_loss / teacher_log_steps:.4f}, steps/sec: {steps_per_sec:.2f}, grad_norm: {teacher_grad_norm:.4f}",
                                accelerator,
                            )
                            (
                                teacher_running_loss,
                                teacher_running_fake_loss,
                                teacher_running_real_loss,
                            ) = (0, 0, 0)
                            teacher_log_steps = 0
                            teacher_start_time = time.time()
                        if teacher_train_steps % discriminator_update_interval == 0:
                            train_generator = True
            cleanup_memory()
        epoch += 1


def _log_training_stats(
    accelerator,
    device,
    args,
    running_loss,
    running_cm_loss,
    running_fm_loss,
    running_layer_losses,
    log_steps,
    start_time,
    grad_norm,
    train_steps,
    scheduler,
):
    """Log training statistics"""
    torch.cuda.synchronize()
    end_time = time.time()
    steps_per_sec = log_steps / (end_time - start_time)
    loss_names = ["loss", "loss_cm", "loss_fm"]
    running_losses = [running_loss, running_cm_loss, running_fm_loss]
    avg_losses = {}
    for name, running_val in zip(loss_names, running_losses):
        avg_val = torch.tensor(running_val / log_steps, device=device)
        avg_val = accelerator.gather(avg_val).sum() / accelerator.num_processes
        avg_losses[name] = avg_val.item()
    avg_loss = [avg_losses[name] for name in loss_names]
    layer_loss = torch.stack(
        [torch.tensor(x, device=device) for x in running_layer_losses]
    )
    avg_layer_loss = layer_loss.mean(dim=0)
    avg_layer_loss = (
        accelerator.gather(avg_layer_loss).view(-1, avg_layer_loss.shape[0]).mean(dim=0)
    )
    lrs = scheduler.get_last_lr()
    lr_str = ", ".join((f"{x:.2e}" for x in lrs))
    log(
        f"(step={train_steps}) loss_cm: {avg_loss[1]:.4f}, loss_fm: {avg_loss[2]:.4f}, steps/sec: {steps_per_sec:.2f}, grad_norm: {grad_norm:.4f}, layer loss: {[f'{x:.4f}' for x in avg_layer_loss]}, lr: [{lr_str}]",
        accelerator,
    )


class ModelPair(torch.nn.Module):

    def __init__(self, model, emas, freezed_teacher, vae):
        super().__init__()
        self.model = model
        self.emas = emas
        self.freezed_teacher = freezed_teacher
        self.vae = vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.emas is not None:
            self.emas.eval()
            self.emas.requires_grad_(False)


def main(args):
    """Main training function"""
    ds_file = "./scripts/config/zero_stage2_config.json"
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_file)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation, deepspeed_plugin=deepspeed_plugin
    )
    device = accelerator.device
    set_seed(args.global_seed + accelerator.process_index)
    checkpoint_dir, visualize_dir, ema_checkpoint_dir = setup_experiment(
        args, accelerator
    )
    auto_resume_checkpoint(args, accelerator)
    log(
        "Args:\n" + "\n".join([f"\t{arg}: {getattr(args, arg)}" for arg in vars(args)]),
        accelerator,
    )
    text_embed_root = getattr(args, "text_embed_root", None)
    loader_train, dataset_train, loader_len = get_blip3o_dataset(
        args,
        args.data_file,
        args.image_root,
        accelerator,
        text_embed_root=text_embed_root,
    )
    if text_embed_root:
        log(
            f"Dataset with precomputed FLUX text embeds: {text_embed_root}", accelerator
        )
    else:
        log(
            "Dataset without --text-embed-root; training uses online FLUX CLIP+T5 each step.",
            accelerator,
        )
    model_root = getattr(
        args, "flux_model_root", os.environ.get("FLUX_MODEL_ROOT", "models/FLUX.1-dev")
    )
    vae = AutoencoderKL.from_pretrained(f"{model_root}/vae", torch_dtype=torch.bfloat16)
    model = create_model(args, accelerator)
    emas_model = None
    emas = None
    freezed_teacher = setup_teacher_model(args, accelerator)
    if args.use_ema:
        emas_model = model
        emas = PostHocEMA(
            emas_model,
            sigma_rels=[0.05, args.sigma_rel],
            update_every=1,
            checkpoint_every_num_steps=5000,
            checkpoint_folder=ema_checkpoint_dir,
            allow_different_devices=True,
            move_ema_to_online_device=False,
        )
        emas.eval()
    modelpair = ModelPair(model, emas, freezed_teacher, vae)
    opt, scheduler, facm_loss, disc_teacher_loss = setup_training_components(
        modelpair.model, modelpair.freezed_teacher, args
    )
    modelpair, scheduler, loader_train, opt = accelerator.prepare(
        modelpair, scheduler, loader_train, opt
    )
    start_epoch, start_step = load_ckpt(
        args, modelpair.model, modelpair.emas, opt, accelerator, scheduler=scheduler
    )
    load_teacher_heads_ckpt(args, modelpair.freezed_teacher, accelerator)
    if args.eval_every != -1:
        eval_prompts = [
            "A parrot standing on a tree branch.",
            "Two red apples on a wooden desk.",
            "A long hair cat sitting on the windowsill",
        ]
        eval_noise = torch.randn(
            (
                len(eval_prompts),
                args.image_size // (8 * 2) * args.image_size // (8 * 2),
                64,
            ),
            device=device,
        )
        eval_cond = load_checkpoint(
            "cache/flux1_eval_prompt_embeds.pt", weights_only=True, map_location=device
        )
        eval_cond.update(
            {"guidance": torch.tensor([args.cfgw] * len(eval_prompts)).to(device)}
        )
    else:
        eval_prompts = None
        eval_noise = None
        eval_cond = None
    log(f"Training until max_steps={args.max_steps}", accelerator)
    if args.use_ema:
        modelpair.emas.to("cpu")
    training_loop(
        args,
        modelpair.model,
        modelpair.emas,
        emas_model,
        opt,
        scheduler,
        facm_loss,
        loader_train,
        accelerator,
        start_epoch,
        start_step,
        device,
        eval_noise,
        eval_cond,
        modelpair.vae,
        visualize_dir,
        checkpoint_dir,
        modelpair.freezed_teacher,
        disc_teacher_loss,
    )
    log("Training complete", accelerator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="output")
    parser.add_argument(
        "--data-file",
        default="data/blip3o/train.jsonl",
        help="BLIP3o JSONL with image_path and caption fields.",
    )
    parser.add_argument("--image-root", default="data/blip3o/images")
    parser.add_argument(
        "--text-embed-root",
        type=str,
        default=None,
        dest="text_embed_root",
        help="BLIP3o FLUX embeddings: mirror image paths under this root with a .pt suffix. Omit to encode captions online.",
    )
    parser.add_argument(
        "--flux-model-root",
        type=str,
        default=os.environ.get("FLUX_MODEL_ROOT", "models/FLUX.1-dev"),
        dest="flux_model_root",
        help="Directory with FLUX tokenizer, text_encoder, tokenizer_2, text_encoder_2, vae.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        dest="max_seq_len",
        help="T5 max sequence length for online text encoding (when --text-embed-root is not set).",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--sigma-rel", type=float, default=0.2)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMA teacher tracking. Use --no-use-ema to disable.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=4e-06,
        dest="disc_lr",
        help="AdamW lr for DiTDiscriminator heads only (student uses --lr).",
    )
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="Flux student init weights in create_model_and_ema (latest.pt with PeftModel state). Set empty string to skip. Separate from --ckpt-path resume.",
    )
    parser.add_argument(
        "--teacher-heads-ckpt",
        type=str,
        default=None,
        dest="teacher_heads_ckpt",
        help="Checkpoint containing teacher_heads for discriminator init. Defaults to --init-ckpt when unset. Does not resume training step.",
    )
    parser.add_argument(
        "--peft-adapter-path",
        type=str,
        default="",
        help="PEFT LoRA adapter directory before loading --init-ckpt. Default (empty): weights/pvd_flux/{--peft-adapter-name}.",
    )
    parser.add_argument(
        "--peft-adapter-name",
        type=str,
        default="part1",
        help="PEFT adapter name passed to PeftModel.set_adapter.",
    )
    parser.add_argument("--t-type", type=str, default="default")
    parser.add_argument("--cfgw", type=float, default=1.75)
    parser.add_argument("--mean", type=float, default=0.8)
    parser.add_argument("--std", type=float, default=1.6)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--distill", action="store_true", default=False)
    parser.add_argument("--max-steps", type=float, default=float("inf"))
    parser.add_argument("--intervals", type=str, default="0.6_0.0")
    parser.add_argument(
        "--adv-weight",
        type=float,
        default=0.1,
        dest="adv_weight",
        help="Weight on generator adversarial term (-E[D(fake)]). Lower (e.g. 0.05–0.1) if G collapses.",
    )
    parser.add_argument(
        "--generator_update_interval",
        type=int,
        default=1,
        help="Generator update interval",
    )
    parser.add_argument(
        "--discriminator_update_interval",
        type=int,
        default=1,
        help="Discriminator update interval",
    )
    args = parser.parse_args()
    main(args)
