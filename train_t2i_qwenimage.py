# PVD Qwen-Image training.
# PVD text-to-image extension; training setup, resume and logging helpers are adapted from FACM train.py (https://github.com/ali-vilab/FACM)

from utils.weight_io import load_checkpoint
import os
import argparse
import json
import time
from contextlib import nullcontext
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from ema_pytorch import KarrasEMA, PostHocEMA
from diffusers.optimization import get_scheduler
from diffusers import QwenImageTransformer2DModel
import torchvision
import gc
from diffusers.models import AutoencoderKLQwenImage
from ldit.transformer_qwenimage_distillation import QwenImageTransformer2DModel_Distill
from ldit.discrimator_qwenimage import DiTDiscriminator
from diffusers.image_processor import VaeImageProcessor
from losses.qwenimage import DiscriminatorLossQwenImage, PVDLossQwenImage
from utils import log, RandomStateManager, _load_optimizer_state, _load_scheduler_state
from utils.t2i import get_blip3o_dataset
from peft import LoraConfig, get_peft_model

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED_TRANSFORMER_PATH = os.environ.get(
    "QWENIMAGE_TRANSFORMER_ROOT", "weights/pvd_qwenimage/backbone.safetensors"
)


def _normalize_text_embed_root(value):
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return os.path.expanduser(os.fspath(value))


def _load_online_text_encoder(args, device, accelerator):
    if _normalize_text_embed_root(args.qwenimage_text_embed_root) is not None:
        log(
            f"Using precomputed QwenImage text embeddings under {args.qwenimage_text_embed_root}",
            accelerator,
        )
    from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = Qwen2Tokenizer.from_pretrained(
        os.path.join(args.model_root, "tokenizer")
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(args.model_root, "text_encoder"), torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    log(f"Online QwenImage text encoding from {args.model_root}", accelerator)
    return (tokenizer, text_encoder)


@torch.no_grad()
def _prepare_text_condition(
    labels, device, img_shapes, tokenizer=None, text_encoder=None
):
    if isinstance(labels, dict):
        embeddings = labels["encoder_hidden_states"].to(
            device=device, dtype=torch.bfloat16
        )
        mask = labels.get("encoder_hidden_states_mask")
    else:
        if tokenizer is None or text_encoder is None:
            raise RuntimeError(
                "Online text encoding requires a QwenImage tokenizer and text encoder."
            )
        from utils.qwenimage import encode_prompt

        embeddings, mask = encode_prompt(
            labels, device=device, tokenizer=tokenizer, text_encoder=text_encoder
        )
    return {
        "encoder_hidden_states": embeddings,
        "encoder_hidden_states_mask": (
            mask.to(device=device) if mask is not None else None
        ),
        "img_shapes": img_shapes,
    }


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
    accelerator.wait_for_everyone()
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
        if emas is not None and hasattr(emas, "checkpoint"):
            emas.checkpoint()
    accelerator.wait_for_everyone()


def _is_posthoc_ema(emas):
    return emas is not None and hasattr(emas, "ema_models")


def _get_eval_ema_model(emas):
    if emas is None:
        return None
    if _is_posthoc_ema(emas):
        return emas.ema_models[0].ema_model
    return emas.ema_model


def _get_reference_ema_model(emas):
    if emas is None:
        return None
    if _is_posthoc_ema(emas):
        return emas.ema_models[1].ema_model
    return emas.ema_model


def _log_incompatible_keys(prefix, incompatible, accelerator):
    if incompatible is None:
        return
    missing_keys = getattr(incompatible, "missing_keys", [])
    unexpected_keys = getattr(incompatible, "unexpected_keys", [])
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        log(f"{prefix} missing keys: {missing_keys}", accelerator)
        log(f"{prefix} unexpected keys: {unexpected_keys}", accelerator)


def _extract_single_ema_state_from_posthoc(posthoc_state, ema_index=1):
    prefix = f"ema_models.{ema_index}."
    single_state = {
        key[len(prefix) :]: value
        for (key, value) in posthoc_state.items()
        if key.startswith(prefix)
    }
    return single_state if len(single_state) > 0 else None


def _load_ema_state(args, checkpoint, emas, accelerator):
    """Load EMA state for either PostHocEMA or single KarrasEMA."""
    if emas is None:
        return
    if "emas" in checkpoint:
        ema_state = checkpoint["emas"]
        try:
            incompatible = emas.load_state_dict(ema_state, strict=False)
            ema_name = "PostHocEMA" if _is_posthoc_ema(emas) else "KarrasEMA"
            log(f"Loading {ema_name} state from checkpoint", accelerator)
            _log_incompatible_keys(ema_name, incompatible, accelerator)
            return
        except Exception as e:
            if not _is_posthoc_ema(emas):
                single_state = _extract_single_ema_state_from_posthoc(
                    ema_state, ema_index=1
                )
                if single_state is not None:
                    incompatible = emas.load_state_dict(single_state, strict=False)
                    log(
                        "Loaded PostHocEMA ema_models[1] into single KarrasEMA shadow",
                        accelerator,
                    )
                    _log_incompatible_keys("KarrasEMA", incompatible, accelerator)
                    return
            log(f"Warning: Cannot load EMA state directly: {e}", accelerator)
            log("Initializing EMA from current model weights", accelerator)
            emas.copy_params_from_model_to_ema()
            return
    if "ema" in checkpoint:
        log(
            "Detected old-style EMA checkpoint, attempting compatible loading...",
            accelerator,
        )
        try:
            if _is_posthoc_ema(emas):
                for i, ema_model in enumerate(emas.ema_models):
                    incompatible = ema_model.ema_model.load_state_dict(
                        checkpoint["ema"], strict=False
                    )
                    log(
                        f"Loading old EMA weights into PostHocEMA model {i}",
                        accelerator,
                    )
                    _log_incompatible_keys(
                        f"PostHocEMA[{i}]", incompatible, accelerator
                    )
            else:
                incompatible = emas.ema_model.load_state_dict(
                    checkpoint["ema"], strict=False
                )
                log("Loading old EMA weights into single KarrasEMA shadow", accelerator)
                _log_incompatible_keys("KarrasEMA", incompatible, accelerator)
            return
        except Exception as e:
            log(f"Warning: Cannot load EMA from old checkpoint: {e}", accelerator)
    log("No EMA data in checkpoint, initializing EMA with current model", accelerator)
    emas.copy_params_from_model_to_ema()


def load_ckpt(args, model, emas, opt, accelerator, scheduler=None):
    """Load checkpoint with compatibility for PostHocEMA and single KarrasEMA."""
    if args.ckpt_path is None:
        return (0, 0)
    checkpoint = load_checkpoint(args.ckpt_path, map_location="cpu", weights_only=True)
    start_epoch = checkpoint.get("epoch", 0)
    start_step = checkpoint.get("step", 0)
    base_model = accelerator.unwrap_model(model)
    if "model" in checkpoint or "ema" in checkpoint:
        if start_step == 0 and "ema" in checkpoint and ("model" not in checkpoint):
            log(
                f"Loading EMA model from {args.ckpt_path}, epoch {start_epoch}, step {start_step}",
                accelerator,
            )
            incompatible = base_model.load_state_dict(checkpoint["ema"], strict=False)
        else:
            model_state = checkpoint.get("model", checkpoint.get("ema"))
            incompatible = base_model.load_state_dict(model_state, strict=False)
            log(
                f"Loading model from {args.ckpt_path}, epoch {start_epoch}, step {start_step}",
                accelerator,
            )
        _log_incompatible_keys("Model", incompatible, accelerator)
    _load_ema_state(args, checkpoint, emas, accelerator)
    _load_optimizer_state(checkpoint, opt, start_step, accelerator)
    _load_scheduler_state(checkpoint, scheduler, start_step, accelerator)
    return (start_epoch, start_step)


def load_teacher_heads_ckpt(args, freezed_teacher, accelerator):
    """Load discriminator heads from checkpoint when available."""
    if freezed_teacher is None or args.ckpt_path is None:
        return
    if not os.path.exists(args.ckpt_path):
        log(
            f"Teacher-head load skipped (ckpt not found): {args.ckpt_path}", accelerator
        )
        return
    ckpt = load_checkpoint(args.ckpt_path, map_location="cpu", weights_only=True)
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
    log(f"Loaded teacher heads from {args.ckpt_path}", accelerator)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        log(f"Teacher heads missing keys: {missing_keys}", accelerator)
        log(f"Teacher heads unexpected keys: {unexpected_keys}", accelerator)


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, height // 2 * (width // 2), num_channels_latents * 4
    )
    return latents


def decode_latent(vae, latents, height, width, vae_scale_factor=8, output_type="pil"):
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    image = image_processor.postprocess(image, output_type=output_type)
    return image


def encode_image(vae, image, height, width, vae_scale_factor=8):
    image = image.unsqueeze(2)
    latents = vae.encode(image, return_dict=False)[0].sample()
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
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
        if (
            getattr(args, "use_ema", False)
            and getattr(args, "ema_mode", "posthoc") == "posthoc"
        ):
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


def _load_pt_pretrained_weights(model, ckpt_path, accelerator):
    try:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        ckpt = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state_dict = {key.replace("module.", "", 1): value for (key, value) in ckpt.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    log(f"Loaded pretrained weights from {ckpt_path}", accelerator)
    log(f"Missing keys: {missing_keys}", accelerator)
    log(f"Unexpected keys: {unexpected_keys}", accelerator)


def _normalize_pretrained_state_dict(state_dict):
    return {key.replace("module.", "", 1): value for (key, value) in state_dict.items()}


def _load_diffusers_transformer_weights(model, transformer_dir, accelerator):
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ImportError as exc:
        raise ImportError(
            "Loading a diffusers transformer directory requires safetensors."
        ) from exc
    index_path = os.path.join(
        transformer_dir, "diffusion_pytorch_model.safetensors.index.json"
    )
    single_file_path = os.path.join(
        transformer_dir, "diffusion_pytorch_model.safetensors"
    )
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        shard_paths = [
            os.path.join(transformer_dir, shard_name) for shard_name in shard_names
        ]
    elif os.path.exists(single_file_path):
        shard_paths = [single_file_path]
    else:
        raise FileNotFoundError(
            f"No diffusers safetensors weights found in transformer directory: {transformer_dir}"
        )
    missing_shards = [
        shard_path for shard_path in shard_paths if not os.path.exists(shard_path)
    ]
    if len(missing_shards) > 0:
        raise FileNotFoundError(
            "Missing safetensors shard(s): " + ", ".join(missing_shards)
        )
    model_keys = set(model.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = set()
    for shard_path in shard_paths:
        state_dict = _normalize_pretrained_state_dict(
            load_safetensors_file(shard_path, device="cpu")
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        loaded_keys.update((key for key in state_dict.keys() if key in model_keys))
        unexpected_keys.update(getattr(incompatible, "unexpected_keys", []))
        del state_dict
        cleanup_memory()
    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(unexpected_keys)
    log(
        f"Loaded pretrained transformer from {transformer_dir} with end_t=True",
        accelerator,
    )
    log(f"Missing keys: {missing_keys}", accelerator)
    log(f"Unexpected keys: {unexpected_keys}", accelerator)


def create_model(args, accelerator):
    config = {
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": False,
        "in_channels": 64,
        "joint_attention_dim": 3584,
        "num_attention_heads": 24,
        "num_layers": 30,
        "out_channels": 16,
        "patch_size": 2,
        "end_t": True,
    }
    pretrained_transformer_path = getattr(
        args, "pretrained_transformer_path", DEFAULT_PRETRAINED_TRANSFORMER_PATH
    )
    if isinstance(
        pretrained_transformer_path, str
    ) and pretrained_transformer_path.lower() in {"", "none", "null"}:
        pretrained_transformer_path = None
    model = QwenImageTransformer2DModel_Distill(**config)
    if pretrained_transformer_path is None:
        log(
            "No pretrained transformer path provided; training from initialization.",
            accelerator,
        )
    elif os.path.isdir(pretrained_transformer_path):
        _load_diffusers_transformer_weights(
            model, pretrained_transformer_path, accelerator
        )
    else:
        if not os.path.exists(pretrained_transformer_path):
            raise FileNotFoundError(
                f"Pretrained transformer path does not exist: {pretrained_transformer_path}"
            )
        _load_pt_pretrained_weights(model, pretrained_transformer_path, accelerator)
    model.enable_fused_attn()
    use_lora = getattr(args, "use_lora", False)
    if use_lora:
        rank = getattr(args, "lora_r", 64)
        lora_alpha = getattr(args, "lora_alpha", 64)
        lora_bias = getattr(args, "lora_bias", "all")
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_qkv",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out.0",
            "to_add_out",
            "img_mod.1",
            "txt_mod.1",
            "txt_net.0.proj",
            "txt_mlp.net.2",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]
        modules_to_save = ["time_text_embed"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias=lora_bias,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        log(
            f"DiT LoRA: r={rank}, alpha={lora_alpha}, bias={lora_bias}; LoRA targets={len(target_modules)} patterns; modules_to_save={len(modules_to_save)}",
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
        num_warmup_steps=max(args.warmup_steps, 0),
        num_training_steps=args.max_steps,
    )
    facm_loss = PVDLossQwenImage()
    disc_teacher_loss = DiscriminatorLossQwenImage()
    return (opt, scheduler, facm_loss, disc_teacher_loss)


def setup_teacher_model(args, accelerator):
    """Setup teacher model for distillation"""
    if not args.distill:
        return None
    freezed_teacher = DiTDiscriminator(
        QwenImageTransformer2DModel.from_pretrained(
            args.model_root, subfolder="transformer", torch_dtype=torch.bfloat16
        ),
        adv_index=[2, 11, 20, 29],
    )

    for param in freezed_teacher.parameters():
        param.requires_grad = False
    freezed_teacher.eval()
    freezed_teacher.heads.to(torch.bfloat16)
    freezed_teacher.heads.requires_grad_(True)
    freezed_teacher.heads.train()
    return freezed_teacher


@torch.no_grad()
def evaluate(args, model, vae, accelerator, train_steps, visualize_dir, noise, cond):
    """Evaluate the model"""
    log(f"Evaluating at step {train_steps}...", accelerator)
    model.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
    batch_size = noise.shape[0]
    img_shapes = [
        [(1, args.image_size // 8 // 2, args.image_size // 8 // 2)]
    ] * batch_size
    intervals = [float(i) for i in args.intervals.split("_")]
    timesteps = torch.tensor(intervals, device=noise.device)
    t, r = (timesteps[0], timesteps[1])
    end_timestep = (t - r).view(1).expand(batch_size).reshape(-1)
    noise = noise.to(device=device, dtype=model_dtype)
    encoder_hidden_states = cond["encoder_hidden_states"].to(
        device=device, dtype=model_dtype
    )
    encoder_hidden_states_mask = cond["encoder_hidden_states_mask"]
    if encoder_hidden_states_mask is not None:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(device=device)
    input_kwargs = {
        "hidden_states": noise,
        "timestep": t.expand(batch_size).reshape(-1),
        "end_timestep": end_timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "img_shapes": img_shapes,
        "return_dict": False,
    }
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=model_dtype)
        if device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    with autocast_ctx:
        pred_v = model(**input_kwargs)[0]
    sampled_x = noise + pred_v * (r - t).view(-1, 1, 1)
    x_hat = decode_latent(
        vae, sampled_x.bfloat16(), args.image_size, args.image_size, output_type="pt"
    )
    x_hat_gathered = accelerator.gather(x_hat)
    if accelerator.is_main_process:
        x_hat_gathered = x_hat_gathered.clamp(0, 1)
        img = torchvision.utils.make_grid(x_hat_gathered[:64], nrow=8)
        img = torchvision.transforms.functional.to_pil_image(img.cpu().float())
        img.save(f"{visualize_dir}/steps{train_steps}.png")
    log(
        f"(steps={train_steps}) images saved to {visualize_dir}/steps{train_steps}.png",
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
        eval_model = (
            _get_eval_ema_model(emas) if args.use_ema and emas is not None else model
        )
        if args.use_ema and emas is not None:
            eval_model.to(accelerator.device)
        with RandomStateManager(eval_seed=args.global_seed + accelerator.process_index):
            evaluate(
                args,
                eval_model,
                vae,
                accelerator,
                train_steps,
                visualize_dir,
                noise=eval_noise,
                cond=eval_cond,
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
    tokenizer, text_encoder = _load_online_text_encoder(args, device, accelerator)
    actual_batch_size = args.global_batch_size // accelerator.num_processes
    img_shapes = [
        [(1, args.image_size // 8 // 2, args.image_size // 8 // 2)]
    ] * actual_batch_size
    negative_condition = _prepare_text_condition(
        [""], device, img_shapes[:1], tokenizer, text_encoder
    )
    if _normalize_text_embed_root(args.qwenimage_text_embed_root) is not None:
        tokenizer = text_encoder = None
        cleanup_memory()
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
        _get_reference_ema_model(emas) if args.use_ema and emas is not None else model
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                img_shapes = [
                    [(1, args.image_size // 16, args.image_size // 16)]
                ] * images.shape[0]
                y = _prepare_text_condition(
                    labels, device, img_shapes, tokenizer, text_encoder
                )
                negative_mask = negative_condition["encoder_hidden_states_mask"]
                un_y = {
                    "encoder_hidden_states": negative_condition[
                        "encoder_hidden_states"
                    ].expand(images.shape[0], -1, -1),
                    "encoder_hidden_states_mask": (
                        negative_mask.expand(images.shape[0], -1)
                        if negative_mask is not None
                        else None
                    ),
                    "img_shapes": img_shapes,
                }
                model_kwargs = dict(y=y, un_y=un_y)
                cleanup_memory()
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
                        if args.use_ema and emas is not None:
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
    args.qwenimage_text_embed_root = _normalize_text_embed_root(
        args.qwenimage_text_embed_root
    )
    ds_file = args.deepspeed_config
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_file)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation, deepspeed_plugin=deepspeed_plugin
    )
    log(f"DeepSpeed hf_ds_config={ds_file}", accelerator)
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
    loader_train, dataset_train, loader_len = get_blip3o_dataset(
        args,
        args.data_file,
        args.image_root,
        accelerator,
        qwenimage_text_embed_root=args.qwenimage_text_embed_root,
    )
    log(f"Dataset loaded", accelerator)
    model_root = args.model_root
    vae = AutoencoderKLQwenImage.from_pretrained(
        f"{model_root}/vae", torch_dtype=torch.bfloat16
    )
    model = create_model(args, accelerator).to(torch.bfloat16)
    emas = None
    freezed_teacher = setup_teacher_model(args, accelerator)
    if args.init_ckpt:
        from utils.qwenimage import load_init_lora_checkpoint

        load_init_lora_checkpoint(args, model, freezed_teacher, accelerator)
    cleanup_memory()
    if args.use_ema:
        log(f"Building EMA mirror model ({args.ema_mode})...", accelerator)
        if args.ema_mode == "single":
            emas = KarrasEMA(
                model,
                sigma_rel=args.sigma_rel,
                update_every=1,
                allow_different_devices=True,
                move_ema_to_online_device=False,
            )
        else:
            if args.sigma_rel == 0.05:
                raise ValueError(
                    "PostHocEMA requires two distinct sigma_rel values. Use --ema-mode single for one EMA shadow, or choose a different --sigma-rel."
                )
            emas = PostHocEMA(
                model,
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
            os.path.join(PROJECT_DIR, "cache", "qwenimage_eval_prompt_embeds.pt"),
            weights_only=True,
            map_location=device,
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
    parser.add_argument("--data-file", default="data/blip3o/train.jsonl")
    parser.add_argument("--image-root", default="data/blip3o/images")
    parser.add_argument(
        "--qwenimage-text-embed-root",
        type=_normalize_text_embed_root,
        default=None,
        help="BLIP3o QwenImage embeddings: mirror image paths under this root with a .pt suffix. Omit to encode captions online.",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("QWENIMAGE_MODEL_ROOT", "models/Qwen-Image"),
    )
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Initialize LoRA and teacher heads without restoring optimizer, EMA or step.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--results-dir", type=str, default="output")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument(
        "--pretrained-transformer-path",
        type=str,
        default=DEFAULT_PRETRAINED_TRANSFORMER_PATH,
        help="Diffusers transformer directory or legacy .pt checkpoint used to initialize the student before training. Set to None in code to skip.",
    )
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=os.path.join(
            PROJECT_DIR, "scripts", "config", "zero_stage2_config.json"
        ),
        help="DeepSpeed JSON (ZeRO stage).",
    )
    parser.add_argument("--sigma-rel", type=float, default=0.2)
    parser.add_argument(
        "--ema-mode",
        type=str,
        default="posthoc",
        choices=["posthoc", "single"],
        help="EMA backend: 'posthoc' keeps two shadows for synthesis; 'single' keeps one KarrasEMA shadow.",
    )
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMA tracking. Use --no-use-ema to disable.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--t-type", type=str, default="default")
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
    parser.add_argument(
        "--lora-r",
        type=int,
        default=256,
        help="LoRA rank (higher -> closer to full fine-tuning on non-saved layers).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=256,
        help="LoRA alpha scaling (often set equal to --lora-r).",
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="all",
        choices=["none", "all", "lora_only"],
        help="Train Linear biases: 'all' is closest to full FT on LoRA-backed layers.",
    )
    parser.add_argument(
        "--use-lora", action="store_true", default=False, help="Use LoRA for training."
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=4e-06,
        dest="disc_lr",
        help="AdamW lr for DiTDiscriminator heads only (student uses --lr).",
    )
    parser.add_argument(
        "--cfgw",
        type=float,
        default=4.0,
        help="Teacher classifier-free guidance scale.",
    )
    args = parser.parse_args()
    main(args)
