# PVD SD3.5 training.
# PVD text-to-image extension; training setup, resume and logging helpers are adapted from FACM train.py (https://github.com/ali-vilab/FACM)

from utils.weight_io import load_checkpoint
import os
import argparse
import time
import gc
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from ema_pytorch import PostHocEMA
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL
import torchvision
from ldit.discrimator import DiTDiscriminator
from ldit.transformer_sd3_distillation import SD3Transformer2DModel_Distill
from ldit.transformer_sd3 import SD3Transformer2DModel
from losses.sd35 import DiscriminatorLossT2I, PVDLossSD35
from utils import log, RandomStateManager, load_ckpt, save_ckpt
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from utils.t2i import decode_latent, encode_latent, encode_prompt, get_blip3o_dataset
from peft import LoraConfig, get_peft_model


def cleanup_memory():
    """Run occasional host/GPU memory cleanup after heavy ops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    missing, unexpected = teacher_model.heads.load_state_dict(
        normalized_state, strict=False
    )
    log(f"Loaded teacher heads from {ckpt_path}", accelerator)
    if missing:
        log(f"Teacher-head missing keys: {missing}", accelerator)
    if unexpected:
        log(f"Teacher-head unexpected keys: {unexpected}", accelerator)


def _core_module_for_dtype(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap PostHocEMA / DDP to read parameter dtype for consistent eval inputs."""
    m = module
    if hasattr(m, "ema_model"):
        m = m.ema_model
    if hasattr(m, "module"):
        m = m.module
    return m


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


def create_model(args, accelerator):
    config = {
        "dual_attention_layers": [0, 1, 2, 3, 4, 5],
        "attention_head_dim": 64,
        "caption_projection_dim": 1536,
        "in_channels": 16,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": 12,
        "out_channels": 16,
        "patch_size": 2,
        "pooled_projection_dim": 2048,
        "pos_embed_max_size": 384,
        "qk_norm": "rms_norm",
        "sample_size": 128,
        "enable_end_timestep": True,
    }
    model = SD3Transformer2DModel_Distill(**config)
    model.enable_fused_attn()
    ckpt_path = os.environ.get(
        "INITIALIZATION_CHECKPOINT", "weights/pvd_sd35m/part1.safetensors"
    )
    try:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)[
            "model"
        ]
    except Exception:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu", weights_only=True)[
            "model"
        ]
    state_dict = {}
    for i in ckpt.keys():
        state_dict.update({i.replace("module.", ""): ckpt[i]})
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    log(f"Loaded pretrained weights from {ckpt_path}", accelerator)
    log(f"Missing keys: {missing_keys}", accelerator)
    log(f"Unexpected keys: {unexpected_keys}", accelerator)
    use_lora = getattr(args, "use_lora", False)
    if use_lora:
        rank = getattr(args, "lora_r", 64)
        lora_alpha = getattr(args, "lora_alpha", 64)
        lora_bias = getattr(args, "lora_bias", "lora_only")
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        modules_to_save = ["time_text_embed"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias=lora_bias,
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
    """Setup optimizer and scheduler (unified student + discriminator heads)."""
    student_params = [p for p in model.parameters() if p.requires_grad]
    disc_lr = getattr(args, "disc_lr", 1e-05)
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
    facm_loss = PVDLossSD35()
    disc_teacher_loss = DiscriminatorLossT2I()
    return (opt, scheduler, facm_loss, disc_teacher_loss)


def setup_teacher_model(args, accelerator):
    """Setup teacher model for distillation"""
    if not args.distill:
        return None
    model_root = os.environ.get("SD35_MODEL_ROOT", "models/stable-diffusion-3.5-medium")
    freezed_teacher = DiTDiscriminator(
        SD3Transformer2DModel.from_pretrained(
            model_root,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        ),
        adv_index=[2, 7, 12, 17, 23],
    )
    for param in freezed_teacher.parameters():
        param.requires_grad = False
    freezed_teacher.eval()
    freezed_teacher.heads.to(torch.bfloat16)
    freezed_teacher.heads.requires_grad_(True)
    freezed_teacher.heads.train()
    return freezed_teacher


@torch.no_grad()
def prepare_negative_condition(device, **text_components):
    """Encode one empty prompt for reuse across all training batches."""
    embeddings, _, pooled, _ = encode_prompt(
        prompt=[""], prompt_2=[""], prompt_3=[""], device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
        **text_components,
    )
    return {"encoder_hidden_states": embeddings, "pooled_projections": pooled}


@torch.no_grad()
def prepare_eval_inputs(args, device, **text_components):
    """Encode fixed evaluation prompts once, without an embedding file."""
    prompts = [
        "A parrot standing on a tree branch.",
        "Two red apples on a wooden desk.",
        "A long hair cat sitting on the windowsill",
    ]
    prompt_embeds, _, pooled_prompt_embeds, _ = encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        **text_components,
    )
    generator = torch.Generator(device=device).manual_seed(args.global_seed)
    noise = torch.randn(
        len(prompts),
        16,
        args.image_size // 8,
        args.image_size // 8,
        device=device,
        generator=generator,
    )
    return noise, {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


@torch.no_grad()
def evaluate(
    args, model, vae, accelerator, train_steps, visualize_dir, noise, cond, cfgw
):
    """Evaluate the model"""
    log(f"Evaluating at step {train_steps}...", accelerator)
    model.eval()
    batch_size = noise.shape[0]
    mdtype = next(_core_module_for_dtype(model).parameters()).dtype
    noise_m = noise.to(dtype=mdtype)
    intervals = [float(i) for i in args.intervals.split("_")]
    timesteps = torch.tensor(intervals, device=noise.device, dtype=mdtype)
    t, r = (timesteps[0], timesteps[1])
    guidance = torch.tensor([cfgw] * batch_size, device=noise.device, dtype=mdtype)
    input_kwargs = {
        "hidden_states": noise_m,
        "encoder_hidden_states": cond["prompt_embeds"].to(
            device=noise.device, dtype=mdtype
        ),
        "pooled_projections": cond["pooled_prompt_embeds"].to(
            device=noise.device, dtype=mdtype
        ),
        "guidance": guidance,
        "timestep": t.repeat(batch_size) * 1000,
        "end_timestep": (t - r).expand(batch_size) * 1000,
    }
    pred_v = model(**input_kwargs)[0]
    sampled_x = noise_m + pred_v * (r - t).view(-1, 1, 1, 1)
    _x_hat = (decode_latent(sampled_x, vae, output_type="latent") + 1) / 2
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
        eval_model = emas.ema_models[1] if args.use_ema and emas is not None else model
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
    vae,
    visualize_dir,
    checkpoint_dir,
    freezed_teacher,
    disc_teacher_loss,
):
    """Main training loop"""
    model_root = os.environ.get("SD35_MODEL_ROOT", "models/stable-diffusion-3.5-medium")
    use_precomputed_text = getattr(args, "text_embed_root", None) is not None
    tokenizer = tokenizer_2 = tokenizer_3 = None
    text_encoder = text_encoder_2 = text_encoder_3 = None
    tokenizer = CLIPTokenizer.from_pretrained(f"{model_root}/tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        f"{model_root}/text_encoder", torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer_2 = CLIPTokenizer.from_pretrained(f"{model_root}/tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        f"{model_root}/text_encoder_2", torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer_3 = T5TokenizerFast.from_pretrained(f"{model_root}/tokenizer_3")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        f"{model_root}/text_encoder_3", torch_dtype=torch.bfloat16
    ).to(device)
    for encoder in (text_encoder, text_encoder_2, text_encoder_3):
        if encoder is not None:
            encoder.eval().requires_grad_(False)
    del encoder
    negative_condition = prepare_negative_condition(
        device, tokenizer=tokenizer, text_encoder=text_encoder,
        tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2,
        tokenizer_3=tokenizer_3, text_encoder_3=text_encoder_3,
    )
    eval_noise = eval_cond = None
    if args.eval_every != -1:
        eval_noise, eval_cond = prepare_eval_inputs(
            args,
            device,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            tokenizer_3=tokenizer_3,
            text_encoder_3=text_encoder_3,
        )
        log("Evaluation conditioning encoded from fixed prompts.", accelerator)
    if use_precomputed_text:
        tokenizer = tokenizer_2 = tokenizer_3 = None
        text_encoder = text_encoder_2 = text_encoder_3 = None
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
        emas.ema_models[1] if args.use_ema and emas is not None else model
    )
    while train_steps < args.max_steps:
        log(f"Starting epoch {epoch}...", accelerator)
        for data in loader_train:
            if train_steps >= args.max_steps:
                break
            with torch.no_grad():
                images, labels = data
                x = encode_latent(vae, images.bfloat16())
                B = images.shape[0]
                if use_precomputed_text:
                    encoder_hidden_states = labels["encoder_hidden_states"].to(
                        device=device, dtype=torch.bfloat16)
                    pooled_projections = labels["pooled_projections"].to(
                        device=device, dtype=torch.bfloat16)
                else:
                    encoder_hidden_states, _, pooled_projections, _ = encode_prompt(
                        prompt=labels, prompt_2=labels, prompt_3=labels,
                        device=device, num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        tokenizer=tokenizer, text_encoder=text_encoder,
                        tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2,
                        tokenizer_3=tokenizer_3, text_encoder_3=text_encoder_3,
                    )
                un_y = {
                    "encoder_hidden_states": negative_condition["encoder_hidden_states"].expand(
                        B, -1, -1).to(dtype=encoder_hidden_states.dtype),
                    "pooled_projections": negative_condition["pooled_projections"].expand(
                        B, -1).to(dtype=pooled_projections.dtype),
                }
                y = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                }
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
                            teacher_running_loss = 0
                            teacher_running_fake_loss = 0
                            teacher_running_real_loss = 0
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
    ds_file = (
        getattr(args, "deepspeed_config", None)
        or "./scripts/config/zero_stage2_config.json"
    )
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
            f"Dataset loaded with precomputed SD3 text embeds under {text_embed_root}",
            accelerator,
        )
    else:
        log("Dataset loaded (on-the-fly text encoding)", accelerator)
    model_root = os.environ.get("SD35_MODEL_ROOT", "models/stable-diffusion-3.5-medium")
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
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="./scripts/config/zero_stage2_config.json",
        dest="deepspeed_config",
        help="DeepSpeed ZeRO config JSON (hf_ds_config).",
    )
    parser.add_argument("--sigma-rel", type=float, default=0.2)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMA teacher tracking. Use --no-use-ema to disable.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--disc-lr", type=float, default=0.0001)
    parser.add_argument(
        "--generator_update_interval",
        type=int,
        default=1,
        help="Train generator this many sync steps before switching to discriminator.",
    )
    parser.add_argument(
        "--discriminator_update_interval",
        type=int,
        default=1,
        help="Train discriminator this many sync steps before switching to generator.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--adv-weight", type=float, default=0.1)
    parser.add_argument("--t-type", type=str, default="default")
    parser.add_argument("--cfgw", type=float, default=1.75)
    parser.add_argument("--mean", type=float, default=0.8)
    parser.add_argument("--std", type=float, default=1.6)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--distill", action="store_true", default=False)
    parser.add_argument("--max-steps", type=float, default=float("inf"))
    parser.add_argument("--intervals", type=str, default="0.6_0.0")
    parser.add_argument(
        "--data-file",
        default="data/blip3o/train.jsonl",
        help="BLIP3o JSONL with image_path and caption fields.",
    )
    parser.add_argument(
        "--image-root",
        default="data/blip3o/images",
        help="Root directory of BLIP3o images.",
    )
    parser.add_argument(
        "--text-embed-root",
        type=str,
        default=None,
        help="BLIP3o SD3.5 embeddings: mirror image paths under this root with a .pt suffix.",
    )
    parser.add_argument(
        "--use-lora", action="store_true", default=False, help="Use LoRA for training."
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
        "--teacher-heads-ckpt",
        type=str,
        default=None,
        dest="teacher_heads_ckpt",
        help="Checkpoint containing teacher_heads for discriminator init.",
    )
    args = parser.parse_args()
    main(args)
