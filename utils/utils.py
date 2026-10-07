# Training, checkpointing and evaluation utilities for PVD.
# Adapted from FACM utils/utils.py (https://github.com/ali-vilab/FACM), with PVD loading and release changes.

from utils.weight_io import load_checkpoint
import os
import zipfile
import tqdm
import datetime
import random
import numpy as np
import torch
import torchvision
from ldit.img_latent_dataset import ImgLatentDataset
from pipelines.c2i import consistency_model_sampler, euler_sampler
from torch.utils.data import Dataset

_cached_latent_stats = None


def log(msg, accelerator=None):
    """Log messages (only in main process)"""
    if accelerator is None or accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("\n".join([f"[{now}] {m}" for m in msg.split("\n")]))
        if hasattr(log, "log_file") and log.log_file is not None:
            with open(log.log_file, "a") as f:
                f.write(f"{msg}\n")
                f.flush()


def mean_flat(x):
    """Take the mean over all non-batch dimensions"""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def get_dataset(args, accelerator, data_dir):
    """Get dataset and dataloader"""
    dataset_train = ImgLatentDataset(
        data_dir=data_dir, latent_norm=True, latent_multiplier=1.0
    )
    actual_batch_size = args.global_batch_size // accelerator.num_processes
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=actual_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        persistent_workers=True,
    )
    loader_len = len(loader_train)
    return (loader_train, dataset_train, loader_len)


def get_latent_stats(x):
    """Get cached latent statistics"""
    global _cached_latent_stats
    if _cached_latent_stats is None:
        latent_stats = load_checkpoint(
            "cache/latents_stats.pt", map_location="cpu", weights_only=True
        )
        _cached_latent_stats = {
            "mean": latent_stats["mean"].detach(),
            "std": latent_stats["std"].detach(),
        }
    mean = _cached_latent_stats["mean"].to(x.device)
    std = _cached_latent_stats["std"].to(x.device)
    return (mean, std)


@torch.no_grad()
def decode_image(z, vae):
    """Decode normalized latent to image"""
    mean, std = get_latent_stats(z)
    z = z * std + mean
    x = vae.decode_to_images(z.float())
    x = x.clamp(-1, 1).to(z.dtype)
    return x.to(z.dtype)


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
        emas_state_dict = emas.state_dict()
        checkpoint = {
            "model": model.state_dict(),
            "emas": emas_state_dict,
            "opt": opt.state_dict(),
            "args": args,
            "epoch": epoch,
            "step": step,
        }
        if freezed_teacher is not None:
            teacher_model = accelerator.unwrap_model(freezed_teacher)
            checkpoint["teacher_heads"] = teacher_model.heads.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, f"{checkpoint_dir}/latest.pt")
        log(f"Checkpoint saved to {checkpoint_path}", accelerator)
        emas.checkpoint()
    accelerator.wait_for_everyone()


def load_ckpt(args, model, emas, opt, accelerator, scheduler=None):
    """Load checkpoint, properly handling PostHocEMA and scheduler state"""
    if args.ckpt_path is None:
        return (0, 0)
    checkpoint = load_checkpoint(args.ckpt_path, map_location="cpu", weights_only=True)
    start_epoch = checkpoint.get("epoch", 0)
    start_step = checkpoint.get("step", 0)
    if "model" in checkpoint or "ema" in checkpoint:
        if start_step == 0 and "ema" in checkpoint:
            log(
                f"Loading EMA model from {args.ckpt_path}, epoch {start_epoch}, step {start_step}",
                accelerator,
            )
            missing_keys, unexpected_keys = model.module.load_state_dict(
                checkpoint["ema"], strict=False
            )
        else:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model"], strict=False
            )
            log(
                f"Loading model from {args.ckpt_path}, epoch {start_epoch}, step {start_step}",
                accelerator,
            )
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            log(f"Missing keys: {missing_keys}", accelerator)
            log(f"Unexpected keys: {unexpected_keys}", accelerator)
    _load_ema_state(checkpoint, emas, accelerator)
    _load_optimizer_state(checkpoint, opt, start_step, accelerator)
    _load_scheduler_state(checkpoint, scheduler, start_step, accelerator)
    return (start_epoch, start_step)


def _load_ema_state(checkpoint, emas, accelerator):
    """Load EMA state from checkpoint"""
    if emas is None:
        return
    if "emas" in checkpoint:
        try:
            missing_keys, unexpected_keys = emas.load_state_dict(
                checkpoint["emas"], strict=False
            )
            log(f"Loading PostHocEMA model from checkpoint", accelerator)
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                log(f"PostHocEMA missing keys: {missing_keys}", accelerator)
                log(f"PostHocEMA unexpected keys: {unexpected_keys}", accelerator)
        except Exception as e:
            log(f"Warning: Cannot load PostHocEMA state: {e}", accelerator)
            log(f"Initializing PostHocEMA with model", accelerator)
            emas.copy_params_from_model_to_ema()
    elif "ema" in checkpoint:
        log(
            f"Detected old version EMA checkpoint, attempting compatible loading...",
            accelerator,
        )
        try:
            for i, ema_model in enumerate(emas.ema_models):
                missing_keys, unexpected_keys = ema_model.ema_model.load_state_dict(
                    checkpoint["ema"], strict=False
                )
                log(f"Loading old EMA weights into PostHocEMA model {i}", accelerator)
        except Exception as e:
            log(f"Warning: Cannot load EMA from old checkpoint: {e}", accelerator)
            emas.copy_params_from_model_to_ema()
    else:
        log(
            f"No EMA data in checkpoint, initializing PostHocEMA with current model",
            accelerator,
        )
        emas.copy_params_from_model_to_ema()


def _load_optimizer_state(checkpoint, opt, start_step, accelerator):
    """Load optimizer state from checkpoint"""
    if "opt" in checkpoint and start_step > 0:
        try:
            opt.load_state_dict(checkpoint["opt"])
            log("Loading optimizer from checkpoint", accelerator)
        except Exception as e:
            log(
                f"Warning: Cannot load optimizer state ({type(e).__name__}): {e!r}",
                accelerator,
            )


def _load_scheduler_state(checkpoint, scheduler, start_step, accelerator):
    """Load scheduler state from checkpoint"""
    if "scheduler" in checkpoint and scheduler is not None and (start_step > 0):
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            log("Loading scheduler state from checkpoint", accelerator)
        except Exception as e:
            log(
                f"Warning: Cannot load scheduler state ({type(e).__name__}): {e!r}",
                accelerator,
            )


class ImagePack(Dataset):
    """Dataset wrapper for FID metric computation"""

    def __init__(self, image_data):
        self.dataset_images = image_data
        self._preprocess_fn = self._create_preprocessing_function()

    def _create_preprocessing_function(self):
        """Create preprocessing function for image normalization"""

        def preprocess(tensor_data):
            scaled_data = tensor_data.mul(255)
            return scaled_data.byte()

        return preprocess

    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, item_idx):
        raw_image = self.dataset_images[item_idx]
        return self._preprocess_fn(raw_image)


@torch.no_grad()
def evaluate(
    args,
    model,
    vae,
    accelerator,
    step,
    visualize_dir,
    noise=None,
    cond=None,
    start=1000,
    cfgw=1,
    fid=False,
    num_samples=5000,
    save_images=False,
    t2i=False,
):
    """Evaluation function"""
    device = accelerator.device
    temp_dir_name = getattr(args, "temp", "temp")
    tmp_dir = os.path.join(
        "output", temp_dir_name, f"fid_image_{args.sampling_steps}_steps", "generated"
    )
    if fid and accelerator.is_main_process:
        tmp_dir = _ensure_unique_dir(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    save_dir = f"{visualize_dir}/sample_{step:07d}"
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    model.eval()
    accelerator.wait_for_everyone()
    with torch.no_grad():
        eval_batch_size = 8 if not fid else 125
        eval_batch_size = 1 if t2i else eval_batch_size
        world_size = accelerator.num_processes
        global_rank = accelerator.process_index
        num_collect = eval_batch_size * world_size
        if not fid:
            return _evaluate_multiple_steps(
                args,
                model,
                vae,
                accelerator,
                step,
                save_dir,
                eval_batch_size,
                device,
                noise,
                cond,
                start,
                cfgw,
            )
        return _evaluate_fid(
            args,
            model,
            vae,
            accelerator,
            step,
            save_dir,
            eval_batch_size,
            world_size,
            global_rank,
            num_collect,
            device,
            num_samples,
            cfgw,
            start,
            tmp_dir,
            save_images,
        )


def _ensure_unique_dir(tmp_dir):
    """Ensure directory name is unique"""
    if os.path.exists(tmp_dir):
        tmp_dir_base = tmp_dir
        counter = 1
        while os.path.exists(tmp_dir):
            tmp_dir = f"{tmp_dir_base}_{counter}"
            counter += 1
    return tmp_dir


def _evaluate_multiple_steps(
    args,
    model,
    vae,
    accelerator,
    step,
    save_dir,
    eval_batch_size,
    device,
    noise,
    cond,
    start,
    cfgw,
):
    """Evaluate with multiple step sizes"""
    step_sizes = [args.sampling_steps]
    if noise is None:
        fixed_noise = torch.randn(
            eval_batch_size,
            32,
            args.image_size // 16,
            args.image_size // 16,
            device=device,
        )
    else:
        fixed_noise = noise.clone()
    if cond is None:
        fixed_cond = torch.randint(
            0, 1000, (eval_batch_size,), device=device, dtype=torch.int64
        )
    else:
        fixed_cond = cond.clone()
    intervals = [float(i) for i in args.intervals.split("_")]
    for num_steps in step_sizes:
        noise_copy = fixed_noise.clone()
        if args.sampling_method == "euler":
            sampled_x = euler_sampler(
                model, noise_copy, fixed_cond, num_steps=num_steps, cfg_scale=cfgw
            )
        else:
            sampled_x = consistency_model_sampler(
                model,
                noise_copy,
                fixed_cond,
                num_steps=num_steps,
                cfg_scale=cfgw,
                intervals=intervals,
            )
        _x_hat = (decode_image(sampled_x, vae) + 1) / 2
        x_hat_gathered = accelerator.gather(_x_hat)
        if accelerator.is_main_process:
            img = torchvision.utils.make_grid(x_hat_gathered[:64], nrow=8)
            img = torchvision.transforms.functional.to_pil_image(img.cpu().float())
            img.save(f"{save_dir}/t{start:04d}_cfgw{cfgw}_steps{num_steps}.png")
        log(
            f"(step={step}) <t={start}> steps={num_steps} images saved to {save_dir}/t{start:04d}_cfgw{cfgw}_steps{num_steps}.png",
            accelerator,
        )
    return 0


def _evaluate_fid(
    args,
    model,
    vae,
    accelerator,
    step,
    save_dir,
    eval_batch_size,
    world_size,
    global_rank,
    num_collect,
    device,
    num_samples,
    cfgw,
    start,
    tmp_dir,
    save_images=False,
):
    all_samples = []
    """Evaluate FID score"""
    conds = torch.arange(1000, device=device, dtype=torch.int64).repeat(100)
    num_batches = int(np.ceil(num_samples / eval_batch_size / world_size))
    bar = tqdm.trange(num_batches, desc=f"FID", disable=not accelerator.is_main_process)
    intervals = [float(i) for i in args.intervals.split("_")]
    for i in bar:
        x_noise = torch.randn(
            eval_batch_size,
            32,
            args.image_size // 16,
            args.image_size // 16,
            device=device,
        )
        cond = conds[
            i * num_collect
            + global_rank * eval_batch_size : i * num_collect
            + (global_rank + 1) * eval_batch_size
        ]
        if args.sampling_method == "euler":
            x_noise = euler_sampler(
                model,
                x_noise,
                cond,
                num_steps=args.sampling_steps,
                cfg_scale=cfgw,
                guidance_low=args.glow,
            )
        else:
            x_noise = consistency_model_sampler(
                model,
                x_noise,
                cond,
                num_steps=args.sampling_steps,
                cfg_scale=cfgw,
                guidance_low=args.glow,
                intervals=intervals,
            )
        _x_hat = (decode_image(x_noise, vae) + 1) / 2
        x_hat_gathered = accelerator.gather(_x_hat).cpu()
        all_samples.append(x_hat_gathered)
        if save_images and accelerator.is_main_process:
            for j in range(x_hat_gathered.shape[0]):
                img_idx = i * num_collect + j
                if img_idx >= num_samples:
                    break
                img = torchvision.transforms.functional.to_pil_image(
                    x_hat_gathered[j].cpu().float()
                )
                img.save(f"{tmp_dir}/{img_idx:05d}.png")
        torch.cuda.empty_cache()
    ret = 0
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if save_images:
            log(
                f"Compressing {len(os.listdir(tmp_dir))} images to zip file...",
                accelerator,
            )
            with zipfile.ZipFile(
                f"{save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip", "w"
            ) as img_zip:
                for filename in os.listdir(tmp_dir):
                    if filename.endswith(".png"):
                        img_zip.write(os.path.join(tmp_dir, filename), filename)
            log(
                f"Individual images saved to {save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip",
                accelerator,
            )
        stat = f"fid-{int(np.ceil(num_samples / 1000))}k-{args.image_size}.npz"
        all_samples = torch.cat(all_samples)[:num_samples]
        image_pack = ImagePack(all_samples) if not args.save_images else tmp_dir
        metrics_dict = __import__("torch_fidelity").calculate_metrics(
            input1=image_pack,
            input2=None,
            fid_statistics_file=f"cache/{stat}",
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        inception_score = metrics_dict["inception_score_mean"]
        inception_score_std = metrics_dict["inception_score_std"]
        ret = fid_score = metrics_dict["frechet_inception_distance"]
        if save_images:
            log(
                f"(step={step}) <t={start}> FID={fid_score:.4f}, IS={inception_score:.4f}±{inception_score_std:.4f}, images saved to {save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip",
                accelerator,
            )
        else:
            log(
                f"(step={step}) <t={start}> FID={fid_score:.4f}, IS={inception_score:.4f}±{inception_score_std:.4f}",
                accelerator,
            )
        del all_samples, metrics_dict, image_pack
    accelerator.wait_for_everyone()
    return ret


class RandomStateManager:
    """
    Random state manager for saving/restoring random states during evaluation to ensure reproducible
    evaluation results without affecting the training process
    """

    def __init__(self, eval_seed=0):
        self.eval_seed = eval_seed
        self.saved_states = {}

    def __enter__(self):
        """Save current random states and set evaluation seed when entering context"""
        self.saved_states["python_random"] = random.getstate()
        self.saved_states["numpy_random"] = np.random.get_state()
        self.saved_states["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            self.saved_states["torch_cuda"] = torch.cuda.get_rng_state_all()
        random.seed(self.eval_seed)
        np.random.seed(self.eval_seed)
        torch.manual_seed(self.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.eval_seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random states when exiting context"""
        random.setstate(self.saved_states["python_random"])
        np.random.set_state(self.saved_states["numpy_random"])
        torch.set_rng_state(self.saved_states["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in self.saved_states:
            torch.cuda.set_rng_state_all(self.saved_states["torch_cuda"])
