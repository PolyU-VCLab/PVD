"""Inference entry point for C2I and standard or Unsplash T2I variants."""

import argparse
from pathlib import Path

WEIGHT_FOLDERS = {
    "c2i": "pvd_c2i",
    "sd35": "pvd_sd35m",
    "sd35_unsplash": "pvd_sd35m_unsplash",
    "flux": "pvd_flux",
    "flux_unsplash": "pvd_flux_unsplash",
    "qwenimage": "pvd_qwenimage",
    "qwenimage_unsplash": "pvd_qwen_unsplash",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        choices=list(WEIGHT_FOLDERS),
        required=True,
    )
    p.add_argument(
        "--model-root",
        help="Local base model directory containing VAE, tokenizers and text encoders",
    )
    p.add_argument("--weights-dir", type=Path, default=Path("weights"))
    p.add_argument("--output", type=Path, default=Path("outputs/sample.png"))
    p.add_argument("--prompt", default="A red apple on a wooden table.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--part1_steps", "--part1-steps", type=int, default=1,
        help="Number of inference steps for part1 (T2I)",
    )
    p.add_argument(
        "--part2_steps", "--part2-steps", type=int, default=1,
        help="Number of inference steps for part2 (T2I)",
    )
    p.add_argument("--class-id", type=int, default=207)
    a = p.parse_args()
    if a.part1_steps < 1:
        p.error("--part1_steps must be positive")
    if a.part2_steps < 1:
        p.error("--part2_steps must be positive")
    if a.task != "c2i" and (not a.model_root):
        p.error("--model-root is required for T2I")
    if a.height % 16 or a.width % 16 or min(a.height, a.width) < 16:
        p.error("Image dimensions must be positive multiples of 16")
    if not 0 <= a.class_id < 1000:
        p.error("--class-id must be in [0, 999]")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("These inference pipelines require a CUDA GPU")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    base_task = a.task.removesuffix("_unsplash")
    w = a.weights_dir / WEIGHT_FOLDERS[a.task]
    backbone = a.weights_dir / WEIGHT_FOLDERS[base_task] / "backbone.safetensors"
    if base_task == "c2i":
        required = [w / "model.safetensors"]
    elif base_task == "sd35":
        required = [w / "part1.safetensors", w / "part2.safetensors"]
    else:
        required = [backbone] + [
            w / part / name
            for part in ("part1", "part2")
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    with torch.inference_mode():
        if base_task == "flux":
            from pipelines.flux import FLUX1Pipeline

            pipe = FLUX1Pipeline(
                a.model_root,
                str(w / "part1"),
                str(w / "part2"),
                str(backbone),
            )
            image = pipe.pvd_infer(
                [a.prompt],
                a.part1_steps,
                a.part2_steps,
                a.height,
                a.width,
                seed=a.seed,
            )
            image.save(a.output)
        elif base_task == "qwenimage":
            from pipelines.qwenimage import PVDQwenImagePipeline

            pipe = PVDQwenImagePipeline(
                model_root=a.model_root, backbone=str(backbone)
            )
            _, image = pipe.pvd_infer(
                [a.prompt],
                str(w / "part1"),
                str(w / "part2"),
                height=a.height,
                width=a.width,
                steps1=a.part1_steps,
                steps2=a.part2_steps,
                seed=a.seed,
            )
            image.save(a.output)
        elif base_task == "sd35":
            from pipelines.sd35 import InferenceConfig, load_sd3_models, run_validation
            from torchvision.utils import save_image

            cfg = InferenceConfig()
            cfg.model_root = a.model_root
            cfg.part1, cfg.part2 = (
                str(w / "part1.safetensors"),
                str(w / "part2.safetensors"),
            )
            cfg.seed = a.seed
            models = load_sd3_models(cfg)
            images = run_validation(
                models,
                [a.prompt],
                cfg,
                a.height,
                a.width,
                num_inference_steps=a.part1_steps,
                num_inference_steps2=a.part2_steps,
            )
            save_image(images, a.output)
        else:
            from ldit.model_manager import create_models_from_config
            from pipelines.c2i import consistency_model_sampler
            from utils import decode_image
            from utils.weight_io import load_weights
            from torchvision.utils import save_image

            vae, model = create_models_from_config(
                "ldit/lightningdit_student.yaml", num_classes=1000
            )
            model.load_state_dict(load_weights(w / "model.safetensors"), strict=True)
            model = model.cuda().eval()
            model.enable_fused_attn()
            g = torch.Generator(device="cuda").manual_seed(a.seed)
            z = torch.randn((1, 32, 16, 16), device="cuda", generator=g)
            y = torch.tensor([a.class_id], device="cuda")
            result = consistency_model_sampler(
                model, z, y, num_steps=1, cfg_scale=1.0, intervals=[0.4, 0.6]
            )
            save_image((decode_image(result, vae) + 1) / 2, a.output)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
