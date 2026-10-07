from utils.weight_io import load_weights
import gc
import torch
from utils.t2i import decode_latent, encode_prompt
from diffusers import AutoencoderKL
from ldit.transformer_sd3_distillation import SD3Transformer2DModel_Distill
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    T5TokenizerFast,
    T5EncoderModel,
)


class InferenceConfig:

    def __init__(self):
        self.model_root = "models/stable-diffusion-3.5-medium"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.seed = 42
        self.guidance_scale = 7.0
        self.vae_scale_factor = 8


def get_model(ckpt_path, num_layers=12, dual_attention_layers=[0, 1, 2, 3, 4, 5]):
    transformer_config = {
        "dual_attention_layers": dual_attention_layers,
        "attention_head_dim": 64,
        "caption_projection_dim": 1536,
        "in_channels": 16,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": num_layers,
        "out_channels": 16,
        "patch_size": 2,
        "pooled_projection_dim": 2048,
        "pos_embed_max_size": 384,
        "qk_norm": "rms_norm",
        "sample_size": 128,
        "enable_end_timestep": True,
    }
    with torch.device("meta"):
        transformer = SD3Transformer2DModel_Distill(**transformer_config)
    transformer.eval()
    transformer.enable_fused_attn()
    transformer.load_state_dict(load_weights(ckpt_path), strict=True, assign=True)
    return transformer


def load_sd3_models(config: InferenceConfig):
    """Load the SD3 models and tokenizers."""
    print(f"Loading SD3 models from {config.model_root}...")
    ckpt_path1 = config.part1
    transformer = get_model(ckpt_path1).to(config.device).to(config.dtype)
    ckpt_path2 = config.part2
    transformer2 = get_model(ckpt_path2).to(config.device).to(config.dtype)
    vae = (
        AutoencoderKL.from_pretrained(
            f"{config.model_root}/vae", torch_dtype=config.dtype
        )
        .to(config.device)
        .to(config.dtype)
    )
    vae.eval()
    tokenizer = CLIPTokenizer.from_pretrained(f"{config.model_root}/tokenizer")
    text_encoder = (
        CLIPTextModelWithProjection.from_pretrained(f"{config.model_root}/text_encoder")
        .to(config.dtype)
        .to(config.device)
    )
    text_encoder.eval()
    tokenizer_2 = CLIPTokenizer.from_pretrained(f"{config.model_root}/tokenizer_2")
    text_encoder_2 = (
        CLIPTextModelWithProjection.from_pretrained(
            f"{config.model_root}/text_encoder_2"
        )
        .to(config.dtype)
        .to(config.device)
    )
    text_encoder_2.eval()
    tokenizer_3 = T5TokenizerFast.from_pretrained(f"{config.model_root}/tokenizer_3")
    text_encoder_3 = (
        T5EncoderModel.from_pretrained(f"{config.model_root}/text_encoder_3")
        .to(config.dtype)
        .to(config.device)
    )
    text_encoder_3.eval()
    return {
        "transformer": transformer,
        "transformer_2": transformer2,
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
        "tokenizer_3": tokenizer_3,
        "text_encoder_3": text_encoder_3,
    }


@torch.no_grad()
def run_validation(
    models,
    prompts,
    config: InferenceConfig,
    height: int,
    width: int,
    num_inference_steps: int = 1,
    num_inference_steps2: int = 1,
    num_images_per_prompt: int = 1,
):
    """Generate validation images."""
    print(f"Starting validation with seed: {config.seed}")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    num_valid_samples = len(prompts)
    batch_size = num_valid_samples * num_images_per_prompt
    noise_gen_cuda = torch.Generator(device=config.device).manual_seed(config.seed)
    latent_shape = (
        batch_size,
        16,
        height // config.vae_scale_factor,
        width // config.vae_scale_factor,
    )
    noisy_input = torch.randn(
        latent_shape, device=config.device, dtype=config.dtype, generator=noise_gen_cuda
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        device=config.device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=False,
        text_encoder=models["text_encoder"],
        tokenizer=models["tokenizer"],
        text_encoder_2=models["text_encoder_2"],
        tokenizer_2=models["tokenizer_2"],
        text_encoder_3=models["text_encoder_3"],
        tokenizer_3=models["tokenizer_3"],
    )
    timesteps = torch.linspace(1.0, 0.6, num_inference_steps + 1, device=config.device)
    timesteps_2 = torch.linspace(
        0.6, 0.0, num_inference_steps2 + 1, device=config.device
    )
    guidance_scale = torch.tensor(
        [config.guidance_scale] * batch_size, device=config.device
    )
    latents = noisy_input.clone()
    transformer = models["transformer"]
    transformer2 = models["transformer_2"]
    transformer.eval()
    transformer2.eval()
    with torch.no_grad():
        for i in range(num_inference_steps):
            timestep = timesteps[i].expand(batch_size) * 1000
            end_timestep = timesteps[i + 1].expand(batch_size) * 1000
            pred_v = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds.to(config.dtype),
                pooled_projections=pooled_prompt_embeds.to(config.dtype),
                timestep=timestep,
                end_timestep=timestep - end_timestep,
                guidance=guidance_scale,
                return_dict=False,
            )[0]
            latents = latents + (timesteps[i + 1] - timesteps[i]) * pred_v
        for i in range(num_inference_steps2):
            timestep = timesteps_2[i].expand(batch_size) * 1000
            end_timestep = timesteps_2[i + 1].expand(batch_size) * 1000
            pred_v = transformer2(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds.to(config.dtype),
                pooled_projections=pooled_prompt_embeds.to(config.dtype),
                timestep=timestep,
                end_timestep=timestep - end_timestep,
                guidance=guidance_scale,
                return_dict=False,
            )[0]
            latents = latents + (timesteps_2[i + 1] - timesteps_2[i]) * pred_v
    images = decode_latent(latents, vae=models["vae"], output_type="latent")
    images = (images.clamp(-1, 1) + 1) / 2.0
    gc.collect()
    torch.cuda.empty_cache()
    return images
