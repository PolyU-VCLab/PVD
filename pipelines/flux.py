from utils.weight_io import load_weights
import torch
from diffusers.image_processor import VaeImageProcessor
from typing import List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torchvision
import gc
from tqdm import tqdm
from ldit.transformer_flux_distillation import FluxTransformer2DModel_Distill
from diffusers import AutoencoderKL
from peft import PeftModel


class FLUX1Pipeline:

    def __init__(self, model_root, part1, part2, backbone):
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self.vae = AutoencoderKL.from_pretrained(f"{model_root}/vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            f"{model_root}/text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(f"{model_root}/tokenizer")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            f"{model_root}/text_encoder_2", torch_dtype=self.dtype
        ).to(self.device)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(f"{model_root}/tokenizer_2")
        self.part1 = part1
        self.part2 = part2
        self.backbone = backbone

    def _prepare_latent_image_ids(self, height, width):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        """Encode prompts using the CLIP and T5 text encoders."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        dtype = self.text_encoder.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        return (prompt_embeds, pooled_prompt_embeds, text_ids)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = self.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

    def _unpack_latents(self, latents, height, width, vae_scale_factor=8):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @torch.no_grad()
    def decode_latent(
        self, latents, height, width, vae_scale_factor=8, output_type="pil"
    ):
        latents = self._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        )
        image = self.vae.decode(latents, return_dict=False)[0]
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        image = image_processor.postprocess(image, output_type=output_type)
        return image

    def get_model(self, ckpt_path, adapter_name=None):
        with torch.device("meta"):
            transformer = FluxTransformer2DModel_Distill(
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
            ).to(self.dtype)
        transformer.load_state_dict(
            load_weights(self.backbone), strict=True, assign=True
        )
        transformer = transformer.to(device=self.device, dtype=self.dtype)
        transformer = PeftModel.from_pretrained(
            transformer, ckpt_path, adapter_name=adapter_name or "default"
        )
        transformer.enable_fused_attn()
        return transformer.eval()

    @torch.no_grad()
    def pvd_infer(
        self,
        prompts,
        num_inference_steps,
        num_inference_steps2,
        height,
        width,
        seed=42,
        guidance=3.5,
    ):
        dtype = self.dtype
        batch_size = len(prompts)
        img_ids = self._prepare_latent_image_ids(height // 8 // 2, width // 8 // 2).to(
            self.device
        )
        noise_gen_cuda = torch.Generator(device="cuda").manual_seed(seed)
        guidance = torch.tensor([guidance] * len(prompts)).to(self.device)
        image_seq = height // (8 * 2) * width // (8 * 2)
        latent_shape = torch.Size([batch_size, image_seq, 64])
        latents = torch.randn(
            latent_shape, generator=noise_gen_cuda, device=self.device
        ).to(dtype)
        prompt_embeds, pooled_prompt_embeds, txt_ids = self.encode_prompt(
            prompt=prompts, prompt_2=prompts, device=self.device
        )
        timesteps1 = torch.linspace(
            1.0, 0.6, num_inference_steps + 1, device=self.device
        )
        timesteps2 = torch.linspace(
            0.6, 0.0, num_inference_steps2 + 1, device=self.device
        )
        ckpt_path1 = self.part1
        transformer1 = self.get_model(ckpt_path1, adapter_name="part1")
        for i in tqdm(range(num_inference_steps)):
            t = timesteps1[i].expand(batch_size)
            end_t = (timesteps1[i] - timesteps1[i + 1]).expand(batch_size)
            input_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_prompt_embeds,
                "timestep": t,
                "end_timestep": end_t,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "guidance": guidance,
                "return_dict": False,
            }
            pred_v = transformer1(**input_kwargs)[0]
            latents = latents + (timesteps1[i + 1] - timesteps1[i]) * pred_v
        del transformer1
        ckpt_path2 = self.part2
        transformer2 = self.get_model(ckpt_path2, adapter_name="part2")
        for i in tqdm(range(num_inference_steps2)):
            t = timesteps2[i].expand(batch_size)
            end_t = (timesteps2[i] - timesteps2[i + 1]).expand(batch_size)
            input_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_prompt_embeds,
                "timestep": t,
                "end_timestep": end_t,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "guidance": guidance,
                "return_dict": False,
            }
            pred_v = transformer2(**input_kwargs)[0]
            latents = latents + (timesteps2[i + 1] - timesteps2[i]) * pred_v
        del transformer2
        images = self.decode_latent(
            latents.float(), height, width, output_type="latent"
        )
        images = (images.clamp(-1, 1) + 1) / 2
        images = torchvision.utils.make_grid(images, nrow=4)
        images = torchvision.transforms.functional.to_pil_image(images.cpu().float())
        gc.collect()
        torch.cuda.empty_cache()
        return images
