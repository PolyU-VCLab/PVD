from __future__ import annotations

"PVD inference for the distilled Qwen-Image backbone and adapters."
from utils.weight_io import load_weights
import gc
import os
import torch
import torchvision
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLQwenImage
from peft import PeftModel
from ldit.transformer_qwenimage_distillation import QwenImageTransformer2DModel_Distill


class PVDQwenImagePipeline:

    def __init__(
        self,
        model_root="models/Qwen-Image",
        device=None,
        dtype=torch.bfloat16,
        backbone="weights/pvd_qwenimage/backbone.safetensors",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_root = model_root
        self.device = torch.device(device)
        self.dtype = dtype
        self.vae_scale_factor = 8
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.tokenizer_max_length = 1024
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            f"{model_root}/vae", torch_dtype=dtype
        ).to(self.device)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_root}/text_encoder", torch_dtype=dtype
        ).to(self.device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(f"{model_root}/tokenizer")
        self.vae.eval()
        self.text_encoder.eval()
        self._pvd_ckpt_cache: dict[str, torch.nn.Module] = {}
        self._pvd_ckpt_cache_enabled = os.environ.get(
            "QWENIMAGE_PVD_CKPT_CACHE", "0"
        ).strip() not in ("0", "false", "False", "no", "NO")
        self.backbone = backbone

    def _get_or_load_pvd_model(self, ckpt_path: str) -> torch.nn.Module:
        if not self._pvd_ckpt_cache_enabled:
            return self.load_model(ckpt_path)
        key = os.path.realpath(os.path.abspath(ckpt_path))
        if key not in self._pvd_ckpt_cache:
            self._pvd_ckpt_cache[key] = self.load_model(ckpt_path)
        return self._pvd_ckpt_cache[key]

    def _extract_masked_hidden(self, hidden_states, mask):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwenimage_prompt_embeds(self, prompt, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text = [self.prompt_template_encode.format(p) for p in prompt]
        text_inputs = self.tokenizer(
            text,
            max_length=self.tokenizer_max_length
            + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(
            hidden_states, text_inputs.attention_mask
        )
        split_hidden = [
            h[self.prompt_template_encode_start_idx :] for h in split_hidden
        ]
        attn_mask_list = [
            torch.ones(h.size(0), dtype=torch.long, device=device) for h in split_hidden
        ]
        max_seq_len = max((h.size(0) for h in split_hidden))
        prompt_embeds = torch.stack(
            [
                torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))])
                for h in split_hidden
            ]
        )
        prompt_embeds_mask = torch.stack(
            [
                torch.cat([m, m.new_zeros(max_seq_len - m.size(0))])
                for m in attn_mask_list
            ]
        )
        return (prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask)

    @torch.no_grad()
    def encode_prompt(self, prompt, device=None, max_sequence_length=1024):
        prompt_embeds, prompt_embeds_mask = self._get_qwenimage_prompt_embeds(
            prompt, device=device, dtype=self.dtype
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None
        return (prompt_embeds, prompt_embeds_mask)

    def _unpack_latents(self, latents, height, width):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, height, width)
        return latents

    @torch.no_grad()
    def decode_latent(self, latents, height, width, output_type="pt"):
        latents = self._unpack_latents(latents, height, width)
        latents_mean = torch.tensor(
            self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=latents.device, dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        return image_processor.postprocess(image, output_type=output_type)

    def _build_model(self):
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
        with torch.device("meta"):
            model = QwenImageTransformer2DModel_Distill(**config)
        model.load_state_dict(load_weights(self.backbone), strict=True, assign=True)
        rope = model.pos_embed
        model.pos_embed = type(rope)(
            theta=10000, axes_dim=[16, 56, 56], scale_rope=True
        )
        model.enable_fused_attn()
        return model.to(device=self.device, dtype=self.dtype).eval()

    def load_model(self, ckpt_path):
        model = self._build_model()
        model = PeftModel.from_pretrained(model, ckpt_path)
        return model.eval()

    def _make_img_shapes(self, batch_size, height, width):
        h = height // self.vae_scale_factor // 2
        w = width // self.vae_scale_factor // 2
        return [[(1, h, w)]] * batch_size

    def _sample_stage(
        self,
        model,
        latents,
        prompt_embeds,
        prompt_embeds_mask,
        img_shapes,
        timesteps,
        height,
        width,
    ):
        batch_size = latents.shape[0]
        for i in tqdm(range(len(timesteps) - 1)):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            input_kwargs = {
                "hidden_states": latents,
                "timestep": t_cur.expand(batch_size),
                "end_timestep": (t_cur - t_next).expand(batch_size),
                "encoder_hidden_states": prompt_embeds,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "img_shapes": img_shapes,
                "return_dict": False,
            }
            pred_v = model(**input_kwargs)[0]
            latents = latents + (t_next - t_cur) * pred_v
        return latents

    @torch.no_grad()
    def pvd_infer(
        self,
        prompts,
        ckpt_path1,
        ckpt_path2,
        steps1=1,
        steps2=1,
        intervals1=(1.0, 0.6),
        intervals2=(0.6, 0.0),
        height=1024,
        width=1024,
        seed=42,
    ):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        batch_size = len(prompts)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompts, device=self.device
        )
        img_shapes = self._make_img_shapes(batch_size, height, width)
        image_seq = (
            height
            // (self.vae_scale_factor * 2)
            * (width // (self.vae_scale_factor * 2))
        )
        latent_shape = (batch_size, image_seq, 64)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            latent_shape, generator=generator, device=self.device, dtype=self.dtype
        )
        timesteps1 = torch.linspace(
            intervals1[0],
            intervals1[1],
            steps1 + 1,
            device=self.device,
            dtype=self.dtype,
        )
        timesteps2 = torch.linspace(
            intervals2[0],
            intervals2[1],
            steps2 + 1,
            device=self.device,
            dtype=self.dtype,
        )
        model1 = self._get_or_load_pvd_model(ckpt_path1)
        latents = self._sample_stage(
            model1,
            latents,
            prompt_embeds,
            prompt_embeds_mask,
            img_shapes,
            timesteps1,
            height,
            width,
        )
        if not self._pvd_ckpt_cache_enabled:
            del model1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model2 = self._get_or_load_pvd_model(ckpt_path2)
        latents = self._sample_stage(
            model2,
            latents,
            prompt_embeds,
            prompt_embeds_mask,
            img_shapes,
            timesteps2,
            height,
            width,
        )
        if not self._pvd_ckpt_cache_enabled:
            del model2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        images = self.decode_latent(latents, height, width, output_type="latent")
        images = (images.clamp(-1, 1) + 1) / 2.0
        grid = torchvision.utils.make_grid(images, nrow=min(4, batch_size))
        grid_pil = torchvision.transforms.functional.to_pil_image(grid.cpu().float())
        return (images, grid_pil)
