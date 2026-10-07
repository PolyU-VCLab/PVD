# PVD SD3.5 training losses.
# PVD text-to-image and discriminator objectives; time-sampling and loss helpers are adapted from FACM losses.py (https://github.com/ali-vilab/FACM)

import torch
import torch.nn.functional as F
import numpy as np
from utils import mean_flat


def _unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def _teacher_reference_model(compiled_model, freezed_teacher):
    if freezed_teacher is None:
        return compiled_model
    return _unwrap_module(freezed_teacher)


class PVDLossSD35:
    """PVD training loss with helpers adapted from FACM"""

    def sample_t(self, device, size=1, type="default", args=None):
        """Sample time values according to different distributions"""
        if type == "default":
            P_mean, P_std = (-args.mean, args.std)
            sigma = torch.randn(size).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()
            samples = torch.arctan(sigma) * (2.0 / np.pi)
        elif type == "log":
            mu, sigma = (-args.mean, args.std)
            normal_samples = torch.randn(size, 1, device=device) * sigma + mu
            samples = 1 / (1 + torch.exp(-normal_samples))
        else:
            raise ValueError(f"Invalid sample type: {type}")
        return 1 - samples.to(device).reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def get_velocity(self, x_t, target, t, y, un_y, model, cfgw, args):
        """Get velocity from teacher model or ground truth with CFG"""
        cuda_state = torch.cuda.get_rng_state()
        input_kwargs = {
            "hidden_states": torch.cat([x_t] * 2).bfloat16(),
            "encoder_hidden_states": torch.cat(
                [un_y["encoder_hidden_states"], y["encoder_hidden_states"]]
            ).bfloat16(),
            "pooled_projections": torch.cat(
                [un_y["pooled_projections"], y["pooled_projections"]]
            ).bfloat16(),
            "timestep": torch.cat([t.view(-1) * 1000] * 2),
            "return_dict": False,
        }
        v = model(**input_kwargs)[0]
        v_uncond, v_cond = torch.chunk(v, 2, dim=0)
        velocity = v_uncond + cfgw * (v_cond - v_uncond)
        return (velocity, cuda_state)

    def norm_l2_loss(self, pred, target, p=0.5, c=0.001):
        """Norm L2 loss with outlier resistance"""
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        loss = e / (e + c).pow(p).detach()
        return loss

    def flow_matching_loss(self, pred, target):
        """Flow Matching loss: MSE + cosine similarity"""
        mse_loss = mean_flat((pred - target) ** 2)
        cos_loss = mean_flat(1 - F.cosine_similarity(pred, target, dim=1, eps=1e-06))
        return mse_loss + cos_loss

    def get_random_t_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u = self.sample_t(device, size=b, type=args.t_type, args=args)
        return t_begin + u * (t_end - t_begin)

    def get_two_timesteps_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u1 = self.get_random_t_between(t_begin, t_end, device, args, b)
        u2 = self.get_random_t_between(t_begin, t_end, device, args, b)
        t, r = (torch.maximum(u1, u2), torch.minimum(u1, u2))
        return (t, r)

    @staticmethod
    def scale_noise(image, timestep, noise):
        xt = (1.0 - timestep) * image + timestep * noise
        return xt.to(image.dtype)

    def __call__(
        self,
        accelerator,
        model,
        compiled_model,
        ema,
        images,
        iters,
        model_kwargs=None,
        args=None,
        freezed_teacher=None,
        epoch=0,
        time_interval=[1.0, 0.0],
    ):
        """PVD training loss computation"""
        if model_kwargs is None:
            model_kwargs = {}
        unwrapped_model = model.module if hasattr(model, "module") else model
        reference_model = _teacher_reference_model(compiled_model, freezed_teacher)
        b, c, h, w = images.shape
        device = images.device
        y, un_y = (model_kwargs["y"], model_kwargs["un_y"])
        guidance = torch.tensor([args.cfgw] * b, device=device)
        noise = torch.randn_like(images)
        t_begin, t_end = (time_interval[0], time_interval[1])
        t, r = self.get_two_timesteps_between(t_begin, t_end, device, args, b)
        x_t = self.scale_noise(images, t, noise)
        target = noise - images
        v, cuda_state = self.get_velocity(
            x_t, target, t, y, un_y, reference_model, args.cfgw, args
        )
        torch.cuda.set_rng_state(cuda_state)
        input_kwargs = {
            "hidden_states": x_t,
            "encoder_hidden_states": y["encoder_hidden_states"],
            "pooled_projections": y["pooled_projections"],
            "guidance": guidance,
            "timestep": t.view(-1) * 1000,
            "end_timestep": (t - t).view(-1) * 1000,
            "return_dict": False,
        }
        F_fm = compiled_model(**input_kwargs)[0]
        fm_loss = self.flow_matching_loss(F_fm, v)

        def model_wrapper(x_input, t_input, r_input):
            torch.cuda.set_rng_state(cuda_state)
            input_kwargs = {
                "hidden_states": x_input,
                "encoder_hidden_states": y["encoder_hidden_states"],
                "pooled_projections": y["pooled_projections"],
                "guidance": guidance,
                "timestep": t_input.view(-1) * 1000,
                "end_timestep": (t_input - r_input).view(-1) * 1000,
                "return_dict": False,
            }
            output = model(**input_kwargs)[0]
            return output

        v_x = v
        v_t = torch.ones_like(t)
        v_r = torch.zeros_like(r)
        with torch.no_grad():
            unwrapped_model.disable_fused_attn()
            F_avg, F_avg_grad = torch.func.jvp(
                model_wrapper, (x_t, t, r), (v_x, v_t, v_r)
            )
            unwrapped_model.enable_fused_attn()
        F_avg = model_wrapper(x_t, t, r)
        F_avg_grad = F_avg_grad.detach()
        F_avg_sg = F_avg.detach()
        v_bar = v + (r - t) * F_avg_grad
        g = F_avg_sg - v_bar
        alpha = 1 - (1 - t) ** args.p
        target = F_avg_sg - alpha * g.clamp(min=-1, max=1)
        beta = torch.cos((1 - t) * np.pi / 2).flatten()
        cm_loss = self.norm_l2_loss(F_avg, target) * beta.flatten()
        t_fake = torch.ones(b, 1, 1, 1).to(device) * t_begin
        r_fake = self.get_random_t_between(t_begin, t_end, device, args, b)
        z_fake = self.scale_noise(images, t_fake, torch.randn_like(images))
        input_kwargs = {
            "hidden_states": z_fake,
            "encoder_hidden_states": y["encoder_hidden_states"],
            "pooled_projections": y["pooled_projections"],
            "guidance": guidance,
            "timestep": t_fake.view(-1) * 1000,
            "end_timestep": (t_fake - r_fake).view(-1) * 1000,
            "return_dict": False,
        }
        F_fm = compiled_model(**input_kwargs)[0]
        x_fake = z_fake + (r_fake - t_fake) * F_fm
        discriminator_input_kwargs = {
            "hidden_states": x_fake.bfloat16(),
            "encoder_hidden_states": y["encoder_hidden_states"].bfloat16(),
            "pooled_projections": y["pooled_projections"].bfloat16(),
            "timestep": r_fake.view(-1) * 1000,
            "return_dict": False,
            "return_discriminator": True,
        }
        pred_fake = reference_model(**discriminator_input_kwargs)
        adv_loss = -torch.mean(pred_fake)
        adv_weight = float(getattr(args, "adv_weight", 0.1))
        layer_loss = [
            cm_loss.detach().mean().item(),
            fm_loss.detach().mean().item(),
            adv_loss.detach().mean().item(),
        ]
        total_loss = cm_loss.mean() + fm_loss.mean() + adv_loss.mean() * adv_weight
        return (total_loss, cm_loss.mean(), fm_loss.mean(), layer_loss)


class DiscriminatorLossT2I:
    """PVD training loss with helpers adapted from FACM"""

    def sample_t(self, device, size=1, type="default", args=None):
        """Sample time values according to different distributions"""
        if type == "default":
            P_mean, P_std = (-args.mean, args.std)
            sigma = torch.randn(size).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()
            samples = torch.arctan(sigma) * (2.0 / np.pi)
        elif type == "log":
            mu, sigma = (-args.mean, args.std)
            normal_samples = torch.randn(size, 1, device=device) * sigma + mu
            samples = 1 / (1 + torch.exp(-normal_samples))
        else:
            raise ValueError(f"Invalid sample type: {type}")
        return 1 - samples.to(device).reshape(-1, 1, 1, 1)

    def norm_l2_loss(self, pred, target, p=0.5, c=0.001):
        """Norm L2 loss with outlier resistance"""
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        loss = e / (e + c).pow(p).detach()
        return loss

    def flow_matching_loss(self, pred, target):
        """Flow Matching loss: MSE + cosine similarity"""
        mse_loss = mean_flat((pred - target) ** 2)
        cos_loss = mean_flat(1 - F.cosine_similarity(pred, target, dim=1, eps=1e-06))
        return mse_loss + cos_loss

    def get_random_t_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u = self.sample_t(device, size=b, type=args.t_type, args=args)
        return t_begin + u * (t_end - t_begin)

    def get_two_timesteps_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u1 = self.get_random_t_between(t_begin, t_end, device, args, b)
        u2 = self.get_random_t_between(t_begin, t_end, device, args, b)
        t, r = (torch.maximum(u1, u2), torch.minimum(u1, u2))
        return (t, r)

    @staticmethod
    def scale_noise(image, timestep, noise):
        xt = (1.0 - timestep) * image + timestep * noise
        return xt.to(image.dtype)

    def __call__(
        self,
        accelerator,
        model,
        compiled_model,
        ema,
        images,
        iters,
        model_kwargs=None,
        args=None,
        freezed_teacher=None,
        epoch=0,
        time_interval=[1.0, 0.0],
    ):
        """PVD training loss computation"""
        if model_kwargs is None:
            model_kwargs = {}
        unwrapped_model = model.module if hasattr(model, "module") else model
        reference_model = _teacher_reference_model(compiled_model, freezed_teacher)
        b, c, h, w = images.shape
        device = images.device
        y = model_kwargs["y"]
        guidance = torch.tensor([args.cfgw] * b, device=device)
        t_begin, t_end = (time_interval[0], time_interval[1])
        r_real = self.get_random_t_between(t_begin, t_end, device, args, b)
        x_true = self.scale_noise(images, r_real, torch.randn_like(images))
        true_input_kwargs = {
            "hidden_states": x_true.bfloat16(),
            "encoder_hidden_states": y["encoder_hidden_states"].bfloat16(),
            "pooled_projections": y["pooled_projections"].bfloat16(),
            "timestep": r_real.view(-1) * 1000,
            "return_dict": False,
            "return_discriminator": True,
        }
        pred_true = reference_model(**true_input_kwargs)
        with torch.no_grad():
            t_fake = torch.ones(b, 1, 1, 1).to(device) * t_begin
            r_fake = self.get_random_t_between(t_begin, t_end, device, args, b)
            z_fake = self.scale_noise(images, t_fake, torch.randn_like(images))
            input_kwargs = {
                "hidden_states": z_fake,
                "encoder_hidden_states": y["encoder_hidden_states"],
                "pooled_projections": y["pooled_projections"],
                "guidance": guidance,
                "timestep": t_fake.view(-1) * 1000,
                "end_timestep": (t_fake - r_fake).view(-1) * 1000,
                "return_dict": False,
            }
            F_fm = unwrapped_model(**input_kwargs)[0]
            x_fake = z_fake + (r_fake - t_fake) * F_fm
        fake_input_kwargs = {
            "hidden_states": x_fake.bfloat16(),
            "encoder_hidden_states": y["encoder_hidden_states"].bfloat16(),
            "pooled_projections": y["pooled_projections"].bfloat16(),
            "timestep": r_fake.view(-1) * 1000,
            "return_dict": False,
            "return_discriminator": True,
        }
        pred_fake = reference_model(**fake_input_kwargs)
        loss_real = torch.mean(F.relu(1.0 - pred_true))
        loss_fake = torch.mean(F.relu(1.0 + pred_fake))
        disc_loss = (loss_real + loss_fake) / 2
        return (disc_loss, loss_real, loss_fake)
