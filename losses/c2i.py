# PVD C2I interval-conditioned training loss.
# Adapted from FACM losses.py (https://github.com/ali-vilab/FACM); retains upstream time-sampling and loss helpers.

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
from utils import mean_flat


def make_overlapping_intervals(
    n_segments: int,
    overlap: Union[float, List[float]] = 0.25,
    segment_lengths: Optional[Union[List[float], torch.Tensor]] = None,
    device: str = "cpu",
):
    "Construct overlapping time intervals [start, end].\n\n    Args:\n        n_segments: Number of intervals.\n        overlap: Global overlap fraction or one value per adjacent interval pair.\n        segment_lengths: Relative interval lengths, normalized to sum to one.\n            None selects equally sized intervals.\n        device: Device for the returned tensor.\n    Returns:\n        Tensor of shape (n_segments, 2) containing interval boundaries.\n"
    if isinstance(overlap, float):
        assert 0 <= overlap <= 1, "Overlap must be in [0, 1]"
        overlap = [overlap] * (n_segments - 1)
    else:
        assert (
            isinstance(overlap, list) and len(overlap) == n_segments - 1
        ), f"Overlap list length must be {n_segments - 1} (n_segments-1)"
        assert all((0 <= o < 1 for o in overlap)), "All overlaps must be in [0, 1)"
    if segment_lengths is None:
        base_len = 1.0 / (n_segments - sum(overlap) * (1.0 / n_segments) * n_segments)
        seg_lengths = [base_len] * n_segments
    else:
        seg_lengths = torch.tensor(
            segment_lengths, dtype=torch.float32, device=device
        ).flatten()
        assert (
            len(seg_lengths) == n_segments
        ), f"Segment lengths must have {n_segments} elements"
        assert torch.all(seg_lengths > 0), "Segment lengths must be positive"
        seg_lengths = (seg_lengths / seg_lengths.sum()).tolist()
    intervals = []
    start = 0.0
    for i in range(n_segments):
        seg_len = seg_lengths[i]
        end = start + seg_len
        intervals.append((start, end))
        if i < n_segments - 1:
            start = end - seg_len * overlap[i]
    intervals[-1] = (intervals[-1][0], 1.0)
    return torch.tensor(intervals, dtype=torch.float32, device=device)


class InsideFACMLoss:
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
    def get_velocity(self, x_t, target, t, y, model, cfgw, args):
        """Get velocity from teacher model or ground truth with CFG"""
        null_label = 1000
        mask = (t.flatten() < args.glow) | (t.flatten() > args.ghigh)
        y_null = torch.where(mask, y, torch.full_like(y, null_label))
        if cfgw > 1.0:
            cuda_state = torch.cuda.get_rng_state()
            v_cond = (
                model(x_t, t.reshape(-1), t.reshape(-1), y) if args.distill else target
            )
            torch.cuda.set_rng_state(cuda_state)
            v_uncond = model(x_t, t.reshape(-1), t.reshape(-1), y_null)
            velocity = v_uncond + cfgw * (v_cond - v_uncond)
        else:
            cuda_state = torch.cuda.get_rng_state()
            velocity = (
                model(x_t, t.reshape(-1), t.reshape(-1), y) if args.distill else target
            )
        return (velocity, cuda_state)

    def norm_l2_loss(self, pred, target, p=0.5, c=0.001):
        """Norm L2 loss with outlier resistance"""
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        loss = e / (e + c).pow(p).detach()
        return loss

    def flow_matching_loss(self, pred, target):
        """Flow Matching loss: MSE + cosine similarity"""
        mse_loss = mean_flat((pred - target) ** 2)
        cos_loss = mean_flat(1 - F.cosine_similarity(pred, target, dim=1))
        return mse_loss + cos_loss

    def get_random_t_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u = self.sample_t(device, size=b, type=args.t_type, args=args)
        return t_begin + u * (t_end - t_begin)

    def get_two_timesteps_between(self, t_begin, t_end, device, args, b):
        """Get random time between t_begin and t_end"""
        u1 = self.get_random_t_between(t_begin, t_end, device, args, b)
        u2 = self.get_random_t_between(t_begin, t_end, device, args, b)
        t, r = (torch.minimum(u1, u2), torch.maximum(u1, u2))
        return (t, r)

    @staticmethod
    def scale_noise(image, timestep, noise):
        return (1.0 - timestep) * noise + timestep * image

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
        intervals=None,
        overlap=0.0,
    ):
        """PVD training loss computation"""
        if model_kwargs is None:
            model_kwargs = {}
        unwrapped_model = model.module if hasattr(model, "module") else model
        reference_model = compiled_model if freezed_teacher is None else freezed_teacher
        b, c, h, w = images.shape
        device = images.device
        num_steps = len(unwrapped_model.div_blocks)
        assert num_steps == len(
            intervals
        ), "Intervals length must match number of steps"
        assert len(intervals) == num_steps, "Segment lengths must match number of steps"
        time_interval = make_overlapping_intervals(
            num_steps, overlap=overlap, segment_lengths=intervals, device=device
        )
        layer_loss = []
        total_fm_loss, total_cm_loss = (0.0, 0.0)
        cache = {"x_latent1": None, "c_pre1": None, "x_latent2": None, "c_pre2": None}
        for i in range(num_steps):
            noise = torch.randn_like(images)
            t_begin, t_end = (time_interval[i][0], time_interval[i][1])
            t, r = self.get_two_timesteps_between(t_begin, t_end, device, args, b)
            x_t = self.scale_noise(images, t, noise)
            target = images - noise
            v, cuda_state = self.get_velocity(
                x_t, target, t, model_kwargs["y"], reference_model, args.cfgw, args
            )
            torch.cuda.set_rng_state(cuda_state)
            F_fm, x_latent1, c_pre1 = compiled_model(
                x_t,
                t.reshape(-1),
                t.reshape(-1),
                model_kwargs["y"],
                index=i,
                x_latent=cache["x_latent1"],
                c_pre=cache["c_pre1"],
            )
            cache["x_latent1"] = x_latent1
            cache["c_pre1"] = c_pre1
            fm_loss = self.flow_matching_loss(F_fm, v)

            def model_wrapper(x_input, t_input, r_input):
                torch.cuda.set_rng_state(cuda_state)
                output, x_latent2, c_pre2 = unwrapped_model(
                    x_input,
                    t_input.reshape(-1),
                    r_input.reshape(-1),
                    model_kwargs["y"],
                    index=i,
                    x_latent=cache["x_latent2"],
                    c_pre=cache["c_pre2"],
                )
                cache["x_latent2"] = x_latent2
                cache["c_pre2"] = c_pre2
                return output

            v_x = v
            v_t = torch.ones_like(t)
            v_r = torch.zeros_like(r)
            unwrapped_model.disable_fused_attn()
            F_avg, F_avg_grad = torch.func.jvp(
                model_wrapper, (x_t, t, r), (v_x, v_t, v_r)
            )
            unwrapped_model.enable_fused_attn()
            F_avg_grad = F_avg_grad.detach()
            F_avg_sg = F_avg.detach()
            v_bar = v + (r - t) * F_avg_grad
            g = F_avg_sg - v_bar
            alpha = 1 - t**args.p
            target = F_avg_sg - alpha * g.clamp(min=-1, max=1)
            beta = torch.cos(t * np.pi / 2).flatten()
            cm_loss = self.norm_l2_loss(F_avg, target) * beta.flatten()
            total_cm_loss = total_cm_loss + cm_loss
            total_fm_loss = total_fm_loss + fm_loss
            layer_loss.append(cm_loss.detach().mean().item())
            layer_loss.append(fm_loss.detach().mean().item())
        total_loss = (total_cm_loss.mean() + total_fm_loss.mean()) / num_steps
        return (total_loss, total_cm_loss.mean(), total_fm_loss.mean(), layer_loss)
