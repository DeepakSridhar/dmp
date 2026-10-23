"""Diffusion Meta-Prompt model: a conditional DDPM over prompt (latent) vectors.

Implements the forward process (Eq. 1) and the denoising objective (Eq. 2) of the paper.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNetModel


class PromptDiffusion(nn.Module):
    """eps-prediction DDPM whose denoiser is a 1D UNet with cross-attention on the task text.

    `x` always has shape (B, channels, length); the conditioning `context` has shape
    (B, tokens, context_dim). `scale_factor` rescales the data to roughly unit variance.
    """

    def __init__(self, unet_config, timesteps=1000, linear_start=0.0015, linear_end=0.0195,
                 scale_factor=1.0, ema_decay=0.9999):
        super().__init__()
        self.unet = UNetModel(**unet_config)
        self.ema_unet = copy.deepcopy(self.unet).requires_grad_(False).eval()
        self.ema_decay = ema_decay
        self.num_timesteps = timesteps
        self.sample_shape = (unet_config["in_channels"], unet_config["image_size"])
        self.register_buffer("scale_factor", torch.tensor(float(scale_factor)))
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float())

    def q_sample(self, x_start, t, noise):
        """Eq. 1: x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) eps."""
        return (self.sqrt_alphas_cumprod[t][:, None, None] * x_start
                + self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise)

    def apply_model(self, x, t, context, use_ema=False):
        return (self.ema_unet if use_ema else self.unet)(x, t, context)

    def loss(self, x_start, context):
        """Eq. 2: E || eps - eps_theta(x_t, tau(y), t) ||^2. `x_start` must already be scaled."""
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        return F.mse_loss(self.apply_model(self.q_sample(x_start, t, noise), t, context), noise)

    @torch.no_grad()
    def update_ema(self):
        self.ema_updates += 1
        n = float(self.ema_updates)
        decay = min(self.ema_decay, (1.0 + n) / (10.0 + n))
        for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_p.sub_((1.0 - decay) * (ema_p - p))
