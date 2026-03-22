"""
Authors: Ruijie He, Botong Cai, Ziqi Yang
"""

import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial


# Helper functions
def default(val, d):
    return val if val is not None else (d() if callable(d) else d)


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2):
    if schedule == "sigmoid":
        start, end, tau = -3, 3, 1
        t = torch.linspace(0, n_timestep, n_timestep + 1, dtype=torch.float64) / n_timestep
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()
    return torch.linspace(linear_start, linear_end, n_timestep)


class Network(nn.Module):
    def __init__(self, unet, beta_schedule):
        super(Network, self).__init__()
        self.denoise_fn = unet
        self.beta_schedule = beta_schedule
        self.loss_fn = nn.MSELoss()

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """ Compute and register noise schedule constants """
        betas = make_beta_schedule(**self.beta_schedule[phase])
        self.betas = betas  # Stored for DPM-Solver
        alphas = 1. - betas.numpy() if isinstance(betas, torch.Tensor) else 1. - betas
        gammas = np.cumprod(alphas, axis=0)

        # Register gammas buffer for training
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32, device=device))

    def forward(self, y_0, y_cond, noise=None):
        """ Training logic with continuous timestep sampling """
        b, *_ = y_0.shape
        # 1. Sample discrete timestep t
        t = torch.randint(1, len(self.gammas), (b,), device=y_0.device).long()

        # 2. Continuous gamma interpolation between t-1 and t
        gamma_t1 = self.gammas[t - 1].view(b, 1)
        gamma_t2 = self.gammas[t].view(b, 1)
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1

        # 3. Forward diffusion: x_t = sqrt(gamma)*x_0 + sqrt(1-gamma)*noise
        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = sample_gammas.view(-1, 1, 1, 1).sqrt() * y_0 + \
                  (1 - sample_gammas).view(-1, 1, 1, 1).sqrt() * noise

        # 4. Predict x_0 and compute MSE loss
        y_0_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        return self.loss_fn(y_0, y_0_hat)

    @torch.no_grad()
    def restoration(self, y_cond):
        """ Inference using DPM-Solver with manual mathematical alignment """
        from dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver

        self.denoise_fn.eval()
        device = y_cond.device

        # 1. Setup noise schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas.to(device))

        # 2. Custom model function wrapper for DPM-Solver
        def custom_model_fn(x, t_continuous):
            # x: Current noisy image [B, 3, H, W]
            # t_continuous: Continuous time [B] from 1.0 (noise) to 0.0 (clean)

            # (A) Get marginal coefficients for current time
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)

            # (B) Convert to gamma_t (equivalent to training sample_gammas)
            gamma_t = (alpha_t ** 2).view(-1, 1).to(device)

            # (C) Concatenate: 9 condition channels + 3 noisy channels = 12 channels
            inp = torch.cat([y_cond, x], dim=1)

            # (D) Predict clean image (x_0)
            x0_pred = self.denoise_fn(inp, gamma_t)

            # (E) Convert predicted x_0 to noise for DPM-Solver engine
            # x_t = alpha_t * x_0 + sigma_t * noise  =>  noise = (x_t - alpha_t * x_0) / sigma_t
            alpha_t = alpha_t.view(-1, 1, 1, 1).to(device)
            sigma_t = sigma_t.view(-1, 1, 1, 1).to(device)

            noise_pred = (x - alpha_t * x0_pred) / sigma_t

            return noise_pred

        # 3. Initialize pure noise
        y_t = torch.randn((y_cond.shape[0], 3, y_cond.shape[2], y_cond.shape[3]), device=device)

        # 4. Instantiate solver with custom function
        dpm_solver = DPM_Solver(custom_model_fn, noise_schedule, algorithm_type="dpmsolver++")

        # 5. Execute 20-step fast sampling
        return dpm_solver.sample(y_t, steps=20, order=2, denoise_to_zero=True)