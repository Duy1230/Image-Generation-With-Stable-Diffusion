import numpy as np
import torch
import torch.nn as nn


class DDPMScheduler:
    def __init__(
        self,
        random_generator,
        train_timesteps=1000,
        diffusion_beta_start=0.00085,
        diffusion_beta_end=0.012
    ):

        self.betas = torch.linspace(
            diffusion_beta_start ** 0.5, diffusion_beta_end ** 0.5, train_timesteps,
            dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumulative_product = torch.cumprod(self.alphas, dim=0)
        self.one_val = torch.tensor(1.0)
        self.prng_generator = random_generator
        self.total_train_timesteps = train_timesteps
        self.schedule_timesteps = torch.from_numpy(
            np.arange(0, train_timesteps)[::-1].copy())

    def set_steps(self, num_sampling_steps=50):
        self.num_sampling_steps = num_sampling_steps
        step_scaling_factor = self.total_train_timesteps // self.num_sampling_steps
        timesteps_for_sampling = (
            np.arange(0, num_sampling_steps) * step_scaling_factor
        ).round()[::-1].copy().astype(np.int64)
        self.schedule_timesteps = torch.from_numpy(timesteps_for_sampling)

    def _get_prior_timestep(self, current_timestep):
        previous_t = current_timestep - self.total_train_timesteps // self.num_sampling_steps
        return previous_t

    def _calculate_variance(self, timestep):
        prev_t = self._get_prior_timestep(timestep)
        alpha_cumprod_t = self.alphas_cumulative_product[timestep]
        alpha_cumprod_t_prev = self.alphas_cumulative_product[prev_t] if prev_t >= 0 else self.one_val
        beta_t_current = 1 - alpha_cumprod_t / alpha_cumprod_t_prev
        variance_value = (1 - alpha_cumprod_t_prev) / \
            (1 - alpha_cumprod_t) * beta_t_current
        variance_value = torch.clamp(variance_value, min=1e-20)
        return variance_value

    def adjust_strength(self, strength_level=1):
        initial_step_index = self.num_sampling_steps - \
            int(self.num_sampling_steps * strength_level)
        self.schedule_timesteps = self.schedule_timesteps[initial_step_index:]
        self.start_sampling_step = initial_step_index  # Lưu lại bước bắt đầu

    def step(self, current_t, current_latents, model_prediction):
        t = current_t
        prev_t = self._get_prior_timestep(t)

        alpha_cumprod_t = self.alphas_cumulative_product[t]
        alpha_cumprod_t_prev = self.alphas_cumulative_product[prev_t] if prev_t >= 0 else self.one_val
        beta_cumprod_t = 1 - alpha_cumprod_t
        beta_cumprod_t_prev = 1 - alpha_cumprod_t_prev
        alpha_t_current = alpha_cumprod_t / alpha_cumprod_t_prev
        beta_t_current = 1 - alpha_t_current

        predicted_original = (current_latents - beta_cumprod_t **
                              0.5 * model_prediction) / alpha_cumprod_t ** 0.5

        original_coeff = (alpha_cumprod_t_prev ** 0.5 *
                          beta_t_current) / beta_cumprod_t
        current_coeff = alpha_t_current ** 0.5 * beta_cumprod_t_prev / beta_cumprod_t

        predicted_prior_mean = original_coeff * \
            predicted_original + current_coeff * current_latents

        variance_term = 0
        if t > 0:
            target_device = model_prediction.device
            noise_component = torch.randn(
                model_prediction.shape,
                generator=self.prng_generator,
                device=target_device,
                dtype=model_prediction.dtype
            )
            variance_term = (self._calculate_variance(t)
                             ** 0.5) * noise_component

        predicted_prior_sample = predicted_prior_mean + variance_term
        return predicted_prior_sample

    def add_noise(self, initial_samples, noise_timesteps):
        alphas_cumprod = self.alphas_cumulative_product.to(
            device=initial_samples.device,
            dtype=initial_samples.dtype
        )
        noise_timesteps = noise_timesteps.to(initial_samples.device)
        sqrt_alpha_cumprod = alphas_cumprod[noise_timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(
            sqrt_alpha_cumprod.shape[0], *([1] * (initial_samples.ndim - 1))
        )
        sqrt_one_minus_alpha_cumprod = (
            1 - alphas_cumprod[noise_timesteps]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(
            sqrt_one_minus_alpha_cumprod.shape[0], *
            ([1] * (initial_samples.ndim - 1))
        )
        random_noise = torch.randn(
            initial_samples.shape, generator=self.prng_generator,
            device=initial_samples.device, dtype=initial_samples.dtype
        )
        noisy_result = sqrt_alpha_cumprod * initial_samples + \
            sqrt_one_minus_alpha_cumprod * random_noise
        return noisy_result, random_noise
