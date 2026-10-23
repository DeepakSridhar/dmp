"""DDIM sampling for DMP with classifier-free guidance, concept composition (Eq. 5)
and test-time steering toward a repository anchor (Eq. 7)."""
import torch


class DDIMSampler:
    def __init__(self, model, use_ema=True):
        self.model = model
        self.use_ema = use_ema

    def _eps(self, x, t, contexts, weights, uncond):
        """Eq. 5: eps = eps(x, null) + sum_i eta_i * (eps(x, y_i) - eps(x, null)).

        One context with weight s is plain classifier-free guidance with scale s; several
        contexts compose concepts; a negative weight negates a concept. Without `uncond`
        the single conditional prediction is returned as is.
        """
        if uncond is None:
            assert len(contexts) == 1, "composition needs an unconditional context"
            return self.model.apply_model(x, t, contexts[0], use_ema=self.use_ema)
        n = len(contexts) + 1
        out = self.model.apply_model(x.repeat(n, 1, 1), t.repeat(n), torch.cat([uncond, *contexts]),
                                     use_ema=self.use_ema)
        eps_uncond, eps_conds = out[: x.shape[0]], out[x.shape[0]:].chunk(len(contexts))
        eps = eps_uncond.clone()
        for weight, eps_cond in zip(weights, eps_conds):
            eps = eps + weight * (eps_cond - eps_uncond)
        return eps

    @torch.no_grad()
    def sample(self, contexts, weights=None, uncond=None, steps=50, eta=0.0, x_T=None,
               anchor=None, steering_strength=0.0, generator=None):
        """Sample (B, channels, length) prompts/latents in the model's (scaled) space.

        contexts: one (B, tokens, dim) tensor or a list of them (composition).
        weights:  guidance scale(s) eta_i for each context (default 1.0 each).
        uncond:   (B, tokens, dim) null conditioning; required for guidance/composition.
        anchor:   optional (B or 1, channels, length) steering target x* in the scaled space.
        steering_strength: lambda in Eq. 7; 0 disables steering.
        """
        model = self.model
        if torch.is_tensor(contexts):
            contexts = [contexts]
        weights = [1.0] * len(contexts) if weights is None else list(weights)
        assert len(weights) == len(contexts)
        batch, device = contexts[0].shape[0], contexts[0].device

        timesteps = torch.arange(0, model.num_timesteps, model.num_timesteps // steps) + 1
        timesteps = timesteps[timesteps < model.num_timesteps]
        alphas = model.alphas_cumprod.cpu()[timesteps]
        alphas_prev = torch.cat([model.alphas_cumprod.cpu()[:1], alphas[:-1]])
        sigmas = eta * ((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)).sqrt()

        x = x_T if x_T is not None else torch.randn((batch, *model.sample_shape), device=device,
                                                      generator=generator)
        steer = anchor is not None and steering_strength > 0.0
        if steer:
            anchor = anchor.to(device=device, dtype=x.dtype)

        for index in reversed(range(len(timesteps))):
            t = torch.full((batch,), int(timesteps[index]), device=device, dtype=torch.long)
            a_t, a_prev, sigma_t = alphas[index].item(), alphas_prev[index].item(), sigmas[index].item()
            eps = self._eps(x, t, contexts, weights, uncond)
            pred_x0 = (x - (1.0 - a_t) ** 0.5 * eps) / a_t ** 0.5
            if steer:  # Eq. 7: pull the denoised estimate toward the anchor
                pred_x0 = (1.0 - steering_strength) * pred_x0 + steering_strength * anchor
            x = a_prev ** 0.5 * pred_x0 + max(1.0 - a_prev - sigma_t ** 2, 0.0) ** 0.5 * eps
            if sigma_t > 0:
                x = x + sigma_t * torch.randn(x.shape, device=device, generator=generator)
        return x
