"""DDPM noise schedule + a DDIM sampler that exposes the denoising chain as an MDP.

Each reverse step x_t -> x_{t-1} is a Gaussian policy pi_theta(x_{t-1} | x_t, t) with mean
mu_theta(x_t, t) and fixed std sigma_t (eta=1 DDIM == DDPM posterior on a sub-sequence).
This is exactly the formulation used by DDPO (Black et al. 2023) and DPOK (Fan et al. 2023).
"""
import math
import torch


def cosine_alphas_cumprod(T=1000, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class Diffusion:
    def __init__(self, T=1000, device="cpu", clip_x0=False, t_min=0):
        self.T = T
        self.clip_x0 = clip_x0
        self.t_min = t_min
        self.device = device
        self.ab = cosine_alphas_cumprod(T).to(device)

    # ---------- training ----------
    def q_sample(self, x0, t, noise):
        ab = self.ab[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0, weights=None):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        per = ((pred - noise) ** 2).mean(dim=(1, 2, 3))
        if weights is None:
            return per.mean()
        return (weights * per).sum() / weights.sum()

    # ---------- sampling as an MDP ----------
    def timesteps(self, n_steps):
        ts = torch.linspace(self.T - 1, self.t_min, n_steps).round().long()
        prev = torch.cat([ts[1:], torch.tensor([-1])])
        return ts.tolist(), prev.tolist()

    def _ab(self, t):
        return self.ab[t] if t >= 0 else torch.tensor(1.0, device=self.device)

    def step_mean_std(self, model, x_t, t, t_prev, eta=1.0):
        """Returns mean, std of pi(x_{t_prev} | x_t) and the predicted x0."""
        ab_t, ab_p = self._ab(t), self._ab(t_prev)
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps = model(x_t, tt)
        x0 = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if self.clip_x0 == 2:  # straight-through clamp: clamped forward, identity backward
            x0 = x0 + (x0.clamp(-1, 1) - x0).detach()
            eps = (x_t - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        elif self.clip_x0:
            x0 = x0.clamp(-1, 1)
            eps = (x_t - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()  # re-derive eps after clipping
        if t_prev < 0:  # final output: clip only for display/reward
            x0 = x0.clamp(-1, 1)
        sigma = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).clamp(min=0).sqrt()
        mean = ab_p.sqrt() * x0 + (1 - ab_p - sigma ** 2).clamp(min=0).sqrt() * eps
        return mean, sigma, x0

    @staticmethod
    def gaussian_logprob(x, mean, sigma):
        # mean over pixels (as in DDPO) keeps the scale independent of resolution
        return (-((x - mean) ** 2) / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def sample(self, model, n, n_steps=20, eta=1.0, return_traj=False, x_T=None):
        """Sample and (optionally) record the trajectory + log-probs of each stochastic step."""
        ts, prevs = self.timesteps(n_steps)
        x = torch.randn(n, 1, 28, 28, device=self.device) if x_T is None else x_T
        xs, lps = [x], []
        for t, tp in zip(ts, prevs):
            mean, sigma, x0 = self.step_mean_std(model, x, t, tp, eta)
            if tp < 0:  # final step: deterministic, output predicted x0
                x = x0
            else:
                x = mean + sigma * torch.randn_like(x)
                lps.append(self.gaussian_logprob(x, mean, sigma))
            xs.append(x)
        if not return_traj:
            return x
        # trajectory: xs[i] -> xs[i+1] for i in 0..n_steps-2 are the stochastic (policy) steps
        return x, torch.stack(xs[:-1], 1), torch.stack(lps, 1), ts, prevs

    def sample_with_grad(self, model, n, n_steps=20, eta=1.0, grad_last_k=1, x_T=None):
        """DRaFT-K style sampling: gradients only through the last K denoising steps."""
        ts, prevs = self.timesteps(n_steps)
        x = torch.randn(n, 1, 28, 28, device=self.device) if x_T is None else x_T
        for i, (t, tp) in enumerate(zip(ts, prevs)):
            with torch.set_grad_enabled(i >= n_steps - grad_last_k):
                mean, sigma, x0 = self.step_mean_std(model, x, t, tp, eta)
                x = x0 if tp < 0 else mean + sigma * torch.randn_like(x)
        return x
