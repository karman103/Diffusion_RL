"""CPU numerical checks against the actual pinned DDIM scheduler."""

import torch
from diffusers import DDIMScheduler

from text_video_grpo.ddim_policy import action_log_probability, transition_parameters


def run() -> None:
    torch.manual_seed(391)
    scheduler = DDIMScheduler.from_pretrained(
        "text_video_grpo/checkpoints/zeroscope_v2_576w/scheduler", local_files_only=True
    )
    scheduler.set_timesteps(40)
    for timestep in (int(scheduler.timesteps[0]), int(scheduler.timesteps[20]), int(scheduler.timesteps[-1])):
        sample = torch.randn(2, 4, 2, 4, 4)
        epsilon = torch.randn_like(sample, requires_grad=True)
        noise = torch.randn_like(sample)
        mean, std = transition_parameters(scheduler, sample, epsilon, timestep, eta=0.5)
        action = mean + std * noise
        expected = scheduler.step(epsilon, timestep, sample, eta=0.5, variance_noise=noise).prev_sample
        assert torch.allclose(action, expected, atol=2e-6), (timestep, (action - expected).abs().max())
        logp = action_log_probability(action.detach(), mean, std)
        expected_logp = (-0.5 * (noise.square() + torch.log(torch.tensor(2 * torch.pi))) - std.log()).flatten(1).sum(1)
        assert torch.allclose(logp, expected_logp, atol=3e-4), (timestep, logp, expected_logp)
        assert torch.isfinite(logp).all()
        (-logp.mean()).backward()
        assert epsilon.grad is not None and torch.isfinite(epsilon.grad).all() and epsilon.grad.abs().sum() > 0
    print("DDIM mean, Gaussian density, and policy gradient: passed at early, middle, final steps")


if __name__ == "__main__":
    run()
