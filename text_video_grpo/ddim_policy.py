"""Conditional Gaussian action density for the pinned ZeroScope DDIM schedule.

This is the density of one stochastic latent transition, not a tractable
likelihood of the decoded video. With eta=0 the transition is deterministic and
has no density with respect to Lebesgue measure, so policy updates require
eta>0. The implementation mirrors Diffusers DDIMScheduler.step for epsilon
prediction, no clipping, and its leading timestep spacing.
"""

import math

import torch


def transition_parameters(scheduler, sample, model_epsilon, timestep: int, eta: float):
    if eta <= 0:
        raise ValueError("A stochastic policy requires eta > 0")
    if scheduler.num_inference_steps is None:
        raise ValueError("Call scheduler.set_timesteps first")
    if scheduler.config.prediction_type != "epsilon" or scheduler.config.clip_sample or scheduler.config.thresholding:
        raise ValueError("This policy is validated only for un-clipped epsilon prediction")
    previous_timestep = int(timestep) - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alphas = scheduler.alphas_cumprod
    alpha_t = alphas[int(timestep)].to(device=sample.device, dtype=torch.float32)
    alpha_prev = (
        alphas[previous_timestep] if previous_timestep >= 0 else scheduler.final_alpha_cumprod
    ).to(device=sample.device, dtype=torch.float32)
    x_t = sample.float()
    epsilon = model_epsilon.float()
    x0 = (x_t - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    variance = ((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev)
    std = eta * variance.clamp_min(0).sqrt()
    if not bool((std > 0).item()):
        raise ValueError(f"Zero variance at timestep {timestep}")
    mean = alpha_prev.sqrt() * x0 + (1 - alpha_prev - std.square()).sqrt() * epsilon
    return mean, std


def action_log_probability(action, mean, std):
    """Per-trajectory log p(x_prev | x_t, text) with spatial dimensions summed."""
    if action.shape != mean.shape or action.ndim < 2:
        raise ValueError("Action and mean must have the same batched shape")
    residual = (action.float() - mean.float()) / std.float()
    element_logp = -0.5 * (residual.square() + math.log(2 * math.pi)) - std.float().log()
    return element_logp.flatten(1).sum(1)
