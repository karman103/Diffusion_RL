"""Stock-DDPM matching and numerical policy checks without a model download."""
import json
import unittest

import torch
from diffusers import DDPMScheduler

from image_video_grpo.core import ImageResidual, clipped_objective, normal_logprob
from image_video_grpo.ddpm_followup.experiment import CFG, PARENT, select_steps
from image_video_grpo.models import ImagePolicy


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, x, t):
        return type("Output", (), {"sample": x * 0.1})()


class PolicyTests(unittest.TestCase):
    def test_stock_step_matches_diffusers(self):
        scheduler = DDPMScheduler.from_config(json.loads((PARENT / "checkpoints/cifar_scheduler.json").read_text()))
        scheduler.set_timesteps(CFG["steps"])
        policy = ImagePolicy(Dummy(), ImageResidual(), CFG["steps"])
        policy.times = scheduler.timesteps.tolist()
        policy.sampler = "ddpm"
        x = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(29))
        for index in (0, 17, 48, 49):
            t = policy.times[index]
            epsilon = x * 0.1
            mean, std = policy.transition(x, index)
            generator = torch.Generator().manual_seed(100 + index)
            expected = scheduler.step(epsilon, t, x, generator=generator).prev_sample
            if std is None:
                actual = mean
            else:
                generator = torch.Generator().manual_seed(100 + index)
                actual = mean + std * torch.randn(x.shape, generator=generator)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_density_ratio_and_zero_advantage(self):
        action = torch.randn(4, 3, 8, 8)
        mean = torch.nn.Parameter(torch.zeros_like(action))
        old = normal_logprob(action, mean.detach(), 0.5)
        new = normal_logprob(action, mean, 0.5)
        torch.testing.assert_close(new, old)
        loss, info = clipped_objective(new, old, torch.zeros(4))
        loss.backward()
        torch.testing.assert_close(mean.grad, torch.zeros_like(mean))
        self.assertEqual(info["clip_fraction"], 0)

    def test_stratified_steps_are_preterminal(self):
        for update in range(CFG["updates"]):
            steps = select_steps(1301, update)
            self.assertTrue(0 <= steps[0] < 16 <= steps[1] < 32 <= steps[2] < 49)


if __name__ == "__main__":
    unittest.main()
