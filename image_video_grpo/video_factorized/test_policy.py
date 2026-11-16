"""Numerical and gradient checks for the exact conditional image policy."""
import unittest

import torch

from image_video_grpo.core import clipped_objective, group_advantages, normal_logprob
from image_video_grpo.video_factorized.model import AdaptedImage, FactorizedPolicy


class FactorizedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AdaptedImage().eval()
        cls.policy = FactorizedPolicy(cls.model, "cpu")

    def test_zero_adapter_equals_frozen_reference(self):
        x = torch.randn(2, 1, 28, 28)
        for step in (0, 10, 18):
            adapted, sd = self.policy.transition(x, step)
            base, sr = self.policy.transition(x, step, adapt=False)
            self.assertTrue(torch.equal(adapted, base))
            self.assertTrue(torch.equal(sd, sr))

    def test_action_density_and_zero_advantage_gradient(self):
        x = torch.randn(4, 1, 28, 28)
        mean, std = self.policy.transition(x, 18)
        action = mean.detach() + std.detach() * torch.randn_like(mean)
        old = normal_logprob(action, mean.detach(), std.detach())
        new = normal_logprob(action, mean, std)
        self.assertEqual(list(new.shape), [4])
        self.assertTrue(torch.isfinite(new).all())
        self.assertLess(float((new - old).detach().abs().max()), 1e-4)
        advantage = group_advantages(torch.ones(4))
        self.assertTrue(torch.equal(advantage, torch.zeros_like(advantage)))
        loss, _ = clipped_objective(new, old, advantage)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.base.parameters()))
        self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
                            for p in self.model.adapter.parameters()))

    def test_finite_video_and_nineteen_stochastic_actions(self):
        videos, canonical, velocity, trajectory = self.policy.rollout(99111, n=2, record=True)
        self.assertEqual(list(videos.shape), [2, 4, 28, 28])
        self.assertEqual(len(trajectory), 19)
        self.assertTrue(torch.isfinite(videos).all())
        self.assertTrue(torch.isfinite(canonical).all())
        self.assertTrue(all(torch.isfinite(row[2]).all() for row in trajectory))
        self.assertEqual(list(velocity.shape), [2, 2])


if __name__ == "__main__":
    unittest.main()
