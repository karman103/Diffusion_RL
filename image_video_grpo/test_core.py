"""Numerical checks that do not depend on pretrained checkpoints."""
import math
import unittest

import torch

from image_video_grpo.core import ImageResidual, VideoResidual, clipped_objective, group_advantages, normal_logprob
from image_video_grpo.models import ImagePolicy, VideoPolicy


class CoreTests(unittest.TestCase):
    def test_exact_joint_density(self):
        x = torch.tensor([[[1.0, -1.0]], [[0.2, 0.3]]])
        m = torch.zeros_like(x)
        s = 0.7
        observed = normal_logprob(x, m, s)
        expected = torch.distributions.Normal(m, s).log_prob(x).flatten(1).sum(1)
        torch.testing.assert_close(observed, expected)

    def test_group_center_and_ties(self):
        a = group_advantages(torch.tensor([1., 2., 3., 4.]))
        self.assertAlmostEqual(float(a.mean()), 0, places=6)
        self.assertAlmostEqual(float(a.std(unbiased=False)), 1, places=6)
        torch.testing.assert_close(group_advantages(torch.ones(4)), torch.zeros(4))

    def test_zero_advantage_has_zero_gradient(self):
        p = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
        loss, _ = clipped_objective(p, torch.zeros(2), torch.zeros(2))
        loss.backward()
        torch.testing.assert_close(p.grad, torch.zeros(2))

    def test_update_prefers_positive_advantage_action(self):
        logp = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([logp], lr=0.1)
        loss, _ = clipped_objective(logp, torch.zeros(2), torch.tensor([1., -1.]))
        loss.backward()
        opt.step()
        self.assertGreater(float(logp[0].detach()), 0)
        self.assertLess(float(logp[1].detach()), 0)

    def test_zero_adapters(self):
        im = ImageResidual()
        x, eps = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        torch.testing.assert_close(im(x, eps), eps)
        vid = VideoResidual()
        vx, v = torch.randn(2, 4, 1, 8, 8), torch.randn(2, 4, 1, 8, 8)
        torch.testing.assert_close(vid(vx, v), v)

    def test_frozen_reference_and_adapter_gradient(self):
        backbone = torch.nn.Conv2d(3, 3, 1).requires_grad_(False)
        adapter = ImageResidual()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            pred = backbone(x)
        out = adapter(x, pred)
        out.square().sum().backward()
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_zero_adapter_matches_reference_transition(self):
        class ImageDummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, x, t):
                return type("Prediction", (), {"sample": torch.zeros_like(x)})()
        class VideoDummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, tokens, ids, sigma):
                return torch.zeros_like(tokens)
        image = ImagePolicy(ImageDummy(), ImageResidual(), 5)
        video = VideoPolicy(VideoDummy(), VideoResidual(), 4, frames=2)
        for policy, x in [(image, torch.randn(2, 3, 32, 32)),
                          (video, torch.rand(2, 2, 1, 64, 64))]:
            mean, std = policy.transition(x, 0, adapt=True)
            reference, ref_std = policy.transition(x, 0, adapt=False)
            torch.testing.assert_close(mean, reference)
            self.assertEqual(float(std), float(ref_std))
            action = mean + std * torch.randn_like(x)
            torch.testing.assert_close(normal_logprob(action, mean, std),
                                       normal_logprob(action, reference, ref_std))


if __name__ == "__main__":
    unittest.main()
