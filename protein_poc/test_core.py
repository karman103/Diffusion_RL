"""Numerical checks that protect the scientific claims, including upstream parity."""
import math
import unittest

import torch

from .core import ProteinPolicy, frames, joint_logprob, geometry
from genie.utils.geo_utils import compute_frenet_frames


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        cls.policy = ProteinPolicy("cpu", steps=10, length=12)

    def test_joint_logprob_is_normal_sum(self):
        x, mu = torch.randn(3, 12, 3), torch.randn(3, 12, 3)
        expected = torch.distributions.Normal(mu, .23).log_prob(x).sum((-1,-2))
        torch.testing.assert_close(joint_logprob(x, mu, .23), expected)

    def test_vectorized_frames_match_upstream(self):
        # Upstream omits dim in torch.cross: batch size 3 wrongly selects the
        # batch axis. Compare against singleton calls, where its axis is valid.
        x = torch.randn(3, 12, 3)
        expected = torch.cat([compute_frenet_frames(row[None], torch.ones(1,12)) for row in x])
        torch.testing.assert_close(frames(x), expected)

    def test_cached_update_matches_actual_network_weight_update(self):
        p = self.policy
        x = torch.randn(2, 12, 3)
        with torch.no_grad():
            p.head.weight.normal_(0, .002); p.head.bias.normal_(0,.002)
            stats = p.frozen_stats(x, 333, 222)
            cached = p.adapted_mean(stats)
            linear = p.model.structure_net.net[-1].bb_update.linear
            original_w, original_b = linear.weight.clone(), linear.bias.clone()
            try:
                linear.weight[3:].add_(p.head.weight)
                linear.bias[3:].add_(p.head.bias)
                actual = p.frozen_stats(x, 333, 222)[0]
                torch.testing.assert_close(cached, actual, rtol=1e-4, atol=2e-5)
            finally:
                linear.weight.copy_(original_w); linear.bias.copy_(original_b)
                p.head.weight.zero_(); p.head.bias.zero_()

    def test_shared_terminal_decoder_ignores_head(self):
        p = self.policy
        x = torch.randn(2,12,3)
        stats = p.frozen_stats(x, 0, -1)
        baseline = p.adapted_mean(stats, enabled=False)
        with torch.no_grad():
            p.head.bias.fill_(10)
        terminal = p.adapted_mean(stats, enabled=False)
        torch.testing.assert_close(terminal, baseline, rtol=0, atol=0)
        self.assertEqual(stats[-1], 0)
        with torch.no_grad():
            p.head.bias.zero_()

    def test_gaussian_kl_matches_distribution_formula(self):
        mu0, mu1 = torch.randn(3,12,3), torch.randn(3,12,3)
        sigma = .3
        exact = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu1,sigma), torch.distributions.Normal(mu0,sigma)
        ).sum((-1,-2))
        actual = (mu1-mu0).square().sum((-1,-2))/(2*sigma**2)
        torch.testing.assert_close(actual, exact)

    def test_gradient_matches_finite_difference(self):
        p = self.policy
        stats = p.frozen_stats(torch.randn(2,12,3), 333,222)
        action = stats[0] + .1*torch.randn_like(stats[0])
        def objective():
            return joint_logprob(action,p.adapted_mean(stats),stats[-1]).sum()
        p.head.zero_grad()
        objective().backward()
        analytical = p.head.bias.grad[0].item()
        with torch.no_grad():
            p.head.bias[0] = .001; plus = objective().item()
            p.head.bias[0] = -.001; minus = objective().item()
            p.head.bias[0] = 0
        self.assertAlmostEqual(analytical,(plus-minus)/.002,delta=.02)

    def test_geometry_separates_straight_chain_and_collapse(self):
        x = torch.zeros(1,12,3)
        x[0,:,0] = torch.arange(12)*3.8
        self.assertEqual(geometry(x)["geometry_pass"].item(),1)
        self.assertEqual(geometry(torch.zeros_like(x))["geometry_pass"].item(),0)


if __name__ == "__main__":
    unittest.main()
