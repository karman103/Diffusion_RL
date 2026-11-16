import unittest

import torch

from image_video_grpo.core import normal_logprob
from image_video_grpo.learned_video import classifiers, metrics
from image_video_grpo.learned_video.data import NDIR, make_clips, mnist
from image_video_grpo.learned_video.diffusion import alphas_bar, sample, schedule, step_mean_std
from image_video_grpo.learned_video.model import VideoUNet


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = mnist("test")
        g = torch.Generator().manual_seed(3)
        cls.d = torch.arange(64) % NDIR
        cls.clips = make_clips(x[:64], cls.d, g)
        cls.proxy, cls.ev = classifiers.load()

    def test_ink_range_covers_real(self):
        ink = ((self.clips + 1) / 2).mean((2, 3, 4))
        self.assertTrue(((ink >= metrics.INK_RANGE[0]) & (ink <= metrics.INK_RANGE[1])).float().mean() > 0.95)

    def test_phase_corr_recovers_motion(self):
        e = metrics.evaluate(self.clips, self.d, self.ev)
        self.assertLess(float(e["angle_err"].median()), 5.0)
        self.assertLess(float(e["speed_err"].median()), 1.0)
        self.assertGreater(float(e["success"].mean()), 0.85)

    def test_reward_prefers_real_over_wrong_direction_and_noise(self):
        r_true = metrics.reward(self.clips, self.d, self.proxy).mean()
        r_wrong = metrics.reward(self.clips, (self.d + 4) % NDIR, self.proxy).mean()
        r_noise = metrics.reward(torch.rand_like(self.clips) * 2 - 1, self.d, self.proxy).mean()
        static = self.clips[:, :1].expand_as(self.clips)
        r_static = metrics.reward(static, self.d, self.proxy).mean()
        self.assertGreater(float(r_true), 0.85)
        self.assertLess(float(r_wrong), 0.5)
        self.assertLess(float(r_noise), 0.05)
        self.assertLess(float(r_static), 0.5)

    def test_recorded_logprob_matches_recomputed(self):
        torch.manual_seed(0)
        net = VideoUNet().eval()
        ab = alphas_bar()
        d = torch.tensor([0, 3])
        _, traj = sample(net, d, 5, ab, steps=6, record=True)
        self.assertEqual(len(traj), 5)
        for t, tp, s, a, lp in traj:
            mean, sigma, _ = step_mean_std(net, s, t, tp, d, ab)
            torch.testing.assert_close(normal_logprob(a, mean, sigma), lp)
        self.assertEqual(schedule(6)[-1][1], -1)


if __name__ == "__main__":
    unittest.main()
