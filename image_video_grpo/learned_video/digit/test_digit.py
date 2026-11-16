import unittest

import torch

from image_video_grpo.learned_video import classifiers
from image_video_grpo.learned_video.data import NDIR, make_clips, mnist
from image_video_grpo.learned_video.digit.common import evaluate, reward
from image_video_grpo.learned_video.model import VideoUNet


class T(unittest.TestCase):
    def test_reward_and_eval_on_real_clips(self):
        x, y = mnist("test")
        g = torch.Generator().manual_seed(4)
        d = torch.arange(128) % NDIR
        c = make_clips(x[:128], d, g)
        cond = torch.stack((d, y[:128]), 1)
        p, e = classifiers.load()
        r_ok = reward(c, cond, p).mean()
        r_wrong_digit = reward(c, torch.stack((d, (y[:128] + 1) % 10), 1), p).mean()
        self.assertGreater(float(r_ok), 0.85)
        self.assertLess(float(r_wrong_digit), 0.7)
        self.assertGreater(float(evaluate(c, cond, e)["success"].mean()), 0.85)

    def test_warm_start_identical_before_training(self):
        torch.manual_seed(0)
        a = VideoUNet().eval(); b = VideoUNet(digits=True).eval()
        b.load_state_dict(a.state_dict(), strict=False)
        x = torch.randn(2, 8, 1, 32, 32); t = torch.tensor([10, 500]); d = torch.tensor([1, 5])
        torch.testing.assert_close(a(x, t, d), b(x, t, torch.stack((d, torch.tensor([3, 7])), 1)))


if __name__ == "__main__":
    unittest.main()
