"""Numerical sanity checks for the declared pilot reward and group baseline."""

import numpy as np

from text_video_grpo.pilot_reward import group_advantages, red_motion_score


def run():
    moving = np.zeros((4, 8, 8, 3), dtype=np.float32)
    static = np.zeros_like(moving)
    for frame in range(4):
        moving[frame, :, frame:frame + 2, 0] = 1.0
        static[frame, :, 0:2, 0] = 1.0
    assert red_motion_score(moving)["reward"] > red_motion_score(static)["reward"]
    assert np.array_equal(group_advantages(np.ones(4)), np.zeros(4))
    advantages = group_advantages(np.array([1, 2, 3, 4]))
    assert abs(float(advantages.mean())) < 1e-6
    assert abs(float(advantages.std()) - 1.0) < 1e-6
    print("pilot red-motion reward and group advantages: passed")


if __name__ == "__main__":
    run()
