"""Regression test for the NumPy output that the pinned Diffusers pipeline returns."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from text_video_grpo.frame_export import save_frames


def run() -> None:
    pixels = np.zeros((3, 8, 8, 3), dtype=np.float32)
    pixels[1, :, :, 0] = 0.5
    pixels[2, :, :, 0] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        report = save_frames(pixels, out, 3)
        assert report["shape"] == [3, 8, 8, 3]
        assert report["raw_dtype"] == "float32"
        assert report["finite"] and report["frame_difference"] > 0
        assert np.array_equal(np.load(out / "frames_raw.npy"), pixels)
        assert Image.open(out / "clip.gif").n_frames == 3
        assert np.asarray(Image.open(out / "filmstrip.png"))[0, 8, 0] == 255
    print("NumPy-frame export, raw preservation, filmstrip, and GIF: passed")


if __name__ == "__main__":
    run()
