"""Audit and export frames returned by Diffusers' text-to-video pipeline."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def save_frames(frames, output: Path, expected_count: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.stack([np.asarray(frame) for frame in frames])
    if arrays.ndim != 4 or arrays.shape[0] != expected_count or arrays.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape {arrays.shape}")
    if not np.isfinite(arrays).all():
        raise ValueError("Decoded frames contain non-finite values")
    np.save(output / "frames_raw.npy", arrays)
    if np.issubdtype(arrays.dtype, np.floating):
        if arrays.min() < -0.001 or arrays.max() > 1.001:
            raise ValueError(f"Decoded float range [{arrays.min()}, {arrays.max()}] is not [0, 1]")
        pixels = np.rint(np.clip(arrays, 0, 1) * 255).astype(np.uint8)
    elif arrays.dtype == np.uint8:
        pixels = arrays
    else:
        raise TypeError(f"Unexpected decoded frame dtype {arrays.dtype}")
    pil_frames = [Image.fromarray(frame, mode="RGB") for frame in pixels]
    Image.fromarray(np.concatenate(pixels[::2], axis=1), mode="RGB").save(output / "filmstrip.png")
    pil_frames[0].save(output / "clip.gif", save_all=True, append_images=pil_frames[1:], duration=160, loop=0)
    return {
        "shape": list(pixels.shape),
        "raw_dtype": str(arrays.dtype),
        "raw_min": float(arrays.min()),
        "raw_max": float(arrays.max()),
        "finite": True,
        "pixel_std": float(pixels.std()),
        "frame_difference": float(np.abs(np.diff(pixels.astype(np.float32), axis=0)).mean()),
        "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
