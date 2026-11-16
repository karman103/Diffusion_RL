"""Render paired raw toy-video tensors as an inspectable animated GIF."""
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from image_video_grpo.video_factorized.model import ROOT


def main():
    run = ROOT / "runs" / "seed_12102"
    zero = torch.load(run / "zero_videos.pt", map_location="cpu", weights_only=True)[0]
    trained = torch.load(run / "trained_videos.pt", map_location="cpu", weights_only=True)[0]
    frames = []
    for time in range(4):
        frame = Image.new("RGB", (252, 156), "#171717")
        draw = ImageDraw.Draw(frame)
        draw.text((12, 10), "Zero adapter", fill="white")
        draw.text((136, 10), "GRPO adapter", fill="white")
        draw.text((89, 136), f"frame {time + 1}/4", fill="white")
        for j, video in enumerate((zero, trained)):
            px = ((video[time].clamp(-1, 1) + 1) * 127.5).byte().numpy()
            tile = Image.fromarray(px).resize((112, 112), Image.Resampling.NEAREST)
            frame.paste(tile.convert("RGB"), (12 + j * 124, 26))
        frames.append(frame)
    frames[0].save(ROOT / "paired_motion.gif", save_all=True, append_images=frames[1:],
                   duration=450, loop=0, optimize=False)


if __name__ == "__main__":
    main()
