#!/usr/bin/env python3
"""Extract a fixed-size RGB frame at a requested timestamp from local media."""
import argparse
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("seconds", type=float)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    if args.png.suffix.lower() != ".png":
        parser.error("the output frame must use the .png extension")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-ss", str(args.seconds), "-i", str(args.video),
        "-map", "0:v:0", "-frames:v", "1",
        "-vf", "scale=448:448:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               "pad=448:448:(ow-iw)/2:(oh-ih)/2",
        "-pix_fmt", "rgb24", "-update", "1", str(args.png),
    ], check=True)
    if not args.png.is_file():
        parser.exit(1, "no frame decoded at the requested timestamp\n")


if __name__ == "__main__":
    main()
