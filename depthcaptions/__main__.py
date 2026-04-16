"""CLI entry point: python -m depthcaptions  or  depthcaptions (after pip install)"""

import argparse
from pathlib import Path

from .config import Config
from .pipeline import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Add 'text behind person' captions to a video."
    )
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path for output video")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size (default: base)")
    parser.add_argument("--font", default=None, help="Path to a .ttf font file")
    parser.add_argument("--opacity", type=float, default=None,
                        help="Text opacity 0.0–1.0 (default: 0.88)")
    parser.add_argument("--y-position", type=float, default=None,
                        help="Text vertical position 0.0–1.0 (default: 0.18)")
    parser.add_argument("--crop", default=None, metavar="W:H",
                        help="Center-crop to aspect ratio before processing, e.g. 4:3 or 1:1")
    args = parser.parse_args()

    cfg = Config(
        whisper_model=args.whisper_model,
        font_path=args.font,
        **({"text_opacity": args.opacity} if args.opacity is not None else {}),
        **({"text_y_position": args.y_position} if args.y_position is not None else {}),
        **({"crop_aspect_ratio": args.crop} if args.crop else {}),
    )

    process_video(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
