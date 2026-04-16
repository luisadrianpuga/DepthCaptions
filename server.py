"""
DepthCaptions MCP server.

Exposes one tool — add_captions — that Claude can call to composite
word-by-word captions behind the person in a video.

Connect to Claude Code by adding to ~/.claude/settings.json:

  "mcpServers": {
    "depthcaptions": {
      "command": "/usr/local/bin/python3",
      "args": ["/path/to/DepthCaptions/server.py"]
    }
  }
"""

from mcp.server.fastmcp import FastMCP

from depthcaptions import process_video
from depthcaptions.config import Config

mcp = FastMCP(
    "DepthCaptions",
    instructions=(
        "Add word-by-word captions that appear behind the person in a video. "
        "Always resolve ~ in paths before passing them. "
        "output_video defaults to input_video stem + '_captioned.mp4' in the same directory."
    ),
)


@mcp.tool()
def add_captions(
    input_video: str,
    output_video: str,
    whisper_model: str = "base",
    text_opacity: float = 0.72,
    text_y_position: float = 0.25,
    font_size_fraction: float = 0.22,
    text_auto_color: bool = True,
    text_smart_position: bool = True,
    font_path: str | None = None,
    segmentation_threshold: float = 0.6,
    mask_blur_radius: int = 5,
    crop_aspect_ratio: str | None = None,
) -> str:
    """
    Add "text behind person" captions to a video.

    Transcribes the audio with Whisper, then for each frame renders the
    currently-spoken word as large semi-transparent text behind the person.

    Args:
        input_video: Path to the source video file.
        output_video: Path to write the captioned video.
        whisper_model: Whisper model size — tiny, base, small, medium, large.
        text_opacity: Text transparency (0.0 = invisible, 1.0 = solid). Default 0.72.
        text_y_position: Vertical position of text (0.0 = top, 1.0 = bottom). Default 0.38.
        text_fill_width: Word scales to fill this fraction of frame width. Default 0.88.
        font_path: Optional path to a .ttf font file. Auto-detected if omitted.
        segmentation_threshold: MediaPipe confidence cutoff for person mask. Default 0.6.
        mask_blur_radius: Feathering on person mask edges in pixels. Default 5.
        crop_aspect_ratio: Center-crop to this aspect ratio before processing, e.g. "4:3" or "1:1". None = no crop.

    Returns:
        Confirmation message with the output path.
    """
    cfg = Config(
        whisper_model=whisper_model,
        text_opacity=text_opacity,
        text_y_position=text_y_position,
        font_size_fraction=font_size_fraction,
        text_auto_color=text_auto_color,
        text_smart_position=text_smart_position,
        font_path=font_path,
        segmentation_threshold=segmentation_threshold,
        mask_blur_radius=mask_blur_radius,
        crop_aspect_ratio=crop_aspect_ratio,
    )

    out = process_video(input_video, output_video, cfg)
    return f"Done! Captioned video saved to: {out}"


if __name__ == "__main__":
    mcp.run()
