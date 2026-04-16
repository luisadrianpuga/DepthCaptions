from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Whisper ---
    whisper_model: str = "base"       # tiny | base | small | medium | large
    whisper_language: str | None = None  # None = auto-detect

    # --- Video ---
    output_fps: float | None = None   # None = match source
    temp_dir: str = "/tmp/depthcaptions"

    # --- Captions ---
    font_path: str | None = None      # None = auto-detect system font
    font_size_fraction: float = 0.22  # fallback font size as fraction of frame height
    text_fill_width: float = 0.88    # scale font so word fills this fraction of frame width
    text_opacity: float = 0.88        # 0.0–1.0
    text_outline_width: int = 0
    text_outline_color: tuple = (0, 0, 0)
    text_y_position: float = 0.18     # 0.0 = top, 1.0 = bottom
    text_auto_color: bool = True      # pick white/dark based on background luminance
    text_color: tuple = (15, 15, 15)  # used only when text_auto_color = False
    text_smart_position: bool = True  # shift text horizontally away from person
    text_position_padding: float = 0.05  # min margin from frame edges (fraction of width)

    # --- Crop ---
    crop_aspect_ratio: str | None = None  # e.g. "4:3", "1:1", "16:9" — None = no crop

    # --- Segmentation ---
    segmentation_threshold: float = 0.6
    mask_blur_radius: int = 5


DEFAULT = Config()
