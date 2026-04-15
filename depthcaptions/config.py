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
    text_color: tuple = (15, 15, 15)
    text_opacity: float = 0.72        # 0.0–1.0
    text_outline_width: int = 0
    text_outline_color: tuple = (0, 0, 0)
    text_y_position: float = 0.38     # 0.0 = top, 1.0 = bottom
    text_fill_width: float = 0.88     # word fills this fraction of frame width

    # --- Segmentation ---
    segmentation_threshold: float = 0.6
    mask_blur_radius: int = 5


DEFAULT = Config()
