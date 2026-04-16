"""
Core "text behind person" compositing pipeline.
"""

import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import whisper
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import Config, DEFAULT

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
)
_MODEL_DIR = Path.home() / ".depthcaptions" / "models"
_MODEL_PATH = _MODEL_DIR / "selfie_segmenter.tflite"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _has_audio(video_path: str) -> bool:
    """Return True if the video file has at least one audio stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def transcribe(video_path: str, cfg: Config) -> list[dict]:
    """Return a flat list of word-level segments: {word, start, end}."""
    print("[1/4] Transcribing audio with Whisper…")
    if not _has_audio(video_path):
        print("    → no audio stream found, skipping transcription")
        return []
    model = whisper.load_model(cfg.whisper_model)
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        language=cfg.whisper_language,
    )
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
    print(f"    → {len(words)} words transcribed")
    return words


# ---------------------------------------------------------------------------
# Word lookup
# ---------------------------------------------------------------------------

def active_word_at(words: list[dict], t: float) -> str | None:
    for w in words:
        if w["start"] <= t <= w["end"]:
            return w["word"]
    return None


def active_word_entry(words: list[dict], t: float) -> dict | None:
    for w in words:
        if w["start"] <= t <= w["end"]:
            return w
    return None


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _find_font_path(cfg: Config) -> str | None:
    if cfg.font_path and Path(cfg.font_path).exists():
        return cfg.font_path
    for candidate in [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def _get_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(font_path, size)


def _best_x(mask: np.ndarray, fw: int, tw: int, y0: int, y1: int, padding: int) -> int:
    """Scan horizontal positions, return x with minimum person-mask overlap.

    A center-bias penalty keeps text roughly centered; it only shifts away
    when avoiding the person is clearly worth it.
    """
    x_min = padding
    x_max = fw - tw - padding
    if x_max <= x_min:
        return (fw - tw) // 2  # word too wide to fit with padding, just center

    center_x = (fw - tw) // 2
    best_x = center_x
    best_score = float("inf")
    step = max(1, fw // 40)
    for x in range(x_min, x_max + 1, step):
        region = mask[y0:y1, x : x + tw]
        overlap = float(region.mean())
        # Penalize distance from center so text stays central unless
        # moving away meaningfully reduces person overlap.
        dist_penalty = (abs(x - center_x) / fw) * 0.4
        score = overlap + dist_penalty
        if score < best_score:
            best_score = score
            best_x = x
    return best_x


def _auto_color(pil_frame: Image.Image, x: int, y: int, tw: int, th: int) -> tuple:
    """Sample background luminance in text region, return contrasting RGB color."""
    fw, fh = pil_frame.size
    region = pil_frame.crop((max(0, x), max(0, y), min(fw, x + tw), min(fh, y + th)))
    pixels = np.array(region.convert("RGB")).astype(float)
    lum = (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).mean() / 255.0
    return (240, 240, 240) if lum < 0.5 else (15, 15, 15)


def _font_size_for_fill(font_path: str | None, word: str, target_w: int, max_h: int) -> int:
    """Return font size so `word` fills target_w pixels wide, capped at max_h tall."""
    probe_size = 100
    font = _get_font(font_path, probe_size)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = dummy.textbbox((0, 0), word, font=font)
    probe_w = bbox[2] - bbox[0]
    probe_h = bbox[3] - bbox[1]
    if probe_w <= 0:
        return probe_size
    size_by_width = int(probe_size * target_w / probe_w)
    size_by_height = int(probe_size * max_h / probe_h) if probe_h > 0 else size_by_width
    return min(size_by_width, size_by_height)


def render_word_on_frame(
    pil_frame: Image.Image,
    word: str,
    font_path: str | None,
    cfg: Config,
    mask: np.ndarray | None = None,
    fixed_x: int | None = None,
) -> Image.Image:
    fw, fh = pil_frame.size
    target_w = int(fw * cfg.text_fill_width)
    max_h = int(fh * 0.40)  # never taller than 40% of frame height
    size = _font_size_for_fill(font_path, word, target_w, max_h)
    font = _get_font(font_path, max(size, 12))

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    y = int(fh * cfg.text_y_position) - th // 2
    y0, y1 = max(0, y), min(fh, y + th)

    if fixed_x is not None:
        x = fixed_x
    elif mask is not None and cfg.text_smart_position:
        padding = int(fw * cfg.text_position_padding)
        x = _best_x(mask, fw, tw, y0, y1, padding)
    else:
        x = (fw - tw) // 2

    color = _auto_color(pil_frame, x, y, tw, th) if cfg.text_auto_color else cfg.text_color

    overlay = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(cfg.text_opacity * 255)
    color_rgba = (*color, alpha)

    if cfg.text_outline_width > 0:
        outline_rgba = (*cfg.text_outline_color, alpha)
        ow = cfg.text_outline_width
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), word, font=font, fill=outline_rgba)
    draw.text((x, y), word, font=font, fill=color_rgba)

    frame = pil_frame.convert("RGBA")
    return Image.alpha_composite(frame, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class PersonSegmenter:
    def __init__(self, cfg: Config):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not _MODEL_PATH.exists():
            print("    Downloading segmentation model…")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"    → saved to {_MODEL_PATH}")

        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self._seg = mp_vision.ImageSegmenter.create_from_options(options)
        self._cfg = cfg

    def get_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._seg.segment(mp_image)
        mask = np.squeeze(result.confidence_masks[0].numpy_view().copy())

        mask = (mask > self._cfg.segmentation_threshold).astype(np.float32)
        if self._cfg.mask_blur_radius > 0:
            pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=self._cfg.mask_blur_radius))
            mask = np.array(pil_mask).astype(np.float32) / 255.0
        return mask

    def close(self):
        self._seg.close()


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def _crop_to_aspect(bgr_frame: np.ndarray, aspect: str) -> np.ndarray:
    """Center-crop a frame to the given aspect ratio string e.g. '4:3'."""
    w_ratio, h_ratio = map(int, aspect.split(":"))
    h, w = bgr_frame.shape[:2]
    target_w = w
    target_h = int(w * h_ratio / w_ratio)
    if target_h > h:
        target_h = h
        target_w = int(h * w_ratio / h_ratio)
    x0 = (w - target_w) // 2
    y0 = (h - target_h) // 2
    return bgr_frame[y0 : y0 + target_h, x0 : x0 + target_w]


def composite_frame(
    bgr_frame: np.ndarray, captioned_pil: Image.Image, mask: np.ndarray
) -> np.ndarray:
    person_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    caption_rgb = np.array(captioned_pil).astype(np.float32)
    m = mask[:, :, np.newaxis]
    blended = np.clip(caption_rgb * (1.0 - m) + person_rgb * m, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_video(input_path: str, output_path: str, cfg: Config = DEFAULT) -> str:
    """
    Process a video: transcribe, composite captions behind the person, save.
    Returns the output path on success.
    """
    input_path = str(Path(input_path).expanduser().resolve())
    output_path = str(Path(output_path).expanduser().resolve())

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    temp_dir = Path(cfg.temp_dir)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        words = transcribe(input_path, cfg)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = cfg.output_fps or fps

        print(f"[2/4] Processing {total_frames} frames at {fps:.2f} fps…")

        font_path = _find_font_path(cfg)
        segmenter = PersonSegmenter(cfg)

        # Cache best-x per word so position is stable for the word's duration
        word_x_cache: dict[float, int] = {}

        frame_idx = 0
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if cfg.crop_aspect_ratio:
                bgr_frame = _crop_to_aspect(bgr_frame, cfg.crop_aspect_ratio)

            t = frame_idx / fps
            pil_frame = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

            mask = segmenter.get_mask(bgr_frame)

            entry = active_word_entry(words, t)
            if entry:
                word = entry["word"].upper()
                key = entry["start"]
                if key not in word_x_cache:
                    # Compute position once on the word's first frame
                    fw, fh = pil_frame.size
                    target_w = int(fw * cfg.text_fill_width)
                    max_h = int(fh * 0.40)
                    size = _font_size_for_fill(font_path, word, target_w, max_h)
                    font = _get_font(font_path, max(size, 12))
                    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                    bbox = dummy_draw.textbbox((0, 0), word, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    y = int(fh * cfg.text_y_position) - th // 2
                    y0, y1 = max(0, y), min(fh, y + th)
                    if cfg.text_smart_position:
                        padding = int(fw * cfg.text_position_padding)
                        word_x_cache[key] = _best_x(mask, fw, tw, y0, y1, padding)
                    else:
                        word_x_cache[key] = (fw - tw) // 2
                captioned = render_word_on_frame(pil_frame, word, font_path, cfg, mask, fixed_x=word_x_cache[key])
            else:
                captioned = pil_frame
            out_frame = composite_frame(bgr_frame, captioned, mask)

            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:07d}.png"), out_frame)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"    {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)", end="\r")

        cap.release()
        segmenter.close()
        print(f"\n    → {frame_idx} frames processed")

        print("[3/4] Reassembling video…")
        _reassemble(str(frames_dir), input_path, output_path, out_fps)
        print(f"[4/4] Done → {output_path}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def _reassemble(frames_dir: str, source_video: str, output_path: str, fps: float) -> None:
    if _has_audio(source_video):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{frames_dir}/frame_%07d.png",
            "-i", source_video,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-shortest",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{frames_dir}/frame_%07d.png",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            output_path,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
