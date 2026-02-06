from __future__ import annotations

import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

from matplotlib.colors import to_rgb
import cyclopts
import ipdb
import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont

RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
Color = Union[RGB, RGBA, str]  # (r,g,b), (r,g,b,a) or "#RRGGBB" / "#RRGGBBAA"


def _parse_color(c: Color, *, default_alpha: int = 255) -> RGBA:
    if isinstance(c, tuple):
        if len(c) == 3:
            return (int(c[0]), int(c[1]), int(c[2]), default_alpha)
        if len(c) == 4:
            return (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        raise ValueError(f"Bad tuple color: {c}")
    if isinstance(c, str):
        s = c.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b, default_alpha)
        if len(s) == 8:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            a = int(s[6:8], 16)
            return (r, g, b, a)
        raise ValueError(f"Bad hex color: {c}")
    raise TypeError(f"Unsupported color type: {type(c)}")

def frame_from_time(t, fps):
    return int(np.floor(t * fps + 1e-9))

def add_colored_text_overlay_per_char_rgb(
        base_clip,
        text: str,
        *,
        font_path: str,
        font_size: int = 48,
        pos: Tuple[int, int] = (40, 40),
        padding: int = 8,
        default_rgb: RGB = (255, 255, 255),
        default_alpha: int = 255,
        bg: Optional[Color] = None,
        # callback returns (n_chars, 3) per frame
        rgb_at_time: Optional[Callable[[float], np.ndarray]] = None,
        line_spacing: Optional[int] = None,  # extra pixels between lines (default: ~0.2*font_size)
        align: str = "left",  # "left" | "center" | "right"
) -> CompositeVideoClip:
    font = ImageFont.truetype(font_path, font_size)
    if line_spacing is None:
        line_spacing = int(round(0.2 * font_size))
    if align not in ("left", "center", "right"):
        raise ValueError("align must be 'left', 'center', or 'right'")

    lines = text.split("\n")

    # Map each line to indices in original text (including '\n' positions)
    line_ranges = []
    idx = 0
    for li, line in enumerate(lines):
        start = idx
        end = start + len(line)
        line_ranges.append((start, end))
        idx = end + (1 if li < len(lines) - 1 else 0)  # skip '\n'

    # Use a reference bbox for consistent line height.
    # (Avoid baseline confusion; PIL draws from top-left.)
    ref_bbox = font.getbbox("Ag")  # good asc/desc coverage
    line_h = int(ref_bbox[3] - ref_bbox[1])
    line_step = line_h + line_spacing

    # Width per line (kerning-aware)
    line_widths = [float(font.getlength(line)) for line in lines]
    text_w = int(np.ceil(max(line_widths) if line_widths else 0.0))
    text_h = line_h * len(lines) + line_spacing * max(0, len(lines) - 1)

    img_w = text_w + 2 * padding
    img_h = text_h + 2 * padding

    # Cache per-line advances (kerning-safe)
    adv_cache = []
    bbox_cache = []
    for line in lines:
        # advances
        prefix = np.zeros(len(line) + 1, dtype=np.float32)
        for k in range(1, len(line) + 1):
            prefix[k] = float(font.getlength(line[:k]))
        adv_cache.append(np.diff(prefix))
        # bbox (used to vertically align top)
        bbox_cache.append(font.getbbox(line) if line else (0, 0, 0, 0))

    def _rgb_for_frame(kk: int) -> np.ndarray:
        if rgb_at_time is None:
            return np.tile(np.array(default_rgb, dtype=np.uint8), (len(text), 1))
        arr = np.asarray(rgb_at_time(kk))
        if arr.shape != (len(text), 3):
            raise ValueError(f"rgb_at_time(frame) must return shape ({len(text)}, 3), got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255)
        return arr

    def rgba_frame(kk: int) -> np.ndarray:
        im = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)

        if bg is not None:
            draw.rectangle([(0, 0), (img_w - 1, img_h - 1)],
                           fill=_parse_color(bg, default_alpha=default_alpha))

        rgb = _rgb_for_frame(kk)

        for li, line in enumerate(lines):
            start, end = line_ranges[li]
            if not line:
                continue

            lw = line_widths[li]
            if align == "left":
                x = float(padding)
            elif align == "center":
                x = float(padding + (text_w - lw) / 2.0)
            else:
                x = float(padding + (text_w - lw))

            # y here is TOP of the line box, not baseline.
            # Shift by -bbox_line[1] to account for negative y in glyph bbox.
            bbox_line = bbox_cache[li]
            y_top = float(padding + li * line_step - bbox_line[1])

            adv = adv_cache[li]

            i = 0
            n = len(line)
            while i < n:
                gi = start + i
                j = i + 1
                while j < n and (rgb[start + j] == rgb[gi]).all():
                    j += 1

                run = line[i:j]
                r, g, b = map(int, rgb[gi])
                draw.text((x, y_top), run, font=font, fill=(r, g, b, int(default_alpha)))

                x += float(adv[i:j].sum())
                i = j

        return np.asarray(im, dtype=np.uint8)

    last_kk = None
    last_rgba = None

    def rgba_cached(kk_):
        nonlocal last_kk, last_rgba
        if last_kk != kk_:
            last_rgba = rgba_frame(kk_)
            last_kk = kk_
        return last_rgba

    def rgb_frame(t: float) -> np.ndarray:
        frame = frame_from_time(t, base_clip.fps)
        return rgba_cached(frame)[:, :, :3]

    def mask_frame(t: float) -> np.ndarray:
        frame = frame_from_time(t, base_clip.fps)
        return rgba_cached(frame)[:, :, 3].astype(np.float32) / 255.0

    duration = base_clip.duration

    overlay_rgb = VideoClip(frame_function=rgb_frame, duration=duration, has_constant_size=True)
    overlay_mask = VideoClip(
        frame_function=mask_frame, duration=duration, is_mask=True, has_constant_size=True
    )

    overlay = overlay_rgb.with_mask(overlay_mask).with_position(pos).with_duration(duration)

    clip = CompositeVideoClip([base_clip, overlay])

    duration = 300 * (1.0 / base_clip.fps)
    clip = clip.subclipped(0, duration)

    return clip

def get_timing_data(seed: int):
    export_dir = "/Users/oswinso/research/me/rraa-rl/media/02-04__delivery/timing_npzs"
    export_dir = pathlib.Path(export_dir)
    npz_path = export_dir / f"timing_seed{seed}.npz"

    npz = np.load(npz_path)
    Tn_achieved = npz["Tn_achieved"]
    T_state = npz["T_state"]

    return Tn_achieved, T_state


# --------------------------
# Example usage
# --------------------------
app = cyclopts.App()


@app.default()
def main(vid_path: pathlib.Path):

    # vid_path should end in seed{}.mp4
    # Extract the number.
    seed_num = int(vid_path.stem.split("seed")[-1])

    first_goal_frame = {
        21: 257
    }

    Tn_achieved, T_state = get_timing_data(seed_num)

    # -------------------------------------------------------------------
    # Find the first index where Tn_achieved is true. We want to compute an offset so that it matches up with
    # first_goal_frame.
    T_achieved = np.any(Tn_achieved, axis=1)
    first_goal_frame_data = np.argmax(T_achieved)
    # first_goal_frame is the "real" frame number.
    offset = first_goal_frame_data - first_goal_frame[seed_num]

    # -------------------------------------------------------------------

    text = "GF( r₁ ) ∧ GF( r₂ ) ∧ GF( rs₁ ) ∧ GF( rs₂ )\n∧ G¬aerial_collide ∧ G¬obstacle ∧ G¬no_fly_zone"
    n = len(text)

    out_path = vid_path.with_name(f"{vid_path.stem}_labeled.mp4")

    # color_true = (np.array(to_rgb("#fd8925")) * 255).astype(np.uint8)
    color_true = (np.array(to_rgb("#ffb17d")) * 255).astype(np.uint8)

    timers = defaultdict(int)
    fps = 50
    n_frames_target = int(0.5 * fps)

    def rgb_at_time(kk_real: int) -> np.ndarray:
        nonlocal timers
        # Start with all-white
        rgb = np.full((n, 3), 255, dtype=np.uint8)

        kk_data = kk_real + offset

        # Before change_idxs[0], only G( ¬collide ) is true.
        # Between change_idxs[0] and change_idxs[1], r₀ becomes true.
        # Between change_idxs[1] and change_idxs[2], F( r₀  ∧ F( r₁ ) ) becomes true.
        # After change_idxs[2], FG( rₕ ) becomes true.

        tmp = "G¬aerial_collide"
        start = text.find(tmp)
        end = text.find(tmp) + len(tmp)
        rgb[start:end, :] = color_true

        tmp = "G¬obstacle"
        start = text.find(tmp)
        end = text.find(tmp) + len(tmp)
        rgb[start:end, :] = color_true

        tmp = "G¬no_fly_zone"
        start = text.find(tmp)
        end = text.find(tmp) + len(tmp)
        rgb[start:end, :] = color_true

        # When achieved[0], color F( r₁ ) for a few frames.
        achieved0 = Tn_achieved[kk_data, 0]
        achieved1 = Tn_achieved[kk_data, 1]

        if achieved0:
            timers["achieved0"] = n_frames_target

        if achieved1:
            timers["achieved1"] = n_frames_target

        if timers["achieved0"] > 0:
            tmp = "F( r₁ )"
            start = text.find(tmp)
            end = text.find(tmp) + len(tmp)
            rgb[start:end, :] = color_true

            timers["achieved0"] -= 1

        if timers["achieved1"] > 0:
            tmp = "F( r₂ )"
            start = text.find(tmp)
            end = text.find(tmp) + len(tmp)
            rgb[start:end, :] = color_true

            timers["achieved1"] -= 1

        return rgb

    clip = VideoFileClip(vid_path)
    out = add_colored_text_overlay_per_char_rgb(
        clip,
        text,
        font_path="/Users/oswinso/Library/Fonts/DejaVuSans.ttf",
        font_size=64,
        pos=(40, 40),
        bg="#00000080",
        rgb_at_time=rgb_at_time,
        align="left"
    )
    out.write_videofile(out_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
