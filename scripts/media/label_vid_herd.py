from __future__ import annotations

import pathlib
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
) -> CompositeVideoClip:
    """
    MoviePy v2-compatible: overlays Unicode text with per-character RGB every frame.

    rgb_at_time(t) must return np.ndarray of shape (len(text), 3) with values in [0,255].
    """

    font = ImageFont.truetype(font_path, font_size)

    # Per-character x-advances (captures kerning) using prefix lengths
    prefix = np.zeros(len(text) + 1, dtype=np.float32)
    for i in range(1, len(text) + 1):
        prefix[i] = float(font.getlength(text[:i]))
    advances = np.diff(prefix)  # (n_chars,)

    bbox = font.getbbox(text)  # (x0,y0,x1,y1) relative to (0,0)
    text_w = int(np.ceil(prefix[-1]))
    text_h = int(bbox[3] - bbox[1])

    img_w = text_w + 2 * padding
    img_h = text_h + 2 * padding

    x0 = float(padding)
    y0 = float(padding - bbox[1])  # baseline shift so glyphs fit

    def _rgb_for_frame(kk: int) -> np.ndarray:
        if rgb_at_time is None:
            return np.tile(np.array(default_rgb, dtype=np.uint8), (len(text), 1))
        arr = np.asarray(rgb_at_time(kk))
        if arr.shape != (len(text), 3):
            raise ValueError(f"rgb_at_time(t) must return shape ({len(text)}, 3), got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255)
        return arr

    def rgba_frame(kk: int) -> np.ndarray:
        im = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)

        if bg is not None:
            draw.rectangle(
                [(0, 0), (img_w - 1, img_h - 1)],
                fill=_parse_color(bg, default_alpha=default_alpha),
            )

        rgb = _rgb_for_frame(kk)  # (n_chars, 3) uint8

        # Draw as runs of identical color (fewer draw calls)
        i = 0
        x = x0
        n = len(text)
        while i < n:
            j = i + 1
            while j < n and (rgb[j] == rgb[i]).all():
                j += 1

            run = text[i:j]
            r, g, b = map(int, rgb[i])
            draw.text((x, y0), run, font=font, fill=(r, g, b, int(default_alpha)))

            x += float(advances[i:j].sum())
            i = j

        return np.asarray(im, dtype=np.uint8)  # (h,w,4)

    last_kk = None
    last_rgba = None

    def rgba_cached(kk_):
        nonlocal last_kk, last_rgba
        if last_kk != kk_:
            last_rgba = rgba_frame(kk_)
            last_kk = kk_
        return last_rgba

    # MoviePy v2 expects RGB frames; alpha goes in a mask clip.
    def rgb_frame(t: float) -> np.ndarray:
        frame = frame_from_time(t, base_clip.fps)
        return rgba_cached(frame)[:, :, :3]

    def mask_frame(t: float) -> np.ndarray:
        frame = frame_from_time(t, base_clip.fps)
        # mask must be float in [0,1]
        return rgba_cached(frame)[:, :, 3].astype(np.float32) / 255.0

    overlay_rgb = VideoClip(frame_function=rgb_frame, duration=base_clip.duration, has_constant_size=True)
    overlay_mask = VideoClip(
        frame_function=mask_frame, duration=base_clip.duration, is_mask=True, has_constant_size=True
    )

    overlay = overlay_rgb.with_mask(overlay_mask).with_position(pos).with_duration(base_clip.duration)

    return CompositeVideoClip([base_clip, overlay])


# --------------------------
# Example usage
# --------------------------
app = cyclopts.App()


@app.default()
def main(vid_path: pathlib.Path):

    # vid_path should end in seed{}.mp4
    # Extract the number.
    seed_num = int(vid_path.stem.split("seed")[-1])

    offsets = {
        0: 91 - 3,
        1: 885 - 856,
        2: 921 - 827,
        3: 1,
        4: 981 - 483,
        5: 929 - 531,
        6: 957 - 579,
        7: 975 - 477
    }
    # real + offset = tgt
    # offset = tgt - real
    idxs_dict = {
        0: [835, 1109, 1581],
        1: [885, 1349, 1812],
        2: [921, 1090, 1559],
        3: [922, 1102, 1547],
        4: [981, 1083, 1653],
        5: [929, 1067, 1553],
        6: [957, 1070, 1525],
        7: [975, 1105, 1601]
    }
    change_idxs = idxs_dict[seed_num]
    assert len(change_idxs) == 3

    text = "G( ¬collide ) ∧ F( r₀  ∧ F( r₁ ) ) ∧ FG( rₕ )"
    n = len(text)

    out_path = vid_path.with_name(f"{vid_path.stem}_labeled.mp4")

    # color_true = (np.array(to_rgb("#fd8925")) * 255).astype(np.uint8)
    color_true = (np.array(to_rgb("#ffb17d")) * 255).astype(np.uint8)

    def rgb_at_time(kk_real: int) -> np.ndarray:
        # Start with all-white
        rgb = np.full((n, 3), 255, dtype=np.uint8)

        kk = kk_real + offsets[seed_num]

        # Before change_idxs[0], only G( ¬collide ) is true.
        # Between change_idxs[0] and change_idxs[1], r₀ becomes true.
        # Between change_idxs[1] and change_idxs[2], F( r₀  ∧ F( r₁ ) ) becomes true.
        # After change_idxs[2], FG( rₕ ) becomes true.

        tmp = "¬collide )"
        end1 = text.find(tmp) + len(tmp)
        rgb[0:end1, :] = color_true

        if change_idxs[0] <= kk < change_idxs[1]:
            tmp2 = "r₀"
            start2 = text.find(tmp2)
            end2 = start2 + len(tmp2)
            rgb[start2:end2, :] = color_true
        elif kk >= change_idxs[1]:
            tmp3 = "F( r₀  ∧ F( r₁ ) )"
            start3 = text.find(tmp3)
            end3 = start3 + len(tmp3)
            rgb[start3:end3, :] = color_true

        if kk >= change_idxs[2]:
            tmp4 = "FG( rₕ )"
            start4 = text.find(tmp4)
            end4 = start4 + len(tmp4)
            rgb[start4:end4, :] = color_true

        # # Example: color a few spans based on time
        # def span(s: str, occurrence: int = 0):
        #     start = -1
        #     idx = 0
        #     for _ in range(occurrence + 1):
        #         start = text.find(s, idx)
        #         if start < 0:
        #             return None
        #         idx = start + 1
        #     return start, start + len(s)
        #
        # spans = [
        #     span("¬collide", 0),
        #     span("r₀", 0),
        #     span("F( r₀  ∧ F( r₁ ) )", 0),
        #     span("FG( rₕ )", 0),
        # ]
        #
        # # Simple color cycle
        # k = int((t * 2) % 4)
        # palette = np.array(
        #     [[255, 80, 80], [80, 255, 80], [80, 150, 255], [255, 180, 80]],
        #     dtype=np.uint8,
        # )
        # for i, sp in enumerate(spans):
        #     if sp is None:
        #         continue
        #     a, b = sp
        #     rgb[a:b] = palette[(k + i) % len(palette)]
        #
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
    )
    out.write_videofile(out_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
