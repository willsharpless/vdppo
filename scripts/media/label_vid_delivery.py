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
    if isinstance(c, (tuple, np.ndarray)):
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
        # ---- NEW: Arrow ----
        arrow_span_at_frame: Optional[Callable[[int], Optional[Tuple[int, int]]]] = None,
        arrow_char: str = "↓",
        arrow_alpha: int = 255,
        arrow_gap: int = 6,  # pixels between arrow and the target line

        # NEW: desired space between top of arrow glyph and top edge of overlay
        arrow_padding_top: int = 8,

        # NEW: arrow color per frame (RGB or RGBA-ish; alpha handled separately unless you pass RGBA/hex8)
        arrow_color_at_frame: Optional[Callable[[int], Color]] = None,
        arrow_default_color: Color = (255, 200, 120),
        arrow_default_alpha: int = 255
) -> CompositeVideoClip:
    if align not in ("left", "center", "right"):
        raise ValueError("align must be 'left', 'center', or 'right'")
    if arrow_padding_top < 0:
        raise ValueError("arrow_padding_top must be >= 0")

    font = ImageFont.truetype(font_path, font_size)
    arrow_font = font  # can be swapped to a different font if desired

    if line_spacing is None:
        line_spacing = int(round(0.2 * font_size))

    lines: list[str] = text.split("\n")

    # Map each line to [start,end) indices in original text (including '\n')
    line_ranges: list[Tuple[int, int]] = []
    idx = 0
    for li, line in enumerate(lines):
        start = idx
        end = start + len(line)
        line_ranges.append((start, end))
        idx = end + (1 if li < len(lines) - 1 else 0)

    # Reference line height (top-left drawing)
    ref_bbox = font.getbbox("Ag")
    line_h = int(ref_bbox[3] - ref_bbox[1])
    line_step = line_h + line_spacing

    # Width per line (kerning-aware)
    line_widths = [float(font.getlength(line)) for line in lines]
    text_w = int(np.ceil(max(line_widths) if line_widths else 0.0))
    text_h = line_h * len(lines) + line_spacing * max(0, len(lines) - 1)

    # Cache per-line prefix, advances, bboxes
    prefix_cache: list[np.ndarray] = []
    adv_cache: list[np.ndarray] = []
    bbox_cache: list[Tuple[int, int, int, int]] = []
    for line in lines:
        if line:
            prefix = np.zeros(len(line) + 1, dtype=np.float32)
            for k in range(1, len(line) + 1):
                prefix[k] = float(font.getlength(line[:k]))
            prefix_cache.append(prefix)
            adv_cache.append(np.diff(prefix))
            bbox_cache.append(font.getbbox(line))
        else:
            prefix_cache.append(np.zeros(1, dtype=np.float32))
            adv_cache.append(np.zeros(0, dtype=np.float32))
            bbox_cache.append((0, 0, 0, 0))

    # ---- NEW: automatically allocate top space for the arrow based on arrow bbox ----
    # PIL bbox is (x0,y0,x1,y1) in font coords; y0 can be negative.
    arrow_bbox = arrow_font.getbbox(arrow_char)
    arrow_h = float(arrow_bbox[3] - arrow_bbox[1])

    # This is the extra top "header" space where the arrow lives.
    # It guarantees the arrow's top is >= arrow_padding_top.
    #
    # Arrow y is computed later as:
    #   ay = line_y_top - arrow_h - arrow_gap - arrow_bbox[1]
    # and line_y_top = header_top + padding + ... - bbox_line[1]
    #
    # For the first line (li=0), worst-case requirement to keep arrow from touching top is:
    #   ay + arrow_bbox[1] >= arrow_padding_top
    # => header_top + padding - bbox_line0[1] - arrow_h - arrow_gap >= arrow_padding_top
    #
    # We'll compute using first non-empty line bbox, fallback to 0 if all empty.
    first_line_idx = next((i for i, ln in enumerate(lines) if ln), None)
    bbox_line0_y1 = bbox_cache[first_line_idx][1] if first_line_idx is not None else 0

    header_top = int(
        np.ceil(
            arrow_padding_top
            - padding
            + bbox_line0_y1
            + arrow_h
            + arrow_gap
        )
    )
    header_top = max(0, header_top)

    img_w = text_w + 2 * padding
    img_h = (text_h + 2 * padding) + header_top

    def _line_x_start(li: int) -> float:
        lw = line_widths[li]
        if align == "left":
            return float(padding)
        if align == "center":
            return float(padding + (text_w - lw) / 2.0)
        return float(padding + (text_w - lw))

    def _line_y_top(li: int) -> float:
        bbox_line = bbox_cache[li]
        return float(header_top + padding + li * line_step - bbox_line[1])

    def _rgb_for_frame(frame_idx: int) -> np.ndarray:
        if rgb_at_time is None:
            return np.tile(np.array(default_rgb, dtype=np.uint8), (len(text), 1))
        arr = np.asarray(rgb_at_time(frame_idx))
        if arr.shape != (len(text), 3):
            raise ValueError(f"rgb_at_time(frame) must return shape ({len(text)}, 3), got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255)
        return arr

    def _arrow_rgba(frame_idx: int) -> RGBA:
        if arrow_color_at_frame is None:
            return _parse_color(arrow_default_color, default_alpha=arrow_default_alpha)
        return _parse_color(arrow_color_at_frame(frame_idx), default_alpha=arrow_default_alpha)

    def _span_center_px(span: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        s, e = span
        if not (0 <= s < e <= len(text)):
            return None
        if text[s:e].strip() == "":
            return None

        # Find line containing s
        li = None
        for k, (a, b) in enumerate(line_ranges):
            if a <= s <= b:
                li = k
                break
        if li is None or not lines[li]:
            return None

        line_start, line_end = line_ranges[li]
        ss = max(s, line_start)
        ee = min(e, line_end)
        if ss >= ee:
            return None

        rel_s = ss - line_start
        rel_e = ee - line_start

        prefix = prefix_cache[li]
        x_line = _line_x_start(li)

        cx = x_line + float(prefix[rel_s] + prefix[rel_e]) / 2.0
        y_top = _line_y_top(li)
        return cx, y_top

    def rgba_frame(frame_idx: int) -> np.ndarray:
        im = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)

        if bg is not None:
            draw.rectangle([(0, 0), (img_w - 1, img_h - 1)],
                           fill=_parse_color(bg, default_alpha=default_alpha))

        rgb = _rgb_for_frame(frame_idx)

        # Draw text
        for li, line in enumerate(lines):
            if not line:
                continue
            x = _line_x_start(li)
            y_top = _line_y_top(li)
            start, _end = line_ranges[li]
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

        # Draw arrow
        if arrow_span_at_frame is not None:
            span = arrow_span_at_frame(frame_idx)
            if span is not None:
                center = _span_center_px(span)
                if center is not None:
                    cx, line_y_top = center
                    arrow_rgba = _arrow_rgba(frame_idx)

                    arrow_w = float(arrow_font.getlength(arrow_char))
                    # reuse arrow_bbox computed above
                    ax = cx - arrow_w / 2.0

                    # Place arrow above the target line with requested gap.
                    ay = line_y_top - arrow_h - float(arrow_gap) - float(arrow_bbox[1])

                    # Enforce padding top exactly (safety clamp)
                    min_ay = float(arrow_padding_top - arrow_bbox[1])
                    if ay < min_ay:
                        ay = min_ay

                    draw.text((ax, ay), arrow_char, font=arrow_font, fill=arrow_rgba)

        return np.asarray(im, dtype=np.uint8)

    # Cache last RGBA frame because MoviePy will call rgb+mask
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

    # duration = 500 * (1.0 / base_clip.fps)
    # clip = clip.subclipped(0, duration)

    return clip

def get_timing_data(seed: int):
    export_dir = "/Users/oswinso/research/me/vdppo/media/02-04__delivery/timing_npzs"
    export_dir = pathlib.Path(export_dir)
    npz_path = export_dir / f"timing_seed{seed}.npz"

    npz = np.load(npz_path)
    Tn_achieved = npz["Tn_achieved"]
    T_state = npz["T_state"]

    return Tn_achieved, T_state

def _clamp01(x: float) -> float:
    return np.clip(x, 0.0, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return (1-t) * a + t * b


def _lerp_color(c0: Color, c1: Color, t: float) -> Color:
    t = _clamp01(t)
    out = np.array([
        int(round(_lerp(c0[0], c1[0], t))),
        int(round(_lerp(c0[1], c1[1], t))),
        int(round(_lerp(c0[2], c1[2], t))),
    ])

    assert out.max() <= 255 and out.min() >= 0, f"Color values must be in 0..255, got {out}"
    return out

def ease_out_cubic(t: float) -> float:
    """Fast start, smooth settle."""
    t = _clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(t: float) -> float:
    """Smooth ramp-up, slower start."""
    t = _clamp01(t)
    return t ** 3


def reward_pulse_color(
    t: float,
    neutral: Color,
    reward: Color,
    reward_overshoot: Color,
    duration_in: float = 0.25,
    hold: float = 0.10,
    duration_out: float = 0.35,
) -> Color:
    """
    Compute color for a neutral -> reward -> neutral "pulse".

    Timeline:
      - [0, duration_in): ease-out from neutral to reward
      - [duration_in, duration_in+hold): hold reward
      - [duration_in+hold, duration_in+hold+duration_out): ease-in back to neutral
      - after that: neutral

    Args:
      t: time since the pulse started (seconds).
      neutral: base (R,G,B).
      reward: peak reward (R,G,B).
      duration_in: time to reach reward.
      hold: time to hold at reward.
      duration_out: time to return to neutral.

    Returns:
      RGB tuple in 0..255.
    """
    if t <= 0:
        return neutral

    t_in_end = duration_in
    t_hold_end = duration_in + hold
    t_out_end = duration_in + hold + duration_out

    if t < t_in_end:
        u = t / max(duration_in, 1e-9)
        return _lerp_color(neutral, reward_overshoot, ease_out_cubic(u))

    if t < t_hold_end:
        return reward_overshoot

    if t < t_out_end:
        u = (t - t_hold_end) / max(duration_out, 1e-9)
        return _lerp_color(reward, neutral, u)

    return neutral

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
        12: 386,
        13: 125,
        14: 61,
        15: 48,
        16: 39,
        17: 311,
        18: 123,
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
    color_true_overshoot = (np.array(to_rgb("#ff8a04")) * 255).astype(np.uint8)

    color_agent1 = (np.array(to_rgb("#4285f4")) * 255).astype(np.uint8)
    color_agent0 = (np.array(to_rgb("#aea1ff")) * 255).astype(np.uint8)

    color_base = (np.array(to_rgb("#ffffff")) * 255).astype(np.uint8)

    timers = defaultdict(int)
    fps = 50

    frames_in = int(0.1 * fps)
    frames_hold = int(0.04 * fps)
    frames_out = int(0.6 * fps)
    n_frames_target = frames_in + frames_hold + frames_out
    # n_frames_target = int(0.5 * fps)

    def rgb_at_time(kk_real: int) -> np.ndarray:
        nonlocal timers
        # Start with all-white
        rgb = np.full((n, 3), 255, dtype=np.uint8)
        rgb[:, :] = color_base

        kk_data = kk_real + offset
        kk_data_next = min(kk_data + 1, T_state.shape[0] - 1)

        state = T_state[kk_data]
        state_next = T_state[kk_data_next]

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
        base0 = (state == 3) and (state != state_next)
        base1 = (state == 0) and (state != state_next)

        if achieved0:
            timers["achieved0"] = n_frames_target

        if achieved1:
            timers["achieved1"] = n_frames_target

        if base0:
            timers["base0"] = n_frames_target

        if base1:
            timers["base1"] = n_frames_target

        timer_dict = {
            "achieved0": "F( r₁ )",
            "achieved1": "F( r₂ )",
            "base0": "F( rs₁ )",
            "base1": "F( rs₂ )",
        }

        for key, phrase in timer_dict.items():
            if timers[key] > 0:
                t = n_frames_target - timers[key]
                # frac = timers[key] / n_frames_target
                # frac = frac ** frac_pow
                # color = (frac * color_true + (1.0 - frac) * color_base).astype(np.uint8)
                color = reward_pulse_color(t, color_base, color_true, color_true_overshoot, frames_in, frames_hold, frames_out)

                start = text.find(phrase)
                end = text.find(phrase) + len(phrase)
                rgb[start:end, :] = color

                timers[key] -= 1

        # if timers["achieved0"] > 0:
        #     frac = timers["achieved0"] / n_frames_target
        #     frac = frac ** frac_pow
        #     color = (frac * color_true + (1.0 - frac) * color_base).astype(np.uint8)
        #
        #     tmp = "F( r₁ )"
        #     start = text.find(tmp)
        #     end = text.find(tmp) + len(tmp)
        #     rgb[start:end, :] = color
        #
        #     timers["achieved0"] -= 1
        #
        # if timers["achieved1"] > 0:
        #     frac = timers["achieved0"] / n_frames_target
        #     frac = frac ** frac_pow
        #     color = (frac * color_true + (1.0 - frac) * color_base).astype(np.uint8)
        #
        #     tmp = "F( r₂ )"
        #     start = text.find(tmp)
        #     end = text.find(tmp) + len(tmp)
        #     rgb[start:end, :] = color_true
        #
        #     timers["achieved1"] -= 1
        #
        # if timers["base0"] > 0:
        #     tmp = "GF( rs₁ )"
        #     start = text.find(tmp)
        #     end = text.find(tmp) + len(tmp)
        #     rgb[start:end, :] = color_true
        #
        #     timers["base0"] -= 1
        #
        # if timers["base1"] > 0:
        #     tmp = "GF( rs₂ )"
        #     start = text.find(tmp)
        #     end = text.find(tmp) + len(tmp)
        #     rgb[start:end, :] = color_true
        #
        #     timers["base1"] -= 1

        return rgb

    def arrow_span_at_frame(kk_real: int):
        kk_data = kk_real + offset
        state = T_state[kk_data]

        # 0: Ag1 to Base (Agent 2)
        # 1: Ag0 to Target 0
        # 2: Ag1 to Target 1
        # 3: Ag0 to Base (Agent 2)

        if state == 0:
            tmp = "GF( rs₂ )"
        elif state == 1:
            tmp = "GF( r₁ )"
        elif state == 2:
            tmp = "GF( r₂ )"
        elif state == 3:
            tmp = "GF( rs₁ )"
        else:
            raise ValueError(f"Bad state: {state}")

        start = text.find(tmp)
        end = text.find(tmp) + len(tmp)
        return start, end

    def arrow_color_at_frame(kk_real: int):
        kk_data = kk_real + offset
        state = T_state[kk_data]

        if state in [0, 2]:
            return color_agent0
        elif state in [1, 3]:
            return color_agent1

        raise ValueError(f"Bad state: {state}")


    clip = VideoFileClip(vid_path)
    out = add_colored_text_overlay_per_char_rgb(
        clip,
        text,
        font_path="/Users/oswinso/Library/Fonts/DejaVuSans.ttf",
        font_size=64,
        pos=(40, 40),
        bg="#00000080",
        rgb_at_time=rgb_at_time,
        align="left",
        arrow_span_at_frame=arrow_span_at_frame,
        arrow_color_at_frame=arrow_color_at_frame,
    )
    out.write_videofile(out_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
