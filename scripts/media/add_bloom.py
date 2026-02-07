#!/usr/bin/env python3
"""
multistep_bloom.py

Apply multi-step bloom (multi-radius glow) to an image.

Typical use: run this on your rendered trails PNG to get a more photographic,
emissive look.

Requires:
  pip install opencv-python numpy cyclopts

Examples:
  python multistep_bloom.py in.png out.png --threshold 0.25 --knee 0.2 --strength 0.8
  python multistep_bloom.py in.png out.png --steps "2,4,8,16" --weights "1,1,0.8,0.6"
  python multistep_bloom.py in.png out.png --mode screen --gamma 2.2
"""

from __future__ import annotations

from loguru import logger
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from cyclopts import App, Parameter

app = App(name="multistep_bloom")


def _gamma_decode(img_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_0_1
    return np.clip(img_0_1, 0.0, 1.0) ** gamma


def _gamma_encode(img_lin_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_lin_0_1
    return np.clip(img_lin_0_1, 0.0, 1.0) ** (1.0 / gamma)


def _luma_bt709(rgb: np.ndarray) -> np.ndarray:
    # rgb shape (H,W,3) in linear [0,1]
    # Note: OpenCV images are BGR; caller should reorder if needed.
    r = rgb[..., 2]
    g = rgb[..., 1]
    b = rgb[..., 0]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _soft_threshold(x: np.ndarray, threshold: float, knee: float) -> np.ndarray:
    """
    Soft-knee thresholding for bloom extraction.
    x: luma or scalar in [0,1]
    threshold: where bloom starts
    knee: softness region size (0 = hard)
    Returns mask-like scalar in [0,1] (not strictly, but generally).
    """
    threshold = float(threshold)
    knee = float(max(knee, 0.0))
    if knee <= 0:
        return np.maximum(x - threshold, 0.0)
    # Smoothly ramps from 0 to (x-threshold) around threshold
    # Similar spirit to Unreal's soft-knee:
    # https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/
    t0 = threshold - knee
    t1 = threshold + knee
    y = np.zeros_like(x, dtype=np.float32)
    # below t0 -> 0
    # above t1 -> x - threshold
    above = x >= t1
    y[above] = x[above] - threshold
    mid = (x > t0) & (x < t1)
    # smoothstep 0..1 across knee region, then scale by knee to match continuity
    m = (x[mid] - t0) / (2.0 * knee)
    m = m * m * (3.0 - 2.0 * m)  # smoothstep
    y[mid] = m * (x[mid] - threshold + knee)  # continuous with above region
    return y


def _parse_csv_floats(s: str) -> list[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("Expected a comma-separated list, got empty.")
    return vals


def _ensure_len_match(values: list[float], n: int, name: str) -> list[float]:
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    raise ValueError(f"{name} must have length 1 or match steps length ({n}); got {len(values)}.")


def _screen(base: np.ndarray, add: np.ndarray) -> np.ndarray:
    # base/add in linear [0,1]
    return 1.0 - (1.0 - base) * (1.0 - add)


@Parameter(name="*")
@dataclass
class Args:
    inp: Path

    # Working space
    gamma: float = 2.2  # decode/encode gamma; use 1.0 if already linear

    # Bloom extraction
    threshold: float = 0.25  # in linear [0,1] luma
    knee: float = 0.20       # softness around threshold

    # Multi-step blur
    steps: str = "2,4,8,16"         # gaussian sigmas in pixels
    weights: str = "1,1,0.8,0.6"    # per-step weights
    strength: float = 0.8           # overall bloom strength multiplier

    # Composition
    mode: str = "add"        # "add" or "screen"
    clamp: bool = True       # clamp to [0,1] after composition

    # Optional: preserve sharp core by mixing original back (usually keep at 1.0)
    core: float = 1.0        # 1.0 keeps original fully; <1 fades original


@app.default
def main(args: Args) -> None:
    if not args.inp.exists():
        raise SystemExit(f"Input not found: {args.inp}")

    mode = args.mode.lower().strip()
    if mode not in ("add", "screen"):
        raise SystemExit("--mode must be 'add' or 'screen'")

    if args.gamma <= 0:
        raise SystemExit("--gamma must be > 0")
    if args.strength < 0:
        raise SystemExit("--strength must be >= 0")
    if not (0.0 <= args.core <= 1.0):
        raise SystemExit("--core must be in [0,1]")

    sigmas = _parse_csv_floats(args.steps)
    wts = _parse_csv_floats(args.weights)
    wts = _ensure_len_match(wts, len(sigmas), "weights")

    img_bgr_u8 = cv2.imread(str(args.inp), cv2.IMREAD_COLOR)
    if img_bgr_u8 is None:
        raise SystemExit(f"Failed to read image: {args.inp}")

    img = img_bgr_u8.astype(np.float32) / 255.0
    img_lin = _gamma_decode(img, args.gamma)

    # Extract bloom source using soft threshold on luma, then scale RGB by it.
    luma = _luma_bt709(img_lin).astype(np.float32)
    kick = _soft_threshold(luma, args.threshold, args.knee)  # scalar >= 0
    # Normalize kick to [0,1]-ish for a mask; prevent division by 0.
    # This keeps color ratios while extracting only highlights.
    denom = np.maximum(luma, 1e-6)
    mask = (kick / denom).astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0)

    src = img_lin * mask[..., None]  # highlight-only RGB

    # Multi-step bloom: blur at multiple sigmas and sum
    bloom = np.zeros_like(img_lin, dtype=np.float32)
    total_w = float(np.sum(wts)) if np.sum(wts) > 0 else 1.0

    for sigma, w in zip(sigmas, wts):
        sigma = float(max(sigma, 0.0))
        w = float(w)
        if w == 0.0:
            continue
        if sigma == 0.0:
            blurred = src
        else:
            # (0,0) lets OpenCV pick kernel size from sigma
            blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
        bloom += (w / total_w) * blurred

    bloom *= float(args.strength)

    # Composite bloom back onto image
    base = img_lin * float(args.core)
    if mode == "add":
        out_lin = base + bloom
    else:
        out_lin = _screen(base, bloom)

    if args.clamp:
        out_lin = np.clip(out_lin, 0.0, 1.0)

    out = _gamma_encode(out_lin, args.gamma)
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    out = args.inp.with_name("{}_bloom.png".format(args.inp.stem))
    ok = cv2.imwrite(out, out_u8)
    logger.success(f"Saved: {out}")


if __name__ == "__main__":
    app()
