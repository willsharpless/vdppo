#!/usr/bin/env python3
"""
Extract background from a video by sampling evenly spaced frames and
computing the per-pixel median.

Requires:
  pip install opencv-python numpy

Example:
  python extract_background_median.py input.mp4 -n 25 -o background.png
"""

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np


def evenly_spaced_indices(total: int, num_samples: int) -> List[int]:
    """Return integer frame indices evenly spaced in [0, total-1]."""
    if total <= 0:
        return []
    if num_samples <= 1:
        return [0]
    num_samples = min(num_samples, total)
    # linspace inclusive endpoints, then round and unique while preserving order
    idx = np.linspace(0, total - 1, num=num_samples, dtype=np.float64)
    idx = np.round(idx).astype(np.int64)
    # Make unique in order (rounding can create duplicates)
    seen = set()
    out = []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    """Seek to a frame and read it. Returns BGR uint8 image."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at index {frame_idx}")
    return frame


def compute_median_background(
    video_path: str,
    num_samples: int,
    max_width: int = 0,
    max_height: int = 0,
) -> np.ndarray:
    """
    Sample frames evenly across the video and compute median background.

    max_width/max_height:
      If > 0, frames will be resized to fit within those bounds (preserving aspect),
      which can greatly reduce memory usage.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Some codecs don't report frame count reliably; fallback to reading sequentially.
        # We'll approximate by sampling based on duration * fps if possible.
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration_sec = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Reset and just read every k-th frame until EOF (fallback method).
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        count = 0
        step = 1
        # Try to get roughly num_samples by adapting step as we go.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % step == 0:
                frames.append(frame)
                if len(frames) >= num_samples:
                    break
            count += 1
            if count > 0 and len(frames) > 0:
                step = max(1, count // max(1, num_samples))
        cap.release()
        if not frames:
            raise RuntimeError("No frames could be read from the video.")
        stack = np.stack([maybe_resize(f, max_width, max_height) for f in frames], axis=0)
        bg = np.median(stack, axis=0).astype(np.uint8)
        return bg

    indices = evenly_spaced_indices(total_frames, num_samples)
    frames = []
    try:
        for idx in indices:
            frame = read_frame_at(cap, idx)
            frame = maybe_resize(frame, max_width, max_height)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No frames sampled; cannot compute background.")

    # Stack to shape: (K, H, W, 3) then median over K
    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)  # uint8 BGR
    return bg


def maybe_resize(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize to fit within max_width/max_height (if provided), preserving aspect."""
    h, w = frame.shape[:2]
    if (max_width and w > max_width) or (max_height and h > max_height):
        scale_w = max_width / w if max_width else 1.0
        scale_h = max_height / h if max_height else 1.0
        scale = min(scale_w, scale_h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("-n", "--num-samples", type=int, default=25,
                    help="Number of evenly sampled frames to use (default: 25)")
    ap.add_argument("-o", "--output", default="background.png",
                    help="Output image path (default: background.png)")
    ap.add_argument("--max-width", type=int, default=0,
                    help="If set, resize frames to fit within this width")
    ap.add_argument("--max-height", type=int, default=0,
                    help="If set, resize frames to fit within this height")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 2
    if args.num_samples <= 0:
        print("Error: --num-samples must be > 0", file=sys.stderr)
        return 2

    bg_bgr = compute_median_background(
        args.video,
        num_samples=args.num_samples,
        max_width=args.max_width,
        max_height=args.max_height,
    )

    ok = cv2.imwrite(args.output, bg_bgr)
    if not ok:
        print(f"Error: failed to write output image: {args.output}", file=sys.stderr)
        return 1

    print(f"Saved background to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
