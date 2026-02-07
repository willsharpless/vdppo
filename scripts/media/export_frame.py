import pathlib
from typing import Annotated, Optional, Tuple

import cv2
import cyclopts
import numpy as np
import tqdm
from cyclopts import Parameter
from loguru import logger

app = cyclopts.App()


@app.default()
def multi(vid_path: pathlib.Path, frames: str, traj_idx: int):
    # frames: "[100 200 300]".
    frames: list[int] = list(map(int, frames.strip("[]").split()))

    # Exports the specified frames as an image.
    cap = cv2.VideoCapture(vid_path)

    for frame_idx in frames:
        out_path = vid_path.parent / f"{vid_path.stem}_mask_corrections" / f"traj{traj_idx}_frame{frame_idx:05d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            logger.warning(f"Frame {frame_idx} already exists at {out_path}, skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}.")

        # vid_path = media/herd_fig/67_crop.mp4
        # we want to save at media/herd_fig/67_crop_mask_corrections/traj{}_frame{:05d}.png
        cv2.imwrite(out_path, frame_bgr)
        logger.success(f"Frame {frame_idx} saved to {out_path}")


@app.command()
def single(vid_path: pathlib.Path, frame_idx: int):
    # Exports the specified frame as an image.
    cap = cv2.VideoCapture(vid_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        raise ValueError(f"frame_idx {frame_idx} is out of bounds for video with {total_frames} frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame at index {frame_idx}.")

    out_path = vid_path.with_name(f"{vid_path.stem}_frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, frame_bgr)
    logger.success(f"Frame {frame_idx} saved to {out_path}")


if __name__ == "__main__":
    app()
