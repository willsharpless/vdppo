import pathlib
from typing import Optional, Tuple

import cv2
import cyclopts
import numpy as np
import tqdm
from loguru import logger

app = cyclopts.App()


@app.default()
def main(vid_path: pathlib.Path, frame_idx: int):
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



if __name__ == '__main__':
    app()