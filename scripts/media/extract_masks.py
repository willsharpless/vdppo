import pathlib
from typing import Annotated, List, Optional, Tuple

import cv2
import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from cyclopts import Parameter
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from segment_anything import SamPredictor, sam_model_registry

from vdppo.common.path_utils import get_root_dir

app = cyclopts.App()


def load_trajectory(traj_path: pathlib.Path) -> List[Optional[Tuple[float, float]]]:
    """Load trajectory from .npy file."""
    data = np.load(traj_path, allow_pickle=True)
    return list(data)


@app.default()
def main(
    vid_path: pathlib.Path,
    traj_paths: Annotated[List[pathlib.Path], Parameter(consume_multiple=True)],
    out_path: Optional[pathlib.Path] = None,
    device: str = "cuda",
):
    """
    Extract object masks for each frame using SAM, based on trajectory points.

    Saves a numpy array of shape (T, H, W) with dtype int, where:
    - 0 = background (no object)
    - 1 = object from 1st trajectory
    - 2 = object from 2nd trajectory
    - etc.
    """
    if not vid_path.exists():
        raise SystemExit(f"Video not found: {vid_path}")

    for traj_path in traj_paths:
        if not traj_path.exists():
            raise SystemExit(f"Trajectory not found: {traj_path}")

    # Load SAM model (vit_b is ~4x faster than vit_h with minimal quality loss)
    ckpt_path = get_root_dir() / "other_ckpts/sam_vit_b_01ec64.pth"
    logger.info(f"Loading SAM model from {ckpt_path}...")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    # Load trajectories
    trajectories = [load_trajectory(traj_path) for traj_path in traj_paths]
    n_trajs = len(trajectories)
    logger.info(f"Loaded {n_trajs} trajectories")

    # Get trajectory lengths and validate
    traj_lengths = [len(t) for t in trajectories]
    min_len = min(traj_lengths)
    max_len = max(traj_lengths)
    if min_len != max_len:
        logger.warning(f"Trajectory lengths vary: {traj_lengths}. Using min length: {min_len}")
    n_frames = min_len

    # Open video
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid_path}")

    vid_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if n_frames > vid_n_frames:
        logger.warning(f"Trajectories have {n_frames} points but video has {vid_n_frames} frames. Truncating.")
        n_frames = vid_n_frames

    logger.info(f"Processing {n_frames} frames at {w}x{h}")

    # Output array: (T, H, W) with object IDs (uint8 is sufficient for <256 objects)
    masks_out = np.zeros((n_frames, h, w), dtype=np.uint8)

    if out_path is None:
        out_path = vid_path.with_name(f"{vid_path.stem}__masks.npz")

    for frame_idx in tqdm.trange(n_frames, desc="Extracting masks"):
        ok, frame_bgr = cap.read()
        if not ok:
            logger.error(f"Failed to read frame {frame_idx}")
            break

        # Convert to RGB for SAM
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Set the image for SAM (only once per frame)
        predictor.set_image(frame_rgb)

        # Get points for this frame from all trajectories
        points = []
        labels = []
        traj_indices = []

        for traj_idx, traj in enumerate(trajectories):
            pt = traj[frame_idx]
            if pt is not None:
                points.append([pt[0], pt[1]])
                labels.append(1)  # 1 = foreground point
                traj_indices.append(traj_idx)

        if len(points) == 0:
            # No valid points for this frame, leave mask as zeros
            continue

        # Predict masks for each point separately to get individual object masks
        for i, (pt, traj_idx) in enumerate(zip(points, traj_indices)):
            point_coords = np.array([[pt[0], pt[1]]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Use the mask with highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]  # (H, W) boolean

            # Assign object ID (1-indexed) where mask is True
            # Later objects overwrite earlier ones in case of overlap
            object_id = traj_idx + 1
            masks_out[frame_idx][mask] = object_id

    cap.release()

    # Save masks with compression (much smaller for sparse mask data)
    np.savez_compressed(out_path, masks=masks_out)

    # Report file size
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    uncompressed_mb = masks_out.nbytes / (1024 * 1024)
    logger.success(f"Saved masks to: {out_path}")
    logger.info(f"Mask array shape: {masks_out.shape}, dtype: {masks_out.dtype}")
    logger.info(f"File size: {file_size_mb:.2f} MB (uncompressed would be {uncompressed_mb:.2f} MB)")
    logger.info(f"Unique values: {np.unique(masks_out)}")


@app.command()
def compress(img_path: pathlib.Path):
    arr = np.load(img_path)
    print("shape: {}, dtype: {}".format(arr.shape, arr.dtype))

    arr = arr.astype(np.uint8)
    # Save compressed
    np.savez_compressed(img_path.with_suffix(".npz"), masks=arr)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
