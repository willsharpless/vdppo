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

from rraa_rl.common.path_utils import get_root_dir

app = cyclopts.App()


def extract_background(cap: cv2.VideoCapture, n_frames_to_use: int = 100) -> np.ndarray:
    """Extract background using median of evenly sampled frames."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_to_use = min(n_frames, n_frames_to_use)
    frame_idxs = np.round(np.linspace(0, n_frames - 1, n_frames_to_use)).astype(int)
    frame_idxs = np.unique(frame_idxs)

    buffer = None
    for ii, frame_idx in enumerate(tqdm.tqdm(frame_idxs, desc="Extracting background")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()

        if buffer is None:
            buffer = np.zeros((len(frame_idxs), *frame_bgr.shape), dtype=frame_bgr.dtype)

        if not ok:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}")

        buffer[ii] = frame_bgr

    background = np.median(buffer, axis=0).astype(np.uint8)
    return background


def load_or_extract_background(vid_path: pathlib.Path, cap: cv2.VideoCapture) -> np.ndarray:
    """Load cached background or extract and cache it."""
    bg_path = vid_path.with_name(f"{vid_path.stem}__bg.png")

    if bg_path.exists():
        logger.info(f"Loading cached background from: {bg_path}")
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            return bg
        logger.warning(f"Failed to load cached background, re-extracting...")

    logger.info("Extracting background...")
    bg = extract_background(cap, n_frames_to_use=100)
    cv2.imwrite(str(bg_path), bg)
    logger.success(f"Saved background to: {bg_path}")
    return bg


def load_trajectory(traj_path: pathlib.Path) -> List[np.ndarray]:
    """Load trajectory from .npy file."""
    data = np.load(traj_path, allow_pickle=True)
    return data


def draw_trajectory_with_gradient(
    ax: plt.Axes,
    trajectory: List[Optional[Tuple[float, float]]],
    start_idx: int,
    end_idx: int,
    color: str,
    linewidth: float = 2,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0,
    alpha_power: float = 2.0,
    segment_length: float = 5.0,
) -> None:
    """
    Draw trajectory segment with opacity gradient using matplotlib.
    Older points (start_idx) are more transparent, newer points (end_idx) are more opaque.
    Uses arc-length parameterization so opacity changes smoothly with distance traveled.
    Resamples trajectory to have uniform segment lengths for consistent visual density.
    """

    min_alpha_orig = min_alpha
    max_alpha_orig = max_alpha
    if alpha_power != 1.0:
        # So that max_alpha remains the same after applying power
        alpha_add = min_alpha - min_alpha**alpha_power
        # max_alpha_new ** alpha_power + alpha_add = max_alpha
        # max_alpha_new = ((max_alpha - alpha_add)) ** (1 / alpha_power)
        max_alpha = ((max_alpha - alpha_add)) ** (1 / alpha_power)
    else:
        alpha_add = 0.0

    # Clamp indices
    start_idx = max(0, start_idx)
    end_idx = min(len(trajectory) - 1, end_idx)

    if start_idx >= end_idx:
        return

    # Get valid points in range
    raw_points = []
    for i in range(start_idx, end_idx + 1):
        pt = trajectory[i]
        if pt is not None:
            raw_points.append((pt[0], pt[1]))

    if len(raw_points) < 2:
        return

    # Convert to numpy array for easier manipulation
    raw_points = np.array(raw_points)  # shape (N, 2)

    # Compute cumulative arc-length at each point
    diffs = np.diff(raw_points, axis=0)  # (N-1, 2)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # (N-1,)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])  # (N,)
    total_length = cum_lengths[-1]

    if total_length < 1e-6:
        return

    # If total_length is very short, then make segment_length even shorter
    segment_length = min(segment_length, total_length / 5)

    # Resample at equal arc-length intervals
    n_segments = max(1, int(np.ceil(total_length / segment_length)))
    sample_distances = np.linspace(0, total_length, n_segments + 1)

    # Interpolate points at these arc-length positions
    resampled_points = []
    for s in sample_distances:
        # Find which segment this distance falls into
        idx = np.searchsorted(cum_lengths, s, side="right") - 1
        idx = np.clip(idx, 0, len(raw_points) - 2)

        # Interpolate within the segment
        s0 = cum_lengths[idx]
        s1 = cum_lengths[idx + 1]
        seg_len = s1 - s0

        if seg_len > 1e-9:
            t = (s - s0) / seg_len
        else:
            t = 0.0

        t = np.clip(t, 0.0, 1.0)
        p0 = raw_points[idx]
        p1 = raw_points[idx + 1]
        interp_pt = p0 + t * (p1 - p0)
        resampled_points.append(interp_pt)

    resampled_points = np.array(resampled_points)  # (n_segments + 1, 2)

    # Build segments from resampled points
    segments = []
    for i in range(len(resampled_points) - 1):
        p0 = resampled_points[i]
        p1 = resampled_points[i + 1]
        segments.append([(p0[0], p0[1]), (p1[0], p1[1])])

    # Calculate alpha based on arc-length position (newer = more opaque)
    # Use midpoint of each segment for alpha calculation
    n_segs = len(segments)
    alphas = []
    for i in range(n_segs):
        # t is the normalized position of segment midpoint
        t = (i + 0.5) / n_segs
        alpha = min_alpha + t * (max_alpha - min_alpha)
        alpha = alpha**alpha_power + alpha_add
        alphas.append(alpha)

    alphas = np.array(alphas)
    if len(alphas) > 0:
        alphas[-1] = max_alpha_orig

    # Create LineCollection with varying alpha
    base_color = to_rgba(color)
    colors = [(base_color[0], base_color[1], base_color[2], a) for a in alphas]

    zorder = 100
    lc = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round", zorder=zorder)
    ax.add_collection(lc)

    # # Draw a circle at the final point
    # if points:
    #     _, final_pt = points[-1]
    #     ax.plot(final_pt[0], final_pt[1], "o", color=color, markersize=linewidth + 3)


def sample_frames_by_arc_length2(
    trajectories: List[np.ndarray],
    start_idx: int,
    end_idx: int,
    n_samples: int,
    arc_length_weight: float = 1.0,
) -> List[np.ndarray]:
    """Compute a set of frame indices for each trajectory independently.
    Blend between time-based and arc-length-based spacing.
    """
    n_points = end_idx - start_idx + 1

    all_frame_indices = []
    for traj in trajectories:
        cum_dist = [0.0]
        for i in range(start_idx + 1, end_idx + 1):
            pt_prev = traj[i - 1]
            pt_curr = traj[i]

            if pt_prev is not None and pt_curr is not None:
                dx = pt_curr[0] - pt_prev[0]
                dy = pt_curr[1] - pt_prev[1]
                dist = np.sqrt(dx * dx + dy * dy)
            else:
                dist = 0.0

            cum_dist.append(cum_dist[-1] + dist)

        cum_dist = np.array(cum_dist)
        total_dist = cum_dist[-1]

        # Normalize arc-length parameterization to [0, 1]
        if total_dist > 1e-6:
            arc_length_param = cum_dist / total_dist
        else:
            arc_length_param = np.linspace(0, 1, n_points)

        # Time parameterization (linear from 0 to 1)
        time_param = np.linspace(0, 1, n_points)

        # Blend between time and arc-length parameterization
        blended_param = (1 - arc_length_weight) * time_param + arc_length_weight * arc_length_param

        # Sample at equal intervals in the blended parameterization
        target_params = np.linspace(0, 1, n_samples)

        # Find frame indices corresponding to each target parameter
        frame_indices = []
        for target_p in target_params:
            # Find the index where blended param exceeds target
            idx = np.searchsorted(blended_param, target_p)
            # Clamp to valid range and convert to frame index
            idx = min(idx, n_points - 1)
            frame_idx = start_idx + idx
            frame_indices.append(frame_idx)

        frame_indices = np.array(frame_indices)
        frame_indices = np.unique(frame_indices)
        assert len(frame_indices) <= n_samples
        all_frame_indices.append(frame_indices)

    return all_frame_indices


def sample_frames_by_arc_length(
    trajectories: List[np.ndarray],
    start_idx: int,
    end_idx: int,
    n_samples: int,
    arc_length_weight: float = 1.0,
) -> List[int]:
    """
    Sample frame indices with blending between time-based and arc-length-based spacing.

    Args:
        trajectories: List of trajectory arrays
        start_idx: Starting frame index
        end_idx: Ending frame index
        n_samples: Number of samples to take
        arc_length_weight: Blend factor between time and arc-length parameterization
            - 0.0 = pure time-based (equal time intervals)
            - 1.0 = pure arc-length-based (equal distance intervals)
            - 0.5 = blend between the two

    Uses the average cumulative distance across all trajectories.
    """
    if start_idx >= end_idx or n_samples < 2:
        return [end_idx]

    n_points = end_idx - start_idx + 1

    # Compute cumulative distance for each trajectory
    all_cum_dists = []

    for traj in trajectories:
        cum_dist = [0.0]
        for i in range(start_idx + 1, end_idx + 1):
            pt_prev = traj[i - 1]
            pt_curr = traj[i]

            if pt_prev is not None and pt_curr is not None:
                dx = pt_curr[0] - pt_prev[0]
                dy = pt_curr[1] - pt_prev[1]
                dist = np.sqrt(dx * dx + dy * dy)
            else:
                dist = 0.0

            cum_dist.append(cum_dist[-1] + dist)

        all_cum_dists.append(np.array(cum_dist))

    # Average cumulative distance across trajectories
    avg_cum_dist = np.mean(all_cum_dists, axis=0)
    total_dist = avg_cum_dist[-1]

    # Normalize arc-length parameterization to [0, 1]
    if total_dist > 1e-6:
        arc_length_param = avg_cum_dist / total_dist
    else:
        arc_length_param = np.linspace(0, 1, n_points)

    # Time parameterization (linear from 0 to 1)
    time_param = np.linspace(0, 1, n_points)

    # Blend between time and arc-length parameterization
    blended_param = (1 - arc_length_weight) * time_param + arc_length_weight * arc_length_param

    # Sample at equal intervals in the blended parameterization
    target_params = np.linspace(0, 1, n_samples)

    # Find frame indices corresponding to each target parameter
    frame_indices = []
    for target_p in target_params:
        # Find the index where blended param exceeds target
        idx = np.searchsorted(blended_param, target_p)
        # Clamp to valid range and convert to frame index
        idx = min(idx, n_points - 1)
        frame_idx = start_idx + idx
        frame_indices.append(frame_idx)

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return unique_indices


def load_masks(mask_path: pathlib.Path) -> np.ndarray:
    """Load masks from .npz file."""
    data = np.load(mask_path)
    return data["masks"]  # shape (T, H, W), dtype uint8


def get_video_frame(vid_path: pathlib.Path, frame_idx: int) -> np.ndarray:
    """Get a specific frame from the video as RGB."""
    cap = cv2.VideoCapture(str(vid_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_or_export_object_image(
    mask_corrections_dir: pathlib.Path,
    frame_idx: int,
    traj_idx: int,
    masks: np.ndarray,
    frame_rgb: np.ndarray,
    delta_frames: int = 0,
) -> List[Optional[np.ndarray]]:
    """
    Get masked object images for each trajectory, using corrected versions if available.

    For each trajectory, checks if traj{i}_frame{idx}.png exists.
    If it exists, loads and uses the corrected RGBA image.
    If not but delta_frames > 0, searches for a nearby corrected image within delta_frames.
    Otherwise, exports the current masked object as an RGBA PNG.

    Args:
        mask_corrections_dir: Directory for mask correction images
        frame_idx: The frame index to get objects for
        masks: The masks array (T, H, W)
        frame_rgb: The RGB frame at frame_idx
        delta_frames: If > 0, search for corrected images within this many frames

    Returns a list of RGBA images (H, W, 4) for each trajectory, or None if no object.
    """
    mask_corrections_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame_rgb.shape[:2]

    object_id = traj_idx + 1  # Object IDs are 1-indexed
    img_path = mask_corrections_dir / f"traj{traj_idx}_frame{frame_idx:05d}.png"

    # First check exact frame, then search nearby if delta_frames > 0
    candidate_path = None
    if img_path.exists():
        candidate_path = img_path
    elif delta_frames > 0:
        # Search for nearby corrected images within delta_frames
        for delta in range(1, delta_frames + 1):
            for nearby_idx in [frame_idx - delta, frame_idx + delta]:
                nearby_path = mask_corrections_dir / f"traj{traj_idx}_frame{nearby_idx:05d}.png"
                if nearby_path.exists():
                    candidate_path = nearby_path
                    logger.debug(f"Using nearby corrected image from frame {nearby_idx} (delta={delta})")
                    break
            if candidate_path is not None:
                break

    if candidate_path is not None:
        # Load corrected RGBA image
        corrected_bgra = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
        if corrected_bgra is not None and corrected_bgra.shape[2] == 4:
            # Convert BGRA to RGBA
            corrected_rgba = cv2.cvtColor(corrected_bgra, cv2.COLOR_BGRA2RGBA)
            logger.debug(f"Using corrected image from {candidate_path}")
            return corrected_rgba
        else:
            logger.warning(f"Failed to load corrected image from {candidate_path}, using original")

    if masks is None:
        # Use the entire image.
        object_mask = np.ones((h, w), dtype=bool)
    else:
        # Create RGBA image for this object
        frame_mask = masks[frame_idx]
        object_mask = frame_mask == object_id

        if not np.any(object_mask):
            # No pixels for this object in this frame
            return None

    # Create RGBA image
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = frame_rgb
    rgba[..., 3] = (object_mask * 255).astype(np.uint8)

    # Save for potential manual correction (BGRA for OpenCV)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(img_path), bgra)
    logger.debug(f"Exported object image to {img_path}")

    if masks is None:
        return None

    return rgba


@app.default()
def main(
    vid_path: pathlib.Path,
    traj_paths: Annotated[List[pathlib.Path], Parameter(consume_multiple=True)],
    mask_path: Optional[pathlib.Path] = None,  # .npz file from extract_masks.py
    colors: Optional[str] = None,  # comma-separated hex colors
    delta_frames: int = 2,  # search for corrected images within this many frames
    skip: int = 0,
    alpha_power: float = 2.0,
):
    """
    Create an overlay image showing trajectories on the video background.

    Draws trajectories from frame_idx - traj_len to frame_idx with an opacity
    gradient where older parts are more transparent. Saves as PDF.

    If mask_path is provided, overlays the actual objects from the video frame
    on top of the background and trajectories.

    The delta_frames parameter allows using corrected mask images from nearby frames
    if the exact frame's correction doesn't exist (useful when corrections are sparse).
    """
    if not vid_path.exists():
        raise SystemExit(f"Video not found: {vid_path}")

    for traj_path in traj_paths:
        if not traj_path.exists():
            raise SystemExit(f"Trajectory not found: {traj_path}")

    # Load masks if provided
    masks = None

    mask_corrections_dir = vid_path.parent / f"{vid_path.stem}_mask_corrections"
    logger.info(f"Mask corrections directory: {mask_corrections_dir}")

    if mask_path is not None:
        if not mask_path.exists():
            raise SystemExit(f"Mask file not found: {mask_path}")
        masks = load_masks(mask_path)
        logger.info(f"Loaded masks from {mask_path}: shape {masks.shape}")

        # Directory for manual mask corrections

    # Open video
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid_path}")

    # Load or extract background
    bg_bgr = load_or_extract_background(vid_path, cap)
    cap.release()

    # Convert BGR to RGB for matplotlib
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

    # Load trajectories
    trajectories = []
    for traj_path in traj_paths:
        traj = load_trajectory(traj_path)
        trajectories.append(traj)
        logger.info(f"Loaded trajectory from {traj_path}: {len(traj)} points")

    # Parse colors
    if colors is not None:
        color_list = [c.strip() for c in colors.split(",")]
    else:
        color_list = ["C1", "C2", "C5"]

    assert "delivery" in str(vid_path)
    frame_idxs = np.array([287, 528, 548, 763, 1088, 1440])
    # Take difference
    traj_lens = np.diff(np.concatenate(([0], frame_idxs)))
    n_overlays = [5, 5, 2, 5, 5, 5]
    out_dir = vid_path.parent

    for ii, (frame_idx, traj_len, n_overlay) in list(enumerate(zip(frame_idxs, traj_lens, n_overlays)))[skip:]:
        # path = out_dir / f"p{ii}.pdf"
        path = out_dir / f"p{ii}.png"

        plot(
            bg_rgb,
            trajectories,
            frame_idx,
            traj_len,
            color_list,
            dpi=700,
            out_path=path,
            vid_path=vid_path,
            masks=masks,
            n_overlay_frames=n_overlay,
            mask_corrections_dir=mask_corrections_dir,
            delta_frames=delta_frames,
            alpha_power=alpha_power,
        )


def plot(
    bg_rgb: np.ndarray,
    trajectories: list[np.ndarray],
    frame_idx: int,
    traj_len: int,
    color_list: List[str],
    dpi: int,
    out_path: pathlib.Path,
    vid_path: Optional[pathlib.Path] = None,
    masks: Optional[np.ndarray] = None,
    n_overlay_frames: int = 8,
    min_overlay_alpha: float = 0.4,
    max_overlay_alpha: float = 0.7,
    arc_length_weight: float = 0.8,
    mask_corrections_dir: Optional[pathlib.Path] = None,
    delta_frames: int = 0,
    alpha_power: float = 2.0,
):
    # Calculate frame range
    start_idx = max(0, frame_idx - traj_len)
    end_idx = frame_idx

    logger.info(f"Drawing trajectories from frame {start_idx} to {end_idx}")

    # Create figure with exact image dimensions
    h, w = bg_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    # Display background
    ax.imshow(bg_rgb)

    # Draw trajectories
    for i, traj in enumerate(trajectories):
        if end_idx >= len(traj):
            logger.warning(f"Trajectory {i} has only {len(traj)} points, frame_idx={frame_idx} is out of range")
            continue

        color = color_list[i % len(color_list)]
        draw_trajectory_with_gradient(ax, traj, start_idx, end_idx, color=color, alpha_power=alpha_power)

    # Compute sample indices with blended time/arc-length parameterization
    logger.debug("n_overlay_frames: {}".format(n_overlay_frames))
    overlay_frame_idxs = sample_frames_by_arc_length2(
        trajectories, start_idx, end_idx, n_overlay_frames, arc_length_weight
    )

    for trajidx, frame_idxs in enumerate(overlay_frame_idxs):
        logger.info("{} frames: {}".format(trajidx, frame_idxs))

    # Overlay objects at multiple points along the trajectory with varying opacity
    # Sample at equal arc-length intervals to avoid clustering when velocity is low
    if vid_path is not None:
        # logger.debug("frame_idxs: {}".format([int(n) for n in overlay_frame_idxs]))

        n_trajectories = len(trajectories)
        for traj_idx, traj in enumerate(trajectories):
            overlay_frame_idxs_this = overlay_frame_idxs[traj_idx]

            # Compute arclengths
            cum_dist = [0.0]
            for i in range(start_idx + 1, end_idx + 1):
                pt_prev = traj[i - 1]
                pt_curr = traj[i]

                if pt_prev is not None and pt_curr is not None:
                    dx = pt_curr[0] - pt_prev[0]
                    dy = pt_curr[1] - pt_prev[1]
                    dist = np.sqrt(dx * dx + dy * dy)
                else:
                    dist = 0.0

                cum_dist.append(cum_dist[-1] + dist)
            cum_dist = np.array(cum_dist)

            # -------------------------------------------
            for i, overlay_idx in enumerate(overlay_frame_idxs_this):
                # Calculate alpha based on arclength position.
                if cum_dist[-1] > 1e-6:
                    arc_length_pos = cum_dist[overlay_idx - start_idx] / cum_dist[-1]
                else:
                    arc_length_pos = 0.0
                alpha = min_overlay_alpha + arc_length_pos * (max_overlay_alpha - min_overlay_alpha)

                if i == len(overlay_frame_idxs_this) - 1:
                    alpha = 1.0

                # Get frame
                frame_rgb = get_video_frame(vid_path, overlay_idx)

                # Get object images (use corrected versions if available, otherwise export)
                if mask_corrections_dir is not None:
                    object_image = get_or_export_object_image(
                        mask_corrections_dir, overlay_idx, traj_idx, masks, frame_rgb, delta_frames
                    )
                else:
                    # Create object images directly from masks
                    frame_mask = masks[overlay_idx]
                    object_id = traj_idx + 1
                    object_mask = frame_mask == object_id
                    if np.any(object_mask):
                        rgba = np.zeros((h, w, 4), dtype=np.uint8)
                        rgba[..., :3] = frame_rgb
                        rgba[..., 3] = (object_mask * 255).astype(np.uint8)
                    object_image = rgba

                # Overlay this object image
                if object_image is None:
                    continue

                img_rgba = object_image.astype(np.float32) / 255.0
                img_rgba[..., 3] *= alpha
                zorder = 4 + i * n_trajectories
                if i == len(overlay_frame_idxs_this) - 1:
                    zorder = zorder + 500
                else:
                    # Put another image above the line.
                    img_rgba2 = img_rgba.copy()
                    img_rgba2[..., 3] = img_rgba[..., 3] * 0.1
                    ax.imshow(img_rgba, zorder=zorder + 100)

                ax.imshow(img_rgba, zorder=zorder)

            #
            # # Scale by alpha.
            # img_rgba[..., 3] *= alpha
            # zorder = 4 + i * n_trajectories
            # if i == len(overlay_frame_idxs) - 1:
            #     zorder = zorder + 500
            # ax.imshow(img_rgba, zorder=zorder)

            # ax.imshow(img_rgb, alpha=alpha, zorder=10 + i * n_trajectories)

        #         overlay_rgba[..., 3] *= alpha
        #
        #         ax.imshow(overlay_rgba, zorder=10 + i * n_trajectories + traj_idx)
        #
        # logger.info(f"Overlaid objects from {len(overlay_frame_idxs)} frames along trajectory")

    # Remove axes and padding
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)

    # Save output as PDF
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    logger.success(f"Saved overlay to: {out_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
