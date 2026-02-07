import pathlib

import cv2
import cyclopts
import ipdb
import numpy as np
import tqdm
from loguru import logger

app = cyclopts.App()


def extract_background(cap: cv2.VideoCapture, n_frames_to_use: int) -> np.ndarray:
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
            buffer = np.zeros((n_frames_to_use, *frame_bgr.shape), dtype=frame_bgr.dtype)
            logger.debug("dtype: {}".format(frame_bgr.dtype))

        if not ok:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}")

        buffer[ii] = frame_bgr

    # Compute median across time.
    background = np.median(buffer, axis=0).astype(buffer.dtype)
    return background


def _gamma_encode(img_linear_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_linear_0_1
    return np.clip(img_linear_0_1, 0.0, 1.0) ** (1.0 / gamma)


def _gamma_decode(img_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_0_1
    return np.clip(img_0_1, 0.0, 1.0) ** gamma


def _ramp_weight(t: float, mode: str) -> float:
    # t in [0,1]
    t = float(np.clip(t, 0.0, 1.0))
    if mode == "none":
        return 1.0
    if mode == "linear":
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unknown time_ramp mode: {mode!r}")


def add_bloom_multiscale_lin(
    img_lin: np.ndarray, thresh: float, sigmas=(2.5, 7.5, 15.0), intensities=(0.4, 0.35, 0.25)
):
    x = img_lin.astype(np.float32, copy=False)
    highlights = np.clip(x - thresh, 0.0, 1.0)

    glow = np.zeros_like(x, dtype=np.float32)
    for s, k in zip(sigmas, intensities):
        glow += cv2.GaussianBlur(highlights, (0, 0), float(s)) * float(k)

    return np.clip(x + glow, 0.0, 1.0)


@app.default()
def main(
    vid_path: pathlib.Path,
    gamma: float = 1.0,
    frac: float = 1.0,
    skip: int = 1,
    start_frame: int = 0,
    # --- masking controls ---
    luma_thresh: float = 0.15,  # absolute threshold in linear [0,1]
    luma_diff_thresh: float = 0.05,  # threshold on luma difference from bg in linear [0,1]
    strength: float = 0.25,  # scale masked contribution (0..1)
    # Time ramp: make early trails dim, late trails bright
    time_ramp: str = "smoothstep",  # none|linear|ease_in|ease_out|smoothstep
    ramp_min: float = 0.15,  # weight at start
    ramp_max: float = 1.00,  # weight at end
    ramp_pow: float = 3.0,
    # Bloom
    bloom_thresh: float = 0.6,  # 0..1, higher = only brightest pixels bloom
    bloom_sigma: float = 6.0,  # blur radius in pixels
    bloom_intensity: float = 0.0,  # 0 disables
):
    """Create light trails from video with brightness-based masking (no recoloring)."""
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid_path}")

    # Extract or load background
    bg_path = vid_path.with_name(f"{vid_path.stem}__bg.png")
    if not bg_path.exists():
        n_frames_to_use = 100
        bg_u8 = extract_background(cap, n_frames_to_use)
        cv2.imwrite(str(bg_path), bg_u8)
        logger.success(f"Saved background to: {bg_path}")
    else:
        bg_u8 = cv2.imread(str(bg_path))
        logger.info(f"Loaded background from: {bg_path}")
    bg_f32 = bg_u8.astype(np.float32) * (1.0 / 255.0)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = int(round(frac * n_frames))
    n_frames = max(n_frames, 1)
    n_frames = n_frames - start_frame

    # Save a frame of the final original image.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + n_frames - 1)
    ok, frame_bgr = cap.read()
    path = vid_path.with_name(f"{vid_path.stem}__final_frame.png")
    cv2.imwrite(str(path), frame_bgr)
    logger.success("Saved final frame ({}) to: {}".format(start_frame + n_frames - 1, path))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # We accumulate log(prod(1 - x)) as sum(log1p(-x))
    log_prod = None
    count = 0

    for frame_idx in tqdm.trange(n_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            logger.error("Failed to read video, exiting...")
            break

        # Only keep every Nth frame (based on frames read since start)
        if (frame_idx % skip) != 0:
            continue

        # float32 [0,1]
        frame_f32 = frame_bgr.astype(np.float32) * (1.0 / 255.0)

        # Subtract background
        frame_without_bg = cv2.subtract(frame_f32, bg_f32)

        # Work in linear space for physically sensible blending if gamma != 1
        x = _gamma_decode(frame_f32, gamma)
        x_without_bg = _gamma_decode(frame_without_bg, gamma)

        # --- Brightness-based masking ---
        # Simple absolute brightness (luma) mask in linear space.
        luma = x.mean(axis=2, keepdims=True)  # shape (H,W,1)
        luma_diff = x_without_bg.mean(axis=2, keepdims=True)
        is_bright = ((luma > luma_thresh) & (luma_diff > luma_diff_thresh)).astype(np.float32)

        # Apply mask + strength so only bright pixels accumulate
        x = x * is_bright * strength

        # --- time ramp weight ---
        t = frame_idx / max(n_frames - 1, 1)
        s = _ramp_weight(t, time_ramp)
        wgt = float(ramp_min + (ramp_max - ramp_min) * s)
        wgt = wgt**ramp_pow
        logger.debug(f"wgt: {wgt}")

        # Clamp strictly inside [0,1] to keep log1p(-x) well-defined and avoid -inf from exact 1.0
        x64 = np.clip(x.astype(np.float64, copy=False), 0.0, 1.0 - 1e-12)

        if log_prod is None:
            log_prod = np.zeros_like(x64, dtype=np.float64)

        # log_prod += log(1 - x) in a stable way
        log_prod += wgt * np.log1p(-x64)
        count += 1

    cap.release()

    if count == 0:
        raise SystemExit("No frames read.")

    # prod(1 - x) = exp(log_prod); screen = 1 - that
    screen_lin = 1.0 - np.exp(log_prod)

    # Back to display space if desired
    out_lin = np.clip(screen_lin, 0.0, 1.0)

    # Bloom (in linear space)
    if bloom_intensity > 0:
        out_lin = add_bloom_multiscale_lin(out_lin, bloom_thresh, bloom_sigma, bloom_intensity)

    out = _gamma_encode(out_lin, gamma)

    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    out_path = vid_path.with_name(f"{vid_path.stem}__trails.png")

    ok = cv2.imwrite(str(out_path), out_u8)
    if ok:
        logger.success(f"Saved to: {out_path}")
    else:
        raise SystemExit(f"Failed to write output: {out_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
