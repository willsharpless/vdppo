import pathlib

import cv2
import cyclopts
import ipdb
import numpy as np
import tqdm
from loguru import logger
from matplotlib.colors import to_rgb

app = cyclopts.App()


def _gamma_encode(img_linear_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_linear_0_1
    return np.clip(img_linear_0_1, 0.0, 1.0) ** (1.0 / gamma)


def _gamma_decode(img_0_1: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_0_1
    return np.clip(img_0_1, 0.0, 1.0) ** gamma


def _rgb_to_hsv_opencv(r: float, g: float, b: float):
    # OpenCV expects BGR input for cvtColor
    bgr = np.array([[[b, g, r]]], dtype=np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), hsv[1], hsv[2]


def _recolor_with_mask_bgr(
    frame_bgr: np.ndarray,
    mask01: np.ndarray,  # float32 [H,W] in [0,1]
    target_h: int,
    target_s: float,
    keep_value: bool = True,
) -> np.ndarray:
    """
    Recolor by setting HSV H/S under mask, with soft alpha blending.

    mask01: float alpha [0,1] (soft ok). We apply alpha blend in BGR.
    """
    if mask01.ndim != 2:
        raise ValueError("mask01 must be [H,W]")

    assert frame_bgr.dtype == np.float32
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # logger.info("h min: {}, max: {}".format(hsv[..., 0].min(), hsv[..., 0].max()))
    hsv2 = hsv.copy()

    # Set hue + saturation where mask > 0
    # We do a hard where() then soften via alpha blend in BGR.
    m = mask01 > 0.0
    hsv2[..., 0] = np.where(m, target_h, hsv2[..., 0])
    # hsv2[..., 1] = np.where(m, target_s, hsv2[..., 1])
    if not keep_value:
        # If you want to force a specific brightness later you could set V here.
        pass

    recolored = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    assert recolored.dtype == np.float32

    a = mask01.astype(np.float32)[..., None]
    out = frame_bgr * (1.0 - a) + recolored * a
    return np.clip(out, 0.0, 1.0)


def extract_background(cap: cv2.VideoCapture, n_frames_to_use: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_to_use = min(n_frames, n_frames_to_use)
    frame_idxs = np.round(np.linspace(0, n_frames - 1, n_frames_to_use)).astype(int)
    frame_idxs = np.unique(frame_idxs)

    buffer = None
    for ii, frame_idx in enumerate(tqdm.tqdm(frame_idxs)):
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
    # --- recolor ---
    mask_vid_path: pathlib.Path = None,
    sat: float = 0.95,  # saturation to apply (0..255)
    mask_blur: float = 0.0,  # gaussian sigma to soften mask edges (0 disables)
    mask_gain: float = 1.0,  # multiply mask alpha (e.g. 1.5), clamped to [0,1]
    # --- masking controls ---
    luma_thresh: float = 0.15,  # absolute threshold in linear [0,1]
    luma_diff_thresh: float = 0.05,  # threshold on luma difference from bg in linear [0,1]
    strength: float = 0.25,  # scale masked contribution (0..1)
    morph_open: int = 0,  # pixels for morphological open (0 disables)
    morph_dilate: int = 0,  # pixels for dilation (0 disables)
    # Time ramp: make early trails dim, late trails bright
    time_ramp: str = "smoothstep",  # none|linear|ease_in|ease_out|smoothstep
    ramp_min: float = 0.15,  # weight at start
    ramp_max: float = 1.00,  # weight at end
    ramp_pow: float = 3.0,
    # Bloom
    bloom_thresh: float = 0.6,  # 0..1, higher = only brightest pixels bloom
    bloom_sigma: float = 6.0,  # blur radius in pixels
    bloom_intensity: float = 0.0,  # 0 disables
    #
    new_color: str = "#348ABD"
):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid_path}")

    mcap = None
    if mask_vid_path is not None:
        mcap = cv2.VideoCapture(str(mask_vid_path))
        if not mcap.isOpened():
            raise SystemExit(f"Could not open mask video: {mask_vid_path}")

    # Basic sanity: size should match
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mcap is not None:
        mw = int(mcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        mh = int(mcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (mw, mh):
            logger.warning(f"Mask video size {(mw, mh)} != video size {(w, h)}; will resize mask each frame.")

    bg_path = vid_path.with_name(f"{vid_path.stem}__bg.png")
    if not bg_path.exists():
        n_frames_to_use = 100
        bg_u8 = extract_background(cap, n_frames_to_use)
        cv2.imwrite(bg_path, bg_u8)
    else:
        bg_u8 = cv2.imread(bg_path)
    bg_f32 = bg_u8.astype(np.float32) * (1.0 / 255.0)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = int(round(frac * n_frames))
    n_frames = max(n_frames, 1)
    n_frames = n_frames - start_frame

    # Save a frame of the final original image.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + n_frames - 1)
    ok, frame_bgr = cap.read()
    path = vid_path.with_name(f"{vid_path.stem}__final_frame.png")
    cv2.imwrite(path, frame_bgr)
    logger.success("Saved final frame ({}) to: {}".format(start_frame + n_frames - 1, path))
    # exit(0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if mcap is not None:
        mcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + n_frames - 1)
        mcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Target hue/sat for recolor
    # new_color = "#348ABD"
    r, g, b = to_rgb(new_color)
    target_h, _, _ = _rgb_to_hsv_opencv(r, g, b)
    logger.info("h: {}".format(target_h))
    # target_s = int(np.clip(sat, 0, 255))
    target_s = sat

    # We accumulate log(prod(1 - x)) as sum(log1p(-x))
    log_prod = None
    count = 0

    # Pre-build morphology kernels if requested
    open_kernel = None
    dilate_kernel = None
    if morph_open and morph_open > 0:
        k = int(morph_open)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        logger.info(f"morph_open: {morph_open}")
    if morph_dilate and morph_dilate > 0:
        k = int(morph_dilate)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

    for frame_idx in tqdm.trange(n_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            logger.error("Failed to read video, exiting...")
            break

        okm, mask_frame = mcap.read()
        if not okm:
            logger.error("Failed to read mask, exiting...")
            break

        # # Save intermediate recolored frame for debugging
        # debug_path = vid_path.with_name(f"{vid_path.stem}__frame_{frame_idx:05d}.png")
        # cv2.imwrite(str(debug_path), frame_bgr)
        # ipdb.set_trace()

        # Only keep every Nth frame (based on frames read since start)
        if (frame_idx % skip) != 0:
            continue

        mask_gray = mask_frame[:, :, 0]

        # Convert mask to float alpha [0,1]
        mask01 = mask_gray.astype(np.float32) / 255.0
        if mask_blur and mask_blur > 0:
            mask01 = cv2.GaussianBlur(mask01, (0, 0), float(mask_blur))
        if mask_gain != 1.0:
            mask01 = np.clip(mask01 * float(mask_gain), 0.0, 1.0)

        # float32 [0,1]
        frame_f32_orig = frame_bgr.astype(np.float32) * (1.0 / 255.0)

        # --- Recolor LED using the mask video (in display/BGR space) ---
        frame_f32 = frame_f32_orig
        # frame_f32 = _recolor_with_mask_bgr(
        #     frame_bgr=frame_f32_orig,
        #     mask01=mask01,
        #     target_h=target_h,
        #     target_s=target_s,
        #     keep_value=True,
        # )

        # # Save intermediate recolored frame for debugging
        # debug_path = vid_path.with_name(f"{vid_path.stem}__frame_{frame_idx:05d}.png")
        # cv2.imwrite(str(debug_path), (frame_f32 * 255.0).astype(np.uint8))
        # ipdb.set_trace()

        # Subtract background
        frame_without_bg = cv2.subtract(frame_f32, bg_f32)

        # Work in linear space for physically sensible blending if gamma != 1
        x = _gamma_decode(frame_f32, gamma)
        x_without_bg = _gamma_decode(frame_without_bg, gamma)

        # --- MASK BEFORE SCREEN ---
        # Simple absolute brightness (luma) mask in linear space.
        # Use mean RGB as a cheap luma proxy; could swap to Rec.709 if you prefer.
        luma = x.mean(axis=2, keepdims=True)  # shape (H,W,1)
        luma_diff = x_without_bg.mean(axis=2, keepdims=True)
        is_bright = ((luma > luma_thresh) & (luma_diff > luma_diff_thresh)).astype(np.float32)  # 0/1 mask, float32

        filter_h = 210
        filter_h_width = 34
        filter_hue = [filter_h - filter_h_width, filter_h + filter_h_width]
        if filter_hue is not None:
            hsv = cv2.cvtColor(frame_f32_orig, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # shape (H,W)
            hue_mask = ((hue >= filter_hue[0]) & (hue <= filter_hue[1])).astype(np.float32)
            is_bright = is_bright * hue_mask[:, :, None]

        # Optional morphology to remove speckles / thicken trails
        if open_kernel is not None or dilate_kernel is not None:
            is_bright_u8 = (is_bright[:, :, 0] * 255).astype(np.uint8)
            if open_kernel is not None:
                is_bright_u8 = cv2.morphologyEx(is_bright_u8, cv2.MORPH_OPEN, open_kernel)
            if dilate_kernel is not None:
                is_bright_u8 = cv2.dilate(is_bright_u8, dilate_kernel, iterations=1)
            is_bright = (is_bright_u8.astype(np.float32) / 255.0)[:, :, None]

        if filter_hue is not None:
            hsv = cv2.cvtColor(frame_f32_orig, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # shape (H,W)
            hue_mask = ((hue >= filter_hue[0]) & (hue <= filter_hue[1])).astype(np.float32)
            is_bright = is_bright * hue_mask[:, :, None]

        # Optional morphology to remove speckles / thicken trails
        if open_kernel is not None:
            is_bright_u8 = (is_bright[:, :, 0] * 255).astype(np.uint8)
            is_bright_u8 = cv2.morphologyEx(is_bright_u8, cv2.MORPH_OPEN, open_kernel)
            is_bright = (is_bright_u8.astype(np.float32) / 255.0)[:, :, None]

        # (Very weakly) include the background by Gaussian blur.
        ksize = 31
        is_bright_blur = cv2.GaussianBlur(is_bright, (ksize, ksize), 0)
        is_bright = np.maximum(0.6 * is_bright_blur[:, :, None], is_bright)

        # # Visualize the mask for each frame. Black if not in is_bright. Use the image if is_bright.
        # viz_img = frame_f32 * is_bright
        # debug_path = vid_path.parent / f"dbg/{vid_path.stem}__mask_viz_{frame_idx:05d}.png"
        # debug_path.parent.mkdir(exist_ok=True)
        # cv2.imwrite(str(debug_path), (np.clip(viz_img, 0.0, 1.0) * 255.0).astype(np.uint8))

        # # Use luma + the mask to define the final mask, recolor frame_bgr.
        # recolor_mask = is_bright.squeeze(-1) * mask01

        # # --- Recolor LED using the mask video (in display/BGR space) ---
        # frame_bgr = _recolor_with_mask_bgr(
        #     frame_bgr=frame_bgr,
        #     mask01=recolor_mask,
        #     target_h=target_h,
        #     target_s=target_s,
        #     keep_value=True,
        # )
        # frame = frame_bgr.astype(np.float32) * (1.0 / 255.0)
        # x = _gamma_decode(frame, gamma)

        # # Save intermediate recolored frame for debugging
        # debug_path = vid_path.with_name(f"{vid_path.stem}__frame_{frame_idx:05d}.png")
        # cv2.imwrite(str(debug_path), (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))
        # ipdb.set_trace()

        # Apply mask + strength so only LEDs (and maybe immediate glow) accumulate
        x = x * is_bright * strength

        # --- time ramp weight ---
        t = frame_idx / (n_frames - 1)
        s = _ramp_weight(t, time_ramp)
        wgt = float(ramp_min + (ramp_max - ramp_min) * s)
        wgt = wgt**ramp_pow
        logger.debug(f"wgt: {wgt}")

        # Clamp strictly inside [0,1] to keep log1p(-x) well-defined and avoid -inf from exact 1.0
        # (If you *want* exact 1.0 to slam to white, you can allow it; but then log becomes -inf.)
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
