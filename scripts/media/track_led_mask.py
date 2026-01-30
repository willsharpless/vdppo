import pathlib
from typing import Optional, Tuple

import cv2
import cyclopts
import numpy as np
import tqdm
from loguru import logger

app = cyclopts.App()


@app.command()
def first_frame(vid_path: pathlib.Path):
    # Exports the first frame of the video as an image.
    cap = cv2.VideoCapture(vid_path)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first frame.")

    out_path = vid_path.with_name(vid_path.stem + "_first_frame.png")
    cv2.imwrite(out_path, frame_bgr)
    logger.success(f"First frame saved to {out_path}")


def _make_soft_circular_mask(h: int, w: int, cx: float, cy: float, radius: int, feather: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if feather <= 0:
        return (dist <= radius).astype(np.float32)

    t = (dist - radius) / feather
    m = np.clip(1.0 - t, 0.0, 1.0)
    m[dist <= radius] = 1.0
    m[dist >= radius + feather] = 0.0
    return m.astype(np.float32)


def _centroid_from_mask(mask_u8: np.ndarray) -> Tuple[float, float]:
    """
    mask_u8: uint8 mask, nonzero means selected region.
    Returns (cx, cy) in pixel coordinates.
    """
    ys, xs = np.nonzero(mask_u8)
    if len(xs) == 0:
        raise SystemExit("Provided init mask contains no white pixels.")
    cx = float(xs.mean())
    cy = float(ys.mean())
    return cx, cy


def _load_init_mask(mask_path: pathlib.Path, target_wh: Tuple[int, int]) -> np.ndarray:
    """
    Loads a user-provided mask image. Converts to single-channel uint8 in {0,255},
    resized to the video frame size if needed.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"Could not read init mask image: {mask_path}")

    w, h = target_wh
    if m.shape[1] != w or m.shape[0] != h:
        # Nearest neighbor keeps mask crisp.
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarize robustly
    m = (m >= 128).astype(np.uint8) * 255
    return m


@app.command()
def track(
    vid_path: pathlib.Path,
    init_mask_path: pathlib.Path,  # <-- user-provided painted mask for frame 0
    out_mask: Optional[pathlib.Path] = None,
    # Output mask shape/feel
    radius: int = 10,
    feather: float = 6.0,
    # Optional gating: only emit mask where the current frame is bright (prevents tinting background)
    v_thresh: int = 160,
    # Tracking params (Lucas–Kanade)
    win_size: int = 21,
    max_level: int = 3,
    lk_iters: int = 30,
    lk_eps: float = 0.01,
    max_track_err: float = 30.0,
    # If tracking is lost, write black masks for remaining frames
    preview: bool = False,
):
    if not vid_path.exists():
        raise SystemExit(f"Input not found: {vid_path}")
    if not init_mask_path.exists():
        raise SystemExit(f"Init mask not found: {init_mask_path}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_mask is None:
        out_mask = vid_path.with_name(f"{vid_path.stem}_led_mask.mp4")

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Failed to read first frame.")

    # Load and binarize the user mask; compute initial centroid
    init_mask = _load_init_mask(init_mask_path, target_wh=(w, h))
    cx0, cy0 = _centroid_from_mask(init_mask)
    p = np.array([[[cx0, cy0]]], dtype=np.float32)

    logger.info(f"Init centroid from mask: ({cx0:.2f}, {cy0:.2f})")
    logger.info(f"Output mask video: {out_mask}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mask), fourcc, fps, (w, h), isColor=False)
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not open writer: {out_mask}")

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    lk_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(lk_iters),
        float(lk_eps),
    )

    def build_mask(frame_bgr: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """
        Build an 8-bit mask for this frame. Soft circular mask around (cx,cy),
        optionally gated by brightness (HSV V channel).
        """
        soft = _make_soft_circular_mask(h, w, cx, cy, radius=radius, feather=feather)  # float [0,1]

        if v_thresh > 0:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            bright = (v >= v_thresh).astype(np.float32)
            soft = soft * bright

        return (np.clip(soft, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # First frame output:
    # You have two reasonable choices:
    # A) Use your painted init mask directly (crisp)
    # B) Use the soft circle around its centroid (consistent with later frames)
    # We'll do B by default for consistency.
    mask0 = build_mask(first, cx0, cy0)
    writer.write(mask0)

    if preview:
        cv2.namedWindow("mask preview", cv2.WINDOW_NORMAL)
        vis0 = cv2.cvtColor(mask0, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(
            vis0,
            (int(round(cx0)), int(round(cy0))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.imshow("mask preview", vis0)
        cv2.waitKey(1)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    remaining = max(0, n_frames - 1) if n_frames > 0 else None

    tracked = True
    it = range(remaining) if remaining is not None else iter(int, 1)

    for _ in tqdm.tqdm(it, total=remaining if remaining is not None else None):
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracked:
            p_next, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                p,
                None,
                winSize=(int(win_size), int(win_size)),
                maxLevel=int(max_level),
                criteria=lk_criteria,
                minEigThreshold=1e-4,
            )

            good = (p_next is not None) and (st is not None) and (st[0, 0] == 1)
            e = float(err[0, 0]) if (err is not None and good) else 0.0

            if (not good) or (err is not None and e > max_track_err):
                tracked = False
                writer.write(np.zeros((h, w), dtype=np.uint8))
            else:
                p = p_next
                cx, cy = float(p[0, 0, 0]), float(p[0, 0, 1])
                m = build_mask(frame, cx, cy)
                writer.write(m)

                if preview:
                    vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                    cv2.drawMarker(
                        vis,
                        (int(round(cx)), int(round(cy))),
                        (0, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                    )
                    cv2.imshow("mask preview", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        logger.warning("Preview aborted with ESC.")
                        break
        else:
            writer.write(np.zeros((h, w), dtype=np.uint8))

        prev_gray = gray

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()

    if tracked:
        logger.success(f"Saved LED mask to: {out_mask}")
    else:
        logger.warning(f"Saved LED mask to: {out_mask} (tracking was lost; mask is black afterward)")


if __name__ == "__main__":
    app()
