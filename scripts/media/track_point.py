import pathlib
from typing import Optional, Tuple

import cv2
import cyclopts
import numpy as np
import tqdm
from loguru import logger

app = cyclopts.App()


class _UserAbort(Exception):
    """Raised when user wants to abort tracking entirely."""

    pass


def _prompt_user_click(
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    window_name: str = "Click to select point",
    allow_skip: bool = True,
) -> Optional[Tuple[float, float]]:
    """
    Display frame and wait for user to click a point.
    Returns (cx, cy) if user clicks, None if user presses ESC to skip (only if allow_skip=True).
    Raises _UserAbort if user presses 'q' to quit entirely.
    """
    clicked_point = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = (float(x), float(y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, on_mouse)

    timestamp_sec = frame_idx / fps
    minutes = int(timestamp_sec // 60)
    seconds = timestamp_sec % 60

    if allow_skip:
        instructions = "Click to track | ESC=skip | Q=quit"
    else:
        instructions = "Click to select initial point | Q=quit"

    display = frame_bgr.copy()
    cv2.putText(
        display,
        f"Frame {frame_idx} ({minutes}:{seconds:05.2f}) | {instructions}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27 and allow_skip:  # ESC
            cv2.destroyWindow(window_name)
            return None
        if key == ord("q") or key == ord("Q"):
            cv2.destroyWindow(window_name)
            raise _UserAbort()
        if clicked_point[0] is not None:
            cv2.destroyWindow(window_name)
            return clicked_point[0]


def _show_frame_and_wait(
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    cx: float,
    cy: float,
    window_name: str = "Tracking",
) -> Tuple[str, Optional[Tuple[float, float]]]:
    """
    Show frame with tracked point and wait for user input.

    Returns:
        (action, point) where action is one of:
        - "accept": user accepted the point (SPACE/ENTER)
        - "override": user clicked to override, point contains new coordinates
        - "skip": user pressed ESC to skip this frame
        - "quit": user pressed Q to quit entirely
    """
    clicked_point = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = (float(x), float(y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    timestamp_sec = frame_idx / fps
    minutes = int(timestamp_sec // 60)
    seconds = timestamp_sec % 60

    display = frame_bgr.copy()

    # Draw the tracked point
    cv2.drawMarker(
        display,
        (int(round(cx)), int(round(cy))),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    # Draw instructions
    instructions = "SPACE=accept | Click=override | ESC=skip | Q=quit"
    cv2.putText(
        display,
        f"Frame {frame_idx} ({minutes}:{seconds:05.2f}) | {instructions}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 32 or key == 13:  # SPACE or ENTER
            return ("accept", None)
        if key == 27:  # ESC
            return ("skip", None)
        if key == ord("q") or key == ord("Q"):
            return ("quit", None)
        if clicked_point[0] is not None:
            return ("override", clicked_point[0])

        # left arrow: go to previous frame
        if key == 81:
            return ("previous", None)


@app.default()
def track(
    vid_path: pathlib.Path,
    out_path: Optional[pathlib.Path] = None,
    # Tracking params (Lucas-Kanade)
    win_size: int = 21,
    max_level: int = 3,
    lk_iters: int = 30,
    lk_eps: float = 0.01,
    max_track_err: float = 30.0,
    interactive: bool = True,
):
    """Track a point through video and save the trajectory.

    If interactive=True (default), shows each frame and waits for user confirmation.
    Press SPACE/ENTER to accept, click to override, ESC to skip, Q to quit.
    """
    if not vid_path.exists():
        raise SystemExit(f"Input not found: {vid_path}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_path is None:
        out_path = vid_path.with_name(f"{vid_path.stem}_trajectory.npy")

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Failed to read first frame.")

    # Have user click to select initial point
    logger.info("Click on the point to track in the first frame...")
    try:
        init_point = _prompt_user_click(first, frame_idx=0, fps=fps, allow_skip=False)
    except _UserAbort:
        cap.release()
        raise SystemExit("User aborted before selecting initial point.")

    cx0, cy0 = init_point
    p = np.array([[[cx0, cy0]]], dtype=np.float32)

    logger.info(f"Initial point: ({cx0:.2f}, {cy0:.2f})")
    logger.info(f"Output trajectory: {out_path}")

    # Collect tracked points: list of (cx, cy) or None if tracking lost
    tracked_points = [(cx0, cy0)]

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    lk_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(lk_iters),
        float(lk_eps),
    )

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    remaining = max(0, n_frames - 1) if n_frames > 0 else None

    user_aborted = False
    it = range(remaining) if remaining is not None else iter(int, 1)
    frame_idx = 0

    try:
        for _ in tqdm.tqdm(it, total=remaining if remaining is not None else None):
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                # Tracking lost - prompt user to re-initialize
                logger.warning(f"Tracking lost at frame {frame_idx}! Prompting user to re-initialize...")
                clicked = _prompt_user_click(frame, frame_idx, fps)
                if clicked is not None:
                    cx, cy = clicked
                    p = np.array([[[cx, cy]]], dtype=np.float32)
                    logger.info(f"Re-initialized tracking at ({cx:.2f}, {cy:.2f})")
                    tracked_points.append((cx, cy))
                else:
                    logger.info("User skipped frame")
                    tracked_points.append(None)
            else:
                p = p_next
                cx, cy = float(p[0, 0, 0]), float(p[0, 0, 1])

                if interactive:
                    # Show frame and wait for user input
                    action, override_pt = _show_frame_and_wait(frame, frame_idx, fps, cx, cy)

                    if action == "quit":
                        raise _UserAbort()
                    elif action == "skip":
                        logger.info(f"User skipped frame {frame_idx}")
                        tracked_points.append(None)
                    elif action == "override":
                        cx, cy = override_pt
                        p = np.array([[[cx, cy]]], dtype=np.float32)
                        logger.info(f"User overrode point at frame {frame_idx}: ({cx:.2f}, {cy:.2f})")
                        tracked_points.append((cx, cy))
                    elif action == "previous":
                        if frame_idx >= 2:
                            # Go back one frame
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 2)
                            tracked_points.pop()  # remove last point
                            frame_idx -= 2
                            prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if frame_idx == 0 else None
                            for i in range(frame_idx):
                                cap.read()  # advance to the correct frame
                                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            p = np.array([[[tracked_points[-1][0], tracked_points[-1][1]]]], dtype=np.float32)
                        else:
                            logger.warning("Already at first frame, cannot go back further.")
                            tracked_points.append((cx, cy))
                    else:  # accept
                        tracked_points.append((cx, cy))
                else:
                    tracked_points.append((cx, cy))

            prev_gray = gray
    except _UserAbort:
        logger.warning("User aborted tracking.")
        user_aborted = True
        # Fill remaining frames with None
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            tracked_points.append(None)

    cap.release()
    cv2.destroyAllWindows()

    # Save tracked points
    np.save(out_path, np.array(tracked_points, dtype=object), allow_pickle=True)

    n_valid = sum(1 for p in tracked_points if p is not None)
    n_total = len(tracked_points)
    if user_aborted:
        logger.warning(f"Saved trajectory to: {out_path} ({n_valid}/{n_total} valid points, user aborted)")
    else:
        logger.success(f"Saved trajectory to: {out_path} ({n_valid}/{n_total} valid points)")


if __name__ == "__main__":
    app()
