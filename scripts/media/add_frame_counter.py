import pathlib
from typing import Optional

import cv2
import cyclopts
import tqdm
from loguru import logger

app = cyclopts.App()


@app.default()
def main(
    vid_path: pathlib.Path,
    out_path: Optional[pathlib.Path] = None,
    font_scale: float = 1.0,
    thickness: int = 2,
    margin: int = 10,
    text_color: str = "white",  # white, black, yellow, green, red
    bg_color: Optional[str] = "black",  # background box color, None for no background
    show_timestamp: bool = False,
):
    """
    Add a frame counter to the top left of a video and export it.
    """
    if not vid_path.exists():
        raise SystemExit(f"Video not found: {vid_path}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if out_path is None:
        out_path = vid_path.with_name(f"{vid_path.stem}_frames{vid_path.suffix}")

    # Color mapping
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
    }
    txt_color_bgr = color_map.get(text_color.lower(), (255, 255, 255))
    bg_color_bgr = color_map.get(bg_color.lower(), (0, 0, 0)) if bg_color else None

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not open writer: {out_path}")

    logger.info(f"Processing {n_frames} frames at {w}x{h}, {fps:.2f} fps")

    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in tqdm.trange(n_frames, desc="Adding frame counter"):
        ok, frame = cap.read()
        if not ok:
            break

        # Build text
        if show_timestamp:
            timestamp_sec = frame_idx / fps
            minutes = int(timestamp_sec // 60)
            seconds = timestamp_sec % 60
            text = f"Frame {frame_idx} ({minutes}:{seconds:05.2f})"
        else:
            text = f"Frame {frame_idx}"

        # Get text size for background box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position (top-left with margin)
        x = margin
        y = margin + text_h

        # Draw background box if requested
        if bg_color_bgr is not None:
            padding = 5
            cv2.rectangle(
                frame,
                (x - padding, y - text_h - padding),
                (x + text_w + padding, y + baseline + padding),
                bg_color_bgr,
                -1,  # filled
            )

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, txt_color_bgr, thickness, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    logger.success(f"Saved to: {out_path}")


if __name__ == "__main__":
    app()
