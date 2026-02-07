import cv2
import numpy as np
import sys

# ---------- Config ----------
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
WINDOW_NAME = "Frame-by-frame viewer"
# ----------------------------

clicked_points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_idx = userdata["frame_idx"]
        print(f"{frame_idx} ({x}, {y})")
        clicked_points.append((x, y))


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of video (or failed to read frame).")
            break

        # Draw frame counter (top-left)
        label = f"Frame: {frame_idx + 1} / {total_frames}"
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        userdata = {"frame_idx": frame_idx}
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, userdata)

        cv2.imshow(WINDOW_NAME, frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 32:  # SPACE
                frame_idx += 1
                break

    cap.release()
    cv2.destroyAllWindows()

    global clicked_points
    clicked_points = np.array(clicked_points)
    print()
    print(repr(clicked_points))


if __name__ == "__main__":
    main()
