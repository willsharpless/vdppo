import argparse
import cv2
import numpy as np
from pathlib import Path
import tqdm
import pathlib

def undistort_video(
    input_path: str,
    output_path: str,
    K: np.ndarray,
    dist: np.ndarray,
    alpha: float = 0.0,
    use_remap: bool = True,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    # Choose codec based on output extension (simple heuristic)
    out_ext = Path(output_path).suffix.lower()
    if out_ext in [".mp4", ".m4v"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Compute new camera matrix.
    # alpha=0 -> crop to valid pixels (no black border)
    # alpha=1 -> keep all pixels, may include black border
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    x, y, rw, rh = roi  # region of interest for alpha<1

    # Precompute remap tables (fastest for video).
    map1 = map2 = None
    if use_remap:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, R=None, newCameraMatrix=newK, size=(w, h), m1type=cv2.CV_16SC2
        )

    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm.trange(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        if use_remap:
            und = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            und = cv2.undistort(frame, K, dist, None, newK)

        # Optional crop to ROI to remove black borders when alpha=0
        # If you want to keep full frame size (same as input), leave this off.
        # If you DO crop, you must change the VideoWriter size too.
        # und = und[y:y+rh, x:x+rw]

        writer.write(und)

    cap.release()
    writer.release()
    print(f"Done. Wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Undistort a video using OpenCV intrinsics/distortion.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0=crop to valid pixels (less black border), 1=keep all pixels (more black border)")
    parser.add_argument("--no-remap", action="store_true",
                        help="Use cv2.undistort per-frame instead of precomputed remap")
    args = parser.parse_args()

    # ---- YOUR CALIBRATION (from your printout) ----
    K = np.array([
        [900.51946647,   0.0,         952.58602189],
        [0.0,            899.52068041, 529.15564578],
        [0.0,            0.0,         1.0],
    ], dtype=np.float64)

    dist = np.array([-0.00151487, -0.0050831, -0.00046027, -0.00394099, 0.00237756],
                    dtype=np.float64)

    path_input = pathlib.Path(args.input)
    path_output = path_input.with_name(path_input.stem + "_undistort.mp4")

    undistort_video(
        input_path=path_input,
        output_path=path_output,
        K=K,
        dist=dist,
        alpha=args.alpha,
        use_remap=not args.no_remap
    )


if __name__ == "__main__":
    main()
