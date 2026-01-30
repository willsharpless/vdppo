import pathlib

import cv2
import cyclopts
import ipdb
import numpy as np
import tqdm
from loguru import logger
from matplotlib.colors import to_rgb

app = cyclopts.App()

@app.default()
def main(
    vid_path: pathlib.Path,
    mask_vid_path: pathlib.Path = None,
):
    """Given a vid and a mask, where the mask is the video but with regions that are blacked out,
    output a video where the blacked region is normal and the rest is blacked out."""
    cap = cv2.VideoCapture(vid_path)
    mcap = cv2.VideoCapture(mask_vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = vid_path.with_name(f"{vid_path.stem}_revmask.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    for frame_idx in tqdm.trange(n_frames):
        ok, frame_bgr = cap.read()
        okm, mask_frame = mcap.read()

        if writer is None:
            w, h = frame_bgr.shape[1], frame_bgr.shape[0]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)

        is_mask = np.all(mask_frame == 0, axis=-1)
        frame_out = frame_bgr.copy()
        frame_out[~is_mask] = 0

        writer.write(frame_out)

    cap.release()
    writer.release()

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
