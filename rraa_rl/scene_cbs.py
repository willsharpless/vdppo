import pathlib

import imageio
import imageio.v2 as imageio
import mujoco
import numpy as np
import tqdm
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from rraa_rl.envs.scene import SceneBase


def animate_scene_traj(
    anim_path: pathlib.Path,
    env: SceneBase,
    T_minstate: SceneBase.MinState,
    T_preds: dict[str, np.ndarray],
    T_labels: list[str] | None = None,
    fps: int | None = None,
    width: int = 1280,
):
    height = int(round(9 / 16 * width))
    T = len(T_minstate.qpos)

    if fps is None:
        dt = env.control_timestep
        fps = int(round(1 / dt))

    m = env.mj_model
    m.vis.global_.offwidth = width
    m.vis.global_.offheight = height
    renderer = mujoco.Renderer(model=m, height=height, width=width)
    camera = "front"

    d = mujoco.MjData(m)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    vid_frames = []
    pbar = tqdm.trange(T, leave=False)
    for kk in pbar:
        d.qpos[:] = T_minstate.qpos[kk]
        d.qvel[:] = T_minstate.qvel[kk]
        mujoco.mj_forward(m, d)

        renderer.update_scene(data=d, camera=camera)
        img = renderer.render()

        # Convert to PIL Image for drawing text
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Draw timestep on top left
        timestep_text = f"t={kk: 3}"
        if T_labels is not None:
            timestep_text += f"\n{T_labels[kk]}"
        draw.text((10, 10), timestep_text, fill=(255, 255, 255), font=font)

        # Draw predicate status on top right
        y_offset = 10
        for pred_name, pred_vals in T_preds.items():
            status = "✓" if pred_vals[kk] > 0.5 else "✗"
            color = (0, 255, 0) if pred_vals[kk] > 0.5 else (255, 0, 0)
            pred_text = f"{pred_name}: {status}"
            # Get text bounding box to right-align
            bbox = draw.textbbox((0, 0), pred_text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((width - text_width - 10, y_offset), pred_text, fill=color, font=font)
            y_offset += 30

        vid_frames.append(np.array(pil_img))

    imageio.mimwrite(anim_path, vid_frames, fps=fps)
    logger.info(f"Saved video to {anim_path}")
