import pickle
from collections import defaultdict

import gymnasium
import imageio
import ipdb
import numpy as np
import tqdm
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from rraa_rl.envs.manipspace.data.button_plan import ButtonPlanOracle
from rraa_rl.envs.manipspace.data.cube_plan import CubePlanOracle
from rraa_rl.envs.manipspace.data.drawer_plan import DrawerPlanOracle
from rraa_rl.envs.manipspace.data.window_plan import WindowPlanOracle
from rraa_rl.envs.manipspace.scene_env import SceneEnv


def main(noise: float = 0.0, noise_smoothing: float = 0.5):
    env = SceneEnv(env_type=None, mode="data_collection")
    env_name = "scene-v0"
    max_episode_steps = 750

    # env = gymnasium.make(
    #     env_name,
    #     terminate_at_goal=False,
    #     mode="data_collection",
    #     max_episode_steps=max_episode_steps,
    # )
    env_unwrapped: SceneEnv = env.unwrapped

    def cube_in_drawer():
        return env_unwrapped._is_in_drawer(env_unwrapped._data.joint("object_joint_0").qpos[:3])

    has_button_states = hasattr(env_unwrapped, "_cur_button_states")

    agents = {
        "cube": CubePlanOracle(env=env, noise=noise, noise_smoothing=noise_smoothing),
        # "button": ButtonPlanOracle(env=env, noise=noise, noise_smoothing=noise_smoothing),
        "drawer": DrawerPlanOracle(env=env, noise=noise, noise_smoothing=noise_smoothing),
        # "window": WindowPlanOracle(env=env, noise=noise, noise_smoothing=noise_smoothing),
    }

    p_stack = 0.5
    rng = np.random.default_rng(seed=45)
    # agent_choices = list(agents.keys())
    # agent_choices = ["cube"]

    agent_choices = ["drawer", "cube", "drawer"]
    agent_labels_to_save = {
        "drawer": ["approach", "grasp_start", "grasp_end", "move", "release"],
        "cube": ["pick_start", "pick_end", "place", "place_start", "place_end"],
    }

    n_episodes = 100

    state_list: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for ii in tqdm.trange(n_episodes):
        ob, info = env.reset(seed=1234 + ii)

        # Unlock the drawer at the start.
        env_unwrapped._cur_button_states[0] = 1
        env_unwrapped._apply_button_states()
        print("button_states: {}".format(env_unwrapped._cur_button_states))

        # env_unwrapped._data.joint("drawer_slide").qpos[0] = 0.0
        env_unwrapped.set_state(env_unwrapped._data.qpos, env_unwrapped._data.qvel, env_unwrapped._cur_button_states)

        force = {"target_task": "drawer", "force_drawer_open": "1"}
        agent_ob, agent_info = env_unwrapped.set_new_target(p_stack=p_stack, force=force)

        state_list_this_ep: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

        # logger.info("button states: {}".format(env_unwrapped._cur_button_states))

        idx = 0
        # target_task = agent_choices[idx % len(agent_choices)]
        # agent = agents[target_task]
        #
        target_task = agent_info["privileged/target_task"]
        logger.info("Starting with target task: {}", target_task)
        agent = agents[target_task]
        agent.reset(agent_ob, agent_info)

        done = False
        step = 0
        ep_qpos = []
        frames = [env.render()]

        while not done:
            # Get an action from the oracle.
            action, label = agent.select_action(ob, info)
            action = np.array(action)

            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = truncated
            # done = terminated or truncated
            # if terminated:
            #     print("terminated at step {}".format(step))
            # if truncated:
            #     print("truncated at step {}".format(step))

            if agent.done:
                idx += 1

                if idx < len(agent_choices):
                    target_task = agent_choices[idx % len(agent_choices)]

                    if target_task == "drawer":
                        force = {"target_task": "drawer"}
                    elif target_task == "cube":
                        force = {"target_task": "cube", "put_in_drawer": "1"}
                    else:
                        force = {}

                    # Set a new task when the current task is done.
                    agent_ob, agent_info = env_unwrapped.set_new_target(p_stack=p_stack, force=force)

                    target_task = agent_info["privileged/target_task"]
                    # target_task = agent_choices[rng.choice(len(agent_choices))]
                    logger.info("Switching to target task: {}", target_task)
                    agent = agents[target_task]
                    agent.reset(agent_ob, agent_info)

            ep_qpos.append(info["prev_qpos"])
            frame = env.render()

            key = (target_task, label)
            if label in agent_labels_to_save[target_task]:
                qpos = env_unwrapped._data.qpos.copy()
                qvel = env_unwrapped._data.qvel.copy()
                state_list_this_ep[key].append((qpos, qvel))

            # Add the label to the frame.
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            if cube_in_drawer():
                draw.text((5, 5), f"{label} | in_drawer", font=font, fill=(0, 255, 0))
            else:
                draw.text((5, 5), label, font=font, fill=(255, 0, 0))

            frame = np.array(im)

            frames.append(frame)
            step += 1

            if idx == len(agent_choices):
                done = True

        # # Check if the cube is in the drawer. If so, add a text saying "success".
        # cube_in_drawer = env_unwrapped._is_in_drawer(env_unwrapped._data.joint("object_joint_0").qpos[:3])
        # if cube_in_drawer:
        #     # Add text to final frame.
        #     im = Image.fromarray(frames[-1])
        #     draw = ImageDraw.Draw(im)
        #     font = ImageFont.load_default()
        #     draw.text((200, 200), "SUCCESS", font=font, fill=(0, 255, 0))
        #     frames[-1] = np.array(im)
        #     logger.success("Cube is in the drawer!")

        # Save only if the cube is in the drawer.
        if cube_in_drawer():
            video_output_path = f"try_collect_{ii}.mp4"
            imageio.mimwrite(video_output_path, frames, fps=30)
            logger.success("Wrote video to {}", video_output_path)

            # Add to overall state list.
            for key, states in state_list_this_ep.items():
                state_list[key].extend(states)

    # Summarize and save the collected states.
    for key, states in state_list.items():
        logger.info("Task: {}, Label: {}, Collected {} states", key[0], key[1], len(states))

    pkl_output_path = "collected_states.pkl"
    with open(pkl_output_path, "wb") as f:
        pickle.dump(state_list, f)

    logger.success("Wrote collected states to {}", pkl_output_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
