import matplotlib.pyplot as plt
import os

def save_reward_plot(rewards, step, cfg):
    """
    Save a plot of the rewards over time.

    Args:
    - rewards (list): List of rewards to plot.
    - step (int): The step at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    substep, save_dir = cfg.SUB_STEPS, cfg.CURR_EXP_PATH
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(range(0, step+substep, substep), rewards)
    plt.title(f"{cfg.ENV}_{cfg.TASK}-{cfg.ALG}-{cfg.BELLMAN}: Training Rewards (single roll-out)")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()

def save_reward_plot_RA(rewards, goals, penalties, step, cfg):
    """
    Save a plot of the rewards, goals & penalties over time.

    Args:
    - rewards (list): List of rewards to plot (purple).
    - rewards (list): List of goals to plot (blue).
    - rewards (list): List of penalties to plot (red).
    - step (int): The step at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    substep, save_dir = cfg.SUB_STEPS, cfg.CURR_EXP_PATH
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(range(0, step+substep, substep), rewards, c='p', linewidth=2)
    if goals:
        plt.plot(range(0, step+substep, substep), goals, c='b')
    if penalties:
        plt.plot(range(0, step+substep, substep), penalties, c='r')
    plt.title(f"{cfg.ENV}_{cfg.TASK}-{cfg.ALG}-{cfg.BELLMAN}: Training Rewards (single roll-out)")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend(['R', 'l', 'g'])

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()

def save_rewards(train_buffer):
    """
    General Function for saving and plotting the current rollout rewards.
    """
    #TODO call correct save_reward_plot fn & save rewards/goals/penalties
    return 