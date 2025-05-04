import matplotlib.pyplot as plt
import os

def save_reward_plot(rewards, step, CONFIG):
    """
    Save a plot of the rewards over time.

    Args:
    - rewards (list): List of rewards to plot.
    - step (int): The step at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    substep, save_dir = CONFIG.SUB_STEPS, CONFIG.CURR_EXP_PATH
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(range(0, step+substep, substep), rewards, color='tab:blue', linewidth=5, label='Reward')
    plt.title(f"{CONFIG.ENV}-{CONFIG.TASK}, {CONFIG.ALG}-{CONFIG.BELLMAN}: Training Roll-out")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()

def save_reward_plot_RA(rewards, goals, penalties, step, CONFIG):
    """
    Save a plot of the rewards, goals & penalties over time.

    Args:
    - rewards (list): List of rewards to plot (purple).
    - rewards (list): List of goals to plot (blue).
    - rewards (list): List of penalties to plot (red).
    - step (int): The step at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    substep, save_dir = CONFIG.SUB_STEPS, CONFIG.CURR_EXP_PATH
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(range(0, step+substep, substep), rewards, color='tab:purple', linewidth=5, label='Reward')
    if goals:
        plt.plot(range(0, step+substep, substep), goals, color='tab:blue', linewidth=3, label='Goal')
    if penalties:
        plt.plot(range(0, step+substep, substep), penalties, color='tab:red', linewidth=3, label='Penalty')
    plt.title(f"{CONFIG.ENV}-{CONFIG.TASK}, {CONFIG.ALG}-{CONFIG.BELLMAN}: Training Roll-out")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend(loc='lower right')

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()

def plot_rewards(train_buffer, step):
    """
    General Function for saving and plotting the current rollout rewards.
    """
    if train_buffer.CONFIG.BELLMAN == 'normal':
        save_reward_plot(train_buffer.rewards, step, train_buffer.CONFIG)
    
    if train_buffer.CONFIG.BELLMAN in ['R', 'A', 'RA']:
        save_reward_plot_RA(train_buffer.rewards, train_buffer.goals, train_buffer.penalties, step, train_buffer.CONFIG)

    