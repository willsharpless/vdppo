import matplotlib.pyplot as plt
import os

def save_reward_plot(rewards, step, substep, save_dir="./models/checkpoints"):
    """
    Save a plot of the rewards over time.

    Args:
    - rewards (list): List of rewards to plot.
    - step (int): The step at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(range(0, step+substep, substep), rewards)
    plt.title(f"Learned Rewards (step {step})")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()
