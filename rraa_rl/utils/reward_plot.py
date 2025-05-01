import matplotlib.pyplot as plt
import os

def save_reward_plot(rewards, timestep, save_dir="./models/checkpoints"):
    """
    Save a plot of the rewards over time.

    Args:
    - rewards (list): List of rewards to plot.
    - timestep (int): The timestep at which this plot is saved.
    - save_dir (str): The directory where the plot will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.plot(rewards)
    plt.title(f"Rewards over time (timestep {timestep})")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    # Save the plot
    plot_path = os.path.join(save_dir, f"model_rewards.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot at {plot_path}")
