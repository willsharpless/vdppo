import ipdb
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Grid size
    n_rows, n_cols = 5, 6

    fig, ax = plt.subplots()
    ax: plt.Axes

    # Dummy gridworld values (optional)
    grid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)

    # Show grid with cell centers at integers
    ax.imshow(grid, origin="lower", extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5))

    # Integer ticks at cell centers
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    # Grid lines at half-integers
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

    ax.grid(False, which="major")
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig("test.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
