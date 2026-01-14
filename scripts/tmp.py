import numpy as np


def main():
    rng = np.random.default_rng(seed=12345)

    num = 10_000_000
    dim = 2

    a = rng.uniform(low=-1, high=1, size=(num, dim))
    b = rng.uniform(low=-1, high=1, size=(num, dim))

    dist = np.linalg.norm(a - b, axis=-1)
    mean_dist = np.mean(dist)
    std_dist = np.std(dist)
    print("Mean: {}, Std: {}".format(mean_dist, std_dist))


if __name__ == "__main__":
    main()
