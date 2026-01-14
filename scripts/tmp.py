import functools as ft

import ipdb
import jax
import numpy as np
from flax import struct


class Static:
    def __init__(self, data):
        self.data = data


@ft.partial(struct.dataclass, frozen=False)
class A:
    x: np.ndarray
    y: Static = struct.field(pytree_node=False)

    @jax.jit
    def f(self, inp):
        print("x: {}, y: {}, inp: {}".format(self.x, self.y, inp))
        return self.x + inp

    def g(self, y_new):
        self.y.data = y_new
        # self_new = self.replace(y=y_new)
        # return self_new


def main():
    s = Static(np.ones(3))
    a = A(np.zeros(3), s)

    for ii in range(3):
        a.f(10 + ii)

    print(a.y)
    a.g(np.ones(4))
    print(a.y)

    for ii in range(3):
        a.f(10 + ii)

    # rng = np.random.default_rng(seed=12345)
    #
    # num = 10_000_000
    # dim = 2
    #
    # a = rng.uniform(low=-1, high=1, size=(num, dim))
    # b = rng.uniform(low=-1, high=1, size=(num, dim))
    #
    # dist = np.linalg.norm(a - b, axis=-1)
    # mean_dist = np.mean(dist)
    # std_dist = np.std(dist)
    # print("Mean: {}, Std: {}".format(mean_dist, std_dist))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
