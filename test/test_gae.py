import ipdb
import numpy as np

from rraa_rl.gae import BellmanMax, gae_generalized


def main():
    gamma = 0.9
    lam = 0.95
    T = 3
    T_V = np.array([0.0, 0.0, 0.0])
    T_V_next = np.array([0.0, 0.0, 0.0])
    T_term = np.zeros(T, dtype=bool)

    T_r = np.array([0.0, 0.0, 1.0])

    assert T_V.shape == T_V_next.shape == T_term.shape == (T,)

    bellman_update = BellmanMax(T_r)
    T_Q_avg = gae_generalized(T_V, T_V_next, T_term, bellman_update, gamma, lam)

    T_Q_avg_expected = np.array([gamma**2 * lam**2 / (1 + lam + lam**2), gamma * lam / (1 + lam), 1.0])
    np.testing.assert_allclose(T_Q_avg, T_Q_avg_expected, rtol=1e-5)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
