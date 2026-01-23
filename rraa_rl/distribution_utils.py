import numpy as np


def get_multidiscrete_min_entropy(n_actions: list[int], max_prob: float) -> float:
    """
    Compute the entropy of a multi-discrete distribution with given action space sizes, where
    one action in each discrete space has probability `max_prob`, and the rest of the probability
    mass is split uniformly among the other actions.
    """
    # total_entropy = 0.0
    # for n in n_actions:
    #     assert n > 1, "Each discrete action space must have at least 2 actions."
    #     other_prob = (1.0 - max_prob) / (n - 1)
    #     entropy = -(max_prob * np.log(max_prob) + (n - 1) * other_prob * np.log(other_prob))
    #     total_entropy += entropy
    # return total_entropy
    n_actions = np.array(n_actions)
    assert np.all(n_actions > 1), "Each discrete action space must have at least 2 actions."
    other_prob = (1.0 - max_prob) / (n_actions - 1)
    entropies = -(max_prob * np.log(max_prob) + (n_actions - 1) * other_prob * np.log(other_prob))
    total_entropy = np.sum(entropies)
    return total_entropy
