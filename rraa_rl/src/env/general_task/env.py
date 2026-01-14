from typing import Any, NamedTuple

import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray


class EnvStep(NamedTuple):
    envstate: Any
    obs: Any
    predicates: dict
    term: bool
    trunc: bool
    info: dict


class Env:
    def step(self, state: Any, action: Any) -> EnvStep:
        raise NotImplementedError("")

    def reset(self, key: PRNGKeyArray) -> Any:
        raise NotImplementedError("")

    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        b_key = jr.split(key, batch_size)
        return jax.vmap(self.reset)(b_key)

    def get_obs(self, state: Any) -> Any:
        raise NotImplementedError("")

    def get_dummy_obs(self) -> Any:
        state = self.reset(jr.key(0))
        return self.get_obs(state)

    @property
    def n_agents(self) -> int:
        """Number of controlled agents."""
        raise NotImplementedError("")
