from typing import Any, Dict, Iterable, TypeVar, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.core import FrozenDict
from jaxtyping import Array, Bool, Float, Int, Shaped

# Jax types.
BoolScalar = Bool[Array, ""] | bool
Shape = tuple[int, ...]
FloatScalar = float | Float[Array, ""]
IntScalar = int | Int[Array, ""]
Arr = Union[np.ndarray, Array]

# Environment types.
# State = Float[Array, 'state_dim']
AbstractState = TypeVar("AbstractState")
AbstractMinState = TypeVar("AbstractMinState")
Observation = Float[Array, "observation_dim"]
Action = Float[Array, "action_dim"]
Reward = FloatScalar
Cost = Float[Array, "M"]
H = Float[Array, "M"]
Term = BoolScalar
Trunc = BoolScalar
Info = Dict[str, Shaped[Array, ""]]

# Batched
# bState = Float[Array, 'b ']
bState = Iterable[AbstractState]
bMinState = Iterable[AbstractMinState]
bObservation = Float[Array, "b observation_dim"]
bAction = Float[Array, "b action_dim"]
bDone = Float[Array, "b"]
bReward = Float[Array, "b"]
bCost = Float[Array, "b M"]
bH = Float[Array, "b M"]
bLatent = Float[Array, "b latent_dim"]
bFloat = Float[Array, "b"]
bBool = Bool[Array, "b"]
bInfo = Iterable[Info]

# Collector
TreeLeaves = tuple[jnp.ndarray, ...]

# Neural network types.
Params = dict[str, Any] | FrozenDict[str, Any]
