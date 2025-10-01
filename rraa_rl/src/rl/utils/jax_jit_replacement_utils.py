"""
File with utility functions for JIT replacement in Gymnax environments.
NOTE: need to jit compile new function, store it in dictionary and the dispatch/properly point to it to replace original jitted method
"""

import types 
import numpy as np 
from gymnax.environments import spaces

from functools import partial
import jax 

def dispatch_jit_method(method_name):
    """Creates a stable JIT wrapper that calls env._patched_methods[name]"""
    @partial(jax.jit, static_argnums=(0,))  
    def wrapper(self, *args, **kwargs):
        return self._patched_methods[method_name](*args, **kwargs)
    return wrapper

def patch_env_methods(env, replacements: dict):
    # Initialize method storage if it doesn't exist
    if not hasattr(env, "_patched_methods"):
        env._patched_methods = {}

    for name, method_fn in replacements.items():
        # Bind and store the new function
        bound_fn = types.MethodType(method_fn, env)
        env._patched_methods[name] = bound_fn

        # Override the original method with the JIT-dispatcher
        setattr(env, name, types.MethodType(dispatch_jit_method(name), env))
