import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class BlockwiseWithMode(tfd.Blockwise):
    """Blockwise that defines mode as concatenation of component modes.

    Assumptions:
      - Each component distribution implements `mode()`
      - The joint mode is the concatenation of component modes
        (i.e., blocks are independent the way Blockwise composes them)
    """

    def _mode(self):
        # `tfd.Blockwise` stores components on `self.distributions` (public property)
        # and typically provides a private `_join(parts)` utility.
        parts = []
        for i, d in enumerate(self.distributions):
            if not hasattr(d, "mode"):
                raise NotImplementedError(f"Component {i} ({type(d).__name__}) does not implement mode().")
            parts.append(d.mode())

        # Use Blockwise's own join logic if present (best for shape correctness).
        join = getattr(self, "_join", None)
        if callable(join):
            return join(parts)

        # Fallback: concatenate along last axis (typical Blockwise event concat).
        # This assumes each part's event is in the last dimension.
        return jnp.array(parts)
