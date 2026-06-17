"""Subgoal-completion progress head for System1 pi0.5.

A state-value scalar in [0,1] predicting how far the current frame is through its
subgoal span. Reads the PaliGemma *prefix* outputs (the clean image+language
representation, noise-independent), NOT the noised action-expert suffix. The
default reads ``stop_grad(prefix)`` so the progress objective never alters the VLM
features (Markovian action policy is preserved); ``prefix_token`` is the only
variant that would let gradients into the backbone (handled in pi0.py).

Readouts (``progress_readout``):
- ``mean_pool``: masked mean over prefix tokens -> MLP -> sigmoid. Cheapest.
- ``shallow_transformer`` (default): a small transformer over the (detached)
  prefix tokens + a learned [PROG] query token; read the [PROG] output -> sigmoid.
  Attention aggregation keeps fine detail a mean would lose.
- ``prefix_token``: handled inside pi0.py (an in-backbone CLS token); this module
  only provides the final projection for that case.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.shared import array_typing as at


class _Block(nnx.Module):
    """One pre-norm transformer block (self-attention + MLP)."""

    def __init__(self, width: int, num_heads: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=width, qkv_features=width, decode=False, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.mlp1 = nnx.Linear(width, 4 * width, rngs=rngs)
        self.mlp2 = nnx.Linear(4 * width, width, rngs=rngs)

    def __call__(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, mask=mask)
        h = self.norm2(x)
        x = x + self.mlp2(nnx.gelu(self.mlp1(h)))
        return x


class ProgressHead(nnx.Module):
    """Progress readout over prefix features -> scalar logit (sigmoid applied by caller)."""

    def __init__(
        self,
        in_features: int,
        *,
        readout: str = "shallow_transformer",
        num_layers: int = 2,
        num_heads: int = 8,
        hidden: int = 256,
        rngs: nnx.Rngs,
    ):
        self.readout = readout
        if readout == "shallow_transformer":
            self.in_proj = nnx.Linear(in_features, hidden, rngs=rngs)
            # Learned [PROG] query token (1, 1, hidden).
            self.prog_token = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (1, 1, hidden)))
            # Store blocks as named attributes (string param paths) rather than a
            # Python list — flax weight-merge flattens with a "/" string separator
            # and chokes on the integer keys a list would produce.
            self.num_layers = num_layers
            for i in range(num_layers):
                setattr(self, f"block_{i}", _Block(hidden, num_heads, rngs=rngs))
            self.out_norm = nnx.LayerNorm(hidden, rngs=rngs)
            self.out = nnx.Linear(hidden, 1, rngs=rngs)
        elif readout in ("mean_pool", "prefix_token"):
            self.mlp1 = nnx.Linear(in_features, hidden, rngs=rngs)
            self.mlp2 = nnx.Linear(hidden, 1, rngs=rngs)
        else:
            raise ValueError(f"unknown progress_readout: {readout}")

    def from_pooled(self, feat: at.Float[at.Array, "b d"]) -> at.Float[at.Array, " b"]:
        """For mean_pool / prefix_token: feat is a single pooled vector per batch."""
        h = nnx.gelu(self.mlp1(feat))
        return jnp.squeeze(self.mlp2(h), axis=-1)

    def from_sequence(
        self,
        tokens: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, " b"]:
        """For shallow_transformer: attend a [PROG] token over the prefix tokens."""
        b = tokens.shape[0]
        x = self.in_proj(tokens)
        prog = jnp.broadcast_to(self.prog_token.value, (b, 1, x.shape[-1]))
        x = jnp.concatenate([prog, x], axis=1)  # [PROG] at position 0
        # Build attention mask: the [PROG] query attends to itself + valid prefix tokens.
        attn_mask = None
        if mask is not None:
            key_valid = jnp.concatenate([jnp.ones((b, 1), dtype=bool), mask], axis=1)  # b, 1+s
            attn_mask = key_valid[:, None, None, :]  # b, 1, 1, 1+s (broadcast over heads+queries)
        for i in range(self.num_layers):
            x = getattr(self, f"block_{i}")(x, mask=attn_mask)
        prog_out = self.out_norm(x[:, 0])  # the [PROG] token output
        return jnp.squeeze(self.out(prog_out), axis=-1)
