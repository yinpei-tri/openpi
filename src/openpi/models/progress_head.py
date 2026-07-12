"""Subgoal-completion progress head for System1 pi0.5.

Predicts how far the current frame is through its subgoal span — either a state-value
scalar in [0,1] (continuous mode, ``num_outputs=1``) or a K-way progress bucket
(classes mode, ``num_outputs=K``; cross-entropy in pi0.py). Reads the PaliGemma
*prefix* outputs (the clean image+language representation, noise-independent), NOT the
noised action-expert suffix. The
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
    """Progress readout over prefix features.

    Emits ``num_outputs`` logits per sample: ``num_outputs=1`` for the continuous
    state-value (caller applies sigmoid), or ``num_outputs=K`` for the K-way progress
    classifier (caller applies softmax / cross-entropy). The readout squeezes the last
    axis only when ``num_outputs == 1`` (continuous), else keeps the class axis.
    """

    def __init__(
        self,
        in_features: int,
        *,
        readout: str = "shallow_transformer",
        num_layers: int = 2,
        num_heads: int = 8,
        hidden: int = 512,
        num_outputs: int = 1,
        rngs: nnx.Rngs,
    ):
        self.readout = readout
        self.num_outputs = num_outputs
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
            self.out = nnx.Linear(hidden, num_outputs, rngs=rngs)
        elif readout in ("mean_pool", "prefix_token"):
            self.mlp1 = nnx.Linear(in_features, hidden, rngs=rngs)
            self.mlp2 = nnx.Linear(hidden, num_outputs, rngs=rngs)
        else:
            raise ValueError(f"unknown progress_readout: {readout}")

    def _squeeze(self, logits: at.Array) -> at.Array:
        # Continuous head (num_outputs==1): drop the trailing unit axis -> [b].
        # Classifier (num_outputs==K): keep the class axis -> [b, K].
        return jnp.squeeze(logits, axis=-1) if self.num_outputs == 1 else logits

    def from_pooled(self, feat: at.Float[at.Array, "b d"]) -> at.Array:
        """For mean_pool / prefix_token: feat is a single pooled vector per batch."""
        h = nnx.gelu(self.mlp1(feat))
        return self._squeeze(self.mlp2(h))

    def from_sequence(
        self,
        tokens: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Array:
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
        return self._squeeze(self.out(prog_out))
