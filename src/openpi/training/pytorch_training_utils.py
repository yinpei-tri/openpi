"""Helpers for the PyTorch trainer.

Currently houses the EMA shadow used to mirror the JAX trainer's optimizer-state EMA.
Kept separate from `scripts/train_pytorch.py` so the entrypoint stays focused on the loop
and so these utilities are unit-testable.
"""

from __future__ import annotations

import logging

import torch
from torch import nn


class EmaModel:
    """Exponential moving average shadow of a model's parameters.

    Mirrors the JAX trainer's update rule:
        ema = decay * ema + (1 - decay) * new

    Stores the shadow on the same device as the base model. EMA params are kept in
    float32 even when the live model trains in bfloat16 — this matches the JAX path
    (where the optimizer state is fp32) and avoids the staircase-quantization that a
    bf16 EMA would introduce over thousands of steps.

    Works with plain `nn.Module` and DDP-wrapped models. Use `full_state_dict()` to
    materialize a CPU, fp32 copy for safetensors checkpointing.
    """

    def __init__(self, model: nn.Module, decay: float, *, device: torch.device | None = None):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        for name, param in self._iter_params(model):
            shadow = param.detach().clone().to(dtype=torch.float32)
            if device is not None:
                shadow = shadow.to(device)
            self._shadow[name] = shadow

    @staticmethod
    def _iter_params(model: nn.Module):
        # Strip the DDP `module.` prefix so EMA keys line up with the unwrapped state dict.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        weight = 1.0 - self.decay
        for name, param in self._iter_params(model):
            shadow = self._shadow[name]
            # `lerp_` computes shadow + weight * (param - shadow); with weight=(1-decay)
            # this matches `decay*shadow + (1-decay)*param`. Cast param to shadow's dtype
            # (fp32) so the arithmetic stays in fp32 even when training in bf16.
            shadow.lerp_(param.detach().to(shadow.dtype), weight)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name not in self._shadow:
                logging.warning("EMA shadow has no entry for %s; skipping", name)
                continue
            self._shadow[name].copy_(tensor.to(self._shadow[name].dtype))

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a CPU, fp32 copy of the shadow — for safetensors checkpointing."""
        return {name: tensor.detach().to("cpu", dtype=torch.float32) for name, tensor in self._shadow.items()}
