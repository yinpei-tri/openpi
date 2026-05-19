"""Helpers for the PyTorch trainer: EMA shadow, FSDP2 wrapping.

Kept separate from `scripts/train_pytorch.py` so the entrypoint stays focused on the loop
and so these utilities are unit-testable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EmaModel:
    """Exponential moving average shadow of a model's parameters.

    Mirrors the JAX trainer's update rule:
        ema = decay * ema + (1 - decay) * new

    Stores the shadow on the same device as the base model. EMA params are kept in
    float32 even when the live model trains in bfloat16 — this matches the JAX path
    (where the optimizer state is fp32) and avoids the staircase-quantization that
    a bf16 EMA would introduce over thousands of steps.

    Works with plain nn.Module, DDP-wrapped, and FSDP2-wrapped models. For FSDP2 the
    shadow stores **sharded** DTensors so memory cost matches FSDP. Materialize a full
    EMA state dict via `full_state_dict()` for checkpointing.
    """

    def __init__(self, model: nn.Module, decay: float, *, device: torch.device | None = None):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        for name, param in self._iter_params(model):
            # Keep EMA in fp32 even if param is bf16. Use empty + copy to preserve DTensor
            # placements when the underlying param is sharded by FSDP2.
            shadow = param.detach().clone().to(dtype=torch.float32)
            if device is not None and not _is_dtensor(shadow):
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
        decay = self.decay
        for name, param in self._iter_params(model):
            shadow = self._shadow[name]
            # `lerp_` computes shadow + weight * (param - shadow); with weight=(1-decay)
            # this matches `decay*shadow + (1-decay)*param`. Cast param to fp32 first so
            # the arithmetic happens in fp32 even when training in bf16.
            shadow.lerp_(param.detach().to(shadow.dtype), 1.0 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name not in self._shadow:
                logging.warning("EMA shadow has no entry for %s; skipping", name)
                continue
            self._shadow[name].copy_(tensor.to(self._shadow[name].dtype))

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a CPU, unsharded copy of the shadow — for safetensors checkpointing."""
        out: dict[str, torch.Tensor] = {}
        for name, tensor in self._shadow.items():
            full = tensor.full_tensor() if _is_dtensor(tensor) else tensor
            out[name] = full.detach().to("cpu", dtype=torch.float32)
        return out


def _is_dtensor(t: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(t, DTensor)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# FSDP2
# ---------------------------------------------------------------------------


def _layer_classes_to_wrap() -> tuple[type, ...]:
    """Return the per-layer module classes worth sharding individually.

    Each transformer block becomes its own FSDP2 unit so that activations from one block
    can be unsharded just-in-time and resharded immediately after. We deliberately do not
    wrap norms/projections — they're tiny and the extra all-gather/reduce-scatter calls
    would dominate.
    """
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

    return (GemmaDecoderLayer, SiglipEncoderLayer)


def build_device_mesh(world_size: int, fsdp_devices: int) -> DeviceMesh:
    """Create a 2D mesh for FSDP x DDP.

    Replicates the JAX trainer's layout: shard params across `fsdp_devices` ranks; replicate
    the shards across `world_size // fsdp_devices` data-parallel groups.

    Returns a mesh with named dims ("dp", "fsdp"). Pass mesh["fsdp"] to fully_shard.
    """
    from torch.distributed.device_mesh import init_device_mesh

    if world_size % fsdp_devices != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by fsdp_devices ({fsdp_devices})"
        )
    dp = world_size // fsdp_devices
    return init_device_mesh("cuda", (dp, fsdp_devices), mesh_dim_names=("dp", "fsdp"))


def apply_fsdp(model: nn.Module, mesh: DeviceMesh, *, param_dtype: torch.dtype) -> nn.Module:
    """Shard `model` in-place using FSDP2 (`fully_shard`).

    Wraps each transformer decoder/encoder layer as its own FSDP unit, then wraps the
    root model. Norm parameters stay in float32; matmul params are cast to `param_dtype`
    on the fly via MixedPrecisionPolicy. Reductions are done in fp32 to keep
    gradient accumulation numerically stable.

    NOTE on the joint forward: pi0/pi05's `compute_layer_complete` accesses submodule
    weights directly (e.g. `layer.self_attn.q_proj(x)`) instead of calling the wrapped
    `GemmaDecoderLayer.forward`. Without intervention this causes a DTensor/Tensor mixed
    op error. We rely on `unshard_layer_for_joint_forward` (called from
    `PaliGemmaWithExpertModel.forward`) to manually unshard each layer before its weights
    are used and reshard immediately after.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp import fully_shard

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        # Cast the FSDP root output back to the input dtype to avoid surprising callers.
        output_dtype=None,
    )

    fsdp_mesh = mesh["fsdp"] if "fsdp" in mesh.mesh_dim_names else mesh
    layer_cls = _layer_classes_to_wrap()

    # Shard inner layers first so the root's flat-param doesn't try to swallow them.
    num_layer_units = 0
    for module in model.modules():
        if isinstance(module, layer_cls):
            fully_shard(module, mesh=fsdp_mesh, mp_policy=mp_policy)
            num_layer_units += 1

    # Shard the root module so the remaining (small) params are also sharded.
    fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)

    if dist.is_initialized() and dist.get_rank() == 0:
        logging.info(
            "FSDP2 wrapped %d transformer layer units (%s); param_dtype=%s",
            num_layer_units,
            ",".join(c.__name__ for c in layer_cls),
            param_dtype,
        )
    return model


def is_fsdp_module(module: nn.Module) -> bool:
    """Return True if `module` was wrapped by `fully_shard` (FSDP2)."""
    try:
        from torch.distributed.fsdp import FSDPModule

        return isinstance(module, FSDPModule)
    except ImportError:
        return False
