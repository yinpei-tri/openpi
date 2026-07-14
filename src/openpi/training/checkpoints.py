from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import jax.experimental.multihost_utils as multihost_utils
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _configure_array_handler_single_host() -> None:
    """Disable orbax's per-process ArrayMetadata store (belt-and-suspenders).

    save_state gathers the pytree to host numpy before saving, so orbax writes a single
    self-contained ocdbt.process_0/ and there is no multi-host coordination anyway. But
    disabling the ArrayMetadata store is harmless insurance against the "primary creates
    array_metadatas/ base dir, others wait on the same path" cross-host wait (which would
    hang / crash finalize on SageMaker's per-node local disks). Must be called before the
    CheckpointManager is built.
    """
    try:
        ocp.type_handlers.register_type_handler(
            jax.Array,
            ocp.type_handlers.ArrayHandler(array_metadata_store=None),
            override=True,
        )
    except Exception:
        logging.warning("Could not configure single-host orbax ArrayHandler; continuing with defaults.")


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    _configure_array_handler_single_host()
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    if checkpoint_dir.exists() and overwrite:
        checkpoint_dir.rmtree()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
    elif checkpoint_dir.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
            "to indicate how to handle it."
        )
    elif not checkpoint_dir.exists() and not resume:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume is an explicit contract: every process must have the same complete local
    # copy of a checkpoint containing both inference params and optimizer train state.
    # Never turn a missing/partial resume directory into a fresh run silently.
    resuming = resume
    if resume:
        latest_step = _collective_validate_resume_checkpoint(checkpoint_dir)
        logging.info(f"Resume preflight passed for checkpoint step {latest_step} at {checkpoint_dir}")

    # Multi-node checkpointing WITHOUT a shared filesystem: save_state gathers the pytree
    # to plain host numpy (process_allgather) before saving, so orbax sees non-distributed
    # arrays and writes a SINGLE self-contained ocdbt.process_0/ from the primary host —
    # no ocdbt.process_1/, no cross-host base-dir wait, no manifest referencing another
    # node's shard. The entrypoint then rank-0-only S3-syncs that complete local dir.
    # Default multiprocessing_options (primary_host=0) is correct for this — process 0 is
    # the writer, save()/restore() are called on all processes and sync via the network
    # coordination service (barriers over all processes, no subset). Single-node unchanged.
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    return mngr, resuming


def _inspect_resume_checkpoint(checkpoint_dir: epath.Path) -> tuple[int, list[str]]:
    """Returns the latest numeric step and structural errors for a local checkpoint."""
    if not checkpoint_dir.exists():
        return -1, [f"checkpoint directory does not exist: {checkpoint_dir}"]

    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        return -1, [f"no numeric checkpoint steps found in {checkpoint_dir}"]

    latest_step = steps[-1]
    step_dir = checkpoint_dir / str(latest_step)
    required = [step_dir / "_CHECKPOINT_METADATA"]
    for item in ("params", "train_state"):
        item_dir = step_dir / item
        required.extend(
            [
                item_dir / "_METADATA",
                item_dir / "manifest.ocdbt",
                item_dir / "ocdbt.process_0" / "manifest.ocdbt",
            ]
        )
    errors = [f"missing required checkpoint object: {path}" for path in required if not path.is_file()]
    return latest_step, errors


def _collective_validate_resume_checkpoint(checkpoint_dir: epath.Path) -> int:
    """Fails all JAX processes if any local resume copy is missing or inconsistent."""
    latest_step, errors = _inspect_resume_checkpoint(checkpoint_dir)
    for error in errors:
        logging.error(error)

    local_status = np.asarray([not errors, latest_step], dtype=np.int32)
    if jax.process_count() > 1:
        statuses = np.asarray(multihost_utils.process_allgather(local_status, tiled=True)).reshape(-1, 2)
    else:
        statuses = local_status.reshape(1, 2)

    if not np.all(statuses[:, 0]):
        raise RuntimeError(
            "Resume checkpoint preflight failed on at least one process. Each host must "
            "download a complete params/ + train_state/ checkpoint before JAX restore."
        )
    steps = statuses[:, 1]
    if not np.all(steps == steps[0]):
        raise RuntimeError(f"Resume checkpoint step differs across processes: {steps.tolist()}")
    return int(steps[0])


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    save_optimizer: bool = False,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)

    # MULTI-NODE: gather the sharded pytree to plain host numpy on every process before
    # saving. This is the crux of correct multi-node checkpointing WITHOUT a shared
    # filesystem: process_allgather turns each device-sharded jax.Array into a full numpy
    # array present on all hosts, so orbax sees NON-distributed arrays and writes a single
    # self-contained ``ocdbt.process_0/`` from the primary host — no ``ocdbt.process_1/``,
    # no cross-node coordination, no manifest referencing another node's shard. The
    # rank-0-only S3 sync then uploads a complete checkpoint. Valid for any mesh (the
    # gather reconstructs the full array regardless of how it was sharded). No-op cost on
    # single-node (allgather over 1 process is a device->host copy we'd do anyway).
    if jax.process_count() > 1:
        train_state = multihost_utils.process_allgather(train_state, tiled=True)
        params = multihost_utils.process_allgather(params, tiled=True)

    # main's refactor: when not saving the optimizer, simply omit train_state from
    # the saved items (rather than zeroing opt_state) — avoids the TrainState
    # typecheck issue the old opt_state={} path hit, so no extra guard needed.
    if not save_optimizer:
        items = {
            "assets": save_assets,
            "params": {"params": params},
        }
    else:
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        }
    checkpoint_manager.save(step, items)


def _reshard_host_tree_to_devices(state, sharding):
    """Rebuild each leaf as a globally-sharded jax.Array from a full host array.

    Our multi-node checkpoint design gathers the pytree to plain host numpy before saving
    (single self-contained ocdbt.process_0). On restore, orbax hands each process back the
    FULL host array for every leaf. Feeding those numpy arrays into the jitted train step —
    whose in_shardings carry the non-trivial FSDP sharding — raises "Passing non-trivial
    shardings for numpy inputs is not allowed". Rebuild each leaf as a global jax.Array:
    make_array_from_callback invokes the callback only for THIS process's addressable index
    slices, reading them out of the full host copy, so there is no cross-host transfer.
    """
    if sharding is None:
        return state

    def put(x, s):
        if s is None:
            return x
        if isinstance(x, jax.Array) and getattr(x, "sharding", None) == s:
            return x  # already a correctly-sharded device array (single-process fast path)
        host = np.asarray(x)
        return jax.make_array_from_callback(host.shape, s, lambda index, host=host: host[index])

    return jax.tree.map(put, state, sharding)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    sharding: training_utils.TrainState | None = None,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        # Restore the full train_state (step + optimizer + running params). `state` here is
        # the eval_shape TEMPLATE (init_train_state returns it unpopulated on resume), so
        # every leaf MUST be overwritten by the checkpoint. If train_state/ is missing (a
        # params-only checkpoint saved with save_optimizer=False), we do NOT fall back to a
        # params-only restore: that would leave step, opt_state, and the running params as
        # ShapeDtypeStruct templates (finding: parameter-only resume is broken), silently
        # continuing training from step 0 with an uninitialized optimizer. Fail loudly
        # instead — resume requires a checkpoint saved with --save-optimizer.
        try:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to restore full train_state (step + optimizer) for --resume. The "
                "checkpoint was likely saved without optimizer state (save_optimizer=False), "
                "which cannot be resumed: step/optimizer/running-params would stay "
                "uninitialized. Re-run the source job with --save-optimizer, or start a fresh "
                "run (load params via the weight_loader instead of --resume)."
            ) from e
    restored_state = _merge_params(restored["train_state"], restored["params"])
    template_paths = [
        jax.tree_util.keystr(path)
        for path, value in jax.tree_util.tree_flatten_with_path(restored_state)[0]
        if isinstance(value, jax.ShapeDtypeStruct)
    ]
    local_ok = np.asarray([not template_paths], dtype=np.int32)
    if jax.process_count() > 1:
        all_ok = np.asarray(multihost_utils.process_allgather(local_ok, tiled=True))
    else:
        all_ok = local_ok
    if template_paths or not np.all(all_ok):
        raise RuntimeError(
            "Resume restore left unmaterialized ShapeDtypeStruct leaves on at least one "
            f"process. Local template paths: {template_paths[:8]}"
        )

    # Multi-node: our save gathers to host numpy, so restore returns FULL host arrays on
    # every process. The jitted train step's in_shardings carry the non-trivial FSDP
    # sharding, and JAX rejects numpy inputs for a non-trivially-sharded arg ("Passing
    # non-trivial shardings for numpy inputs is not allowed"). Rebuild each leaf as a
    # globally-sharded jax.Array matching train_state_sharding before returning.
    restored_state = _reshard_host_tree_to_devices(restored_state, sharding)

    jax.block_until_ready(restored_state)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("openpi_resume_restore_complete")
    logging.info(f"Restored full train state at step {int(restored_state.step)} on all processes")
    return restored_state


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
