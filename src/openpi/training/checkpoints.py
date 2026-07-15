from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import jax.experimental.multihost_utils as multihost_utils
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
    except Exception:  # noqa: BLE001 - best-effort; never block training on this
        logging.warning("Could not configure single-host orbax ArrayHandler; continuing with defaults.")


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    _configure_array_handler_single_host()
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


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
    # Only gather what we actually SAVE: when save_optimizer=False we skip the train_state
    # gather (it would all-gather ~20GB+ of optimizer + running params to host RAM on every
    # node just to discard it — wasted bandwidth + a needless cross-node collective). params
    # here is the EMA (inference) weights.
    if jax.process_count() > 1:
        params = multihost_utils.process_allgather(params, tiled=True)
        if save_optimizer:
            train_state = multihost_utils.process_allgather(train_state, tiled=True)

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


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    # NOTE: on restore, orbax logs an INFO line like
    #   "No metadata found for any process_index, checkpoint_dir=.../params. ...
    #    If the checkpoint does not contain jax.Array then it is expected. ... if no
    #    error is raised then it is a bug."
    # This is EXPECTED and benign here, not a bug. We save with the per-process
    # ArrayMetadata store disabled (see _configure_array_handler_single_host): the pytree
    # is gathered to host numpy and written as a single self-contained ocdbt.process_0/,
    # so no `array_metadatas/` dir is produced. orbax simply doesn't find that optional
    # metadata and falls back to reading arrays whole (the always-correct path). As long
    # as the restore completes without raising (it does — the arrays live in ocdbt.*),
    # the weights are fully intact.
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        # Try restoring with full train_state (including optimizer). If the checkpoint
        # was saved without optimizer state, fall back to restoring only params.
        try:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
        except Exception:
            logging.warning("Could not restore optimizer state from checkpoint, restoring params only")
            restored = checkpoint_manager.restore(
                step,
                items={
                    "params": {"params": params},
                },
            )
            restored["train_state"] = train_state
    return _merge_params(restored["train_state"], restored["params"])


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
