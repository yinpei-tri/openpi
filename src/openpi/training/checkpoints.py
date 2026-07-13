from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _configure_array_handler_single_host() -> None:
    """Configure orbax's jax.Array handler for SINGLE-HOST checkpoint writes.

    Two settings, both needed so process 0 alone writes a COMPLETE checkpoint to its own
    local disk (no shared filesystem, no cross-node coordination):

    - ``array_metadata_store=None``: disables the per-process ArrayMetadata store. That
      store makes the primary host create an ``array_metadatas/`` base dir and other
      hosts WAIT for it on the SAME path — impossible across SageMaker's per-node local
      disks (times out; also caused the finalize JSONDecodeError on the S3-synced mount).
      Optional subchunk metadata; safe to drop.

    - ``use_replica_parallel=False``: by default orbax splits a replicated array's WRITE
      across replica hosts to go faster (process 0 writes ~5 GB, process 1 the other
      ~7 GB). On per-node filesystems those halves land in different places and the
      final dir gets only one -> incomplete, unloadable checkpoint. With this off, the
      single writing host emits ALL bytes, so process 0's dir is a complete checkpoint.

    Valid because the model is REPLICATED across nodes here (FSDP shards on the fsdp axis
    only; the batch/node axis is a data-parallel replica), so process 0 holds the whole
    model. NOT valid if params are ever sharded across nodes (e.g. fsdp == total devices
    spanning nodes). Must be called before the CheckpointManager is built.
    """
    try:
        ocp.type_handlers.register_type_handler(
            jax.Array,
            ocp.type_handlers.ArrayHandler(array_metadata_store=None, use_replica_parallel=False),
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

    # Multi-node: scope the save to process 0 ONLY (active_processes={0}). The model is
    # replicated across nodes (FSDP shards on the fsdp axis; the node axis is a
    # data-parallel replica), so process 0 holds the whole model and — with
    # use_replica_parallel=False above — writes a COMPLETE checkpoint to its own local
    # disk with no cross-node coordination or shared filesystem. save()/restore() are
    # still CALLED on all processes (orbax syncs the active subset internally). On a
    # single process this is a no-op. WARNING: only correct while params are replicated
    # across nodes — do NOT use with params sharded across nodes.
    mp_options = None
    if jax.process_count() > 1:
        mp_options = ocp.options.MultiprocessingOptions(primary_host=0, active_processes={0})

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
            multiprocessing_options=mp_options,
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
