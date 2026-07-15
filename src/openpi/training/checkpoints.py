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
    """Disable orbax's per-process ArrayMetadata store — MULTI-NODE ONLY.

    save_state gathers the pytree to host numpy before saving (see save_state), so orbax
    writes a single self-contained ocdbt.process_0/ and there is no multi-host coordination
    anyway. Disabling the ArrayMetadata store is belt-and-suspenders insurance against the
    "primary creates array_metadatas/ base dir, others wait on the same path" cross-host wait
    (which would hang / crash finalize on SageMaker's per-node local disks — there is no
    shared FS). Must be called before the CheckpointManager is built.

    GATED on process_count() > 1: this globally overrides orbax's jax.Array handler for the
    process, which would CHANGE the single-node checkpoint format (dropping array_metadatas/).
    Single-node must keep orbax's default handler untouched, so callers skip this.
    """
    if jax.process_count() <= 1:
        return
    try:
        ocp.type_handlers.register_type_handler(
            jax.Array,
            ocp.type_handlers.ArrayHandler(array_metadata_store=None),
            override=True,
        )
    except Exception:
        logging.warning("Could not configure multi-host orbax ArrayHandler; continuing with defaults.")


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    _configure_array_handler_single_host()  # no-op single-node (see gate inside)
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

    # MULTI-NODE (process_count > 1): gather the sharded pytree to plain host numpy on every
    # process before saving. This is the crux of correct multi-node checkpointing WITHOUT a
    # shared filesystem: process_allgather turns each device-sharded jax.Array into a full
    # numpy array present on all hosts, so orbax sees NON-distributed arrays and writes a
    # single self-contained ``ocdbt.process_0/`` from the primary host — no ``ocdbt.process_1/``,
    # no cross-node coordination, no manifest referencing another node's shard. The entrypoint's
    # rank-0-only S3 sync then uploads a complete checkpoint. Valid for any mesh (the gather
    # reconstructs the full array regardless of sharding). Only gather what we actually SAVE:
    # with save_optimizer=False we gather ONLY params (the EMA inference weights) and skip the
    # train_state gather (it would all-gather ~20GB+ of optimizer + running params to host RAM
    # on every node just to discard it — wasted bandwidth + a needless cross-node collective).
    # SINGLE-NODE: process_count()==1, so this whole block is skipped — behavior is byte-for-byte
    # identical to before (orbax writes the device-sharded arrays directly, as it always did).
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

    # NOTE: on restore, orbax may log an INFO line like
    #   "No metadata found for any process_index, checkpoint_dir=.../params. ... If the
    #    checkpoint does not contain jax.Array then it is expected. ... if no error is
    #    raised then it is a bug."
    # This is EXPECTED and benign for checkpoints written by this module: we save with the
    # per-process ArrayMetadata store disabled (see _configure_array_handler_single_host)
    # and, under multi-node, gather to host numpy first, so no `array_metadatas/` dir is
    # produced. orbax simply doesn't find that optional metadata and falls back to reading
    # arrays whole (the always-correct path). As long as restore completes without raising
    # (it does — the arrays live in ocdbt.*), the weights are fully intact.
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        # Resume MUST restore the full train_state (step + optimizer + running params).
        # The old code silently fell back to a params-only restore on ANY exception and
        # returned the unpopulated ShapeDtypeStruct template as train_state — which
        # meant a --resume silently continued from step 0 with an uninitialized optimizer
        # (Adam moments = template, not the saved moments). That is a data/optimizer-loss
        # footgun, especially on preemptible spot instances where resume is routine. Fail
        # loudly instead: a resume checkpoint must have been saved with save_optimizer=True.
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
                "Failed to restore full train_state (step + optimizer) for --resume. The checkpoint "
                "was likely saved without optimizer state (save_optimizer=False), which cannot be "
                "resumed (step/optimizer/running-params would stay uninitialized). Re-run the source "
                "job with save_optimizer=True, or start a fresh run (load params via the weight_loader "
                "instead of --resume)."
            ) from e
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
