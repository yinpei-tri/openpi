import contextlib
import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.distributed as _distributed
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def robocasa_exp_tag(config: _config.TrainConfig) -> str:
    """Settings tag appended to exp_name for RoboCasa System1 runs.

    Ablations are driven by CLI overrides on a single config, so the only thing that
    distinguishes one run's checkpoint dir / wandb name from another is exp_name.
    Append a DETERMINISTIC tag derived from the ablation knobs so runs never collide
    and are self-describing. Deterministic => same settings reproduce the same path,
    so --resume still works. Returns "" for non-RoboCasa configs.
    """
    data = config.data  # the RoboCasaDataConfig factory (has the knobs directly)
    # RoboCasa runs are identified by the system1_full knob `prompt_source`.
    if not hasattr(data, "prompt_source") or not hasattr(data, "shards"):
        return ""  # not a RoboCasa run
    m = config.model
    parts = [
        f"src-{data.prompt_source}",
        f"pad-{getattr(data, 'subgoal_action_pad', 'subgoal')}",
        ("repad" if getattr(data, "repad_actions", False) else "bakedpad"),
        ("anchor" if getattr(m, "use_anchor_images", False) else "noanchor"),
        # Prompt-content flags default ON, so render BOTH states explicitly — an absent
        # tag would be ambiguous once the default is on.
        ("taskgoal" if getattr(data, "include_task_goal", False) else "notaskgoal"),
        ("anchorstate" if getattr(data, "include_anchor_state", False) else "noanchorstate"),
        ("cond" if getattr(data, "include_conditioning", False) else "nocond"),
        ("grip" if getattr(data, "include_gripper_flag", False) else "nogrip"),
    ]
    if getattr(m, "use_progress_head", False):
        # mode (classes/continuous) + readout + insulation (stop-grad) + loss weight,
        # so ablations over any of these get distinct checkpoint/wandb names.
        sg = "sg" if getattr(m, "progress_stop_gradient", True) else "nosg"
        mode = getattr(m, "progress_mode", "continuous")
        parts.append(f"prog-{mode}-{m.progress_readout}-{sg}-w{getattr(m, 'progress_loss_weight', 1.0):g}")
    else:
        parts.append("noprog")
    return "_".join(parts)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True
) -> bool:
    if not enabled:
        return False

    # W&B is process-0-only, non-essential telemetry. Never let it tear down one member of
    # a distributed job while the other processes continue into collectives.
    try:
        ckpt_dir = config.checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        if resuming:
            run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
            # resume="allow" (not "must"): resume the run if it exists on the server, else
            # start a fresh run with this id. "must" hard-fails when the prior run was
            # never registered (e.g. the source job was stopped early). Pass name= too so
            # the run shows our exp_name, not the AWS Batch job string, on the resume path.
            wandb.init(id=run_id, name=config.exp_name, resume="allow", project=config.project_name)
        else:
            # Force our OWN run id + name. On SageMaker/AWS Batch the runtime injects
            # WANDB_RUN_ID / WANDB_NAME env vars (the ugly "AWSBatch...-ip-..." string);
            # passing id/name explicitly to wandb.init overrides that env so the run is
            # named after exp_name and the id we persist is one WE control.
            run_id = wandb.util.generate_id()
            wandb.init(
                id=run_id,
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

        if log_code:
            wandb.run.log_code(epath.Path(__file__).parent.parent)
    except Exception:
        logging.exception("wandb.init failed; continuing with W&B logging disabled")
        with contextlib.suppress(Exception):
            wandb.finish(exit_code=1, quiet=True)
        return False
    return True


def log_wandb(data: dict[str, Any], *, step: int, enabled: bool) -> None:
    if not enabled:
        return
    try:
        wandb.log(data, step=step)
    except Exception:
        logging.exception("wandb.log failed; continuing training")


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # Ask the model for per-component metrics (e.g. flow_loss / progress_loss) so we
        # can log them separately. Only Pi0 supports the return_metrics kwarg; other
        # models (e.g. Pi0FAST) keep the original signature, so fall back without it.
        try:
            out = model.compute_loss(rng, observation, actions, train=True, return_metrics=True)
        except TypeError:
            out = model.compute_loss(rng, observation, actions, train=True)
        chunked_loss, metrics = out if isinstance(out, tuple) else (out, {})
        return jnp.mean(chunked_loss), metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        # Per-component losses + diagnostics from the model (flow_loss, progress_loss,
        # progress_mae, progress_pred/target_mean, ...). Logged individually to wandb.
        **loss_metrics,
    }
    return new_state, info


def main(config: _config.TrainConfig, tentative_run: bool = False):
    init_logging()
    logging.info(
        f"Running on: {platform.node()} | jax process {jax.process_index()}/{jax.process_count()} "
        f"| local devices {jax.local_device_count()} | global devices {jax.device_count()}"
    )

    # For RoboCasa System1, ablations are CLI overrides on one config, so append a
    # deterministic settings tag to exp_name -> distinct, self-describing checkpoint
    # dirs + wandb names that never collide. No-op for other configs / if already tagged.
    tag = robocasa_exp_tag(config)
    if tag and config.exp_name and f"__{tag}" not in config.exp_name:
        config = dataclasses.replace(config, exp_name=f"{config.exp_name}__{tag}")
        logging.info(f"RoboCasa exp_name tagged -> {config.exp_name}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if tentative_run:
        checkpoint_manager, resuming = None, False
    else:
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Multi-node sanity: prove the global batch is sharded across ALL devices (16 for
    # 2x8) AND that each process/device holds DIFFERENT data. Printed once on every
    # process, so the SageMaker logs from both nodes show the split + distinct content.
    _first = next(iter(batch[0].images.values()))
    try:
        _n_global_shards = len(_first.addressable_shards) if hasattr(_first, "addressable_shards") else -1
        # Content fingerprint per LOCAL device shard: mean of each addressable slice.
        # Different values across shards (and across processes) => genuinely different
        # data on each of the 16 GPUs, not a replicated/duplicated batch.
        shard_means = []
        for sh in getattr(_first, "addressable_shards", []):
            d = np.asarray(sh.data, dtype=np.float32)
            shard_means.append(round(float(d.mean()), 4))
        logging.info(
            f"[data-dist] proc={jax.process_index()}/{jax.process_count()} "
            f"global_batch_shape={_first.shape} "
            f"local_shard_shape={_first.addressable_data(0).shape if hasattr(_first, 'addressable_data') else 'n/a'} "
            f"n_addressable_shards={_n_global_shards} devices={jax.device_count()} mesh={mesh.shape}"
        )
        # Per-device content fingerprints: distinct values = distinct data per GPU.
        logging.info(f"[data-dist] proc={jax.process_index()} per-shard image means={shard_means}")
    except Exception as _e:  # noqa: BLE001 - debug only
        logging.info(f"[data-dist] shard introspection skipped: {_e}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader, sharding=train_state_sharding
        )

    # Only process 0 logs. Initialize telemetry after the distributed checkpoint protocol
    # has completed, then synchronize so every process enters training in the same phase.
    wandb_active = init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled and not tentative_run and jax.process_index() == 0,
    )
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("openpi_wandb_init_complete")

    if wandb_active:
        try:
            images_to_log = [
                wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
                for i in range(min(5, len(next(iter(batch[0].images.values())))))
            ]
            log_wandb({"camera_views": images_to_log}, step=0, enabled=True)
        except Exception:
            logging.exception("Could not construct W&B camera preview; continuing training")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    tentative_run_step = start_step + 10

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            log_wandb(reduced_info, step=step, enabled=wandb_active)
            infos = []
        batch = next(data_iter)

        if tentative_run and step > tentative_run_step:
            logging.info("==========Tentative run completed==========")
            break

        if checkpoint_manager and ((step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step, save_optimizer=config.save_optimizer)

    if checkpoint_manager:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # Multi-node: initialize the JAX distributed runtime ONCE, at process start, BEFORE
    # any JAX device op (including the tentative run below, which initializes the XLA
    # backend via jax.device_count / jit). Calling it inside main() was too late — the
    # tentative run already brought up XLA in this same process, so the real run's
    # initialize() raised "must be called before any JAX calls". No-op single-node.
    _distributed.maybe_init_distributed()
    config = _config.cli()
    main(config, tentative_run=True)
    time.sleep(20)
    main(config)
