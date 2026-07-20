import contextlib
import dataclasses
import functools
import json
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

# --- RoboCasa System1 experiment-name settings tag -----------------------------------
# The trainer appends `__<tag>` to exp_name so each run's checkpoint dir / wandb name is
# unique + self-describing. The tag has ALWAYS-shown axes (the experiment variables) and
# DEVIATION tokens (prompt-content knobs, shown only when they differ from the defaults).
# Hyperparameters we've frozen (progress_loss_weight=1, progress_hidden=512, no stop-grad,
# LR warmup1k->flat 5e-5, subgoal action-pad, base_pose on, ...) are NOT in the tag.
#
# ALWAYS shown (order: prog, gran, verb):
#   prog{act|reg|cls}   progress predictor  (progact = progress-as-action head-off;
#                       progreg = continuous regression head; progcls = 10-way classes head)
#   gran{fine|crse|both}  subgoal granularity   (placeholder: only `fine` is in the data now)
#   verb{simp|rich|both}  subgoal verbosity     (placeholder: only `simp` is in the data now)
# DEVIATION tokens (order below), each shown only when the knob != its default:
_ROBOCASA_PROMPT_DEVIATIONS = [
    # (attr, default, token-when-different). All default TRUE (prompt-content on).
    # Conditioning: `nocond` drops the whole line; `noexec`/`noestl` drop only the Executed
    # Step / Estimated Length field (both suppressed when nocond is set — see below).
    ("include_conditioning", True, "nocond"),
    ("include_executed_step", True, "noexec"),
    ("include_est_length", True, "noestl"),
    # State has TWO orthogonal toggles: `nostate` drops the ENTIRE state block (Initial +
    # Current); `noanchorstate` drops only the anchor/Initial half (Current State stays).
    ("include_state", True, "nostate"),
    ("include_anchor_state", True, "noanchorstate"),
    ("include_task_goal", True, "notask"),
    ("use_anchor_images", True, "noanchor"),  # read from model (data mirrors it)
    ("include_gripper_flag", True, "nogrip"),
]

# Placeholder gran/verb derivation from the current single 3-way `prompt_source`. The data
# only carries the terse fine-step subgoal today, so `subgoal` -> (fine, simp) is the only
# combo that actually occurs; the others are mapped for when the data grows (then this is
# replaced by real `gran`/`verb` data knobs). Unknown values fall back to (fine, simp).
_PROMPT_SOURCE_TO_GRAN_VERB = {
    "subgoal": ("fine", "simp"),
    "subgoal_detail": ("fine", "rich"),
    "milestone": ("crse", "simp"),
}


def robocasa_exp_tag(config: _config.TrainConfig) -> str:
    """Settings tag appended to exp_name for RoboCasa System1 runs.

    Format: ``prog<x>_gran<y>_verb<z>[_<deviations>]`` — the three always-shown axes
    (progress predictor, subgoal granularity, subgoal verbosity) plus a deviation token for
    each prompt-content knob that differs from its default. Deterministic => same settings
    reproduce the same path, so --resume still works. Returns "" for non-RoboCasa configs.

    Example: ``progreg_granfine_verbsimp`` (regression, fine+simple subgoal, full prompt);
    ``progcls_granfine_verbsimp_notask_nostate`` (classes, subgoal-only prompt, no state).
    """
    data = config.data  # the RoboCasaDataConfig factory (has the knobs directly)
    # RoboCasa runs are identified by the system1_full knob `prompt_source`.
    if not hasattr(data, "prompt_source") or not hasattr(data, "shards"):
        return ""  # not a RoboCasa run
    m = config.model

    # 1) Progress predictor (always shown, exactly one).
    if getattr(data, "progress_as_action", False):
        predictor = "act"  # progress is a 12th action dim; head is off
    elif not getattr(m, "use_progress_head", True):
        predictor = "none"  # no progress signal at all (edge case)
    elif getattr(m, "progress_mode", "classes") == "continuous":
        predictor = "reg"
    else:
        predictor = "cls"  # classes (default); binary retired
    parts = [f"prog{predictor}"]

    # 2) Subgoal granularity + verbosity (always shown). Placeholder: derived from the
    #    single prompt_source knob until the data supports the full gran x verb grid.
    gran, verb = _PROMPT_SOURCE_TO_GRAN_VERB.get(getattr(data, "prompt_source", "subgoal"), ("fine", "simp"))
    parts.append(f"gran{gran}")
    parts.append(f"verb{verb}")

    # 3) Prompt-content deviations (shown only when the knob != default).
    for attr, default, token in _ROBOCASA_PROMPT_DEVIATIONS:
        # `noexec`/`noestl` (drop a single conditioning field) are redundant when `nocond`
        # (drop the whole line) is already set — suppress them so the tag never shows both.
        if attr in ("include_executed_step", "include_est_length") and not getattr(data, "include_conditioning", True):
            continue
        obj = m if attr == "use_anchor_images" else data
        if getattr(obj, attr, default) != default:
            parts.append(token)
    return "_".join(parts)


# gran/verb -> prompt_source (inverse of _PROMPT_SOURCE_TO_GRAN_VERB).
_GRAN_VERB_TO_PROMPT_SOURCE = {v: k for k, v in _PROMPT_SOURCE_TO_GRAN_VERB.items()}


def robocasa_config_from_tag(tag: str, base_config_name: str = "pi05_robocasa_system1") -> "_config.TrainConfig":
    """Inverse of ``robocasa_exp_tag``: rebuild the exact TrainConfig from a settings tag.

    The training runs use ONE root config (``pi05_robocasa_system1``) + CLI overrides, and only
    the resulting tag (``prog<x>_gran<y>_verb<z>[_<dev>...]``) is recorded — in the checkpoint
    dir name. This reconstructs the config those overrides produced, so inference can serve any
    ablation from the root config alone. Single source of truth with ``robocasa_exp_tag`` (same
    deviation table, same gran/verb map), so tag->config->tag round-trips.

    ``tag`` is the part AFTER the ``<exp_name>__`` prefix, e.g.
    ``progreg_granfine_verbsimp_nostate``. Accepts a full dir name too (splits on ``__``).
    """
    import dataclasses as _dc

    if "__" in tag:
        tag = tag.split("__", 1)[1]  # tolerate a full "<exp>__<tag>" dir name
    tokens = tag.split("_")
    cfg = _config.get_config(base_config_name)
    model_over: dict = {}
    data_over: dict = {}

    for tok in tokens:
        if tok.startswith("prog"):
            p = tok[len("prog"):]
            if p == "cls":
                model_over.update(progress_mode="classes", use_progress_head=True)
                data_over.update(progress_as_action=False)
            elif p == "reg":
                model_over.update(progress_mode="continuous", use_progress_head=True)
                data_over.update(progress_as_action=False)
            elif p == "act":
                model_over.update(use_progress_head=False)
                data_over.update(progress_as_action=True)
            elif p == "none":
                model_over.update(use_progress_head=False)
                data_over.update(progress_as_action=False)
        elif tok.startswith("gran") or tok.startswith("verb"):
            pass  # resolved together below
        else:
            # a deviation token -> flip its knob to (not default). model vs data per the table.
            for attr, default, token in _ROBOCASA_PROMPT_DEVIATIONS:
                if tok == token:
                    (model_over if attr == "use_anchor_images" else data_over)[attr] = not default
                    break

    # gran/verb -> prompt_source
    gran = next((t[len("gran"):] for t in tokens if t.startswith("gran")), "fine")
    verb = next((t[len("verb"):] for t in tokens if t.startswith("verb")), "simp")
    ps = _GRAN_VERB_TO_PROMPT_SOURCE.get((gran, verb))
    if ps is not None:
        data_over["prompt_source"] = ps

    model = _dc.replace(cfg.model, **model_over) if model_over else cfg.model
    data = _dc.replace(cfg.data, **data_over) if data_over else cfg.data
    return _dc.replace(cfg, model=model, data=data)


def resolve_robocasa_config(checkpoint_dir) -> "_config.TrainConfig | None":
    """Reconstruct a RoboCasa TrainConfig for a checkpoint dir, for INFERENCE/serving.

    Precedence (self-describing first, dir-name second):
      1. ``<ckpt>/config.json`` (written by training going forward) -> its ``base_config`` +
         ``robocasa_tag`` fed through ``robocasa_config_from_tag`` (authoritative).
      2. else parse the settings tag out of the dir name (``<exp>__<tag>``) -> same inverter
         (fallback for the current m0717 checkpoints, which predate config.json).
    Returns None if neither yields a RoboCasa tag (caller should fall back to an explicit config).
    Accepts a step dir (``.../29999``) or the run root; searches both for config.json.
    """
    import json as _json

    p = epath.Path(str(checkpoint_dir))
    for cand in (p, p.parent):  # config.json lives at the run root; a step dir is one level down
        cj = cand / "config.json"
        if cj.exists():
            try:
                meta = _json.loads(cj.read_text())
                if meta.get("robocasa_tag"):
                    return robocasa_config_from_tag(meta["robocasa_tag"],
                                                    meta.get("base_config", "pi05_robocasa_system1"))
            except Exception:
                pass  # fall through to dir-name parsing
    # dir-name fallback: find a path component containing the "__<tag>" settings suffix.
    for part in reversed(p.parts):
        if "__" in part and any(part.split("__", 1)[1].startswith(f"prog{x}") for x in ("cls", "reg", "act", "none")):
            return robocasa_config_from_tag(part)
    return None


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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # W&B is process-0-only, non-essential telemetry. NEVER let a transient W&B failure
    # (auth/network/stale run id) raise on process 0 and kill a distributed job while the
    # other processes proceed into collectives. On any error, fall back to disabled logging
    # and continue training. This stays a purely LOCAL, process-0 concern — NO cross-process
    # barrier here (a barrier around wandb is exactly what deadlocked the multi-node run
    # before). The other processes independently call wandb.init(disabled). Single-node:
    # same behavior as before, just guarded.
    try:
        ckpt_dir = config.checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        if resuming:
            run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

        if log_code:
            wandb.run.log_code(epath.Path(__file__).parent.parent)
    except Exception:
        logging.exception("wandb.init failed; continuing with W&B logging DISABLED (training unaffected).")
        with contextlib.suppress(Exception):
            wandb.init(mode="disabled")


def log_wandb(data: dict, *, step: int) -> None:
    """wandb.log that never raises — telemetry must not crash/desync training."""
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


def _fold_progress_ratios(info: dict) -> dict:
    """Convert the progress head's additive per-class/per-bin counters into ratios.

    The model logs, per window-averaged step, additive scalars:
      - classification: ``pcls_num/{c}`` (correct) and ``pcls_den/{c}`` (samples) per class
      - regression:     ``pbin_err/{b}`` (sum |err|) and ``pbin_cnt/{b}`` (samples) per bin
    Both numerator and denominator were mean-averaged over the log window, so the
    1/n_batches factor cancels and mean(num)/mean(den) is the exact window-level ratio.
    We emit ``progress_acc_class/{c}`` and ``progress_mae_bin/{b}`` and drop the raw
    counters. den==0 (a class/bin unseen in the whole window) -> NaN, which wandb skips.
    """
    out = {}
    pairs = {}  # ratio_key -> [num, den]
    confusion = {}  # binary-mode tp/fp/fn/tn counts
    for k, v in info.items():
        if k.startswith("pcls_num/"):
            pairs.setdefault(f"progress_acc_class/{k.split('/', 1)[1]}", [None, None])[0] = v
        elif k.startswith("pcls_den/"):
            pairs.setdefault(f"progress_acc_class/{k.split('/', 1)[1]}", [None, None])[1] = v
        elif k.startswith("pbin_err/"):
            pairs.setdefault(f"progress_mae_bin/{k.split('/', 1)[1]}", [None, None])[0] = v
        elif k.startswith("pbin_cnt/"):
            pairs.setdefault(f"progress_mae_bin/{k.split('/', 1)[1]}", [None, None])[1] = v
        elif k in ("pbin_tp", "pbin_fp", "pbin_fn", "pbin_tn"):
            confusion[k] = float(v)
        else:
            out[k] = v
    for ratio_key, (num, den) in pairs.items():
        if num is None or den is None:
            continue
        out[ratio_key] = float(num) / float(den) if float(den) > 0 else float("nan")
    # Binary "finished" head: derive precision/recall/F1/accuracy + the positive rates
    # from the additive confusion counts. Accuracy is misleading at ~4:1 imbalance, so
    # precision/recall/F1 are the metrics to watch; pred_pos_rate flags collapse.
    if len(confusion) == 4:
        tp, fp, fn, tn = (confusion["pbin_tp"], confusion["pbin_fp"], confusion["pbin_fn"], confusion["pbin_tn"])
        total = tp + fp + fn + tn
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        out["progress_precision"] = prec
        out["progress_recall"] = rec
        out["progress_f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        out["progress_acc"] = (tp + tn) / total if total > 0 else float("nan")
        out["progress_pred_pos_rate"] = (tp + fp) / total if total > 0 else float("nan")
        out["progress_true_pos_rate"] = (tp + fn) / total if total > 0 else float("nan")
    return out


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

    _robocasa_config_descriptor = None  # set below for RoboCasa runs; copied into each step dir
    if tentative_run:
        checkpoint_manager, resuming = None, False
    else:
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )
        # Write a machine-readable config descriptor next to the checkpoints so inference can
        # reconstruct the EXACT config WITHOUT parsing the dir name. We persist the RECONSTRUCTION
        # INPUTS (base config name + settings tag) rather than a full dataclass dump: the tag is
        # the single source of truth (robocasa_config_from_tag inverts it, validated to round-trip),
        # and it stays serializable + human-readable. Primary process only; best-effort.
        # Written in two places: (1) HERE at the run root, up front, so it survives even if
        # training crashes before the first save; (2) later, into each FINALIZED step dir
        # (after wait_until_finished, below) so a single uploaded step
        # dir (e.g. .../99) is self-describing on its own. resolve_robocasa_config reads either.
        if jax.process_index() == 0 and tag:
            _robocasa_config_descriptor = json.dumps({
                "base_config": "pi05_robocasa_system1",
                "robocasa_tag": tag,
                "exp_name": config.exp_name,
            }, indent=2)
            try:
                (epath.Path(config.checkpoint_dir) / "config.json").write_text(_robocasa_config_descriptor)
            except Exception:
                logging.warning("Could not write config.json to the checkpoint dir; continuing.")
    # Multi-node: only the PRIMARY process (index 0) logs to wandb. Otherwise every
    # process starts its own run and you get N identical curves (the metrics are the same
    # replicated value — see below). Single-node: process_index()==0, so unchanged.
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled and not tentative_run and jax.process_index() == 0,
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Multi-node sanity: prove the global batch is sharded across ALL devices AND that each
    # process/device holds DIFFERENT data. Printed once on every process, so the SageMaker
    # logs from every node show the split + distinct content. Debug-only; guarded so it can
    # never crash the job. No-op / harmless single-node (one process, one set of shards).
    _first = next(iter(batch[0].images.values()))
    try:
        _n_shards = len(_first.addressable_shards) if hasattr(_first, "addressable_shards") else -1
        # Per-device content fingerprint: mean of each addressable slice. Distinct values
        # across shards (and across processes) => genuinely different data on each GPU, not
        # a replicated/duplicated batch (which would mean the shard split lost process id).
        shard_means = [
            round(float(np.asarray(sh.data, dtype=np.float32).mean()), 4)
            for sh in getattr(_first, "addressable_shards", [])
        ]
        logging.info(
            f"[data-dist] proc={jax.process_index()}/{jax.process_count()} global_batch_shape={_first.shape} "
            f"n_addressable_shards={_n_shards} devices={jax.device_count()} mesh={mesh.shape}"
        )
        logging.info(f"[data-dist] proc={jax.process_index()} per-shard image means={shard_means}")
    except Exception as _e:
        logging.info(f"[data-dist] shard introspection skipped: {_e}")

    # Log images from first batch to sanity check. Guarded: image construction / wandb.log
    # must never crash the (multi-node) job. Runs on all processes — a no-op where wandb is
    # disabled (non-primary) — so there is no asymmetric collective / barrier.
    try:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        log_wandb({"camera_views": images_to_log}, step=0)
    except Exception:
        logging.exception("W&B camera preview failed; continuing training")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    tentative_run_step = start_step + 10

    # Data-resume: on a real resume (not the tentative run), skip the shards already
    # consumed before start_step so a preempted/spot run doesn't replay the first shards
    # (which would over-sample early data and under-sample the tail on a 1-2 epoch run).
    # shards_consumed ~= start_step * global_batch_size / samples_per_shard. The loader
    # (RoboCasa WebDataset) applies a per-worker skip derived from this; other loaders
    # (LeRobot/RLDS) no-op. Only meaningful when resuming with a positive step.
    if resuming and start_step > 0:
        samples_per_shard = _data_loader.robocasa_samples_per_shard(config)
        if samples_per_shard and hasattr(data_loader, "set_resume_shards_consumed"):
            shards_consumed = (start_step * config.batch_size) // samples_per_shard
            data_loader.set_resume_shards_consumed(shards_consumed)
            logging.info(
                f"Data-resume: start_step={start_step}, batch={config.batch_size}, "
                f"samples/shard={samples_per_shard} -> skipping ~{shards_consumed} global shards"
            )
            data_iter = iter(data_loader)  # rebuild so the skip takes effect
            batch = next(data_iter)

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # Timing diagnostics (added to wandb as t/*). We can't split forward vs backward:
    # they're fused in one jitted value_and_grad and XLA interleaves their ops, and JAX
    # async-dispatch means per-op timers would only measure dispatch, not compute. What we
    # CAN measure cheaply and honestly, with ZERO added throughput cost:
    #   - data_wait_s: wall time blocked in next(data_iter) — worker starvation / S3 IO.
    #   - step_s: total per-step wall time, averaged over the log window.
    #   - compute_s: step_s - data_wait_s — the GPU compute+dispatch remainder.
    #   - data_frac: data_wait_s / step_s — fraction of the step lost to data loading.
    # Throughput (step_s) over a window is wall-clock-accurate despite async dispatch
    # because the logging block's jax.device_get(reduced_info) is a natural sync barrier
    # at every log_interval (it depends on all infos in the window). No per-step
    # block_until_ready needed, so pipelining/overlap is untouched.
    data_wait_accum = 0.0
    window_start = time.perf_counter()
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))  # syncs the window
            n_win = len(infos)
            elapsed = time.perf_counter() - window_start
            # Skip the first logged step: it includes XLA compilation of ptrain_step,
            # which would dwarf steady-state numbers.
            if step > start_step and n_win > 0:
                step_s = elapsed / n_win
                data_wait_s = data_wait_accum / n_win
                reduced_info["t/step_s"] = step_s
                reduced_info["t/data_wait_s"] = data_wait_s
                reduced_info["t/compute_s"] = max(step_s - data_wait_s, 0.0)
                reduced_info["t/data_frac"] = data_wait_s / step_s if step_s > 0 else 0.0
            # Fold the progress-head per-class / per-bin num/den scalars into ratios.
            # The model emits additive counts (pcls_num/den, pbin_err/cnt) that were
            # mean-averaged over the window above; the 1/n_batches cancels in the ratio,
            # so mean(num)/mean(den) is the correct window-level accuracy/MAE. Empty
            # classes/bins (den==0) are reported as NaN (no samples => undefined), which
            # wandb simply skips in the plot. Raw counts are dropped from the log.
            #
            # MULTI-NODE: every scalar in `info` is ALREADY globally reduced — ptrain_step
            # is jitted with out_shardings=replicated, so XLA's collectives reduce every
            # jnp.mean/jnp.sum inside the step (loss, grad_norm, progress counts) over the
            # WHOLE global batch (all nodes' devices) and replicate the result to process 0.
            # So this fold computes precision/recall/F1/MAE over the full global batch with
            # no extra gather; process 0 just logs the already-global value.
            reduced_info = _fold_progress_ratios(reduced_info)
            # Console line stays compact: skip the 10 per-class / per-bin breakdowns
            # (they go to wandb). Keep the scalar summaries.
            info_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in reduced_info.items()
                if not k.startswith(("progress_acc_class/", "progress_mae_bin/"))
            )
            pbar.write(f"Step {step}: {info_str}")
            log_wandb(reduced_info, step=step)
            infos = []
            data_wait_accum = 0.0
            window_start = time.perf_counter()
        _t_data = time.perf_counter()
        batch = next(data_iter)
        data_wait_accum += time.perf_counter() - _t_data

        if tentative_run and step > tentative_run_step:
            logging.info("==========Tentative run completed==========")
            break

        if checkpoint_manager and (
            (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        ):
            _checkpoints.save_state(
                checkpoint_manager, train_state, data_loader, step, save_optimizer=config.save_optimizer
            )

    if checkpoint_manager:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()
        # Drop config.json INTO each finalized step dir (.../<step>/config.json) so an
        # individually-uploaded step checkpoint is self-describing. Must run AFTER
        # wait_until_finished: orbax stages each step in a `<step>.orbax-checkpoint-tmp-*`
        # dir and only renames it to `<step>/` at finalize, so writing earlier would race
        # (and could create a `<step>/` that collides with orbax's rename). Primary process
        # only, best-effort; the run-root copy written up front is the fallback.
        if jax.process_index() == 0 and _robocasa_config_descriptor is not None:
            try:
                root = epath.Path(config.checkpoint_dir)
                for step in checkpoint_manager.all_steps():
                    (root / str(step) / "config.json").write_text(_robocasa_config_descriptor)
            except Exception:
                logging.warning("Could not write per-step config.json; the run-root copy still applies.")


if __name__ == "__main__":
    # Configure logging BEFORE maybe_init_distributed so its coordinator/rank INFO lines
    # (the ones you need to diagnose a cross-host init hang) actually reach the console —
    # otherwise they hit root's default WARNING level and are dropped. main() calls
    # init_logging() again, which is idempotent (just re-sets the formatter).
    init_logging()
    # Multi-node: initialize the JAX distributed runtime ONCE, at process start, BEFORE
    # any JAX device op (including the tentative run below, which brings up the XLA backend
    # via jax.device_count / jit). Calling it inside main() would be too late — the
    # tentative run already initialized XLA in this same process, so the real run's
    # initialize() would raise "must be called before any JAX calls". No-op single-node.
    _distributed.maybe_init_distributed()
    config = _config.cli()
    main(config, tentative_run=True)
    time.sleep(20)
    main(config)
