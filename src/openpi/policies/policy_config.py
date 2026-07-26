import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    use_ema: bool = True,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        use_ema: For PyTorch checkpoints, overlay `ema.safetensors` on top of the live weights when
            present. Matches the JAX path, which serializes EMA params into `params/`. No-op for
            JAX checkpoints (their `params/` is already the EMA when EMA was enabled).

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        ema_path = os.path.join(checkpoint_dir, "ema.safetensors")
        ema_path = ema_path if (use_ema and os.path.exists(ema_path)) else None
        model = train_config.model.load_pytorch(train_config, weight_path, ema_path=ema_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # RoboCasa `noanchorstate` fix (scoped by the checkpoint's robocasa_tag ONLY). This variant sets
    # include_anchor_state=False -> the model uses a 14-d state (no anchor half). But its shipped
    # norm_stats.json has a 28-d state (the 14-d current-state stats TILED x2, produced when the
    # norm-stats pass ran with include_anchor_state=True). The OUTPUT Unnormalize (which does NOT
    # slice) then hits a (14,) vs (28,) broadcast error. The two 28-d halves are IDENTICAL, so the
    # correct 14-d stats are exactly the first half — truncate to it. Guarded to ONLY the
    # `noanchorstate` tag so it can never touch the anchor-on checkpoints (v1-v11) whose 28-d stats
    # are correct. (A prior version keyed on data_config.state_split, which is NOT a field on the
    # created DataConfig -> it read None for everyone and wrongly truncated all ckpts.)
    _rc_tag = ""
    try:
        import json as _json
        for _cand in (pathlib.Path(str(checkpoint_dir)), pathlib.Path(str(checkpoint_dir)).parent):
            _cj = _cand / "config.json"
            if _cj.exists():
                _rc_tag = _json.loads(_cj.read_text()).get("robocasa_tag", "") or ""
                break
        if not _rc_tag:
            _rc_tag = str(checkpoint_dir)  # fall back to dir name (carries the tag)
    except Exception:
        _rc_tag = str(checkpoint_dir)
    if ("noanchorstate" in _rc_tag and norm_stats is not None
            and isinstance(norm_stats, dict) and norm_stats.get("state") is not None):
        import numpy as _np
        ss = norm_stats["state"]
        cur = ss.mean.shape[-1] if getattr(ss, "mean", None) is not None else None
        if cur is not None and cur % 2 == 0:
            half = cur // 2
            if _np.allclose(ss.mean[:half], ss.mean[half:]):  # genuinely a tiled 28-d entry
                _half = lambda a: (a[:half] if a is not None else a)  # noqa: E731
                norm_stats = dict(norm_stats)
                norm_stats["state"] = transforms.NormStats(
                    mean=_half(ss.mean), std=_half(ss.std), q01=_half(ss.q01), q99=_half(ss.q99))

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # At SERVING/eval time, ask TokenizePrompt to also emit the exact assembled prompt string
    # (real discretized state ints) so Policy.infer can return the TRUE model input for logging.
    # dataclasses are frozen, so rebuild each TokenizePrompt with the flag flipped on.
    model_input_transforms = [
        dataclasses.replace(t, emit_prompt_text=True) if isinstance(t, transforms.TokenizePrompt) else t
        for t in data_config.model_transforms.inputs
    ]

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_input_transforms,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
