import dataclasses
import os
import pathlib
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_wandb_init_failure_is_nonfatal(tmp_path, monkeypatch):
    # A transient wandb.init failure on the (process-0) logger must NOT raise — it would
    # kill one process of a distributed job and hang the rest at the shutdown barrier.
    # init_wandb catches it and falls back to a disabled run.
    (tmp_path / "wandb_id.txt").write_text("missing-run")
    config = types.SimpleNamespace(checkpoint_dir=tmp_path, project_name="test")

    calls = []

    def fake_init(**kwargs):
        # First (real) init raises; the disabled fallback (mode="disabled") succeeds.
        if kwargs.get("mode") == "disabled":
            calls.append("disabled")
            return
        raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(train.wandb, "init", fake_init)
    # Must not raise, and must fall back to a disabled init.
    train.init_wandb(config, resuming=True, enabled=True)
    assert calls == ["disabled"]


def test_wandb_log_failure_is_nonfatal(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("wandb.log down")

    monkeypatch.setattr(train.wandb, "log", boom)
    # Must not raise.
    train.log_wandb({"loss": 1.0}, step=1)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        # Resume needs the full train_state (step + optimizer): restore_state raises if it
        # only finds params/ (a params-only checkpoint cannot be resumed). save_optimizer
        # defaults to False, so the resume leg below would otherwise fail.
        save_optimizer=True,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
