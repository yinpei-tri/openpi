import dataclasses
import os
import pathlib
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_wandb_init_failure_is_nonfatal(tmp_path, monkeypatch):
    (tmp_path / "wandb_id.txt").write_text("missing-run")
    config = types.SimpleNamespace(checkpoint_dir=tmp_path, project_name="test")

    def fail_init(**_kwargs):
        raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(train.wandb, "init", fail_init)
    monkeypatch.setattr(train.wandb, "finish", lambda **_kwargs: None)

    assert not train.init_wandb(config, resuming=True)

    missing_config = types.SimpleNamespace(checkpoint_dir=tmp_path / "missing", project_name="test")
    assert not train.init_wandb(missing_config, resuming=True)


def test_wandb_log_failure_is_nonfatal(monkeypatch):
    monkeypatch.setattr(train.wandb, "log", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()))
    train.log_wandb({"loss": 1.0}, step=1, enabled=True)


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
