import dataclasses

import numpy as np
import pytest

from openpi.training import checkpoints


def _write_checkpoint_file(root, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")


def test_inspect_resume_checkpoint_requires_full_state(tmp_path):
    step = tmp_path / "1000"
    _write_checkpoint_file(step, "_CHECKPOINT_METADATA")
    for item in ("params", "train_state"):
        _write_checkpoint_file(step, f"{item}/_METADATA")
        _write_checkpoint_file(step, f"{item}/manifest.ocdbt")
        _write_checkpoint_file(step, f"{item}/ocdbt.process_0/manifest.ocdbt")

    latest, errors = checkpoints._inspect_resume_checkpoint(tmp_path)  # noqa: SLF001
    assert latest == 1000
    assert errors == []
    assert checkpoints._collective_validate_resume_checkpoint(tmp_path) == 1000  # noqa: SLF001

    (step / "train_state" / "_METADATA").unlink()
    _, errors = checkpoints._inspect_resume_checkpoint(tmp_path)  # noqa: SLF001
    assert any("train_state/_METADATA" in error for error in errors)
    with pytest.raises(RuntimeError, match="preflight failed"):
        checkpoints._collective_validate_resume_checkpoint(tmp_path)  # noqa: SLF001


@dataclasses.dataclass
class _State:
    step: np.ndarray
    params: dict
    opt_state: dict
    ema_params: dict | None


def test_ema_and_train_state_split_round_trip():
    state = _State(
        step=np.asarray(7),
        params={"w": np.asarray([1.0])},
        opt_state={"m": np.asarray([2.0])},
        ema_params={"w": np.asarray([9.0])},
    )

    train_state, inference_params = checkpoints._split_params(state)  # noqa: SLF001
    assert train_state.ema_params is None
    np.testing.assert_array_equal(train_state.params["w"], [1.0])
    np.testing.assert_array_equal(inference_params["w"], [9.0])

    restored = checkpoints._merge_params(train_state, {"params": inference_params})  # noqa: SLF001
    np.testing.assert_array_equal(restored.params["w"], [1.0])
    np.testing.assert_array_equal(restored.ema_params["w"], [9.0])
    np.testing.assert_array_equal(restored.opt_state["m"], [2.0])
    assert int(restored.step) == 7
