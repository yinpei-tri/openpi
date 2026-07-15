import dataclasses

import numpy as np

from openpi.training import checkpoints


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
