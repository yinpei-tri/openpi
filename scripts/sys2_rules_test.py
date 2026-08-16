# ruff: noqa: SLF001

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
import sys2_rules as rules
from sys2_rules_exp_ArrangeTea import _rule_arrangetea_kettle_inner_tray


def test_repeat_cap_advances_to_next_fine_step(monkeypatch):
    monkeypatch.setattr(rules, "MAX_SAME_SUBGOAL", 3)
    plan = """- [~] M1: do two things
  * [~] M1.1: reach to the cup
  * [ ] M1.2: grasp the cup"""
    state = {}

    # Reach/carry uses the tighter cap of two, so the third identical proposal advances.
    assert rules._rule_repeat_cap("AnyTask", plan, "reach to the cup", 50, state) == {}
    assert rules._rule_repeat_cap("AnyTask", plan, "continue to reach to the cup", 50, state) == {}
    result = rules._rule_repeat_cap("AnyTask", plan, "reach to the cup again", 50, state)

    assert result["subgoal"] == "grasp the cup"
    assert "* [x] M1.1: reach to the cup" in result["plan"]
    assert "* [~] M1.2: grasp the cup" in result["plan"]


def test_repeat_cap_declines_then_stops_at_true_end_of_plan(monkeypatch):
    monkeypatch.setattr(rules, "MAX_SAME_SUBGOAL", 3)
    monkeypatch.setattr(rules, "MAX_CAP_DECLINES", 4)
    plan = """- [~] M1: close the door
  * [~] M1.1: push the door closed"""
    state = {}

    for _ in range(3):
        assert rules._rule_repeat_cap("AnyTask", plan, "push the door closed", 50, state) == {}
    for decline in range(1, 4):
        result = rules._rule_repeat_cap("AnyTask", plan, "push the door closed", 50, state)
        assert result["interventions"][0]["kind"] == "cap_declined"
        assert state["rep_declines"] == decline

    result = rules._rule_repeat_cap("AnyTask", plan, "push the door closed", 50, state)
    assert result["stop_episode"] is True
    assert result["interventions"][0]["kind"] == "max_cap"


@pytest.mark.parametrize(
    "subgoal",
    [
        "turn on the sink faucet handle",
        "continue to turn on the sink faucet handle",
        "push the sink faucet handle to turn it on",
        "continue to push the sink faucet to turn it on",
        "turn the sink faucet on",
        "finish turning on the sink faucet handle",
    ],
)
def test_sink_faucet_est_matches_activation_phrasings(subgoal):
    result = rules._rule_sink_faucet_est("AnyTask", "", subgoal, 50, {})

    assert result["est_proposal"] == 100
    assert result["interventions"][0]["rule"] == "sink_faucet_est"


@pytest.mark.parametrize(
    "subgoal",
    [
        "reach to the sink faucet handle",
        "grasp the sink faucet handle",
        "hold the gripper on the faucet handle",
        "turn off the sink faucet",
        "push the microwave start button to turn it on",
    ],
)
def test_sink_faucet_est_rejects_other_operations(subgoal):
    assert rules._rule_sink_faucet_est("AnyTask", "", subgoal, 50, {}) == {}


def test_sink_faucet_est_is_a_floor():
    assert rules._rule_sink_faucet_est("AnyTask", "", "turn the sink faucet on", 125, {}) == {}


def test_microwave_est_bump_is_task_specific():
    active = rules._rule_microwave_est_bump("TurnOnMicrowave", "", "press the microwave start button", 50, {})
    assert active["est_proposal"] == 75

    assert rules._rule_microwave_est_bump("SteamInMicrowave", "", "press the microwave start button", 50, {}) == {}
    assert rules._rule_microwave_est_bump("GetToastedBread", "", "press the toaster lever", 50, {}) == {}
    assert rules._rule_microwave_est_bump("TurnOnSinkFaucet", "", "turn on the sink faucet handle", 50, {}) == {}


@pytest.mark.parametrize(
    ("task", "subgoal"),
    [
        ("TurnOnMicrowave", "continue to press the start button"),
        ("TurnOnElectricKettle", "continue to press the switch down"),
        ("PrepareCoffee", "continue to press the start button"),
        ("NavigateKitchen", "continue to press the kettle switch"),
    ],
)
def test_press_again_rewrites_both_prompt_fields(task, subgoal):
    result = rules._rule_press_again(task, "", subgoal, 50, {})

    expected = subgoal.removeprefix("continue to ") + " again"
    assert result["subgoal"] == expected
    assert result["subgoal_detail"] == expected
    assert rules._norm(result["subgoal"]) == rules._norm(subgoal)


def test_press_again_does_not_use_broad_task_name_substrings():
    assert rules._rule_press_again(
        "CoffeeSetupMug", "", "continue to press the mug against the dispenser", 50, {},
    ) == {}


@pytest.mark.parametrize(
    ("subgoal", "expected"),
    [
        ("lift and carry the kettle to the tray",
         "lift and carry the kettle to the inner side of the tray"),
        ("carry the kettle to the tray",
         "lift and carry the kettle to the inner side of the tray"),
        ("continue to lift and carry the kettle to the tray",
         "continue to lift and carry the kettle to the inner side of the tray"),
        ("continue to carry the kettle to the tray.",
         "continue to lift and carry the kettle to the inner side of the tray"),
    ],
)
def test_arrangetea_kettle_carry_targets_inner_side(subgoal, expected):
    result = _rule_arrangetea_kettle_inner_tray(
        "ArrangeTea", "unchanged plan", subgoal, 100, {},
    )

    assert result["subgoal"] == expected
    assert result["subgoal_detail"] == expected
    assert result["interventions"][0]["kind"] == "subgoal_override"
    assert "plan" not in result
    assert "est_proposal" not in result


@pytest.mark.parametrize(
    ("task", "subgoal"),
    [
        ("ArrangeTea", "lower the kettle onto the tray and release"),
        ("ArrangeTea", "carry the mug to the tray"),
        ("ArrangeTea", "carry the kettle to the counter"),
        ("ArrangeBreadBasket", "lift and carry the kettle to the tray"),
    ],
)
def test_arrangetea_kettle_inner_tray_rule_is_narrow(task, subgoal):
    assert _rule_arrangetea_kettle_inner_tray(
        task, "unchanged plan", subgoal, 100, {},
    ) == {}


def test_arrangetea_kettle_inner_tray_rule_is_registered():
    result = rules.apply_rules(
        "ArrangeTea", plan="unchanged plan",
        subgoal="lift and carry the kettle to the tray",
        subgoal_detail="move the kettle over the tray", est=100, state={},
        general=True, task_tier=True,
    )

    expected = "lift and carry the kettle to the inner side of the tray"
    assert result["subgoal"] == expected
    assert result["subgoal_detail"] == expected
    assert result["plan"] == "unchanged plan"
    assert result["est"] == 100
    assert any(i["rule"] == "arrangetea_kettle_inner_tray"
               for i in result["interventions"])
    assert rules._rule_press_again(
        "CuttingToolSelection", "", "continue to press the cucumber against the board", 50, {},
    ) == {}


def test_regrasp_hit_suppresses_sink_est_for_injected_turn():
    grasp_plan = """- [~] M1: operate faucet
  * [~] M1.1: grasp the sink faucet handle
  * [ ] M1.2: turn on the sink faucet handle"""
    turn_plan = """- [~] M1: operate faucet
  * [x] M1.1: grasp the sink faucet handle
  * [~] M1.2: turn on the sink faucet handle"""
    state = {}
    rules.apply_rules(
        "WashLettuce", plan=grasp_plan, subgoal="grasp the sink faucet handle",
        subgoal_detail="grasp the sink faucet handle", est=50, state=state,
        general=True, task_tier=False,
    )
    state["grip_width"] = 0.0

    result = rules.apply_rules(
        "WashLettuce", plan=turn_plan, subgoal="turn on the sink faucet handle",
        subgoal_detail="turn on the sink faucet handle", est=50, state=state,
        general=True, task_tier=False,
    )

    assert result["subgoal"] == "grasp the sink faucet handle again"
    assert result["est"] == rules.REGRASP_NEAR_EST
    assert state["regrasp_recovery_hit"] is True
    assert not any(i["rule"] == "sink_faucet_est" for i in result["interventions"])
    held = rules.pending_resume(state)
    assert held["subgoal"] == "turn on the sink faucet handle"
    assert held["est"] == 50


def test_regrasp_offset2_replays_intervening_motion_before_held_subgoal():
    grasp_plan = """- [~] M1: place the bottle
  * [~] M1.1: grasp the bottle
  * [ ] M1.2: carry the bottle to the tray
  * [ ] M1.3: release the bottle on the tray"""
    carry_plan = """- [~] M1: place the bottle
  * [x] M1.1: grasp the bottle
  * [~] M1.2: carry the bottle to the tray
  * [ ] M1.3: release the bottle on the tray"""
    release_plan = """- [~] M1: place the bottle
  * [x] M1.1: grasp the bottle
  * [x] M1.2: carry the bottle to the tray
  * [~] M1.3: release the bottle on the tray"""
    state = {}

    rules.apply_rules(
        "AnyTask", plan=grasp_plan, subgoal="grasp the bottle",
        subgoal_detail="close the gripper around the bottle", est=60, state=state,
        general=True, task_tier=False,
    )
    rules.record_executed_subgoal(
        state, subgoal="grasp the bottle",
        subgoal_detail="close the gripper around the bottle", est=60,
    )
    state["grip_width"] = 0.04
    rules.apply_rules(
        "AnyTask", plan=carry_plan, subgoal="carry the bottle to the tray",
        subgoal_detail="move the bottle left over the tray", est=110, state=state,
        general=True, task_tier=False,
    )
    rules.record_executed_subgoal(
        state, subgoal="carry the bottle to the tray",
        subgoal_detail="move the bottle left over the tray", est=110,
    )
    state["grip_width"] = 0.001

    result = rules.apply_rules(
        "AnyTask", plan=release_plan, subgoal="release the bottle on the tray",
        subgoal_detail="open the gripper over the tray", est=70, state=state,
        general=True, task_tier=False,
    )

    assert result["subgoal"] == "reach and grasp the bottle again"
    assert result["est"] == rules.REGRASP_FAR_EST
    assert any(i["kind"] == "tx_replay_queued" for i in result["interventions"])

    replay = rules.pending_resume(state)
    assert replay == {
        "key": "rg_M1.1", "rule": "regrasp_recovery",
        "subgoal": "carry the bottle to the tray",
        "subgoal_detail": "move the bottle left over the tray",
        "est": 110, "held_kind": "offset2_replay", "tx_label": "tx_sg_replay",
    }
    assert state["rg_M1.1_held"] == "release the bottle on the tray"

    held = rules.pending_resume(state)
    assert held["subgoal"] == "release the bottle on the tray"
    assert held["subgoal_detail"] == "open the gripper over the tray"
    assert held["est"] == 70
    assert held["held_kind"] == "s2_resume"
    assert "rg_M1.1_held" not in state
    assert rules.pending_resume(state) is None


def test_regrasp_offset2_requeries_when_prior_action_is_unsafe_to_replay():
    state = {
        "rg_turn": 2,
        "rg_active_fid": "M1.1",
        "rg_M1.1_gturn": 1,
        "rg_M1.1_bar": rules.REGRASP_MISS_WIDTH,
        "rg_M1.1_fid": "M1.1",
        "rg_M1.1_text": "grasp the bottle",
        "grip_width": 0.001,
    }
    rules.record_executed_subgoal(
        state, subgoal="lower and release the bottle",
        subgoal_detail="open the gripper over the tray", est=50,
    )
    plan = """- [~] M1: place the bottle
  * [x] M1.1: grasp the bottle
  * [~] M1.2: release the bottle on the tray"""

    result = rules.apply_rules(
        "AnyTask", plan=plan, subgoal="release the bottle on the tray",
        subgoal_detail="open the gripper over the tray", est=50, state=state,
        general=True, task_tier=False,
    )

    assert result["subgoal"] == "reach and grasp the bottle again"
    assert any(i["kind"] == "tx_replay_declined" for i in result["interventions"])
    assert rules.pending_resume(state) is None
    assert "rg_M1.1_held" not in state


def test_skip_failed_regrasp_requires_judge_and_no_physical_recovery():
    plan = """- [~] M1: pick up ketchup
  * [x] M1.1: reach to ketchup
  * [~] M1.2: grasp ketchup
  * [ ] M1.3: lift ketchup"""

    assert rules._rule_skip_failed_regrasp(
        rules.PPC2C, plan, "grasp ketchup again", 50, {"judge": None}) == {}
    assert rules._rule_skip_failed_regrasp(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": True},
    ) == {}
    # FAIL CLOSED: no recorded width is not evidence that the object is held.
    assert rules._rule_skip_failed_regrasp(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": False},
    )["interventions"][0]["kind"] == "skip_declined"
    # ...and the gripper reading empty means System2 was RIGHT, so its re-grasp must run.
    assert rules._rule_skip_failed_regrasp(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": False, "grip_width": 0.001},
    )["interventions"][0]["kind"] == "skip_declined"
    # Only a HELD object licenses the skip.
    result = rules._rule_skip_failed_regrasp(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": False, "grip_width": 0.045},
    )
    assert result["subgoal"] == "lift ketchup"


def test_skip_failed_regrasp_is_enabled_for_sink_to_counter():
    plan = """- [~] M1: pick up the egg
  * [x] M1.1: reach to the egg
  * [~] M1.2: grasp the egg
  * [ ] M1.3: lift the egg"""
    result = rules._rule_skip_failed_regrasp(
        "PickPlaceSinkToCounter", plan, "grasp the egg again", 50,
        {"judge": "subgoal_failed", "grip_width": 0.04},
    )
    assert result["subgoal"] == "lift the egg"


@pytest.mark.parametrize("subgoal", ["grasp the red mug", "grasp mug", "Grasp the blue mug"])
def test_coffee_normal_grasp_has_est_floor_75(subgoal):
    result = rules._rule_coffee_grasp_est(rules.COFFEE, "", subgoal, 50, {})
    assert result["est_proposal"] == 75
    assert result["interventions"][0]["rule"] == "coffee_grasp_est"

    assert rules._rule_coffee_grasp_est(rules.COFFEE, "", subgoal, 75, {}) == {}
    assert rules._rule_coffee_grasp_est(rules.COFFEE, "", subgoal, 100, {}) == {}


@pytest.mark.parametrize(
    "subgoal",
    [
        "continue to grasp the red mug",
        "grasp the red mug again",
        "grasp the red mug handle",
        "reach and grasp the red mug",
        "grasp the red mug.",
    ],
)
def test_coffee_normal_grasp_uses_raw_anchored_subgoal(subgoal):
    assert rules._rule_coffee_grasp_est(rules.COFFEE, "", subgoal, 50, {}) == {}


COFFEE_GRASP_PLAN = """- [~] M1: grasp the mug
  * [x] M1.1: reach for the mug
  * [~] M1.2: grasp the mug
- [ ] M2: place the mug under the coffee machine dispenser"""


@pytest.mark.parametrize("judge", [None, "subgoal_incomplete", "subgoal_complete"])
def test_coffee_skip_failed_requires_failed_judge(judge):
    assert rules._rule_coffee_skip_failed(
        rules.COFFEE, COFFEE_GRASP_PLAN, "grasp the mug again", 75,
        {"judge": judge, "grip_width": 0.02},
    ) == {}


@pytest.mark.parametrize("width", [None, 0.001])
def test_coffee_skip_failed_preserves_real_or_unverified_regrasp(width):
    result = rules._rule_coffee_skip_failed(
        rules.COFFEE, COFFEE_GRASP_PLAN, "grasp the mug again", 75,
        {"judge": "subgoal_failed", "grip_width": width},
    )
    assert result["interventions"][0]["kind"] == "skip_declined"
    assert "subgoal" not in result


def test_coffee_skip_failed_yields_to_an_injected_physical_recovery():
    assert rules._rule_coffee_skip_failed(
        rules.COFFEE, COFFEE_GRASP_PLAN, "grasp the mug again", 75,
        {"judge": "subgoal_failed", "grip_width": 0.02, "regrasp_recovery_hit": True},
    ) == {}


@pytest.mark.parametrize(("width", "expected"), [(0.001, "grasp the mug again"),
                                                   (0.02, "lift and carry the mug")])
def test_coffee_regrasp_and_skip_rules_coordinate_in_registry_order(width, expected):
    state = {"judge": "task_begin", "grip_width": 0.079}
    rules.apply_rules(
        rules.COFFEE, plan=COFFEE_GRASP_PLAN, subgoal="grasp the mug",
        subgoal_detail="grasp the mug", est=50, state=state,
        general=True, task_tier=True,
    )
    state["judge"] = "subgoal_failed"
    state["grip_width"] = width

    result = rules.apply_rules(
        rules.COFFEE, plan=COFFEE_GRASP_PLAN, subgoal="grasp the mug again",
        subgoal_detail="grasp the mug again", est=75, state=state,
        general=True, task_tier=True,
    )

    assert result["subgoal"].startswith(expected)
    if width < rules.regrasp_bar("grasp the mug"):
        assert state["rg_M1.2_injected"] == 1  # the re-grasp will actually execute
    else:
        assert "rg_M1.2_injected" not in state  # skipped attempt was refunded


def test_coffee_skip_failed_rewrites_only_safe_shape_and_refunds_regrasp():
    state = {
        "judge": "subgoal_failed", "grip_width": 0.02,
        "rg_M1.2_injected": 1,
    }
    result = rules._rule_coffee_skip_failed(
        rules.COFFEE, COFFEE_GRASP_PLAN, "grasp the mug again", 75, state,
    )

    assert result["subgoal"] == "lift and carry the mug to the coffee machine dispenser"
    assert result["est_assign"] == rules.COFFEE_M2_EST
    assert "- [x] M1: grasp the mug" in result["plan"]
    assert "- [~] M2: place the mug under the coffee machine dispenser" in result["plan"]
    assert "rg_M1.2_injected" not in state
    assert any(i["kind"] == "regrasp_credit_refund" for i in result["interventions"])


def test_coffee_skip_failed_refuses_to_overwrite_unrelated_or_unrolled_milestone():
    unrelated = """- [x] M1: grasp the mug
- [~] M2: press the coffee machine start button
  * [~] M2.1: press the coffee machine start button
- [ ] M3: retract the arm"""
    result = rules._rule_coffee_skip_failed(
        rules.COFFEE, unrelated, "grasp the mug again", 75,
        {"judge": "subgoal_failed", "grip_width": 0.02},
    )
    assert result["interventions"][0]["kind"] == "shape_declined"

    unrolled_next = """- [~] M1: grasp the mug
  * [~] M1.1: grasp the mug
- [ ] M2: place the mug under the coffee machine dispenser
  * [ ] M2.1: carry the mug to the dispenser"""
    result = rules._rule_coffee_skip_failed(
        rules.COFFEE, unrolled_next, "grasp the mug again", 75,
        {"judge": "subgoal_failed", "grip_width": 0.02},
    )
    assert result["interventions"][0]["kind"] == "shape_declined"


def test_drawer_authored_m11_base_alignment_has_est_floor_75():
    plan = """- [~] M1: open the drawer
  * [~] M1.1: reposition the base to align with the drawer
  * [ ] M1.2: reach to the drawer handle
  * [ ] M1.3: pull the drawer open"""
    result = rules._rule_drawer_base_align_est(
        "PickPlaceDrawerToCounter", plan,
        "reposition the base to align with the drawer", 50, {},
    )
    assert result["est_proposal"] == 75
    assert result["interventions"][0]["rule"] == "drawer_base_align_est"
    assert rules._rule_drawer_base_align_est(
        "PickPlaceDrawerToCounter", plan,
        "reposition the base to align with the drawer", 100, {},
    ) == {}


def test_drawer_inserted_m11_base_alignment_gets_same_est_floor():
    plan = """- [~] M1: open the drawer
  * [~] M1.1: reach to the drawer handle
  * [ ] M1.2: pull the drawer open"""
    result = rules.apply_rules(
        "PickPlaceDrawerToCounter", plan=plan,
        subgoal="reach to the drawer handle",
        subgoal_detail="reach to the drawer handle", est=50, state={},
        general=False, task_tier=True,
    )
    assert result["subgoal"] == rules.BASE_ALIGN_TEXT
    assert result["est"] == 75
    assert sum(
        i["rule"] == "drawer_base_align_est" and i["kind"] == "est_proposed"
        for i in result["interventions"]
    ) == 1


def test_drawer_alignment_est_requires_m11_alignment_step():
    plan = """- [~] M1: open the drawer
  * [x] M1.1: reposition the base to align with the drawer
  * [~] M1.2: reach to the drawer handle
  * [ ] M1.3: pull the drawer open"""
    assert rules._rule_drawer_base_align_est(
        "PickPlaceDrawerToCounter", plan, "reach to the drawer handle", 50, {},
    ) == {}


def test_current_general_registry():
    assert tuple(fn.__name__ for fn in rules._MANDATORY_RULES) == ("_rule_repeat_cap",)
    assert tuple(fn.__name__ for fn in rules._GENERAL_RULES) == (
        "_rule_press_again",
        "_rule_regrasp_recovery",
        "_rule_sink_faucet_est",
    )
    assert rules._rule_repeat_cap not in rules._RULES
    assert all(fn in rules._RULES for fn in rules._GENERAL_RULES + rules._TASK_RULES)
    assert rules._rule_microwave_est_bump in rules._TASK_RULES
    assert rules._rule_microwave_est_bump in rules._RULES


def test_rule_config_describes_selected_tiers():
    base = rules.rule_config(general=False, task_tier=False)
    assert base["mandatory_rules"] is True
    assert base["behavior_version"] == 2
    assert base["regrasp_recovery_version"] == "offset2-replay-v1"
    assert base["general_rules"] is False
    assert base["task_rules"] is False
    assert base["active_rules"]["mandatory"] == ["repeat_cap"]

    full = rules.rule_config(general=True, task_tier=True)
    assert full["active_rules"]["general"] == [
        "press_again", "regrasp_recovery", "sink_faucet_est"]
    assert "skip_failed_regrasp" in full["active_rules"]["task"]
