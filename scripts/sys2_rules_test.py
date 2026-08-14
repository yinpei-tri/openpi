# ruff: noqa: SLF001

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
import sys2_rules as rules


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


def test_ppc2c_skip_failed_requires_judge_and_no_physical_recovery():
    plan = """- [~] M1: pick up ketchup
  * [x] M1.1: reach to ketchup
  * [~] M1.2: grasp ketchup
  * [ ] M1.3: lift ketchup"""

    assert rules._rule_ppc2c_skip_failed(
        rules.PPC2C, plan, "grasp ketchup again", 50, {"judge": None}) == {}
    assert rules._rule_ppc2c_skip_failed(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": True},
    ) == {}
    result = rules._rule_ppc2c_skip_failed(
        rules.PPC2C, plan, "grasp ketchup again", 50,
        {"judge": "subgoal_failed", "regrasp_recovery_hit": False},
    )
    assert result["subgoal"] == "lift ketchup"


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
        "_rule_microwave_again",
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
    assert base["general_rules"] is False
    assert base["task_rules"] is False
    assert base["active_rules"]["mandatory"] == ["repeat_cap"]

    full = rules.rule_config(general=True, task_tier=True)
    assert full["active_rules"]["general"] == [
        "microwave_again", "regrasp_recovery", "sink_faucet_est"]
    assert "ppc2c_skip_failed" in full["active_rules"]["task"]
