# ruff: noqa: SLF001

import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
import sys2_rules as rules
import sys2_client
from sys2_rules_exp_ArrangeTea import _rule_arrangetea_kettle_inner_tray


@pytest.mark.parametrize("closing_tag", ["subgoal", "goal"])
def test_exec_parser_accepts_supported_subgoal_closing_tags(closing_tag):
    raw = (
        "<thought>I should move the mug.</thought>"
        "<judge>subgoal_incomplete</judge>"
        "<plan_update></plan_update>"
        "<estimated_step>75</estimated_step>"
        f"<subgoal>carry the mug to the dispenser</{closing_tag}>"
        "<subgoal_detail>move the held mug forward</subgoal_detail>"
    )

    parsed = sys2_client.parse_exec(raw)

    assert parsed["subgoal"] == "carry the mug to the dispenser"
    assert parsed["subgoal_detail"] == "move the held mug forward"


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


def test_final_retry_snapshots_coffee_milestone_after_task_rewrite():
    state = {"judge": "subgoal_failed", "grip_width": 0.02}

    result = rules.apply_rules(
        rules.COFFEE, plan=COFFEE_GRASP_PLAN, subgoal="grasp the mug again",
        subgoal_detail="try to grasp the mug again", est=75, state=state,
        general=True, task_tier=True, task_status="ongoing",
    )

    cached = state["_final_milestone_start"]
    assert result["subgoal"] == "lift and carry the mug to the coffee machine dispenser"
    assert cached["mid"] == "M2"
    assert cached["plan"] == result["plan"]
    assert cached["subgoal"] == result["subgoal"]
    assert cached["subgoal_detail"] == result["subgoal_detail"]
    assert cached["est"] == rules.COFFEE_M2_EST


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
    assert rules._HORIZON_RULES == (rules._rule_final_milestone_retry,)
    assert rules._rule_repeat_cap not in rules._RULES
    assert all(fn in rules._RULES for fn in (
        rules._GENERAL_RULES + rules._TASK_RULES + rules._HORIZON_RULES))
    assert rules._rule_microwave_est_bump in rules._TASK_RULES
    assert rules._rule_microwave_est_bump in rules._RULES


def test_rule_config_describes_selected_tiers():
    base = rules.rule_config(general=False, task_tier=False)
    assert base["schema_version"] == 2
    assert base["mandatory_rules"] is True
    assert base["behavior_version"] == 10
    assert base["regrasp_recovery_version"] == "offset2-replay-v1"
    assert base["final_milestone_retry_version"] == "official-step-loop-v7"
    assert base["final_milestone_retry_max"] == rules.FINAL_MILESTONE_RETRY_MAX
    assert base["final_milestone_retry_unlimited"] == (
        rules.FINAL_MILESTONE_RETRY_MAX == 0)
    assert base["general_rules"] is False
    assert base["task_rules"] is False
    assert base["last_milestone_retry"] is False
    assert base["active_rules"]["mandatory"] == ["repeat_cap"]
    assert base["active_rules"]["horizon_post"] == []

    full = rules.rule_config(general=True, task_tier=True)
    assert full["active_rules"]["general"] == [
        "press_again", "regrasp_recovery", "sink_faucet_est"]
    assert "skip_failed_regrasp" in full["active_rules"]["task"]
    assert full["active_rules"]["horizon_post"] == ["final_milestone_retry"]

    isolated = rules.rule_config(
        general=False, task_tier=False, last_milestone_retry=True)
    assert isolated["general_rules"] is False
    assert isolated["last_milestone_retry"] is True
    assert isolated["active_rules"]["horizon_post"] == ["final_milestone_retry"]


def test_legacy_retry_environment_gate_no_longer_overrides_recorded_cli_selection():
    env = os.environ.copy()
    env["SYS2_RULES_FINAL_MILESTONE_RETRY"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "examples" / "robocasa")
    code = """
import sys2_rules as rules
assert not hasattr(rules, 'FINAL_MILESTONE_RETRY_ON')
cfg = rules.rule_config(general=False, task_tier=False, last_milestone_retry=True)
assert cfg['last_milestone_retry'] is True
assert cfg['active_rules']['horizon_post'] == ['final_milestone_retry']
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


_FINAL_START = """- [x] M1: prepare the object
  * [x] M1.1: grasp the kettle
- [~] M2: put the kettle on the tray
  * [~] M2.1: carry the kettle to the inner side of the tray
  * [ ] M2.2: release the kettle
  * [ ] M2.3: retract the robot arm"""

_FINAL_DONE = """- [x] M1: prepare the object
  * [x] M1.1: grasp the kettle
- [x] M2: put the kettle on the tray
  * [x] M2.1: carry the kettle to the inner side of the tray
  * [x] M2.2: release the kettle
  * [x] M2.3: retract the robot arm"""

_FINAL_RETRACTING = """- [x] M1: prepare the object
  * [x] M1.1: grasp the kettle
- [~] M2: put the kettle on the tray
  * [x] M2.1: carry the kettle to the inner side of the tray
  * [x] M2.2: release the kettle
  * [~] M2.3: retract the robot arm"""


def _cache_final_start(state):
    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_START,
        subgoal="carry the kettle to the inner side of the tray",
        subgoal_detail="carefully carry the kettle to the inner side of the tray",
        est=125, state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert result["interventions"] == []


def test_final_milestone_retry_suppresses_false_finish_and_restores_carry():
    state = {}
    _cache_final_start(state)
    state["_resume_pending"] = {
        "key": "rg_M2.1", "rule": "regrasp_recovery", "subgoal": "stale held output",
        "_injection_credit_key": "rg_M2.1_injected",
    }
    state["rg_M2.1_held"] = "stale held output"
    state["rg_M2.1_injected"] = 1
    state["judge"] = "task_finish"
    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert result["plan"] == _FINAL_START
    assert result["subgoal"] == "reach and grasp the kettle"
    assert result["subgoal_detail"] == "reach and grasp the kettle"
    assert result["est"] == 125
    assert result["tx_label"] == "tx_final_milestone_retry"
    assert result["stop_episode"] is False
    assert result["suppress_task_finish"] is True
    assert any(i["kind"] == "task_finish_suppressed" for i in result["interventions"])
    assert any(i["kind"] == "tx_canceled" for i in result["interventions"])
    assert state["_resume_pending"]["subgoal"] == (
        "carry the kettle to the inner side of the tray")
    assert "rg_M2.1_held" not in state
    assert "rg_M2.1_injected" not in state

    held = rules.pending_resume(state)
    assert held["subgoal"] == "carry the kettle to the inner side of the tray"
    assert held["subgoal_detail"] == (
        "carefully carry the kettle to the inner side of the tray")
    assert held["est"] == 125
    assert held["held_kind"] == "final_milestone_replay"
    assert rules.pending_resume(state) is None


def test_final_retry_refunds_superseded_regrasp_recovery_credit():
    start = """- [~] M2: finish object placement
  * [~] M2.1: carry the pan to the counter
  * [ ] M2.2: grasp the pan
  * [ ] M2.3: retract the arm"""
    done = """- [x] M2: finish object placement
  * [x] M2.1: carry the pan to the counter
  * [x] M2.2: grasp the pan
  * [x] M2.3: retract the arm"""
    state = {}
    rules.apply_rules(
        "AnyTask", plan=start, subgoal="carry the pan to the counter",
        subgoal_detail="carry the pan to the counter", est=100,
        state=state, general=True, task_tier=False, task_status="ongoing",
        last_milestone_retry=True,
    )
    # Reproduce a recovery becoming eligible on the same fake-finish pass where the horizon rule
    # restarts M2. regrasp_recovery books the credit first; the post-rule must cancel and refund it.
    state.update({
        "judge": "task_finish",
        "rg_turn": 1,
        "rg_active_fid": "M2.2",
        "rg_M2.2_fid": "M2.2",
        "rg_M2.2_gturn": 1,
        "rg_M2.2_text": "grasp the pan",
        "rg_M2.2_bar": 0.015,
        "grip_width": 0.001,
    })

    result = rules.apply_rules(
        "AnyTask", plan=done, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
        last_milestone_retry=True,
    )

    kinds = [(i["rule"], i["kind"]) for i in result["interventions"]]
    assert ("regrasp_recovery", "tx_sg_failed") in kinds
    assert ("final_milestone_retry", "tx_canceled") in kinds
    assert "rg_M2.2_injected" not in state
    assert "rg_M2.2_held" not in state
    assert state["_resume_pending"]["key"] == "final_milestone_retry"


def test_final_milestone_retry_repeats_when_rule_cap_is_unlimited(monkeypatch):
    monkeypatch.setattr(rules, "FINAL_MILESTONE_RETRY_MAX", 0)
    state = {}
    _cache_final_start(state)
    state["judge"] = "task_finish"
    first = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert first["suppress_task_finish"] is True
    assert rules.pending_resume(state)["subgoal"] == (
        "carry the kettle to the inner side of the tray")

    second = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert second["suppress_task_finish"] is True
    assert second["subgoal"] == "reach and grasp the kettle"
    assert state["_final_milestone_start"]["retry_count"] == 2


def test_final_milestone_retry_positive_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(rules, "FINAL_MILESTONE_RETRY_MAX", 1)
    state = {}
    _cache_final_start(state)
    state["judge"] = "task_finish"
    first = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert first["suppress_task_finish"] is True
    assert rules.pending_resume(state)["subgoal"] == (
        "carry the kettle to the inner side of the tray")

    second = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert second["suppress_task_finish"] is False
    assert not any(i["rule"] == "final_milestone_retry" for i in second["interventions"])


def test_final_milestone_retry_continued_retract_triggers_after_first_retract():
    state = {}
    _cache_final_start(state)
    state["judge"] = None

    rules.record_executed_subgoal(
        state, subgoal="retract the robot arm", subgoal_detail="retract the robot arm", est=50)
    after_one = rules.apply_rules(
        "AnyTask", plan=_FINAL_RETRACTING, subgoal="continue to retract the robot arm",
        subgoal_detail="continue to retract the robot arm", est=50,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert after_one["subgoal"] == "reach and grasp the kettle"
    assert after_one["plan"] == _FINAL_START
    assert after_one["suppress_task_finish"] is False
    assert rules.pending_resume(state)["subgoal"] == (
        "carry the kettle to the inner side of the tray")


@pytest.mark.parametrize(
    "terminal_subgoal",
    [
        "continue to release the pan and retract.",
        "finish the task",
    ],
)
def test_final_milestone_retry_accepts_broader_terminal_phrasing(terminal_subgoal):
    state = {}
    _cache_final_start(state)
    state["judge"] = "subgoal_incomplete"

    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_RETRACTING, subgoal=terminal_subgoal,
        subgoal_detail=terminal_subgoal, est=50,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )

    assert result["plan"] == _FINAL_START
    assert result["subgoal"] == "reach and grasp the kettle"
    assert result["stop_episode"] is False


def test_horizon_retry_can_run_without_the_general_or_task_tiers():
    state = {"judge": "subgoal_incomplete"}
    rules.apply_rules(
        "AnyTask", plan=_FINAL_START,
        subgoal="carry the kettle to the inner side of the tray",
        subgoal_detail="carry the kettle to the inner side of the tray", est=125,
        state=state, general=False, task_tier=False, task_status="ongoing",
        last_milestone_retry=True,
    )

    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_RETRACTING, subgoal="finish the task",
        subgoal_detail="finish the task", est=50,
        state=state, general=False, task_tier=False, task_status="ongoing",
        last_milestone_retry=True,
    )

    assert result["plan"] == _FINAL_START
    assert result["subgoal"] == "reach and grasp the kettle"


def test_ordinary_terminal_cap_keeps_six_attempts_before_new_milestone_cycle(monkeypatch):
    monkeypatch.setattr(rules, "MAX_SAME_SUBGOAL", 3)
    monkeypatch.setattr(rules, "MAX_CAP_DECLINES", 4)
    plan = """- [~] M1: close the appliance door
  * [~] M1.1: push the appliance door closed"""
    state = {"judge": "subgoal_incomplete"}

    # Three in-cap attempts plus three executed declines. The fourth decline is the non-executed
    # max_cap decision, which the horizon post-rule converts into attempt 1 of a fresh M1 cycle.
    for _ in range(6):
        result = rules.apply_rules(
            "AnyTask", plan=plan, subgoal="push the appliance door closed",
            subgoal_detail="push the appliance door closed", est=75,
            state=state, general=False, task_tier=False, task_status="ongoing",
            last_milestone_retry=True,
        )
        assert result["stop_episode"] is False
        assert not any(i["rule"] == "final_milestone_retry" for i in result["interventions"])

    restarted = rules.apply_rules(
        "AnyTask", plan=plan, subgoal="push the appliance door closed",
        subgoal_detail="push the appliance door closed", est=75,
        state=state, general=False, task_tier=False, task_status="ongoing",
        last_milestone_retry=True,
    )

    assert restarted["stop_episode"] is False
    assert restarted["subgoal"] == "push the appliance door closed"
    assert any(i["kind"] == "max_cap" for i in restarted["interventions"])
    assert any(i["kind"] == "max_cap_suppressed" for i in restarted["interventions"])
    assert state["rep_n"] == 1
    assert state["rep_declines"] == 0
    assert state["_final_milestone_start"]["retry_count"] == 1


def test_final_milestone_retry_cancels_terminal_max_cap(monkeypatch):
    monkeypatch.setattr(rules, "MAX_SAME_SUBGOAL", 3)
    monkeypatch.setattr(rules, "MAX_CAP_DECLINES_RETRACT", 0)
    state = {}
    _cache_final_start(state)
    state["judge"] = "subgoal_incomplete"

    for _ in range(3):
        before_cap = rules.apply_rules(
            "AnyTask", plan=_FINAL_RETRACTING, subgoal="retract the robot arm",
            subgoal_detail="retract the robot arm", est=50,
            state=state, general=True, task_tier=False, task_status="ongoing",
        )
        assert before_cap["stop_episode"] is False

    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_RETRACTING, subgoal="retract the robot arm",
        subgoal_detail="retract the robot arm", est=50,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )

    assert result["stop_episode"] is False
    assert result["plan"] == _FINAL_START
    assert result["subgoal"] == "reach and grasp the kettle"
    assert any(i["kind"] == "max_cap" for i in result["interventions"])
    assert any(i["kind"] == "max_cap_suppressed" for i in result["interventions"])


def test_final_milestone_retry_third_bare_retract_is_fallback_trigger():
    state = {}
    _cache_final_start(state)
    state["judge"] = None

    rules.record_executed_subgoal(
        state, subgoal="retract the robot arm", subgoal_detail="retract the robot arm", est=50)
    rules.record_executed_subgoal(
        state, subgoal="finish retracting the arm", subgoal_detail="finish retracting the arm",
        est=50)
    third = rules.apply_rules(
        "AnyTask", plan=_FINAL_RETRACTING, subgoal="retract the arm",
        subgoal_detail="retract the arm", est=50,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert third["plan"] == _FINAL_START
    assert third["subgoal"] == "reach and grasp the kettle"
    assert third["suppress_task_finish"] is False
    assert state["_executed_retract_streak"] == 0


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("carry the pan to the stove", "the pan"),
        ("lift and carry the lid to the blender pitcher", "the lid"),
        ("continue to carry the bowl into the fridge", "the bowl"),
        ("carry and lower the visible basket to the dining counter", "the visible basket"),
        ("continue to carry and align the straw over the glass cup", "the straw"),
        ("lower the squash and sweet potato onto the highest rack and release",
         "the squash and sweet potato"),
        ("lower and release the left bun onto the grey plate", "the left bun"),
    ],
)
def test_final_milestone_retry_extracts_named_grasp_target(text, target):
    assert rules._final_retry_grasp_target(text) == target


@pytest.mark.parametrize(
    ("text", "milestone"),
    [
        ("reach to the microwave start button", "press the microwave start button"),
        ("search for the coffee machine", "navigate to the coffee machine"),
        ("reposition the base to the glass", "place the ice cube into the glass"),
        ("move the base left", "place the bottle in the glass group"),
        ("continue to move to the colander", "move to the colander in the sink"),
        ("release the sponge", "release the sponge on the cutting board"),
        ("lower the basket onto the counter", "place the basket on the counter"),
        ("lift the ice cube", "place the ice cube into the glass"),
        ("finish placing the basket on the counter", "place the basket on the counter"),
    ],
)
def test_final_milestone_retry_does_not_invent_grasp_target_for_nonobject_action(
        text, milestone):
    assert rules._final_retry_grasp_target(text, milestone) is None


def test_final_milestone_retry_directly_replays_nonobject_first_subgoal():
    start = """- [~] M1: press the microwave start button
  * [~] M1.1: reach to the microwave start button
  * [ ] M1.2: press the microwave start button"""
    done = """- [x] M1: press the microwave start button
  * [x] M1.1: reach to the microwave start button
  * [x] M1.2: press the microwave start button"""
    state = {}
    rules.apply_rules(
        "AnyTask", plan=start, subgoal="reach to the microwave start button",
        subgoal_detail="carefully reach to the microwave start button", est=75,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    state["judge"] = "task_finish"
    result = rules.apply_rules(
        "AnyTask", plan=done, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert result["plan"] == start
    assert result["subgoal"] == "reach to the microwave start button"
    assert result["subgoal_detail"] == "carefully reach to the microwave start button"
    assert result["est"] == 75
    assert result["suppress_task_finish"] is True
    assert "_resume_pending" not in state


def test_final_milestone_retry_does_not_checkpoint_a_nonfinal_milestone():
    plan = """- [~] M1: move the lemon
  * [~] M1.1: carry the lemon to the pitcher
  * [ ] M1.2: retract the arm
- [ ] M2: stir the lemonade"""
    state = {}
    rules.apply_rules(
        "AnyTask", plan=plan, subgoal="carry the lemon to the pitcher",
        subgoal_detail="carry the lemon to the pitcher", est=100,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert "_final_milestone_start" not in state

    rules.record_executed_subgoal(
        state, subgoal="retract the arm", subgoal_detail="retract the arm", est=50)
    continued = rules.apply_rules(
        "AnyTask", plan=plan, subgoal="continue to retract the arm",
        subgoal_detail="continue to retract the arm", est=50,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert continued["subgoal"] == "continue to retract the arm"
    assert not any(
        i["rule"] == "final_milestone_retry" for i in continued["interventions"])
    assert "_resume_pending" not in state

    state["judge"] = "task_finish"
    result = rules.apply_rules(
        "AnyTask", plan=plan, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="ongoing",
    )
    assert result["suppress_task_finish"] is False


def test_final_milestone_retry_requires_environment_ongoing():
    state = {}
    _cache_final_start(state)
    state["judge"] = "task_finish"
    result = rules.apply_rules(
        "AnyTask", plan=_FINAL_DONE, subgoal="", subgoal_detail="", est=None,
        state=state, general=True, task_tier=False, task_status="finished",
    )
    assert result["suppress_task_finish"] is False
