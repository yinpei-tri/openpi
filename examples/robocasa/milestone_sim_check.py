"""MILESTONE-level sim-grounded success check for RoboCasa rollouts — ORACLE-REFERENCED.

Why milestone (not child subtask): a child span's predicate is checked at the child's END frame,
but the physical state change settles in the NEXT child (grasp ends before lift; push ends before
the door settles) -> false negatives even on a perfect oracle replay. A MILESTONE ("pick up the
pan", "open the fridge door") ends after the motion has settled, so the predicate is reliable.

Design (per the user): use the ORACLE trajectory as the REFERENCE. Instead of self-chosen absolute
thresholds, compare the rollout's settled milestone-end state to the GT (oracle) state at the same
milestone-end frame. The caller supplies `ref` = a dict of oracle end-state signals (captured by
running milestone_eval in --oracle mode), and this module reads the rollout's live env state and
compares.

Verdict: {"verdict": "success"|"failure"|"unknown", "rule": str, "target": str, "detail": str}.
"unknown" (precision-first) whenever no clean predicate applies or the target can't be resolved.

Goal-primitive routing (the milestone's last non-move_to/navigate/reach/retract/hold child prim):
  grasp                 -> in contact + finger opening within 0.01 of oracle's (NO abs <0.035)
  pick_up (grasp+lift)  -> grasp + object Δz >= 0.5 * oracle Δz (both vs milestone-start z)
  place / release       -> check_obj_in_receptacle + gripper->obj dist >= oracle dist - tol
  open/close/pull/push  -> per-door joint qpos within tol of oracle's settled value (no 0.9/0.05)
  turn / press          -> fixture on/off state flag == oracle's
  navigate / search     -> base xy within 0.30 m of oracle end-xy AND cos(Δyaw vs oracle) >= 0.90
  hold/stir/scrub/dump/other -> unknown (defer to Gemini)

`capture_ref_state(env, milestone)` builds the oracle REF dict (call at the oracle milestone end).
`milestone_check(env, milestone, ref)` returns the verdict for a policy rollout milestone end.
"""

from __future__ import annotations

import numpy as np
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T

# reuse the (already-calibrated) resolvers from the child-level check
from subtask_sim_check import resolve_object
from subtask_sim_check import resolve_fixture
from subtask_sim_check import _per_door_openness
from subtask_sim_check import _call_env

# ---- tolerances. Design: the SUBTASK success criterion is intentionally LESS STRICT than the
# episode _check_success, so it pinpoints REAL per-subgoal failures (not settling/pose nitpicks);
# the episode signal is AND-combined at the end to catch task-level failure. ----
LIFT_FRAC = 0.5          # (milestone pick_up only) rollout Δz >= this * oracle Δz
JOINT_TOL = 0.10         # |joint_qpos - oracle_joint| normalized-qpos match tol (open/close/pull/push)
NAV_DIST_TOL = 0.30      # base xy within this of oracle end-xy (m)
NAV_COS_TOL = 0.17       # cos(base_yaw - oracle_yaw) >= this  (~±80°, loose — heading roughly right)

_MANIP_PRIMS = ("grasp", "pick_up", "place", "release", "open", "close", "pull", "push",
                "turn", "press")
_SKIP_PRIMS = ("move_to", "navigate", "reach", "retract", "hold", "carry")


def _V(verdict, rule, target="", detail=""):
    return {"verdict": verdict, "rule": rule, "target": target, "detail": detail}


# --------------------------------------------------------------------------- state accessors
_FINGER_JOINTS = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]


def _finger_opening(env) -> float | None:
    """Sum of |finger joint qpos| — a monotone proxy for how open the gripper is (symmetric
    fingers, e.g. +0.0206/-0.0205). Larger = more open."""
    try:
        return float(sum(abs(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(j)])
                         for j in _FINGER_JOINTS))
    except Exception:
        return None


def _obj_z(env, name: str) -> float | None:
    try:
        return float(env.sim.data.body_xpos[env.obj_body_id[name]][2])
    except Exception:
        return None


def _gripper_obj_dist(env, name: str) -> float | None:
    try:
        obj = env.sim.data.body_xpos[env.obj_body_id[name]]
        eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return float(np.linalg.norm(np.asarray(eef) - np.asarray(obj)))
    except Exception:
        return None


def _gripper_contact_obj(env) -> str | None:
    """The env.objects name currently in contact with the right gripper (the manipulated object),
    resolved by SIM CONTACT rather than text — robust when the milestone text names a synonym the
    object langs don't contain (e.g. 'pan' == env object 'vegetable_container'). Returns the single
    contacting object, or None if zero / ambiguous (>=2)."""
    objs = getattr(env, "objects", {}) or {}
    if not objs:
        return None
    try:
        grip = env.robots[0].gripper["right"]
    except Exception:
        return None
    hits = []
    for n, o in objs.items():
        try:
            if env.check_contact(grip, o):
                hits.append(n)
        except Exception:
            pass
    return hits[0] if len(hits) == 1 else None


def _base_pose(env):
    """(xy: np.array[2], yaw: float) of the mobile base."""
    try:
        rid = env.sim.model.body_name2id("mobilebase0_base")
        xy = np.asarray(env.sim.data.body_xpos[rid][:2], float)
        yaw = float(T.mat2euler(np.array(env.sim.data.body_xmat[rid]).reshape(3, 3))[2])
        return xy, yaw
    except Exception:
        return None, None


def _fixture_state_flag(env, fx):
    """Best-effort on/off boolean for turn/press: faucet water_on / stove knobs-on / appliance
    turned_on. Returns bool or None."""
    # faucet / sink handle
    for meth in ("get_handle_state",):
        fn = getattr(fx, meth, None)
        if fn is not None:
            try:
                st = _call_env(fn, env)
                if isinstance(st, dict) and "water_on" in st:
                    return bool(st["water_on"])
            except Exception:
                pass
    # stove burners
    fn = getattr(fx, "get_knobs_state", None)
    if fn is not None:
        try:
            ks = _call_env(fn, env)
            if isinstance(ks, dict) and ks:
                return any(abs(float(v)) > 0.3 for v in ks.values())
        except Exception:
            pass
    # generic appliance turned_on
    fn = getattr(fx, "get_state", None)
    if fn is not None:
        try:
            st = fn()
            if isinstance(st, dict) and "turned_on" in st:
                return bool(st["turned_on"])
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- goal primitive
# Meaningful spans that get a per-span subtask verdict (finestep scores each of these; skips the
# rest = move_to/reach/retract/carry/hold/stir/scrub/dump). "navigate"/"search"/"goto" = base moves.
MEANINGFUL_PRIMS = ("grasp", "place", "release", "open", "close", "pull", "push", "turn", "press",
                    "navigate", "search", "goto")


def span_goal_primitive(primitive: str) -> str | None:
    """Normalize a SINGLE fine-subgoal primitive to its subtask-check goal (or None if not scorable).
    Unlike milestone_goal_primitive (which reduces a whole child LIST + detects grasp+lift=pick_up),
    this scores ONE span by its own primitive — finestep grasp stays 'grasp' (no pick_up)."""
    p = (primitive or "").lower()
    if p in ("goto",):
        return "navigate"
    return p if p in MEANINGFUL_PRIMS else None


def milestone_goal_primitive(child_prims: list[str]) -> str:
    """The milestone's goal = its LAST child primitive that isn't pure repositioning. Also detect
    the grasp+lift 'pick up' pattern: a grasp followed only by move_to/lift children."""
    prims = list(child_prims)
    goal = None
    for p in reversed(prims):
        if p not in _SKIP_PRIMS:
            goal = p
            break
    if goal is None:
        goal = prims[-1] if prims else "other"
    # grasp that ends the milestone via a trailing lift/move => "pick up"
    if goal == "grasp":
        gi = max(i for i, p in enumerate(prims) if p == "grasp")
        if any(prims[j] == "move_to" for j in range(gi + 1, len(prims))):
            return "pick_up"
    return goal


# --------------------------------------------------------------------------- REF capture
def capture_ref_state(env, milestone_text: str, goal_prim: str, start_obj_z: float | None,
                      held_obj: str | None = None) -> dict:
    """Capture the ORACLE end-state signals for a milestone (call at the oracle milestone-end frame,
    after settling). `start_obj_z` = the manipulated object's z at the milestone START (for lift Δz).
    `held_obj` = the object the gripper held DURING the milestone (resolved by contact by the caller);
    used for grasp/pick_up/place/release where text resolution is unreliable.
    Returns a small JSON-able dict the policy check compares against."""
    ref: dict = {"goal_prim": goal_prim, "milestone_text": milestone_text}
    if goal_prim in ("grasp", "pick_up"):
        ref["finger_opening"] = _finger_opening(env)
        # prefer contact-at-end, then the object held during the milestone, then text.
        obj = _gripper_contact_obj(env) or held_obj
        if obj is None:
            o, conf = resolve_object(env, milestone_text)
            obj = o if conf >= 1.0 else None
        ref["obj"] = obj
        if obj:
            ref["obj_z_end"] = _obj_z(env, obj)
            ref["obj_z_start"] = start_obj_z
    elif goal_prim in ("place", "release"):
        # object is released by the end, so use the object held DURING the milestone (by contact).
        obj = held_obj
        if obj is None:
            o, conf = resolve_object(env, milestone_text)
            obj = o if conf >= 1.0 else None
        ref["obj"] = obj
        if obj:
            ref["gripper_obj_dist"] = _gripper_obj_dist(env, obj)
            ref["receptacle"] = _receptacle_moved_into(env, milestone_text, obj)
    elif goal_prim in ("open", "close", "pull", "push"):
        fx, nm, conf = resolve_fixture(env, milestone_text)
        ref["fixture"] = nm if conf >= 1.0 else None
        if fx is not None:
            ref["door_openness"] = _per_door_openness(env, fx)  # {joint: qpos}
    elif goal_prim in ("turn", "press"):
        fx, nm, conf = resolve_fixture(env, milestone_text)
        ref["fixture"] = nm if conf >= 1.0 else None
        if fx is not None:
            ref["state_flag"] = _fixture_state_flag(env, fx)
    elif goal_prim in ("navigate", "search"):
        xy, yaw = _base_pose(env)
        ref["base_xy"] = xy.tolist() if xy is not None else None
        ref["base_yaw"] = yaw
    return ref


# --------------------------------------------------------------------------- policy check
def subtask_check(env, subgoal_text: str, ref: dict, last_grip: float | None = None) -> dict:
    """SUBTASK success verdict for a POLICY rollout span end, vs the oracle REF. Intentionally LOOSE
    (see tolerances) — pinpoints real per-subgoal failure; the caller AND-combines with episode
    _check_success at the terminal span. `last_grip` = the last COMMANDED gripper action (+1 close /
    -1 open), used for grasp (must be closing) and place/release (must be opening)."""
    gp = (ref or {}).get("goal_prim", "other")

    if gp in ("grasp", "pick_up"):
        obj = ref.get("obj")
        if not obj:
            return _V("unknown", "grasp_unresolved", detail="no manipulable env.object resolved")
        try:
            in_contact = env.check_contact(env.robots[0].gripper["right"], env.objects[obj])
        except Exception:
            return _V("unknown", "grasp_error", target=obj)
        # LOOSE grasp: in contact AND the gripper is COMMANDED CLOSED (action > 0). Uses the commanded
        # action (policy intent, stable/binary) rather than finger qpos vs oracle (noisy, mid-transition).
        closing = (last_grip is None) or (last_grip > 0)
        grasped = bool(in_contact and closing)
        cstr = (f"contact={in_contact}{'' if in_contact else ' -> NOT touching obj'}; "
                f"grip_cmd={'close' if (last_grip is None or last_grip>0) else 'OPEN'}"
                f"{'' if closing else ' -> not closing'}")
        if gp == "grasp":
            why = "" if grasped else "  FAIL because " + ("not in contact" if not in_contact else "gripper not closing")
            return _V("success" if grasped else "failure", "grasp", target=obj, detail=cstr + why)
        # pick_up (milestone only): grasp + lifted vs oracle Δz
        zc, z0, ze = _obj_z(env, obj), ref.get("obj_z_start"), ref.get("obj_z_end")
        if zc is None or z0 is None or ze is None:
            return _V("success" if grasped else "failure", "grasp", target=obj, detail=cstr)  # no z -> grasp only
        oracle_dz, roll_dz = ze - z0, zc - z0
        need = LIFT_FRAC * oracle_dz
        lifted = oracle_dz <= 1e-4 or roll_dz >= need
        v = "success" if (grasped and lifted) else "failure"
        why = "" if v == "success" else "  FAIL because " + (
            "not grasped" if not grasped else f"not lifted (Δz {roll_dz:.3f} < {LIFT_FRAC}×oracle {need:.3f})")
        return _V(v, "pick_up", target=obj,
                  detail=f"{cstr}; lift Δz rollout={roll_dz:.3f} vs oracle={oracle_dz:.3f}{why}")

    if gp in ("place", "release"):
        obj = ref.get("obj")
        if not obj:
            return _V("unknown", "place_unresolved", detail="no held env.object resolved")
        recep = ref.get("receptacle") or _resolve_receptacle(env, subgoal_text, exclude=obj)
        if not recep:
            return _V("unknown", "place_no_receptacle", target=obj,
                      detail=f"no target receptacle for {obj} (fixture target? -> unknown)")
        # LOOSE place: object in CONTACT with the receptacle (drop the xy-radius term) AND the gripper
        # is COMMANDED OPEN (action < 0) — i.e. it let go. Contact-only is far less strict than
        # check_obj_in_receptacle (which also requires within 0.7*recep-radius of the center).
        try:
            in_contact = bool(env.check_contact(env.objects[obj], env.objects[recep]))
        except Exception:
            return _V("unknown", "place_contact_error", target=f"{obj}->{recep}")
        opened = (last_grip is None) or (last_grip < 0)
        v = "success" if (in_contact and opened) else "failure"
        why = "" if v == "success" else "  FAIL because " + (
            f"{obj} not touching {recep}" if not in_contact else "gripper not opened (still holding)")
        return _V(v, "place_released", target=f"{obj}->{recep}",
                  detail=f"obj-recep contact={in_contact}; grip_cmd={'open' if opened else 'CLOSE'}{why}")

    if gp in ("open", "close", "pull", "push"):
        fxnm = ref.get("fixture")
        ref_door = ref.get("door_openness") or {}
        if not fxnm or not ref_door:
            return _V("unknown", "fixture_unresolved", detail="no fixture/door-joint resolved")
        fx, _, _ = resolve_fixture(env, subgoal_text)
        if fx is None:
            return _V("unknown", "fixture_unresolved")
        cur = _per_door_openness(env, fx)
        if not cur:
            return _V("unknown", "fixture_no_joint", target=fxnm)
        # compare each joint present in BOTH to the oracle's settled value; report the worst joint
        common = [j for j in ref_door if j in cur]
        if not common:
            return _V("unknown", "fixture_no_common_joint", target=fxnm)
        worst = max(common, key=lambda j: abs(cur[j] - ref_door[j]))
        wdiff = abs(cur[worst] - ref_door[worst])
        v = "success" if wdiff <= JOINT_TOL else "failure"
        jn = worst.split("_")[-2] if "_" in worst else worst   # short joint name
        why = "" if v == "success" else (f"  FAIL because joint '{jn}' rollout={cur[worst]:.3f} "
                                         f"vs oracle={ref_door[worst]:.3f} |Δ|={wdiff:.3f} > tol {JOINT_TOL}")
        return _V(v, "fixture_joint_vs_oracle", target=fxnm,
                  detail=f"worst joint '{jn}': rollout={cur[worst]:.3f} vs oracle={ref_door[worst]:.3f} "
                         f"|Δ|={wdiff:.3f} (tol {JOINT_TOL}){why}")

    if gp in ("turn", "press"):
        fxnm = ref.get("fixture")
        ref_flag = ref.get("state_flag")
        if not fxnm or ref_flag is None:
            return _V("unknown", "fixture_state_unresolved", detail="no on/off state flag resolved")
        fx, _, _ = resolve_fixture(env, subgoal_text)
        cur = _fixture_state_flag(env, fx) if fx is not None else None
        if cur is None:
            return _V("unknown", "fixture_no_state", target=fxnm)
        v = "success" if bool(cur) == bool(ref_flag) else "failure"
        why = "" if v == "success" else f"  FAIL because state rollout={bool(cur)} != oracle={bool(ref_flag)}"
        return _V(v, "fixture_state_vs_oracle", target=fxnm,
                  detail=f"on/off rollout={bool(cur)} vs oracle={bool(ref_flag)}{why}")

    if gp in ("navigate", "search"):
        rxy, ryaw = ref.get("base_xy"), ref.get("base_yaw")
        if rxy is None or ryaw is None:
            return _V("unknown", "nav_no_ref")
        xy, yaw = _base_pose(env)
        if xy is None:
            return _V("unknown", "nav_no_base")
        dist = float(np.linalg.norm(xy - np.asarray(rxy)))
        cos = float(np.cos(yaw - ryaw))
        dist_ok, ori_ok = dist <= NAV_DIST_TOL, cos >= NAV_COS_TOL
        v = "success" if (dist_ok and ori_ok) else "failure"
        why = "" if v == "success" else "  FAIL because " + (
            f"base too far (dist {dist:.3f} > {NAV_DIST_TOL})" if not dist_ok
            else f"orientation off (cos {cos:.3f} < {NAV_COS_TOL})")
        return _V(v, "navigate_vs_oracle",
                  detail=f"base xy dist to oracle={dist:.3f} (tol {NAV_DIST_TOL}); "
                         f"yaw cos={cos:.3f} (tol {NAV_COS_TOL}){why}")

    return _V("unknown", f"primitive_{gp}", detail="no sim predicate for this primitive")


def _receptacle_moved_into(env, milestone_text: str, obj: str) -> str | None:
    """Oracle-time: the receptacle the object ended up in — the OTHER env object it's in contact
    with / inside at the milestone end (via check_obj_in_receptacle), else the text-named one."""
    objs = getattr(env, "objects", {}) or {}
    for n in objs:
        if n == obj:
            continue
        try:
            if OU.check_obj_in_receptacle(env, obj, n):
                return n
        except Exception:
            pass
    return _resolve_receptacle(env, milestone_text, exclude=obj)


def _resolve_receptacle(env, milestone_text: str, exclude: str) -> str | None:
    """Resolve a SECOND env object as the placement receptacle (the object the milestone text names
    that isn't the one being placed). Returns a name or None."""
    objs = getattr(env, "objects", {}) or {}
    names = [n for n in objs if n != exclude]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    txt = (milestone_text or "").lower()
    from subtask_sim_check import _obj_langs
    langs = _obj_langs(env)
    hits = []
    for n in names:
        lang = langs.get(n, n)
        toks = [t for t in lang.replace("_", " ").split() if len(t) > 2]
        if any(t in txt for t in toks) or n.lower() in txt:
            hits.append(n)
    return hits[0] if len(hits) == 1 else None


# Back-compat alias (old name).
milestone_check = subtask_check
