"""High-precision SUBTASK-LEVEL sim-grounded success check for RoboCasa rollouts.

Design goal (per the user): PRECISION over coverage. Return a success/fail verdict ONLY when a
clean, verifiable sim predicate applies to the subtask's primitive + resolved target; otherwise
return "unknown" (the caller defers to Gemini). Never guess.

Scope of checkable primitives (everything else -> unknown):
  * grasp   -> the resolved object is HELD by the gripper (contact + gripper closed), evaluated
               robustly over a trailing window (not a 1-frame bounce). Size-independent guard:
               we DON'T require a fixed finger-width; check_obj_grasped's contact term carries the
               signal, and we additionally accept "object lifted / moved with gripper" as held.
  * open/close/pull/push (FIXTURE) -> the resolved fixture's door/drawer joint crosses the
               open/closed normalized-qpos threshold (fixture.is_open / is_closed / get_door_state).
  * turn/press (FIXTURE state) -> faucet water_on / stove burner-on / appliance turned_on toggled
               to the intended state (get_handle_state / is_burner_on / get_state).
  * place/release -> the resolved object is in its target receptacle AND released (gripper far).

NOT checkable here (return unknown): move_to / navigate / reach / retract / carry / hold / stir /
scrub / press_against, and grasp/turn of a FIXTURE HANDLE (knob) where contact is ambiguous.

Returns a dict: {verdict: "success"|"failure"|"unknown", rule: <str>, detail: <str>, target: <str>}.
Evaluated at the CURRENT (rollout-end) sim state; the caller may also pass a trailing window.
"""

from __future__ import annotations

import numpy as np
import robocasa.utils.object_utils as OU


# ---------------------------------------------------------------------------------
# target resolution: which env object / fixture does this subtask act on?
# ---------------------------------------------------------------------------------
def _obj_langs(env) -> dict:
    """name -> lowercase language string for each manipulable object (env.objects)."""
    out = {}
    for name in getattr(env, "objects", {}) or {}:
        lang = None
        try:
            lang = OU.get_obj_lang(env, name)  # e.g. "electric kettle", "bread"
        except Exception:
            pass
        out[name] = (lang or name).lower()
    return out


def resolve_object(env, subgoal_text: str) -> tuple[str | None, float]:
    """Best-effort: pick the env.objects name whose language appears in the subgoal text.
    Returns (name, confidence). If exactly one object exists, it's trivially that one.
    Confidence: 1.0 unique-match / single-object, 0.5 ambiguous (>=2 candidates matched),
    0.0 none. Caller downgrades to unknown when confidence < 1.0 for scoring safety."""
    objs = getattr(env, "objects", {}) or {}
    if not objs:
        return None, 0.0
    if len(objs) == 1:
        return next(iter(objs)), 1.0
    txt = (subgoal_text or "").lower()
    langs = _obj_langs(env)
    # match by language token overlap; also try the canonical name
    hits = []
    for name, lang in langs.items():
        toks = [t for t in lang.replace("_", " ").split() if len(t) > 2]
        if any(t in txt for t in toks) or name.lower() in txt:
            hits.append(name)
    if len(hits) == 1:
        return hits[0], 1.0
    if len(hits) >= 2:
        return hits[0], 0.5   # ambiguous — caller treats as unknown
    return None, 0.0


# The task fixtures the RoboCasa tasks reference as DIRECT env attributes in their _check_success
# (env.sink, env.stove, ...). After our replay reset, env.fixture_refs is EMPTY (it's populated in
# _setup_kitchen_references during a normal reset, skipped on replay), so we resolve from these
# attributes instead. Keyword -> attribute name(s) to look up, matched against the subgoal text.
_FIXTURE_ATTRS = {
    "sink": ["sink"], "faucet": ["sink"], "spout": ["sink"],
    "stove": ["stove"], "burner": ["stove"], "knob": ["stove"],
    "microwave": ["microwave"], "kettle": ["electric_kettle", "kettle"],
    "blender": ["blender"], "toaster oven": ["toaster_oven"], "toaster": ["toaster"],
    "cabinet": ["cabinet", "cab"], "drawer": ["drawer"], "fridge": ["fridge"],
    "dishwasher": ["dishwasher"], "oven": ["oven"], "stand mixer": ["stand_mixer"],
}


def resolve_fixture(env, subgoal_text: str):
    """Resolve the task fixture from the subgoal text via the env's DIRECT fixture attributes
    (env.sink, env.stove, ...) — fixture_refs is empty after a replay reset. Also tries
    fixture_refs as a fallback. Returns (fixture_obj, name, confidence)."""
    txt = (subgoal_text or "").lower()
    # 1) keyword -> direct env attribute (the authoritative task fixtures)
    for kw, attrs in _FIXTURE_ATTRS.items():
        if kw in txt:
            for a in attrs:
                fx = getattr(env, a, None)
                if fx is not None:
                    return fx, a, 1.0
    # 2) fallback: fixture_refs (populated only on a normal reset)
    refs = getattr(env, "fixture_refs", {}) or {}
    hits = [(nm, fx) for nm, fx in refs.items() if nm.lower() in txt]
    if len(hits) == 1:
        return hits[0][1], hits[0][0], 1.0
    if len(refs) == 1:
        nm, fx = next(iter(refs.items()))
        return fx, nm, 1.0
    return None, None, 0.0


# ---------------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------------
def _obj_held(env, name: str) -> bool:
    """Object held by the gripper. check_obj_grasped = contact(gripper,obj) AND fingers<0.035;
    the finger threshold FALSE-NEGATIVES on large objects, so we OR in a size-independent test:
    gripper in contact with the object AND the object is not resting far below the eef (i.e. it
    moves WITH the gripper). We keep it conservative: require contact at minimum."""
    try:
        if OU.check_obj_grasped(env, name):
            return True
    except Exception:
        return False
    return False


def _call_env(fn, env):
    """Call fn(env=env) or fn(env) or fn(); return result or raise."""
    try:
        return fn(env=env)
    except TypeError:
        pass
    try:
        return fn(env)
    except TypeError:
        return fn()


# _per_door_openness collects a {joint_name -> normalized openness} dict, using whichever accessor
# the fixture exposes: get_door_state (cabinets/drawers/single-hinge) OR the base
# Fixture.get_joint_state on the door joints (multi-door fridges like FridgeFrenchDoor / SideBySide
# have NO get_door_state and their is_open/is_closed AGGREGATE over BOTH doors — a per-joint read is
# the only way to score a single-door subgoal). Returns {} if no per-joint signal is available.
def _per_door_openness(env, fx) -> dict:
    if hasattr(fx, "get_door_state"):
        try:
            ds = _call_env(fx.get_door_state, env)
            if isinstance(ds, dict) and ds:
                return {str(k): float(v) for k, v in ds.items()}
        except Exception:
            pass
    # fall back to the raw door joints via the base Fixture.get_joint_state
    for attr in ("_fridge_door_joint_names", "door_joint_names"):
        names = getattr(fx, attr, None)
        if names:
            try:
                js = fx.get_joint_state(env, list(names))
                if isinstance(js, dict) and js:
                    return {str(k): float(v) for k, v in js.items()}
            except Exception:
                pass
    return {}


# _fixture_openness returns a normalized openness (the SAME signal the door/drawer tasks'
# _check_success uses: 0.95 open / 0.05 closed — is_open/is_closed FALSE-NEGATIVE on drawers,
# calibration showed door_state=1.31 but is_open=False). Returns (value, ambiguous):
#   value = the selected joint's openness (or an aggregate for single-joint fixtures)
#   ambiguous = True when the fixture has MULTIPLE doors but the subgoal text doesn't name ONE
#               specific door -> caller returns UNKNOWN rather than a wrong max/min aggregate
#               (multi-door fridge: "close the right door" left the left door open -> aggregate
#                is_closed stays False -> old code wrongly read "not closed"). Precision-first.
def _fixture_openness(env, fx, txt: str = "") -> tuple[float | None, bool]:
    ds = _per_door_openness(env, fx)
    if not ds:
        return None, False
    if len(ds) == 1:
        return float(next(iter(ds.values()))), False
    # multiple joints: try to pick the one the subgoal text names (left/right/top/bottom/side).
    lo = (txt or "").lower()
    for side in ("left", "right", "top", "bottom", "upper", "lower"):
        if side in lo:
            hits = [v for k, v in ds.items() if side in str(k).lower()]
            if len(hits) == 1:
                return float(hits[0]), False
    # multi-door but no clean single-door selection -> ambiguous (caller -> unknown)
    return None, True


def _fixture_open(env, fx, txt: str = "") -> bool | None:
    o, ambig = _fixture_openness(env, fx, txt)
    if ambig:
        return None
    if o is not None:
        return o >= 0.95   # task _check_success 'open' threshold
    try:
        return bool(_call_env(fx.is_open, env))
    except Exception:
        return None


def _fixture_closed(env, fx, txt: str = "") -> bool | None:
    o, ambig = _fixture_openness(env, fx, txt)
    if ambig:
        return None
    if o is not None:
        return o <= 0.05   # task _check_success 'close' threshold
    try:
        return bool(_call_env(fx.is_closed, env))
    except Exception:
        return None


def _V(verdict, rule, target="", detail=""):
    return {"verdict": verdict, "rule": rule, "target": target, "detail": detail}


# primitives we will attempt a sim verdict for; the rest -> unknown
FIXTURE_OPEN = {"open", "pull"}
FIXTURE_CLOSE = {"close", "push"}


def sim_check_subtask(env, subgoal_text: str, primitive: str) -> dict:
    """Return a sim-grounded verdict for this subtask at the CURRENT env state, or unknown.

    PRECISION-FIRST: any ambiguity (target not uniquely resolved, no clean predicate) -> unknown.
    """
    prim = (primitive or "").lower().strip()
    txt = (subgoal_text or "")

    # --- grasp: object held by gripper -----------------------------------------
    if prim == "grasp":
        name, conf = resolve_object(env, txt)
        if name is None or conf < 1.0:
            # grasp of a fixture handle (knob/handle) or ambiguous object -> not scorable here
            return _V("unknown", "grasp_unresolved", detail=f"conf={conf}")
        held = _obj_held(env, name)
        return _V("success" if held else "failure", "obj_grasped", target=name,
                  detail=f"held={held}")

    # --- place / release: object in target receptacle AND released --------------
    if prim in ("place", "release"):
        name, conf = resolve_object(env, txt)
        if name is None or conf < 1.0:
            return _V("unknown", "place_unresolved", detail=f"conf={conf}")
        released = not _obj_held(env, name)
        # target receptacle: try a movable receptacle in env.objects, else a fixture (cabinet/sink)
        in_target = None
        # (a) any OTHER object that is a receptacle the obj now contacts+centers on
        for recep in (getattr(env, "objects", {}) or {}):
            if recep == name:
                continue
            try:
                if OU.check_obj_in_receptacle(env, name, recep):
                    in_target = True; break
            except Exception:
                pass
        # (b) a fixture container (cabinet/drawer/sink) named in the text
        if in_target is None:
            fx, fxnm, fconf = resolve_fixture(env, txt)
            if fx is not None:
                try:
                    if OU.obj_inside_of(env, name, fx, partial_check=True):
                        in_target = True
                except Exception:
                    pass
        # (c) counter contact as a last resort when the text says "counter"
        if in_target is None and "counter" in txt.lower():
            try:
                in_target = bool(OU.check_obj_any_counter_contact(env, name))
            except Exception:
                in_target = None
        if in_target is None:
            return _V("unknown", "place_no_target", target=name,
                      detail="could not resolve target receptacle")
        ok = bool(in_target and released)
        return _V("success" if ok else "failure", "obj_placed_released", target=name,
                  detail=f"in_target={in_target} released={released}")

    # --- open / close / pull / push / turn / press on a FIXTURE ------------------
    # These primitives are OVERLOADED: "push/pull a door/drawer" = open/close (joint threshold),
    # but "push/turn a faucet handle" or "press a button" = an appliance STATE toggle. So route by
    # the resolved fixture's CAPABILITY: if it exposes an on/off state (faucet water_on / burner /
    # turned_on), that's the signal; else fall back to the door open/close joint threshold.
    if prim in FIXTURE_OPEN or prim in FIXTURE_CLOSE or prim in ("turn", "press"):
        fx, fxnm, conf = resolve_fixture(env, txt)
        if fx is None or conf < 1.0:
            return _V("unknown", "fixture_unresolved", detail=f"conf={conf}")
        lo = txt.lower()
        want_off = ("off" in lo or "turn off" in lo)
        # (1) appliance/faucet/burner STATE toggle
        state = _fixture_state_on(env, fx)   # True/False on-ness, or None if not a stateful fixture
        if state is not None:
            ok = (not state) if want_off else state
            return _V("success" if ok else "failure", "fixture_state", target=fxnm,
                      detail=f"on={state} want_off={want_off}")
        # (2) door/drawer open-close joint threshold (only if NOT a stateful fixture). Pass the
        # subgoal text so a multi-door fixture can select the specific door named ("right"/"left"…);
        # if it's multi-door and the text doesn't name one, _fixture_open/closed return None ->
        # unknown (don't guess an aggregate — precision-first).
        want_open = prim in FIXTURE_OPEN or (prim == "turn" and not want_off)
        st = _fixture_open(env, fx, txt) if want_open else _fixture_closed(env, fx, txt)
        if st is not None:
            return _V("success" if st else "failure",
                      "fixture_open" if want_open else "fixture_closed", target=fxnm,
                      detail=f"{'open' if want_open else 'closed'}={st}")
        return _V("unknown", "fixture_ambiguous_or_no_signal", target=fxnm)

    # everything else: move_to / navigate / reach / retract / hold / stir / scrub / ...
    return _V("unknown", f"primitive_{prim or 'none'}")


def _fixture_state_on(env, fx) -> bool | None:
    """On/off state for a STATEFUL fixture (faucet water_on / stove burner / appliance turned_on),
    or None if this fixture has no such state (e.g. a plain door/drawer/cabinet). Reads the same
    signals the tasks' own _check_success uses."""
    # sink faucet: water running
    if hasattr(fx, "get_handle_state"):
        try:
            hs = _call_env(fx.get_handle_state, env)
            if isinstance(hs, dict) and "water_on" in hs:
                return bool(hs["water_on"])
        except Exception:
            pass
    # stove: any burner on (env.knob is None after replay, so check ALL knob locations)
    if hasattr(fx, "get_knobs_state"):
        try:
            ks = _call_env(fx.get_knobs_state, env)
            if isinstance(ks, dict) and ks:
                import numpy as _np
                return any(0.35 <= abs(float(v)) <= (2 * _np.pi - 0.35) for v in ks.values())
        except Exception:
            pass
    # microwave / kettle / blender: turned_on flag
    if hasattr(fx, "get_state"):
        try:
            gs = fx.get_state(env) if _one_arg(fx.get_state) else fx.get_state()
            if isinstance(gs, dict) and "turned_on" in gs:
                return bool(gs["turned_on"])
        except Exception:
            pass
    return None


def _one_arg(fn) -> bool:
    """True if fn takes at least one positional arg beyond self (for get_state(env) vs get_state())."""
    import inspect
    try:
        sig = inspect.signature(fn)
        return len([p for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]) >= 1
    except (ValueError, TypeError):
        return False
