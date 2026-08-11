"""HARDCODED per-task rules that revise System2's output before System1 executes it.

WHY THIS IS A SEPARATE FILE. These are deliberate, task-specific overrides of a learned planner
-- not model behaviour and not a bug fix. Keeping them out of ``combined_eval.py`` means:
  * the unmodified pipeline stays readable and is what runs when no rule matches;
  * every intervention is recorded, so a result is never quietly attributable to a hidden hack
    (see ``scripts/report_rule_interventions.py``);
  * a rule can be added/removed without touching the rollout loop.

Runs on the EXEC path only, once per turn, AFTER ``apply_plan_update`` has merged System2's
``<plan_update>`` into the running checklist. Four kinds of revision:

    plan_revised      the checklist is rewritten (persists! see below)
    subgoal_override  a different instruction is handed to System1 than System2 asked for
    est_override      System2's <estimated_step> is replaced (changes the segment budget)
    subgoal_skipped   the turn runs NO System1 segment; the step is marked done and System2 is
                      re-asked next turn

PLAN REVISIONS PERSIST. The returned plan replaces the caller's running checklist, so it becomes
the context System2 conditions on for every later turn -- exactly like a model-authored
``plan_update``. Two consequences worth stating:
  * rules must be IDEMPOTENT: they are re-evaluated every turn against the latest plan, and a
    revision already present must be a no-op (verified below -- 5/20 PickPlaceDrawerToCounter
    plans already contain the base-alignment step System2 wrote itself).
  * a rule is therefore self-healing: if System2's next ``<plan_update>`` drops the inserted step,
    the rule re-inserts it on the following turn.

Every rule here was checked against the recorded s1-progact270k rollouts before being written --
the match strings and milestone ids are the ones the model actually emits, not guesses.
"""

from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------------------------
# plan-checklist helpers
#
# Format (author: System2's own cold plan; see sys2_client.SYS_PLAN):
#     - [~] M1: pull the drawer further open
#       * [~] M1.1: reach to the drawer handle
#       * [ ] M1.2: pull the drawer open
# A milestone line is "- [mark] Mk:"; its fine steps are "  * [mark] Mk.n:". Marks are
# ' ' (todo), '~' (in progress), 'x' (done).

_MILE_RE = re.compile(r"^\s*-\s*\[(.)\]\s*(M\d+)\s*:\s*(.*)$")
_FINE_RE = re.compile(r"^\s*\*\s*\[(.)\]\s*(M\d+\.\d+)\s*:\s*(.*)$")


def _blocks(plan: str) -> list[dict]:
    """-> [{mark, mid, text, fine:[{mark, fid, text}]}] preserving order."""
    out: list[dict] = []
    for line in (plan or "").splitlines():
        if not line.strip():
            continue
        m = _MILE_RE.match(line)
        if m:
            out.append({"mark": m.group(1), "mid": m.group(2), "text": m.group(3).strip(),
                        "fine": []})
            continue
        f = _FINE_RE.match(line)
        if f and out:
            out[-1]["fine"].append({"mark": f.group(1), "fid": f.group(2),
                                    "text": f.group(3).strip()})
    return out


def _render(blocks: list[dict]) -> str:
    """Inverse of _blocks, in System2's own layout (2-space indent for fine steps)."""
    lines: list[str] = []
    for b in blocks:
        lines.append(f"- [{b['mark']}] {b['mid']}: {b['text']}")
        for f in b["fine"]:
            lines.append(f"  * [{f['mark']}] {f['fid']}: {f['text']}")
    return "\n".join(lines)


def current_fine_id(plan: str) -> str | None:
    """The fine step System2 is working on: first '~', else first ' ' (todo). None if all done.

    Used to key rules on a milestone POSITION (e.g. "M2.1") rather than on subgoal wording, which
    varies run to run (M2.1 was phrased 6 different ways across 20 CoffeeSetupMug episodes).
    """
    b = _blocks(plan)
    for blk in b:
        for f in blk["fine"]:
            if f["mark"] == "~":
                return f["fid"]
    for blk in b:
        for f in blk["fine"]:
            if f["mark"] == " ":
                return f["fid"]
    return None


def _norm(s: str | None) -> str:
    """Lowercase, strip a leading 'continue to ' / trailing 'again', collapse spaces.

    System2 re-issues the same instruction with those affixes when a step is only partway done
    ("continue to turn on the sink faucet handle"), and a rule keyed on the bare phrasing must
    still fire.
    """
    t = (s or "").strip().lower().rstrip(".")
    t = re.sub(r"^continue\s+to\s+", "", t)
    t = re.sub(r"\s+again$", "", t)
    return re.sub(r"\s+", " ", t).strip()


# Matches the retract-arm family. Anchored on the verb so "retract the arm from the microwave"
# and "continue to retract the arm" both hit, while an unrelated subgoal mentioning an arm does
# not.
_RETRACT_RE = re.compile(r"^retract(ing)?\s+(the\s+)?(robot\s+)?arm\b")

# Which tasks have retract steps stripped from the plan. Overridable via SYS2_RULES_STRIP_RETRACT
# (comma list, or empty to disable) so the strip can be scoped per run WITHOUT editing this file --
# MEASURED, one variable at a time, on the same 8 TurnOnMicrowave episodes:
#     keep retract, no rephrase    4/8   mean 4.0 turns
#     strip retract, no rephrase   0/8   mean 9.0 turns  (8/8 max_turns)
#     strip retract + rephrase     1/8   mean 8.6 turns  (7/8 max_turns)
#     keep retract + rephrase      5/8   mean 3.8 turns
# Stripping is what breaks it, and rephrasing does not rescue it -- the cause is structural, not
# wording. Retract is System2's NEXT step after pressing; delete it and the planner has nothing to
# advance to, so it loops on "press ... again" until the turn budget runs out, while the microwave
# only registers as on once the arm withdraws. TurnOnMicrowave is therefore EXCLUDED by default.
# OpenStandMixerHead keeps the strip (3/7 -> 5/7): the push alone completes that task.
_STRIP_RETRACT_TASKS = tuple(
    t.strip() for t in os.environ.get(
        "SYS2_RULES_STRIP_RETRACT", "OpenStandMixerHead").split(",") if t.strip())

# Max consecutive turns a rule may skip System1. NO RULE CURRENTLY SKIPS: the mechanism froze the
# env (success is polled only on executed steps), so retract handling moved to plan surgery
# instead -- see _rule_strip_retract_plan. The plumbing is kept because `skip_s1` is a generic
# capability, but a future skipping rule must account for the frozen-env effect.
MAX_CONSEC_SKIPS = 2

# Max consecutive turns System1 may be handed the SAME subgoal before the repeat cap advances the
# plan (see _rule_repeat_cap). Env-overridable so the threshold can be swept without a code edit.
MAX_SAME_SUBGOAL = int(os.environ.get("SYS2_RULES_MAX_SAME_SUBGOAL", "3"))


# --------------------------------------------------------------------------------------------
# the rules


def _rule_drawer_base_align(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceDrawerToCounter: guarantee a base-alignment step at the head of M1.

    Observed: 15/20 episodes planned M1 as exactly two fine steps (reach handle -> pull open) and
    the robot reached from a pose the drawer was not aligned to; the other 5 episodes had System2
    itself insert "reposition the base to align with the drawer" first. This rule makes that
    third step unconditional.

    Guards:
      * only when M1 has exactly 2 fine steps and none already mentions repositioning/aligning
        the base -> idempotent, and never fights a plan that already has it;
      * only when NO M1 fine step is marked done, so a mid-milestone plan is not reset (the
        inserted step demotes its siblings to todo, which would discard completed work).
    On the turn it fires it ALSO overrides the subgoal, because leaving System2's "reach to the
    drawer handle" to execute first is precisely the ordering the rule exists to prevent.
    """
    iv: list[dict] = []
    if task != "PickPlaceDrawerToCounter":
        return {}
    blocks = _blocks(plan)
    m1 = next((b for b in blocks if b["mid"] == "M1"), None)
    if m1 is None or len(m1["fine"]) != 2:
        return {}
    if any(re.search(r"reposition|align", f["text"], re.I) for f in m1["fine"]):
        return {}          # already present (model-authored or ours from an earlier turn)
    if any(f["mark"] == "x" for f in m1["fine"]):
        return {}          # milestone already in progress -- do not rewind it

    NEW = "reposition the base to align with the drawer"
    old_fine = [f"{f['fid']}: {f['text']}" for f in m1["fine"]]
    m1["fine"] = [{"mark": "~", "fid": "M1.1", "text": NEW},
                  {"mark": " ", "fid": "M1.2", "text": m1["fine"][0]["text"]},
                  {"mark": " ", "fid": "M1.3", "text": m1["fine"][1]["text"]}]
    new_plan = _render(blocks)
    iv.append({"rule": "drawer_base_align", "kind": "plan_revised",
               "detail": "inserted M1.1 base alignment; M1 2 fine steps -> 3",
               "before": old_fine,
               "after": [f"{f['fid']}: {f['text']}" for f in m1["fine"]]})
    out = {"plan": new_plan, "interventions": iv}
    # Hand System1 the new first step this turn, with the requested budget. The detail is synced
    # HERE, by the rule that changed the ACTION -- a stale detail would still describe reaching for
    # the handle. Rules that only rephrase (see _rule_microwave_again) must NOT touch the detail.
    out["subgoal"] = NEW
    out["subgoal_detail"] = NEW
    out["est"] = 75
    iv.append({"rule": "drawer_base_align", "kind": "subgoal_override",
               "detail": "execute the inserted base-alignment step before reaching",
               "before": subgoal, "after": NEW})
    iv.append({"rule": "drawer_base_align", "kind": "est_override",
               "detail": "base alignment budget", "before": est, "after": 75})
    return out


def _rule_strip_retract_plan(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnMicrowave / OpenStandMixerHead: DELETE retract-arm steps from the checklist.

    Plan surgery, at the source. The cold plan reads

        - [~] M1: open the stand mixer head
          * [~] M1.1: reach to the stand mixer head
          * [ ] M1.2: push the stand mixer head open
          * [ ] M1.3: retract the arm        <- removed

    so System2, conditioning on the revised checklist from task_begin onward, never proposes the
    step in the first place. Surviving siblings are renumbered to stay contiguous.

    WHY NOT the previous approach: the first version left the plan alone and skipped System1 on
    each retract subgoal as it arrived. That was wrong twice over. (a) It only reacted after
    System2 had already spent a turn proposing the step, so retract subgoals still filled the
    turn list. (b) A skipped turn never steps the env, and ``_check_success`` is polled per
    executed step -- so skipping FROZE the world and made success undetectable. Measured: 20
    skipped vs 10 executed anyway (a skip cap turned it into a 2-on/1-off cycle), and all four
    TurnOnMicrowave successes latched 10-13 steps INTO an executed retract, i.e. only because the
    cap defeated the rule.

    Deliberately NOT applied to CoffeeSetupMug, which also emits retract subgoals but where the
    retract precedes carrying the mug and is load-bearing.
    """
    if task not in _STRIP_RETRACT_TASKS:
        return {}
    blocks = _blocks(plan)
    removed: list[str] = []
    for b in blocks:
        keep = []
        for f in b["fine"]:
            # Only strip a step that has NOT been executed. One already marked done is history:
            # deleting it would rewrite what happened, and renumbering around it would silently
            # change which id the remaining steps refer to.
            if f["mark"] != "x" and _RETRACT_RE.match(_norm(f["text"])):
                removed.append(f"{f['fid']}: {f['text']}")
                continue
            keep.append(f)
        if len(keep) != len(b["fine"]):
            for n, f in enumerate(keep, start=1):      # renumber to stay contiguous
                f["fid"] = f"{b['mid']}.{n}"
            b["fine"] = keep
    if not removed:
        return {}          # nothing to strip -> idempotent no-op on later turns
    return {"plan": _render(blocks),
            "interventions": [{"rule": "strip_retract_plan", "kind": "plan_revised",
                               "detail": "deleted retract-arm step(s) from the checklist so "
                                         "System2 never proposes them",
                               "before": removed, "after": None}]}


def _rule_flag_retract_emitted(task: str, plan: str, subgoal: str, est, state) -> dict:
    """Record when System2 asks to retract ANYWAY, despite the step being gone from the plan.

    Purely observational -- it does not change behaviour. System2 also reads the video, so it can
    re-propose a retract from what it sees even with no such step in the checklist; this counts
    how often the plan edit fails to prevent that.

    The subgoal is then EXECUTED normally rather than skipped, deliberately: skipping stops the
    env, and ``_check_success`` only advances on executed steps (see _rule_strip_retract_plan).
    Executing is also exactly what the baseline did, so it adds no new risk.
    """
    if task not in ("TurnOnMicrowave", "OpenStandMixerHead"):
        return {}
    if not _RETRACT_RE.match(_norm(subgoal)):
        return {}
    return {"interventions": [{"rule": "retract_emitted_despite_plan", "kind": "observed",
                               "detail": "System2 proposed a retract with no such step in the "
                                         "plan; executing it (NOT skipping -- a skip would "
                                         "freeze the env and block success detection)",
                               "before": subgoal, "after": subgoal}]}


def _rule_microwave_again(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnMicrowave: rewrite "continue to X" -> "X again" in the System1 prompt.

    Both phrasings occur in this task's recorded rollouts, but they are not equally represented:
    "press the microwave start button again" (9x) / "press the start button again" (2x) appear as
    their own subgoals, so the "... again" form is in-distribution for System1, while "continue
    to ..." is the wrapper System2 adds when it judges a step only partway done.

    This changes ONLY the instruction string handed to System1. System2's own subgoal is recorded
    verbatim (turn.json "s2"), the checklist is untouched, and the est/budget is unchanged -- so the
    planner's behaviour on later turns is unaffected except through what System1 actually does.

    Guards: an existing trailing "again" is not doubled, and a subgoal without the "continue to"
    wrapper is left exactly as-is.
    """
    if task != "TurnOnMicrowave":
        return {}
    m = re.match(r"^\s*continue\s+to\s+(.+)$", subgoal or "", re.I)
    if not m:
        return {}
    body = m.group(1).strip().rstrip(".")
    new = body if re.search(r"\bagain$", body, re.I) else f"{body} again"
    if new == subgoal:
        return {}
    return {"subgoal": new,
            "interventions": [{"rule": "microwave_again", "kind": "subgoal_override",
                               "detail": "'continue to X' -> 'X again' (the in-distribution "
                                         "phrasing for System1 on this task)",
                               "before": subgoal, "after": new}]}


def _rule_repeat_cap(task: str, plan: str, subgoal: str, est, state) -> dict:
    """ALL tasks: after MAX_SAME_SUBGOAL consecutive turns on one subgoal, advance to the next
    fine step instead of re-issuing it.

    Measured over the 1000-episode s1-progact270k run, success collapses with the length of the
    longest run of consecutive turns on the same subgoal (normalising away "continue to"/"again"):

        1-2 repeats  495 eps  74% success
        3 repeats     41 eps  51%
        4-8 repeats  213 eps  ~12%
        9+ repeats   251 eps   2.4%   (243 of them end in max_turns)

    92% of all wins come from episodes that never repeated more than twice, so a run of 4+ is
    almost always System1 stuck rather than progress accumulating.

    STIR IS EXEMPT. For continuous-effort actions the repetition IS the task, and stirring is the
    clearest case: StirVegetables won 6 times at repeat depths 3,3,3,6,6,7. Capping it would
    destroy those. (Other continuous-effort verbs -- push-door-closed, turn-knob -- also win deep,
    notably ArrangeTea at depths 9-14; they are NOT exempt here, per the requested rule, so this
    cap is expected to cost those wins. See the note in the module docstring.)

    On trigger: mark the exhausted step done and hand System1 the NEXT fine step. Marking it done
    matters -- it is what makes System2 move on as well, instead of re-proposing the same step and
    forcing us to override every remaining turn.

    NO NEXT STEP -> DO NOTHING. Advancing off the last step would leave System2 with nothing to
    propose; that is exactly how stripping retract from TurnOnMicrowave turned 4/8 into 0/8, with
    the planner looping to max_turns. Better to keep re-issuing than to strand it.
    """
    n = _norm(subgoal)
    if not n:
        return {}
    prev, count = state.get("rep_sg"), state.get("rep_n", 0)
    count = count + 1 if n == prev else 1
    state["rep_sg"], state["rep_n"] = n, count
    if count <= MAX_SAME_SUBGOAL:
        return {}
    if "stir" in n:
        return {"interventions": [{"rule": "repeat_cap", "kind": "cap_exempt",
                                   "detail": f"repeat #{count} but 'stir' is a continuous-effort "
                                             "action -- the repetition IS the task",
                                   "before": subgoal, "after": subgoal}]}
    # Locate the current fine step and the one after it.
    blocks = _blocks(plan)
    flat = [(b, f) for b in blocks for f in b["fine"]]
    cur_id = current_fine_id(plan)
    idx = next((i for i, (_, f) in enumerate(flat) if f["fid"] == cur_id), None)
    if idx is None or idx + 1 >= len(flat):
        return {"interventions": [{"rule": "repeat_cap", "kind": "cap_declined",
                                   "detail": f"repeat #{count} but no next fine step exists -- "
                                             "advancing would strand System2 with nothing to "
                                             "propose (see TurnOnMicrowave retract finding)",
                                   "before": subgoal, "after": subgoal}]}
    _, cur_f = flat[idx]
    _, nxt_f = flat[idx + 1]
    cur_f["mark"] = "x"                      # so System2 advances too, not just System1
    nxt_f["mark"] = "~"
    for b in blocks:                          # close a milestone whose steps are all done
        if b["fine"] and all(f["mark"] == "x" for f in b["fine"]):
            b["mark"] = "x"
    new_sg = nxt_f["text"]
    state["rep_sg"], state["rep_n"] = _norm(new_sg), 1
    return {"plan": _render(blocks), "subgoal": new_sg, "subgoal_detail": new_sg,
            "interventions": [
                {"rule": "repeat_cap", "kind": "plan_revised",
                 "detail": f"repeat #{count} > MAX_SAME_SUBGOAL={MAX_SAME_SUBGOAL}: marked "
                           f"{cur_f['fid']} done, advanced to {nxt_f['fid']}",
                 "before": f"{cur_f['fid']}: {cur_f['text']}",
                 "after": f"{nxt_f['fid']}: {nxt_f['text']}"},
                {"rule": "repeat_cap", "kind": "subgoal_override",
                 "detail": "hand System1 the next fine step instead of the repeated one",
                 "before": subgoal, "after": new_sg}]}


def _rule_sink_faucet_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnSinkFaucet: the turn-the-handle subgoal always gets 100 steps.

    System2 budgeted this 50 (18x) or re-issued it as "continue to ..." (33x) -- i.e. it kept
    running out of segment before the handle was over. 100 covers it in one segment.
    """
    if task != "TurnOnSinkFaucet":
        return {}
    if _norm(subgoal) != "turn on the sink faucet handle":
        return {}
    if est == 100:
        return {}
    return {"est": 100,
            "interventions": [{"rule": "sink_faucet_est", "kind": "est_override",
                               "detail": "handle rotation needs a full segment",
                               "before": est, "after": 100}]}


def _rule_coffee_m2_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: EVERY M2.x step gets 100 steps -- except a retract-arm step.

    Keyed on the milestone POSITION rather than wording: M2.1 alone was phrased 6 different ways
    across 20 episodes ("lift and carry the mug to the coffee machine dispenser", "carry and place
    the red mug under ...", ...), so text matching would be fragile. Across M2 System2 budgeted 50
    or 75 and essentially never 100 (M2.1: 50x16/75x4, M2.2: 50x32/75x4, M2.3: 50x13).

    The retract exemption is by request: unlike the carry/lower/release steps, a retract is a short
    move away from the machine and does not need a full segment. NOTE this task keeps its retract
    steps -- only TurnOnMicrowave/OpenStandMixerHead strip them, because here the retract is
    load-bearing (it precedes carrying the mug).
    """
    if task != "CoffeeSetupMug":
        return {}
    fid = current_fine_id(plan)
    if not fid or not fid.startswith("M2."):
        return {}
    if _RETRACT_RE.match(_norm(subgoal)):
        return {"interventions": [{"rule": "coffee_m2_est", "kind": "est_exempt",
                                   "detail": f"{fid} is a retract-arm step -- left at System2's "
                                             "estimate, no full segment needed",
                                   "before": est, "after": est}]}
    if est == 100:
        return {}
    return {"est": 100,
            "interventions": [{"rule": "coffee_m2_est", "kind": "est_override",
                               "detail": f"{fid} (M2 milestone) needs a full segment",
                               "before": est, "after": 100}]}


# Order matters: the plan rewrite runs first so later rules see the revised checklist.
_RULES = (_rule_drawer_base_align, _rule_strip_retract_plan, _rule_flag_retract_emitted,
          _rule_microwave_again, _rule_repeat_cap, _rule_sink_faucet_est,
          _rule_coffee_m2_est)

# Tasks with task-SPECIFIC rules. _rule_repeat_cap additionally applies to EVERY task.
TASKS_WITH_RULES = ("PickPlaceDrawerToCounter", "TurnOnMicrowave", "TurnOnSinkFaucet",
                    "OpenStandMixerHead", "CoffeeSetupMug", "<all: repeat_cap>")


def apply_rules(task: str, *, plan: str, subgoal: str, subgoal_detail: str, est,
                state: dict | None = None) -> dict:
    """Revise one turn's System2 output. Returns what the caller should actually use.

    ``state`` is a per-EPISODE dict the caller threads through every turn (skip counters live
    there). Returns:
        {plan, subgoal, subgoal_detail, est, skip_s1, interventions:[...]}
    ``interventions`` is empty when nothing fired -- that is the no-rule path, byte-identical to
    running without this module.
    """
    st = state if state is not None else {}
    cur = {"plan": plan, "subgoal": subgoal, "subgoal_detail": subgoal_detail, "est": est,
           "skip_s1": False}
    ivs: list[dict] = []
    for fn in _RULES:
        r = fn(task, cur["plan"], cur["subgoal"], cur["est"], st)
        if not r:
            continue
        ivs.extend(r.get("interventions", []))
        for k in ("plan", "subgoal", "est", "skip_s1"):
            if k in r:
                cur[k] = r[k]
        if cur["skip_s1"]:
            break            # nothing else applies to a turn that runs no segment
    st["consec_skips"] = (st.get("consec_skips", 0) + 1) if cur["skip_s1"] else 0
    cur["interventions"] = ivs
    return cur
