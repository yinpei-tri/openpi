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


def current_milestone_text(plan: str) -> str:
    """Text of the milestone that owns the current fine step ("" if none).

    Lets a rule key on the PHASE of a composite task rather than the task name -- e.g. the
    "turn on the sink faucet" milestone inside RinseSinkBasin is the same physical act as the whole
    of the atomic TurnOnSinkFaucet task.
    """
    cur = current_fine_id(plan)
    if not cur:
        return ""
    for b in _blocks(plan):
        if any(f["fid"] == cur for f in b["fine"]):
            return b["text"].strip().lower()
    return ""


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
# Tighter cap for a pure "reach to/for X": positioning either converges quickly or not at all.
REACH_SAME_SUBGOAL = int(os.environ.get("SYS2_RULES_MAX_SAME_REACH", "2"))
# OpenStandMixerHead est_length floor. est_length is a POLICY CONDITIONING tag (rendered into
# System1's prompt as "Estimated Length"), not only a budget multiplier -- see _rule_mixer_est_floor.
MIXER_EST_FLOOR = int(os.environ.get("SYS2_RULES_MIXER_EST_FLOOR", "75"))
# est_length BUCKET LADDER, as used by the training data. A bump means "the next bucket up".
EST_BUCKETS = (50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500,
               600, 800, 1000, 1300)
# At and above this, est is left alone: the bump is a precision aid for short/medium motions, and
# the rungs above 500 are large jumps on spans that are already long.
EST_BUMP_CEILING = int(os.environ.get("SYS2_RULES_EST_BUMP_CEILING", "500"))
# Which tasks get the bump. "all" = every task; the default is the three SINGLE-CONTACT PRECISION
# tasks, measured one variable at a time (v3 -> v4, same 20 episodes per task, bump the only change):
#     TurnOnMicrowave   11 -> 15   (+4)   press a button
#     GetToastedBread   13 -> 16   (+3)   press a lever
#     TurnOnSinkFaucet  17 -> 19   (+2)   turn a handle
#     OpenStandMixerHead 20 -> 20  ( 0)   already saturated (its own est floor covers it)
#     CoffeeSetupMug    11 -> 11   ( 0)
#     PickPlaceDrawerToCounter 19 -> 18 (-1)
#     ArrangeTea         9 ->  5   (-4)   <-- long-horizon, TURN-bound not precision-bound
# ArnrangeTea shows the boundary: every one of its failures is max_turns at 23t, and the bump raised
# steps/turn 112 -> 140 and steps/episode 2282 -> 3038 while reaching the SAME 5.2 of 6.4 milestones.
# Slower, more careful motion buys nothing when the binding constraint is the turn budget, and costs
# the episodes that were finishing just inside it.
# NOT extended to composite tasks by milestone. A milestone gate was tried (matching "turn on the
# sink faucet" / "press the microwave start button" inside RinseSinkBasin, WashLettuce,
# SteamInMicrowave, PreSoakPan, WashFruitColander, WaffleReheat -- 102 episodes) and reverted before
# measurement, because a probe on two similar tasks came back flat:
#     PickPlaceCounterToCabinet  15/20 -> 15/20   (fixed 3, broke 3)
#     SlideDishwasherRack        15/20 -> 16/20   (fixed 2, broke 1)
# The bump was demonstrably ACTIVE there -- steps/turn 44 -> 56 and 72 -> 115 -- so System1 did move
# slower and the outcome simply did not follow. Those tasks fail by running out of turns
# (all-max_turns), not by stopping short of a small contact, which is the only failure the bump
# addresses. The composite sink/microwave tasks are long and max_turns-prone in the same way, so the
# extension was judged unsupported. Recoverable from 61679dd if it is worth measuring later.
_EST_BUMP_TASKS = tuple(
    t.strip() for t in os.environ.get(
        "SYS2_RULES_EST_BUMP_TASKS",
        "TurnOnMicrowave,TurnOnSinkFaucet,GetToastedBread").split(",") if t.strip())


def bump_est(est):
    """Next bucket up, or ``est`` unchanged at/above EST_BUMP_CEILING (or if not a number).

    A value that is not itself a bucket (System2 occasionally writes e.g. 60) snaps UP to the next
    bucket, which is the in-distribution neighbour rather than an invented number.
    """
    if not isinstance(est, int) or est >= EST_BUMP_CEILING:
        return est
    return next((b for b in EST_BUCKETS if b > est), est)


_REACH_RE = re.compile(r"^reach\s+(to|for)\b")


# --------------------------------------------------------------------------------------------
# the rules


# Drawer milestone shape: a "reach ... drawer handle" step followed by a "pull/open ... drawer"
# step. Matching the SUBGOAL TEXT rather than a task name generalises the rule from the atomic
# PickPlaceDrawerToCounter to every composite task that opens a drawer (measured drawer-subgoal
# turns: SetUpCuttingStation 172, DeliverStraw 116, OpenDrawer 78, CuttingToolSelection 12).
_DRAWER_REACH_RE = re.compile(r"(reach|move|extend).*drawer|drawer handle")
_DRAWER_PULL_RE = re.compile(r"(pull|open|slide).*drawer|drawer.*(open|out)")
# Already-has-base-alignment guard. MUST cover the model's own phrasings or the rule would insert a
# duplicate: it writes "reposition the base to face the drawer" (53x) and "adjust the base to face
# the drawer" (51x) as well as "reposition the base to align with the drawer" (19x).
_BASE_ALIGN_RE = re.compile(r"reposition|align|adjust|face the|orient")
BASE_ALIGN_TEXT = "reposition the base to align with the drawer"

# Which tasks get the base-alignment insert. SCOPED to the atomic task by request: the rule was
# briefly gated on subgoal text alone, which would also have fired on every composite task that
# opens a drawer (SetUpCuttingStation 172 drawer-subgoal turns, DeliverStraw 116, OpenDrawer 78,
# CuttingToolSelection 12). Its 1/6 -> 6/6 result was measured ONLY on PickPlaceDrawerToCounter,
# so widening it was an untested extrapolation. Add task names here (or via the env var) to widen
# once there is evidence for it.
_DRAWER_ALIGN_TASKS = tuple(
    t.strip() for t in os.environ.get(
        "SYS2_RULES_DRAWER_ALIGN_TASKS", "PickPlaceDrawerToCounter").split(",") if t.strip())


def _rule_drawer_base_align(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceDrawerToCounter (see _DRAWER_ALIGN_TASKS): guarantee a base-alignment step in front
    of a reach->pull drawer milestone.

    Observed there: 15/20 episodes planned the drawer milestone as exactly two fine steps (reach
    handle -> pull open) and the robot reached from a pose the drawer was not aligned to; the other
    5 had System2 insert a base-alignment step itself. Forcing that step took the previously-failing
    episodes from 1/6 (rules-off control) to 6/6 -- the strongest single effect measured.

    The milestone is located by SHAPE rather than assumed to be M1, so the rule still works if the
    planner numbers the drawer milestone differently; but the TASK gate is deliberate, because the
    6/6 result was measured on this task alone.

    Guards:
      * the milestone must have exactly 2 fine steps matching reach-drawer then pull-drawer;
      * none may already mention repositioning/aligning/adjusting/facing -> idempotent, and never
        duplicates a step System2 wrote itself;
      * none may be marked done, so a milestone already under way is not rewound (the insert
        demotes its siblings to todo, which would discard completed work).
    The subgoal is overridden ONLY when the current step is the one being displaced -- otherwise
    the plan is fixed up for later and this turn runs untouched.
    """
    if task not in _DRAWER_ALIGN_TASKS:
        return {}
    blocks = _blocks(plan)
    cur_id = current_fine_id(plan)
    for b in blocks:
        if len(b["fine"]) != 2:
            continue
        t0, t1 = b["fine"][0]["text"].lower(), b["fine"][1]["text"].lower()
        if not (_DRAWER_REACH_RE.search(t0) and _DRAWER_PULL_RE.search(t1)):
            continue
        if any(_BASE_ALIGN_RE.search(f["text"].lower()) for f in b["fine"]):
            continue
        if any(f["mark"] == "x" for f in b["fine"]):
            continue
        old_fine = [f"{f['fid']}: {f['text']}" for f in b["fine"]]
        displaced_first = (cur_id == b["fine"][0]["fid"])
        mid = b["mid"]
        b["fine"] = [{"mark": "~" if displaced_first else " ", "fid": f"{mid}.1",
                      "text": BASE_ALIGN_TEXT},
                     {"mark": " ", "fid": f"{mid}.2", "text": b["fine"][0]["text"]},
                     {"mark": " ", "fid": f"{mid}.3", "text": b["fine"][1]["text"]}]
        iv = [{"rule": "drawer_base_align", "kind": "plan_revised",
               "detail": f"inserted {mid}.1 base alignment; {mid} 2 fine steps -> 3",
               "before": old_fine,
               "after": [f"{f['fid']}: {f['text']}" for f in b["fine"]]}]
        out = {"plan": _render(blocks), "interventions": iv}
        if displaced_first:
            # Execute the inserted step NOW: letting System2's "reach to the drawer handle" run
            # first is exactly the ordering this rule exists to prevent. The detail is synced here,
            # by the rule that changed the ACTION (a stale detail would still describe reaching).
            out["subgoal"] = BASE_ALIGN_TEXT
            out["subgoal_detail"] = BASE_ALIGN_TEXT
            out["est_proposal"] = 75
            iv.append({"rule": "drawer_base_align", "kind": "subgoal_override",
                       "detail": "execute the inserted base-alignment step before reaching",
                       "before": subgoal, "after": BASE_ALIGN_TEXT})
            iv.append({"rule": "drawer_base_align", "kind": "est_override",
                       "detail": "base alignment budget", "before": est, "after": 75})
        return out
    return {}


def _rule_strip_retract_plan(task: str, plan: str, subgoal: str, est, state) -> dict:
    """_STRIP_RETRACT_TASKS (default: OpenStandMixerHead only): DELETE retract-arm steps from the
    checklist.

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
    # Only where the strip actually ran: this rule's whole claim is "System2 asked to retract even
    # though the step is gone from the plan". On a task that KEEPS its retract step (TurnOnMicrowave,
    # excluded from the strip since 3f6f1f2) that claim is false, and logging it would misreport
    # normal planning as an anomaly.
    if task not in _STRIP_RETRACT_TASKS:
        return {}
    if not _RETRACT_RE.match(_norm(subgoal)):
        return {}
    return {"interventions": [{"rule": "retract_emitted_despite_plan", "kind": "observed",
                               "detail": "System2 proposed a retract with no such step in the "
                                         "plan; executing it (NOT skipping -- a skip would "
                                         "freeze the env and block success detection)",
                               "before": subgoal, "after": subgoal}]}


def _rule_microwave_again(task: str, plan: str, subgoal: str, est, state) -> dict:
    """ANY microwave subgoal: rewrite "continue to press X" -> "press X again" for System1.

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
    # ANY task whose subgoal is about the MICROWAVE (not just the atomic task): SteamInMicrowave
    # 145 microwave-button turns, WaffleReheat 36, PrepareCoffee 11. Deliberately NOT global -- the
    # "start button" phrasing also belongs to the coffee machine (129 turns), and this rephrase is
    # only motivated where the "... again" form was observed in-distribution.
    # Gate: the SUBGOAL names the microwave, OR the TASK does. Both are needed -- the atomic task
    # drops the word ("press the start button" 6x, "press the start button again" 2x) so a
    # subgoal-only gate would miss its own re-issues, while a task-only gate would miss the
    # composite tasks (SteamInMicrowave 145 microwave-button turns, WaffleReheat 36).
    if "microwave" not in _norm(subgoal) and "microwave" not in (task or "").lower():
        return {}
    # PRESS only, by request: "continue to press X" -> "press X again". The "... again" form was
    # observed in-distribution specifically for the button press ("press the microwave start button
    # again" 9x, "press the start button again" 2x); there is no such evidence for other verbs, so
    # "continue to push the microwave door closed" is left exactly as System2 wrote it.
    m = re.match(r"^\s*continue\s+to\s+(press\b.+)$", subgoal or "", re.I)
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

    Two thresholds: MAX_SAME_SUBGOAL (3) in general, REACH_SAME_SUBGOAL (2) for a pure
    "reach to/for X" -- see the comment at the cap selection below.

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
    # A pure REACH gets a tighter cap. Reaching is positioning, not effort accumulation: if System1
    # has not arrived in two turns it is not converging, and more reaching only burns turns. The
    # clearest case is ArrangeTea ep0, which spent turns 9-22 on "continue to reach to the right
    # cabinet door" -- 14 repeats -- while "push the right cabinet door closed" sat untouched.
    # Matches "reach to"/"reach for" only, NOT compounds like "reach and grasp the kettle", where
    # the grasp is the real work.
    cap = REACH_SAME_SUBGOAL if _REACH_RE.match(n) else MAX_SAME_SUBGOAL
    if count <= cap:
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
    """ANY task: the turn-on-the-sink-faucet-handle subgoal always gets 100 steps.

    System2 budgeted this 50 (18x) or re-issued it as "continue to ..." (33x) -- i.e. it kept
    running out of segment before the handle was over. 100 covers it in one segment.
    """
    # ANY task: gated on the SUBGOAL, not the task name, so the composite tasks that turn on the
    # same faucet are covered too (measured faucet-subgoal turns: WashLettuce 138, RinseSinkBasin
    # 99, WashFruitColander 94, PreSoakPan 89, vs 51 on the atomic task). The phrasing is stable --
    # "turn on the sink faucet handle" accounts for 471 of the turns across all five.
    if _norm(subgoal) != "turn on the sink faucet handle":
        return {}
    if est == 100:
        return {}
    return {"est_proposal": 100,
            "interventions": [{"rule": "sink_faucet_est", "kind": "est_proposed",
                               "detail": "handle rotation needs a full segment",
                               "before": est, "after": 100}]}


def _rule_est_bump(task: str, plan: str, subgoal: str, est, state) -> dict:
    """_EST_BUMP_TASKS (default: the 3 single-contact precision tasks): raise est_length one bucket -- 50->75, 75->100, ... 400->500.
    Retract-arm subgoals are exempt, and 500+ is left alone.

    MECHANISM. est_length is not only a budget multiplier: it is a POLICY CONDITIONING tag, rendered
    into System1's prompt as "Estimated Length". A larger value tells System1 the motion should take
    longer, so it commands smaller per-step deltas -- it moves more slowly and therefore more
    precisely. That is the effect being exploited here, not the extra step allowance.

    EVIDENCE, on TurnOnMicrowave (all 20 episodes, same episodes as the baseline):
        baseline (no rules)            12/20
        rephrase + repeat cap          11/20
        + est bump 50->75, 75->100     18/20   press segments 35.7 -> 44.6 mean steps,
                                               task_finish failures 9 -> 2
    The failure there was System1 stopping short of depressing the button while its progress head
    reported stop_rule; the longer conditioned motion actually depresses it. Consistent with
    OpenStandMixerHead, whose est floor of 75 (from System2's flat 50) took it 13/20 -> 20/20.

    SCOPE CAUTION. That evidence is ONE task and ONE failure mode (under-shooting a contact). A task
    that fails for the opposite reason -- overshooting, or releasing late -- could regress, and every
    segment that runs to budget gets up to 2x more sim steps. Hence _EST_BUMP_TASKS, so this can be
    measured broadly before being trusted broadly.

    PROGREG INTERACTION, worth knowing before running the regression head with this on: for
    progreg, est_length ALSO selects the progress threshold
    (stop_criterion.PROGREG_THRESH_BY_EST -- 50/75 -> 0.88, 100 -> 0.92, else 0.95), so a bump makes
    the stop rule STRICTER (75->100 raises the bar 0.88 -> 0.92). These runs are progact, where the
    threshold is a flat 0.95 and only the conditioning and budget channels apply.
    """
    if "all" not in _EST_BUMP_TASKS and task not in _EST_BUMP_TASKS:
        return {}
    if _RETRACT_RE.match(_norm(subgoal)):
        return {"interventions": [{"rule": "est_bump", "kind": "est_exempt",
                                   "detail": "retract-arm step -- a short move away, left at "
                                             "System2's estimate",
                                   "before": est, "after": est}]}
    new = bump_est(est)
    if new == est:
        return {}
    return {"est_proposal": new,
            "interventions": [{"rule": "est_bump", "kind": "est_proposed",
                               "detail": f"one bucket up ({est} -> {new}); conditioning tag, so "
                                         "System1 moves slower and more precisely",
                               "before": est, "after": new}]}


def _rule_mixer_est_floor(task: str, plan: str, subgoal: str, est, state) -> dict:
    """OpenStandMixerHead: every subgoal gets est_length of AT LEAST 75.

    A floor, not a fixed value: an estimate already >= 75 is left alone. System2 gave this task 50
    for every single subgoal (reach 20/20, push 20/20, retract 12/12), so in practice the floor
    raises all of them to 75.

    WHY THIS IS NOT JUST A BUDGET CHANGE. The budget was never the binding constraint here -- with
    est 50 and horizon_mult 2 the budget is 100, while the measured segments ran mean 49-50 steps
    and max 60, ending on the stop rule or env_success and NEVER on budget. What makes this rule
    bite is that ``est_length`` is also a POLICY CONDITIONING tag: it is sent to the server in the
    infer dict and rendered into System1's prompt as "Estimated Length" (see
    openpi robocasa_policy.py PROMPT_TAGS / build_prompt). So the floor changes the action
    distribution System1 samples from, telling it to plan a longer motion, independently of how many
    steps it is allowed.

    Corollary worth remembering when reading results: est overrides on OTHER tasks (faucet, coffee)
    act through the same two channels, and for progreg they additionally relax the progress
    threshold (stop_criterion.PROGREG_THRESH_BY_EST). This run is progact, so only the conditioning
    and budget channels apply.
    """
    if task != "OpenStandMixerHead":
        return {}
    cur = est if isinstance(est, int) else None
    if cur is not None and cur >= MIXER_EST_FLOOR:
        return {}
    return {"est_proposal": MIXER_EST_FLOOR,
            "interventions": [{"rule": "mixer_est_floor", "kind": "est_proposed",
                               "detail": f"floor est_length at {MIXER_EST_FLOOR} (conditioning "
                                         "tag, not just budget)",
                               "before": est, "after": MIXER_EST_FLOOR}]}


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
    return {"est_proposal": 100,
            "interventions": [{"rule": "coffee_m2_est", "kind": "est_proposed",
                               "detail": f"{fid} (M2 milestone) needs a full segment",
                               "before": est, "after": 100}]}


# --------------------------------------------------------------------------------------------
# ACTION-level overrides. Everything above revises System2's output; this one reaches into
# System1's actions instead, so it is applied by the rollout loop per step rather than per turn.

# GetToastedBread M1.1/M1.2 = "reach to the toaster lever" (68x) then "press/push the toaster lever
# down" (47x/21x). Pressing a lever wants a CLOSED gripper -- open fingers straddle the lever
# instead of bearing on it -- so the gripper command is pinned closed for both steps. Task was 8/20.
_TOASTER_GRIP_STEPS = ("M1.1", "M1.2")


def action_overrides(task: str, plan: str, subgoal: str) -> dict:
    """Per-STEP overrides applied to System1's commanded action for this segment.

    Returns e.g. ``{"grip": 1.0}`` to pin the gripper command (+1 = close). Applied to the whole
    predicted chunk, so quiescence/lookahead and the recorded actions all see the same values --
    an override applied only at step time would make the stop criterion reason about actions that
    were never executed.

    Empty dict == no override, which is the untouched path.
    """
    if task == "GetToastedBread" and current_fine_id(plan) in _TOASTER_GRIP_STEPS:
        return {"grip": 1.0,
                "why": f"GetToastedBread {current_fine_id(plan)}: pin gripper CLOSED to press "
                       "the toaster lever"}
    return {}


# Order matters: the plan rewrite runs first so later rules see the revised checklist.

# =============================================================================================
# GRADUATED per-task rules. Each arrived via sys2_rules_exp.py and is promoted here only after a
# run measured it. The measurement tables are kept inline so a rule is never re-litigated from
# memory. Both sets are gated on ONE task and share the two helpers below.
# =============================================================================================


def _split_fine_step(plan: str, subgoal: str, match_re, second_text_fn, first_text_fn=None,
                     milestone: str | None = None, rule: str = "split"):
    """Split the first matching fine step into two, renumbering that milestone.

    Skips steps already marked done ([x]) -- that is history, and renumbering around it would change
    which id the later steps refer to. Idempotent: once split, nothing matches. When the split step
    is the CURRENT one, System1 is also handed the FIRST half this turn, so it does not perform the
    half it was just told to defer.
    """
    blocks = _blocks(plan)
    cur = current_fine_id(plan)
    for b in blocks:
        if milestone is not None and b["mid"] != milestone:
            continue
        for i, f in enumerate(b["fine"]):
            if f["mark"] == "x":
                continue
            m = match_re.match(f["text"])
            if not m:
                continue
            head = (first_text_fn or (lambda mm: mm.group("head").strip()))(m)
            tail = second_text_fn(m)
            was_current = (f["fid"] == cur)
            new_fine = list(b["fine"])
            new_fine[i] = {"mark": f["mark"], "fid": "", "text": head}
            new_fine.insert(i + 1, {"mark": " ", "fid": "", "text": tail})
            for k, ff in enumerate(new_fine, start=1):
                ff["fid"] = f"{b['mid']}.{k}"
            b["fine"] = new_fine
            iv = [{"rule": rule, "kind": "plan_revised",
                   "detail": f"split {f['fid']} into two steps",
                   "before": f"{f['fid']}: {f['text']}",
                   "after": [f"{x['fid']}: {x['text']}" for x in new_fine[i:i + 2]]}]
            out = {"plan": _render(blocks), "interventions": iv}
            if was_current:
                out["subgoal"] = head
                out["subgoal_detail"] = head
                iv.append({"rule": rule, "kind": "subgoal_override",
                           "detail": "run only the first half this turn",
                           "before": subgoal, "after": head})
            return out
    return {}


def _advance_current_step(plan: str, subgoal: str, rule: str, detail: str):
    """Mark the current fine step done and hand System1 the NEXT one.

    This is how "skip this subgoal" is implemented. It is deliberately NOT ``skip_s1``: a skipped
    turn never steps the env, and ``_check_success()`` is polled only on executed steps, so skipping
    FREEZES the world and makes success undetectable (measured on TurnOnMicrowave: strip-retract via
    skipping took 4/8 -> 0/8, every episode running to max_turns). Advancing keeps the env moving.

    Declines when there is no next step: advancing off the last one strands System2 with nothing to
    propose, which is the same failure.
    """
    blocks = _blocks(plan)
    flat = [(b, f) for b in blocks for f in b["fine"]]
    cur = current_fine_id(plan)
    idx = next((i for i, (_, f) in enumerate(flat) if f["fid"] == cur), None)
    if idx is None or idx + 1 >= len(flat):
        return {"interventions": [{"rule": rule, "kind": "advance_declined",
                                   "detail": "no next fine step -- advancing would strand System2",
                                   "before": subgoal, "after": subgoal}]}
    _, cur_f = flat[idx]
    _, nxt_f = flat[idx + 1]
    cur_f["mark"] = "x"
    nxt_f["mark"] = "~"
    for b in blocks:
        if b["fine"] and all(f["mark"] == "x" for f in b["fine"]):
            b["mark"] = "x"
    return {"plan": _render(blocks), "subgoal": nxt_f["text"], "subgoal_detail": nxt_f["text"],
            "interventions": [
                {"rule": rule, "kind": "plan_revised", "detail": detail,
                 "before": f"{cur_f['fid']}: {cur_f['text']}",
                 "after": f"{nxt_f['fid']}: {nxt_f['text']}"},
                {"rule": rule, "kind": "subgoal_override",
                 "detail": "run the next step instead", "before": subgoal,
                 "after": nxt_f["text"]}]}


# =============================================================================================
# GRADUATED: PickPlaceCounterToCabinet   (baseline 15/20, verified rules 14/20)
#
# No rule has ever modified this task: repeat_cap is the only one that applies and it fired 33
# cap_declined / 0 advances, because the step it wants to cap is the LAST in the plan.
#
# All 6 failures in the verified run are max_turns at 14t, every one stuck repeating "continue to
# retract the arm from the cabinet" for 6-9 turns AFTER the object was already carried and placed.
# Separately, 7 turns carried judge=subgoal_failed, all of them "grasp X again".
#
# Plan shape: M1 pick up X (reach / grasp / lift) ; M2.1 carry X to the cabinet ;
#             M2.2 place X in the cabinet and release ; M2.3 retract the arm from the cabinet.
# =============================================================================================

PPC2C = "PickPlaceCounterToCabinet"

# 1. judge == subgoal_failed. All 7 observed were "grasp X again" -- the same false positive seen on
#    CoffeeSetupMug, where the object is in fact already held, so re-doing the grasp repeats a
#    finished action. Skip it by ADVANCING the plan (see _advance_current_step for why not skip_s1).
_REGRASP_RE = re.compile(r"^\s*(continue\s+to\s+)?grasp\b.*\bagain\b\s*$", re.I)

# 2. The grasp is the pose every later step inherits. est_length is a POLICY CONDITIONING tag
#    (rendered into System1's prompt as "Estimated Length"), so a floor makes the grasp slower and
#    more precise rather than merely longer. Uses the shared ladder; bump_est(50) == 75.
_GRASP_RE = re.compile(r"^grasp\b")
GRASP_EST_FLOOR = 75

# 3. One extra "continue to <M2.1>" step after M2.1. M2.1 is the CARRY ("carry X to the cabinet"),
#    and the carry is what has to get the object far enough inside; a second segment gives it another
#    go before the plan moves on to the release. Inserted into the plan, so System2 conditions on it
#    from the next turn.
_CONTINUE_PREFIX = "continue to "


def _rule_ppc2c_skip_failed(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceCounterToCabinet: a re-grasp after subgoal_failed advances the plan instead."""
    if task != PPC2C:
        return {}
    if not _REGRASP_RE.match(subgoal or ""):
        return {}
    return _advance_current_step(
        plan, subgoal, "exp_ppc2c_skip_failed",
        f"subgoal_failed re-grasp is a false positive (judge={state.get('judge')!r}); "
        "advance rather than repeat a finished grasp")


def _rule_ppc2c_grasp_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceCounterToCabinet: floor a "grasp ..." subgoal at est 75 (one bucket up from 50)."""
    if task != PPC2C or not _GRASP_RE.match(_norm(subgoal)):
        return {}
    new = max(bump_est(est), GRASP_EST_FLOOR) if isinstance(est, int) else GRASP_EST_FLOOR
    if isinstance(est, int) and est >= new:
        return {}
    return {"est_proposal": new,
            "interventions": [{"rule": "ppc2c_grasp_est", "kind": "est_proposed",
                               "detail": f"grasp sets the pose everything downstream inherits "
                                         f"({est} -> {new}); conditioning tag",
                               "before": est, "after": new}]}


def _rule_ppc2c_extra_carry(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceCounterToCabinet: add one extra "continue to <M2.1>" step after M2.1.

    Idempotent: the inserted step already starts with "continue to", so it never matches itself, and
    the rule does nothing once the extra step exists. Skips a done M2.1 -- once the carry has been
    executed there is nothing to extend.
    """
    if task != PPC2C:
        return {}
    blocks = _blocks(plan)
    for b in blocks:
        if b["mid"] != "M2" or not b["fine"]:
            continue
        first = b["fine"][0]
        if first["fid"] != "M2.1" or first["mark"] == "x":
            return {}
        if first["text"].lower().startswith(_CONTINUE_PREFIX):
            return {}
        nxt = b["fine"][1]["text"].lower() if len(b["fine"]) > 1 else ""
        if nxt.startswith(_CONTINUE_PREFIX):
            return {}                      # already inserted on an earlier turn
        extra = _CONTINUE_PREFIX + first["text"]
        new_fine = list(b["fine"])
        new_fine.insert(1, {"mark": " ", "fid": "", "text": extra})
        for k, ff in enumerate(new_fine, start=1):
            ff["fid"] = f"{b['mid']}.{k}"
        b["fine"] = new_fine
        return {"plan": _render(blocks),
                "interventions": [{"rule": "ppc2c_extra_carry", "kind": "plan_revised",
                                   "detail": "give the carry a second segment before the release",
                                   "before": f"M2.1: {first['text']}",
                                   "after": f"M2.2: {extra}"}]}
    return {}


# =============================================================================================
# GRADUATED: CoffeeSetupMug   (baseline 10/20, verified rules 11/20)
#
# Seven runs of the same 20 episodes:
#     config                                  success   task_finish
#     baseline (no rules)                      10/20         8
#     verified rules only                      11/20         8
#     splits only                              10/20         8
#     grasp est only                           10/20        10
#     splits + grasp est                       13/20         3
#     splits + grasp est, repeat               12/20         4
#     splits + grasp est, repeat               12/20         5
#
# The success counts alone are marginal (+2.1 mean, and two runs of IDENTICAL code agreed on only
# 15/20 episodes). What justifies keeping this is the INTERACTION: task_finish -- the diagnosed
# failure signature for this task, both milestones done and arm retracted with _check_success()
# never firing -- collapses to 3-5 only when BOTH ingredients are present, and stays at 8-10 in all
# four configurations missing either one. Three runs with both, four without, and the groups do not
# overlap on either metric. Reading: the splits fix WHEN the release happens, the grasp est fixes
# THE POSE IT INHERITS FROM; neither is sufficient alone.
# =============================================================================================

COFFEE = "CoffeeSetupMug"

# "<action> and release" as a TRAILING clause -- 46 of 49 occurrences on this task, all in M2
# ("lower the mug under the coffee machine dispenser and release" 30x, "place the mug ... and
# release" 10x). The 3 remaining are "lower and release the red mug ..." where the release sits
# inside the verb phrase; splitting that would need the object rewritten, so it is left alone.
_COFFEE_AND_RELEASE_RE = re.compile(r"^(?P<head>.+?)\s+and\s+release\s*$", re.I)
COFFEE_RELEASE_SUBGOAL = "release the mug"

# "reach and grasp <object>" -> "reach to <object>" + "grasp <object>". Rare here (4x and 1x in the
# two runs) -- this task usually plans reach and grasp as separate steps already -- but System2 does
# occasionally emit the compound.
_COFFEE_REACH_GRASP_RE = re.compile(r"^reach\s+and\s+grasp\s+(?P<obj>.+?)\s*$", re.I)

# The grasp sets the pose every later step inherits. M1 step, so it cannot collide with the verified
# coffee_m2_est, which is M2-only. Uses the shared ladder; bump_est(50) == 75, and System2 budgeted
# the grasp 50 in all 42 observed firings.
_COFFEE_GRASP_RE = re.compile(r"^grasp\b")


def _rule_coffee_split_release(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: M2 "<action> and release" -> "<action>" + "release the mug"."""
    if task != COFFEE:
        return {}
    return _split_fine_step(plan, subgoal, _COFFEE_AND_RELEASE_RE,
                            second_text_fn=lambda m: COFFEE_RELEASE_SUBGOAL,
                            milestone="M2", rule="exp_coffee_split_release")


def _rule_coffee_split_reach_grasp(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: "reach and grasp X" -> "reach to X" + "grasp X"."""
    if task != COFFEE:
        return {}
    return _split_fine_step(
        plan, subgoal, _COFFEE_REACH_GRASP_RE,
        first_text_fn=lambda m: f"reach to {m.group('obj').strip()}",
        second_text_fn=lambda m: f"grasp {m.group('obj').strip()}",
        rule="exp_coffee_split_reach_grasp")


def _rule_coffee_grasp_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: bump a "grasp ..." subgoal one est bucket (50 -> 75, 75 -> 100, ...)."""
    if task != COFFEE or not _COFFEE_GRASP_RE.match(_norm(subgoal)):
        return {}
    new = bump_est(est)
    if new == est:
        return {}
    return {"est_proposal": new,
            "interventions": [{"rule": "coffee_grasp_est", "kind": "est_proposed",
                               "detail": "grasp sets the pose everything downstream inherits; "
                                         f"one bucket up ({est} -> {new}), conditioning tag",
                               "before": est, "after": new}]}


_RULES = (_rule_drawer_base_align, _rule_strip_retract_plan, _rule_flag_retract_emitted,
          _rule_microwave_again, _rule_repeat_cap, _rule_sink_faucet_est,
          _rule_est_bump, _rule_mixer_est_floor, _rule_coffee_m2_est,
          # GRADUATED (see the tables above)
          _rule_coffee_split_release, _rule_coffee_split_reach_grasp, _rule_coffee_grasp_est,
          _rule_ppc2c_skip_failed, _rule_ppc2c_extra_carry, _rule_ppc2c_grasp_est)

# EXPERIMENTAL PATCH LAYER. Unverified per-task rules live in sys2_rules_exp.py and are appended
# here if that module is importable. This is the ONLY hook they need: an experiment is added,
# revised or thrown away by editing that one file, and DELETING it reverts to the verified rule set
# above -- the import simply fails and nothing is appended. Kept last so an experiment sees the
# checklist the verified rules produced, and set SYS2_RULES_NO_EXP=1 to ignore the file without
# deleting it (e.g. to re-measure the verified baseline).
if not os.environ.get("SYS2_RULES_NO_EXP"):
    try:
        from sys2_rules_exp import EXP_RULES as _EXP_RULES
    except Exception:  # noqa: BLE001 - absent or broken patch file must never break a real run
        _EXP_RULES = ()
    _RULES = _RULES + tuple(_EXP_RULES)
else:
    _EXP_RULES = ()

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
    est_proposals: list[tuple] = []
    for fn in _RULES:
        r = fn(task, cur["plan"], cur["subgoal"], cur["est"], st)
        if not r:
            continue
        ivs.extend(r.get("interventions", []))
        if "est_proposal" in r:
            # Collected, NOT applied: several rules may propose an est for the same turn (a
            # task-specific one and the universal bucket bump), and applying them in sequence would
            # CHAIN -- the faucet rule's 100 would then be bumped again to 125, which is neither
            # proposal. They are resolved once, below, by taking the largest.
            est_proposals.append((r["est_proposal"], r["interventions"][0]["rule"]
                                  if r.get("interventions") else "?"))
        for k in ("plan", "subgoal", "subgoal_detail", "skip_s1"):
            if k in r:
                cur[k] = r[k]
        if cur["skip_s1"]:
            break            # nothing else applies to a turn that runs no segment
    # EST RESOLUTION: largest of System2's own estimate and every proposal. "Whichever is larger"
    # is the rule because a larger est_length conditions System1 to move slower, i.e. more
    # precisely -- so between a task-specific value and the universal bump, the bigger one is the
    # stronger version of the same intent. All proposals are computed from System2's ORIGINAL est,
    # so the order rules run in cannot change the outcome.
    if est_proposals:
        base = est if isinstance(est, int) else 0
        best, who = max(est_proposals, key=lambda pr: pr[0])
        if best > base:
            cur["est"] = best
            names = "+".join(sorted({w for _, w in est_proposals}))
            ivs.append({"rule": "est_resolve", "kind": "est_override",
                        "detail": f"largest of {sorted({p for p, _ in est_proposals})} "
                                  f"proposed by {names}" + (f"; winner {who}"
                                                            if len(est_proposals) > 1 else ""),
                        "before": est, "after": best})
    st["consec_skips"] = (st.get("consec_skips", 0) + 1) if cur["skip_s1"] else 0
    cur["interventions"] = ivs
    return cur
