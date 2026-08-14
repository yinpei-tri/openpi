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
# How many END-OF-PLAN cap declines to tolerate before force-stopping the episode as "max_cap".
# An episode that trips the cap with nothing left to advance into is effectively dead: measured over
# the two 1500-episode v2 sweeps, such episodes succeed 1.5-2.3% of the time against ~65% for episodes
# that never trip it. The DEFAULT 4 gives SIX executed attempts on the terminal subgoal (3 inside the
# cap plus 3 executed declines; the stopping turn itself runs no System1 segment). Measured cost of
# that cut, stir excluded: 5 wins on qwen35 (0.33%) and 5 on qwen3vl (0.33%) -- the episodes that
# needed a 7th attempt. Tightening to 3 declines / 5 attempts would cost qwen3vl 7 wins (0.47%) by
# additionally losing TurnOnSinkFaucet ep01 and WashFruitColander ep10, and stopping at the FIRST
# decline would cost 0.93% / 1.80%. Six is the setting the session owner chose.
MAX_CAP_DECLINES = int(os.environ.get("SYS2_RULES_MAX_CAP_DECLINES", "4"))
# A RETRACT terminal step gets NO declines at all: it stops the moment the cap is exceeded, i.e. 3
# executed attempts rather than 5. A retract either latches immediately or never -- measured over the
# two 1500-episode sweeps, retract-terminal loops beyond 3 repeats number 240 episodes on qwen35 and 82
# on qwen3vl and yield exactly ONE win each (SetUpCuttingStation ep22, PickPlaceCounterToCabinet ep22),
# against 505/549 episodes at a single retract turn yielding 282/316 wins. So cutting here stops 203 and
# 82 more episodes for 0.07% of the run apiece. Tasks most affected: PrepareCoffee, SeparateFreezerRack,
# GarnishPancake, PanTransfer, CategorizeCondiments, ScrubCuttingBoard, StoreLeftoversInBowl.
MAX_CAP_DECLINES_RETRACT = int(os.environ.get("SYS2_RULES_MAX_CAP_DECLINES_RETRACT", "0"))
# _norm has already stripped a leading "continue to", so this matches the re-issues too.
_RETRACT_TERMINAL_RE = re.compile(r"^retract\b")
# OpenStandMixerHead est_length floor. est_length is a POLICY CONDITIONING tag (rendered into
# System1's prompt as "Estimated Length"), not only a budget multiplier -- see _rule_mixer_est_floor.
MIXER_EST_FLOOR = int(os.environ.get("SYS2_RULES_MIXER_EST_FLOOR", "75"))
# est_length BUCKET LADDER, as used by the training data. A bump means "the next bucket up".
EST_BUCKETS = (50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500,
               600, 800, 1000, 1300)
# At and above this, est is left alone: the bump is a precision aid for short/medium motions, and
# the rungs above 500 are large jumps on spans that are already long.
EST_BUMP_CEILING = int(os.environ.get("SYS2_RULES_EST_BUMP_CEILING", "500"))
# Historical scope of the bucket bump, measured one variable at a time (v3 -> v4, same 20 episodes
# per task, bump the only change):
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
#
# The rule is now scoped to TurnOnMicrowave alone. TurnOnSinkFaucet is covered by the semantic
# sink_faucet_est text gate below, including composite tasks that contain the same operation, while
# GetToastedBread has dedicated wait and slot-positioning rules. Keeping those three mechanisms
# separate avoids stacking two estimate rules on the same turn and makes their attribution clear.


def bump_est(est):
    """Next bucket up, or ``est`` unchanged at/above EST_BUMP_CEILING (or if not a number).

    A value that is not itself a bucket (System2 occasionally writes e.g. 60) snaps UP to the next
    bucket, which is the in-distribution neighbour rather than an invented number.
    """
    if not isinstance(est, int) or est >= EST_BUMP_CEILING:
        return est
    return next((b for b in EST_BUCKETS if b > est), est)


# Positioning/transport moves, which get the TIGHTER repeat cap (REACH_SAME_SUBGOAL). The name is
# historical -- it started as reach-only. CARRY was added because the same argument applies: moving
# an object from A to B either converges or it does not, and re-issuing it is not effort
# accumulation the way stirring or pressing a door is. "lift and carry ..." is included for the
# same reason -- it is the identical motion with a lift prefix, and excluding it would make the cap
# depend on System2's wording (3022 turns say "carry ...", another 1420 say "lift and carry ...").
# Still EXCLUDES "reach and grasp ..." -- there the grasp is the real work and must not be rushed.
_REACH_RE = re.compile(r"^(reach\s+(to|for)|(lift\s+and\s+)?carry)\b")


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
DRAWER_ALIGN_EST_FLOOR = 75

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
            iv.append({"rule": "drawer_base_align", "kind": "subgoal_override",
                       "detail": "execute the inserted base-alignment step before reaching",
                       "before": subgoal, "after": BASE_ALIGN_TEXT})
        return out
    return {}


def _rule_drawer_base_align_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceDrawerToCounter: floor the M1.1 base-alignment step at est 75.

    Kept separate from ``_rule_drawer_base_align`` so it covers both sources of the step: a
    checklist where System2 authored M1.1 itself, and the M1.1 inserted by that rule immediately
    before this one runs.  Position plus alignment wording prevents an unrelated M1.1 from being
    raised.
    """
    if task not in _DRAWER_ALIGN_TASKS or current_fine_id(plan) != "M1.1":
        return {}
    text = _norm(subgoal)
    if "drawer" not in text or not _BASE_ALIGN_RE.search(text):
        return {}
    if isinstance(est, int) and est >= DRAWER_ALIGN_EST_FLOOR:
        return {}
    return {
        "est_proposal": DRAWER_ALIGN_EST_FLOOR,
        "interventions": [{
            "rule": "drawer_base_align_est",
            "kind": "est_proposed",
            "detail": f"M1.1 base alignment gets an est floor of {DRAWER_ALIGN_EST_FLOOR}",
            "before": est,
            "after": DRAWER_ALIGN_EST_FLOOR,
        }],
    }


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
    # A NEW subgoal starts a FRESH decline budget. Without this reset rep_declines accumulated over the
    # whole episode, so a second terminal subgoal inherited the first one's declines and stopped after 5
    # executed attempts instead of the documented 6 -- and an episode whose plan System2 later extended
    # could be terminated by declines belonging to an already-abandoned step.
    if n != prev:
        state["rep_declines"] = 0
    state["rep_sg"], state["rep_n"] = n, count
    # A pure REACH gets a tighter cap. Reaching is positioning, not effort accumulation: if System1
    # has not arrived in two turns it is not converging, and more reaching only burns turns. The
    # clearest case is ArrangeTea ep0, which spent turns 9-22 on "continue to reach to the right
    # cabinet door" -- 14 repeats -- while "push the right cabinet door closed" sat untouched.
    # Matches "reach to"/"reach for" only, NOT compounds like "reach and grasp the kettle", where
    # the grasp is the real work.
    # Name the cap that is actually in force, not just the general one: a pure reach is capped at
    # REACH_SAME_SUBGOAL and reporting it as MAX_SAME_SUBGOAL produced the nonsense "repeat #3 >
    # MAX_SAME_SUBGOAL=3" in the audit trail, which reads as if a non-reach had been mis-classified.
    _is_reach = bool(_REACH_RE.match(n))
    cap = REACH_SAME_SUBGOAL if _is_reach else MAX_SAME_SUBGOAL
    _capname = "REACH_SAME_SUBGOAL" if _is_reach else "MAX_SAME_SUBGOAL"
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
        # NO NEXT FINE STEP. That used to end the rule, on the reasoning that advancing would strand
        # System2 -- but System2 unrolls fine steps ONE MILESTONE AT A TIME, so the last unrolled step
        # is routinely NOT the end of the plan. Measured on the qwen35 v2 sweep: of 2879 such declines,
        # 636 (47 episodes) had a later unfinished MILESTONE sitting right there, and those episodes
        # succeeded 1/47. ArrangeBreadBasket ep0 is the type case: 16 declines and 19 turns burnt on
        # "continue to pull the left cabinet door open" while M2..M7 were still pending.
        _num = lambda mid: int(re.sub(r"[^0-9]", "", mid) or 0)
        cur_b = next((b for b, f in flat if f["fid"] == cur_id), None) if idx is not None else \
            next((b for b in blocks if b["mark"] == "~"), next((b for b in blocks if b["mark"] == " "), None))
        later = [b for b in blocks
                 if cur_b is not None and _num(b["mid"]) > _num(cur_b["mid"]) and b["mark"] != "x"]
        if later and cur_b is not None:
            # Close the current milestone so System2 unrolls the NEXT one. The subgoal is left alone:
            # this turn still executes (skipping freezes the env), and System2 sees the [x] next turn
            # and proposes a step from the following milestone.
            for f in cur_b["fine"]:
                f["mark"] = "x"
            cur_b["mark"] = "x"
            # Reset the repeat counter: the checklist has moved on, so the next turn starts a fresh
            # count. This also BOUNDS the branch -- without it, re-applying the rules to the revised
            # plan would find the next bare milestone and close that one too, walking the whole plan.
            state["rep_n"] = 1
            return {"plan": _render(blocks), "requery_s2": True,
                    "interventions": [{"rule": "repeat_cap", "kind": "milestone_closed",
                                       "detail": f"repeat #{count} and no next fine step, but "
                                                 f"{later[0]['mid']} is still pending -- closing "
                                                 f"{cur_b['mid']} and re-asking System2, so this turn "
                                                 "runs ITS next-milestone subgoal instead of the "
                                                 "exhausted one",
                                       "before": f"{cur_b['mid']} [{cur_b['mark']}]",
                                       "after": f"{cur_b['mid']} [x], next {later[0]['mid']}"}]}
        # TRULY the end of the plan: nothing to advance into, so the loop cannot be broken by moving
        # the checklist. Tolerate a few declines, then stop the episode rather than burn the remaining
        # turn budget on a step that is not converging.
        #
        # NOTE ON COUNTING: the stopping turn runs NO System1 segment (combined_eval sets s1=None and
        # breaks), so the number of EXECUTED attempts is cap + limit, not cap + limit + 1:
        #     ordinary terminal step   3 in-cap + 3 declines = 6 executed, stop on the 7th S2 turn
        #     retract terminal step    3 in-cap + 0 declines = 3 executed, stop on the 4th S2 turn
        _retract = bool(_RETRACT_TERMINAL_RE.match(n))
        limit = MAX_CAP_DECLINES_RETRACT if _retract else MAX_CAP_DECLINES
        n_dec = state["rep_declines"] = state.get("rep_declines", 0) + 1
        # >= not >: limit 3 stops ON the 3rd decline (5 executed attempts, the configuration
        # the 0.33%/0.47% cost was measured against), and limit 0 stops on the first (3 executed).
        if n_dec >= (limit or 1):
            return {"stop_episode": True,
                    "interventions": [{"rule": "repeat_cap", "kind": "max_cap",
                                       "detail": f"repeat #{count}, decline #{n_dec} at the END of "
                                                 f"the plan ({'retract terminal, ' if _retract else ''}"
                                                 f"limit {limit}) -- "
                                                 "no step or milestone left to advance into, so the "
                                                 "episode is stopped instead of running to max_turns",
                                       "before": subgoal, "after": "STOP (max_cap)"}]}
        return {"interventions": [{"rule": "repeat_cap", "kind": "cap_declined",
                                   "detail": f"repeat #{count}, decline #{n_dec} of {limit} -- no "
                                             "next fine step AND no later "
                                             "milestone; advancing would strand System2 (see the "
                                             "TurnOnMicrowave retract finding)",
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
                 "detail": f"repeat #{count} > {_capname}={cap}: marked "
                           f"{cur_f['fid']} done, advanced to {nxt_f['fid']}",
                 "before": f"{cur_f['fid']}: {cur_f['text']}",
                 "after": f"{nxt_f['fid']}: {nxt_f['text']}"},
                {"rule": "repeat_cap", "kind": "subgoal_override",
                 "detail": "hand System1 the next fine step instead of the repeated one",
                 "before": subgoal, "after": new_sg}]}


_SINK_FAUCET_ON_RE = re.compile(
    r"^(?:"
    r"(?:finish\s+)?turn(?:ing)?\s+on\s+the\s+sink\s+faucet(?:\s+handle)?"
    r"|(?:finish\s+)?turn(?:ing)?\s+the\s+sink\s+faucet(?:\s+handle)?\s+on"
    r"|push(?:ing)?\s+the\s+sink\s+faucet(?:\s+handle)?\s+to\s+turn\s+it\s+on"
    r")$"
)


def _rule_sink_faucet_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """ANY task: a subgoal semantically turning on the sink faucet gets est_length >= 100.

    System2 budgeted this 50 (18x) or re-issued it as "continue to ..." (33x) -- i.e. it kept
    running out of segment before the handle was over. 100 covers it in one segment.
    """
    # ANY task: gated on the SUBGOAL, not the task name, so the composite tasks that turn on the
    # same faucet are covered too (measured faucet-subgoal turns: WashLettuce 138, RinseSinkBasin
    # 99, WashFruitColander 94, PreSoakPan 89, vs 51 on the atomic task). Match both model families:
    # Qwen3.5 usually emits "turn on the sink faucet handle", while Qwen3-VL often emits
    # "push the sink faucet handle to turn it on". Deliberately exclude reach/grasp/hold/release,
    # which mention the same fixture but are not the contact that turns on the water.
    # A width-gated recovery may have borrowed this turn and replaced the faucet activation with a
    # re-grasp. The estimate belongs to the faucet action System2 proposed, not to that injected
    # recovery; the held faucet action resumes next turn with System2's own estimate. Keep the two
    # transactions separate instead of attaching 100 to "grasp ... again".
    if state.get("regrasp_recovery_hit"):
        return {}
    if not _SINK_FAUCET_ON_RE.match(_norm(subgoal)):
        return {}
    if isinstance(est, int) and est >= 100:
        return {}
    return {"est_proposal": 100,
            "interventions": [{"rule": "sink_faucet_est", "kind": "est_proposed",
                               "detail": "sink-faucet activation needs est_length >= 100",
                               "before": est, "after": 100}]}


def _rule_microwave_est_bump(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnMicrowave only: raise est_length one bucket -- 50->75, 75->100, ... 400->500.
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

    SCOPE CAUTION. That evidence is ONE task and ONE failure mode (under-shooting a contact), so the
    rule is gated on exactly that task. TurnOnSinkFaucet uses sink_faucet_est instead, and
    GetToastedBread uses its dedicated wait/slot rules.

    PROGREG INTERACTION, worth knowing before running the regression head with this on: for
    progreg, est_length ALSO selects the progress threshold
    (stop_criterion.PROGREG_THRESH_BY_EST -- 50/75 -> 0.88, 100 -> 0.92, else 0.95), so a bump makes
    the stop rule STRICTER (75->100 raises the bar 0.88 -> 0.92). These runs are progact, where the
    threshold is a flat 0.95 and only the conditioning and budget channels apply.
    """
    if task != "TurnOnMicrowave":
        return {}
    if _RETRACT_RE.match(_norm(subgoal)):
        return {"interventions": [{"rule": "microwave_est_bump", "kind": "est_exempt",
                                   "detail": "retract-arm step -- a short move away, left at "
                                             "System2's estimate",
                                   "before": est, "after": est}]}
    new = bump_est(est)
    if new == est:
        return {}
    return {"est_proposal": new,
            "interventions": [{"rule": "microwave_est_bump", "kind": "est_proposed",
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
    # EXEMPT: retract-arm AND a bare "release ..." step. Both are short terminal motions -- opening
    # the fingers, or backing away -- rather than the carry/lower work the 100 was chosen for, so a
    # full segment buys nothing and only spends steps against the task's 600-step official horizon.
    # Anchored with ^release so a COMBINED step ("lower the mug under the dispenser and release",
    # which System2 emits 47 times) still gets the full segment: that one does have to travel.
    _short = _RETRACT_RE.match(_norm(subgoal)) or _COFFEE_RELEASE_RE.match(_norm(subgoal))
    if _short:
        return {"interventions": [{"rule": "coffee_m2_est", "kind": "est_exempt",
                                   "detail": f"{fid} is a retract/release step -- left at System2's "
                                             "estimate, no full segment needed",
                                   "before": est, "after": est}]}
    if est == COFFEE_M2_EST:
        return {}
    return {"est_proposal": COFFEE_M2_EST,
            "interventions": [{"rule": "coffee_m2_est", "kind": "est_proposed",
                               "detail": f"{fid} (M2 milestone) needs a full segment",
                               "before": est, "after": COFFEE_M2_EST}]}


# --------------------------------------------------------------------------------------------
# ACTION-level overrides. Everything above revises System2's output; this one reaches into
# System1's actions instead, so it is applied by the rollout loop per step rather than per turn.

# GetToastedBread M1.1/M1.2 = "reach to the toaster lever" (68x) then "press/push the toaster lever
# down" (47x/21x). Pressing a lever wants a CLOSED gripper -- open fingers straddle the lever
# instead of bearing on it -- so the gripper command is pinned closed for both steps. Task was 8/20.
_TOASTER_GRIP_STEPS = ("M1.1", "M1.2")


# action_overrides() is called DIRECTLY by combined_eval, not through _RULES, so it needs its own
# tier check: without it a "general rules only" arm still applied the GetToastedBread gripper pin,
# and the arm was not general-only at all.
#
# The REGISTRY check below is no longer sufficient on its own. It was written when the arm was chosen
# by editing _RULES, so "is a task rule registered?" answered "is this a task-rules arm?". The arm is
# now chosen at RUNTIME by --task-rules, and _RULES always contains the task tier, so the caller must
# pass the tier in. Kept as a second condition so an unregistered task tier still disables it.
def _task_rules_active() -> bool:
    return any(f in _RULES for f in _TASK_RULES)


def action_overrides(task: str, plan: str, subgoal: str, *, task_tier: bool = True) -> dict:
    """Per-STEP overrides applied to System1's commanded action for this segment.

    Returns e.g. ``{"grip": 1.0}`` to pin the gripper command (+1 = close). Applied to the whole
    predicted chunk, so quiescence/lookahead and the recorded actions all see the same values --
    an override applied only at step time would make the stop criterion reason about actions that
    were never executed.

    ``task_tier`` is the runtime arm: False on a mandatory-only or general-only run, which must not
    get this GetToastedBread-specific pin.

    Empty dict == no override, which is the untouched path.
    """
    if not (task_tier and _task_rules_active()):
        return {}
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


def _advance_or_close_milestone(plan: str, subgoal: str, state, rule: str, detail: str,
                                state_key: str, max_closes: int = 2):
    """``_advance_current_step``, but when there is NO next fine step, close the MILESTONE instead.

    System2 unrolls fine steps ONE MILESTONE AT A TIME, so "the current step is the last fine step"
    usually does NOT mean "the end of the plan" -- the next milestone is sitting right there with no
    fine steps yet. Plain _advance_current_step declines in that situation, which is why a
    skip-this-step rule can fire and change nothing (measured on CoffeeSetupMug: 26 firings, 26
    advance_declined, 0 skips).

    So: close the current milestone and set ``requery_s2``, which makes combined_eval ask System2
    again THIS turn with the revised checklist and execute the subgoal it returns -- i.e. the first
    step of the next milestone. Same mechanism repeat_cap uses for its own exhausted-step case.

    BOUNDED by ``max_closes`` per episode via ``state[state_key]``: without a bound, re-applying the
    rules to the revised plan could find the next bare milestone and close that one too, walking the
    whole checklist in a single turn.
    """
    r = _advance_current_step(plan, subgoal, rule, detail)
    if not any(i.get("kind") == "advance_declined" for i in r.get("interventions", [])):
        return r                                      # a next fine step existed; ordinary advance
    if state.get(state_key, 0) >= max_closes:
        return r                                      # budget spent; leave the decline recorded
    blocks = _blocks(plan)
    cur_id = current_fine_id(plan)
    def _num(mid):
        return int(re.sub(r"[^0-9]", "", mid) or 0)
    cur_b = next((b for b in blocks if any(f["fid"] == cur_id for f in b["fine"])), None) \
        or next((b for b in blocks if b["mark"] == "~"), None) \
        or next((b for b in blocks if b["mark"] == " "), None)
    if cur_b is None:
        return r
    later = [b for b in blocks if _num(b["mid"]) > _num(cur_b["mid"]) and b["mark"] != "x"]
    if not later:
        return r                                      # genuinely the end of the plan
    for f in cur_b["fine"]:
        f["mark"] = "x"
    cur_b["mark"] = "x"
    state[state_key] = state.get(state_key, 0) + 1
    return {"plan": _render(blocks), "requery_s2": True,
            "interventions": [{"rule": rule, "kind": "milestone_closed",
                               "detail": f"{detail} -- no next fine step, but {later[0]['mid']} is "
                                         f"still pending, so {cur_b['mid']} is closed and System2 is "
                                         "re-asked this turn for a step from the next milestone "
                                         f"(close {state[state_key]}/{max_closes})",
                               "before": f"{cur_b['mid']} [{cur_b['mark']}]",
                               "after": f"{cur_b['mid']} [x], next {later[0]['mid']}"}]}


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
# extra_carry ALTERS the checklist (it inserts a real "continue to <M2.1>" step), and that is the
# point rather than a side effect: adding a step means the repeated retract is no longer the LAST
# plan step, so repeat_cap can ADVANCE instead of declining, which is where retract turns 64 -> 24
# came from. A borrowed-turn replacement was tried and rejected -- one injection 16/20, two with
# System2's subgoal held 16/20 and 17/20, all leaving retract at 40-49 -- so this is FIXED here with
# no toggle. The comparison is settled; recover the toggle from git if it ever needs re-running.

# 1. judge == subgoal_failed. All 7 observed were "grasp X again" -- the same false positive seen on
#    CoffeeSetupMug, where the object is in fact already held, so re-doing the grasp repeats a
#    finished action. Skip it by ADVANCING the plan (see _advance_current_step for why not skip_s1).
# NAMED DISTINCTLY on purpose: a later _REGRASP_RE (the general re-grasp recovery, "^(reach and )?
# grasp\b") used to SHADOW this one, because module-level names are resolved at call time and the
# later assignment wins. That made ppc2c_skip_failed match a FIRST grasp and advance past it, i.e.
# the task stopped grasping at all before lifting. Verified and fixed; do not reintroduce the
# shared name.
_REGRASP_AGAIN_RE = re.compile(r"^\s*(continue\s+to\s+)?grasp\b.*\bagain\b\s*$", re.I)

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
    # This rule handles System2's FALSE-POSITIVE failure judgement. If the physical width-gated
    # recovery actually fired, its re-grasp must win; otherwise this rule would immediately rewrite
    # the injected recovery into the next step and the dropped object would never be recovered.
    if state.get("judge") != "subgoal_failed" or state.get("regrasp_recovery_hit"):
        return {}
    if not _REGRASP_AGAIN_RE.match(subgoal or ""):
        return {}
    return _advance_current_step(
        plan, subgoal, "ppc2c_skip_failed",
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
# est every M2.x step gets (and what the subgoal_failed rewrite pins M2.1 to).
COFFEE_M2_EST = int(os.environ.get("SYS2_COFFEE_M2_EST", "100"))
# A BARE release step -- exempt from the M2 est like a retract. Anchored so
# "lower ... and release" (a travel + release) is NOT exempt.
_COFFEE_RELEASE_RE = re.compile(r"^release\b")

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

# The normal grasp sets the pose every later step inherits. Match the RAW (unnormalized) subgoal and
# anchor it at ``mug`` deliberately: ``continue to grasp ...`` and ``grasp ... mug again`` are
# recovery / continuation requests, not the ordinary grasp turn this estimate was chosen for.
# This M1 step cannot collide with coffee_m2_est, which is M2-only.
_COFFEE_GRASP_RE = re.compile(r"^grasp\b.*\bmug$", re.IGNORECASE)
COFFEE_GRASP_EST_FLOOR = 75


def _rule_coffee_split_release(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: M2 "<action> and release" -> "<action>" + "release the mug"."""
    if task != COFFEE:
        return {}
    return _split_fine_step(plan, subgoal, _COFFEE_AND_RELEASE_RE,
                            second_text_fn=lambda m: COFFEE_RELEASE_SUBGOAL,
                            milestone="M2", rule="coffee_split_release")


def _rule_coffee_split_reach_grasp(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: "reach and grasp X" -> "reach to X" + "grasp X"."""
    if task != COFFEE:
        return {}
    return _split_fine_step(
        plan, subgoal, _COFFEE_REACH_GRASP_RE,
        first_text_fn=lambda m: f"reach to {m.group('obj').strip()}",
        second_text_fn=lambda m: f"grasp {m.group('obj').strip()}",
        rule="coffee_split_reach_grasp")


def _rule_coffee_grasp_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: floor a normal raw ``grasp ... mug`` subgoal at est 75."""
    # Do not call _norm here: stripping "continue to" / "again" would turn a continuation or
    # recovery request into a false match for the normal grasp rule.
    if task != COFFEE or not _COFFEE_GRASP_RE.match((subgoal or "").strip()):
        return {}
    if isinstance(est, int) and est >= COFFEE_GRASP_EST_FLOOR:
        return {}
    return {"est_proposal": COFFEE_GRASP_EST_FLOOR,
            "interventions": [{"rule": "coffee_grasp_est", "kind": "est_proposed",
                               "detail": "grasp sets the pose everything downstream inherits; "
                                         f"est floor {COFFEE_GRASP_EST_FLOOR}, conditioning tag",
                               "before": est, "after": COFFEE_GRASP_EST_FLOOR}]}


# ---- skip the false-positive re-grasp ------------------------------------------------------------
# System2 judges the mug grasp `subgoal_failed` and asks to "grasp the mug again" on 24-25 of 30
# episodes -- but the grasp HAS succeeded, and the judgement is a false positive. Measured on the
# qwen3vl -v2 run, end-of-segment gripper width in SUCCESSFUL episodes only:
#     reach (gripper open)                n=17  median 0.0797
#     grasp (the one judged failed)       n=17  median 0.0115   range 0.0052 - 0.0204
#     carry/place (mug demonstrably held) n=26  median 0.0145
# The grasp and the carry sit in the SAME band, so ~0.012 is simply what this mug reads when held by
# its thin handle -- it is below the 0.015 "closed on nothing" bar that suits bulkier objects, which
# is why it looks like a miss. The widths recorded at the 24-25 "grasp X again" turns (0.005-0.021)
# are indistinguishable from that holding band.
#
# COST OF OBEYING IT: each re-grasp burns a turn and ~110 executed steps (observed 77-250) on a task
# whose budget is 12 turns and whose mean is 7.5-7.9, with 4 of 30 episodes dying on max_turns in the
# -v2 arm. So the wasted turn is not free.
#
# Advance the plan rather than repeat the grasp -- the same mechanism as the graduated
# _rule_ppc2c_skip_failed, and deliberately NOT skip_s1 (a skipped turn does not step the env, so
# _check_success can never fire; measured 4/8 -> 0/8 on TurnOnMicrowave).
COFFEE_SKIP_FAILED = os.environ.get("SYS2_COFFEE_SKIP_FAILED", "1") not in ("", "0")


def _rule_coffee_skip_failed(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: a "grasp the mug again" re-grasp advances the plan instead of repeating.

    Matched on the "... again" form only (``_REGRASP_AGAIN_RE``), so a genuine FIRST grasp is never
    skipped. Declines when the grasp is the last fine step, since advancing off the end would strand
    System2 with nothing to propose.
    """
    if task != COFFEE or not COFFEE_SKIP_FAILED:
        return {}
    if not _REGRASP_AGAIN_RE.match(subgoal or ""):
        return {}
    # DETERMINISTIC REWRITE, no re-query. Re-asking System2 was tried and it simply refused: handed
    # the closed checklist it answered with a plan_update RE-OPENING the milestone --
    #     <thought>the gripper closed too early and only achieved a shallow, insecure hold...</thought>
    #     <plan_update>- [~] M1: grasp the red mug / * [~] M1.2: grasp the red mug</plan_update>
    #     <subgoal>grasp the red mug again</subgoal>
    # -- and since apply_plan_update merges that, and combined_eval allows only ONE re-query per turn,
    # System1 still ran the re-grasp (ep0 t2: 192 wasted steps). System2's judgement here is wrong --
    # ep0's gripper width was 0.0204, inside the holding band -- but it is wrong CONFIDENTLY and
    # REPEATEDLY, so the rule cannot rely on asking it again.
    #
    # This task's shape is fixed and known, so the next milestone is written out directly. The
    # phrasings are System2's OWN most frequent ones for these steps, so System1 sees in-distribution
    # text: "lift and carry the mug to the coffee machine dispenser" (92 occurrences), "lower the mug
    # under the coffee machine dispenser" (47 as "... and release"), "release the mug" (37), "retract
    # the arm" (170).
    obj = None
    m_obj = re.match(r"^\s*(?:continue\s+to\s+)?grasp\s+(.*?)\s+again\s*$", subgoal or "", re.IGNORECASE)
    if m_obj:
        obj = m_obj.group(1).strip()
    blocks = _blocks(plan)
    cur_id = current_fine_id(plan)
    cur_b = next((b for b in blocks if any(f["fid"] == cur_id for f in b["fine"])), None) \
        or next((b for b in blocks if b["mark"] == "~"), None)
    if cur_b is None:
        return {}
    if obj is None:      # fall back to the milestone's own wording ("grasp the mug", "pick up the mug")
        m2 = re.match(r"^(?:grasp|pick\s+up)\s+(.*)$", cur_b["text"].strip(), re.IGNORECASE)
        obj = m2.group(1).strip() if m2 else "the mug"

    def _num(mid):
        return int(re.sub(r"[^0-9]", "", mid) or 0)

    nxt = next((b for b in blocks if _num(b["mid"]) > _num(cur_b["mid"]) and b["mark"] != "x"), None)
    if nxt is None:
        return {}                                  # no milestone to move into
    if state.get("coffee_skip_closes", 0) >= 1:
        return {}                                  # ONE rewrite per episode; do not re-plan repeatedly
    steps = [f"lift and carry {obj} to the coffee machine dispenser",
             f"lower {obj} under the coffee machine dispenser",
             f"release {obj}",
             "retract the arm"]
    for f in cur_b["fine"]:                        # the grasp milestone is finished
        f["mark"] = "x"
    cur_b["mark"] = "x"
    nxt["mark"] = "~"                              # and the next one is now ongoing...
    nxt["fine"] = [{"mark": ("~" if i == 0 else " "), "fid": f"{nxt['mid']}.{i + 1}", "text": t}
                   for i, t in enumerate(steps)]   # ...with the known decomposition written out
    state["coffee_skip_closes"] = state.get("coffee_skip_closes", 0) + 1
    new_sg = steps[0]
    # est is ASSIGNED, not left to resolution. On this turn the est proposals were computed from the
    # subgoal System2 asked for ("grasp the mug again"), so coffee_grasp_est's grasp bump was winning
    # est_resolve and the CARRY inherited it -- observed 75 in ep0 and 100 in ep1, i.e. an accidental
    # value derived from a subgoal that no longer exists. est_assign is applied after est_resolve, so
    # this pins the rewritten step regardless of what the other rules proposed or what order they ran
    # in. COFFEE_M2_EST (100) is the same value coffee_m2_est gives every other M2.x step.
    return {"plan": _render(blocks), "subgoal": new_sg, "subgoal_detail": new_sg,
            "est_assign": COFFEE_M2_EST,
            "interventions": [
                {"rule": "coffee_skip_failed", "kind": "plan_revised",
                 "detail": "System2 judged the mug grasp failed, but a held mug reads ~0.012 on this "
                           f"thin handle (same band as the carry). {cur_b['mid']} marked done and "
                           f"{nxt['mid']} written out as its known 4 steps, no re-query (System2 was "
                           "asked and re-opened the milestone instead)",
                 "before": f"{cur_b['mid']} [~], {nxt['mid']} bare",
                 "after": f"{cur_b['mid']} [x], {nxt['mid']} [~] with "
                          f"{nxt['mid']}.1..{nxt['mid']}.{len(steps)}"},
                {"rule": "coffee_skip_failed", "kind": "subgoal_override",
                 "detail": "run the first step of the next milestone instead of the re-grasp",
                 "before": subgoal, "after": new_sg}]}


DRAWER = "PickPlaceDrawerToCounter"
_DRAWER_TARGET_RE = re.compile(r"^grasp\b", re.IGNORECASE)
_AGAIN_SUFFIX = " again"
# Fingers closed below this = closed on nothing (graduated drawer re-grasp below).
DRAWER_MISS_WIDTH = float(os.environ.get("SYS2_RULES_DRAWER_MISS_WIDTH", "0.010"))


# GRADUATED: PickPlaceDrawerToCounter -- a MONITORED recovery re-grasp that never touches the plan.
#
# Measured on the same 20 episodes:
#     baseline (no rules)                          14/20   mean  7.6 turns
#     verified rules only                          16/20   mean  8.7   (also 19, 18, 18 in three
#                                                                        further runs of that config)
#     + "grasp X again" INSERTED into the plan      17/20   mean 10.7  <- rejected: +2 turns, and
#                                                                        inside the verified spread
#     + this rule, run a                           19/20   mean  8.2
#     + this rule, run b                           20/20   mean  8.2  <- replicated, no turn cost
#
# WHY NOT PLAN SURGERY HERE, when PickPlaceCounterToCabinet needs exactly that: a CONTINUATION is
# part of the intended sequence and benefits from being a real step (and gives repeat_cap a non-last
# step to advance into); a RECOVERY is a response to a physical failure and should not rewrite the
# plan System2 authored. So this one is monitored and injected, and the carry continuation is not.
#
# The gate is deliberately tight and therefore rare -- it tripped on 2 of 20 grasp segments in one
# run and 1 of 20 in each of these two. At n=20 its own contribution cannot be separated from the
# verified rules; what is established is that it costs nothing when it does not fire. Widening is NOT
# free: 0.015 would catch two successful measuring-cup grasps, and re-grasping a held object risks
# dropping it.
# =============================================================================================


def _target_step(plan: str, milestone: str | None, target_re):
    """The first not-yet-done fine step matching ``target_re``, or None.

    ``milestone=None`` searches every milestone -- needed when System2 numbers the step differently
    across episodes (ScrubCuttingBoard puts the scrub at M2.2 in most plans and M3.2 in others).
    """
    for b in _blocks(plan):
        if milestone is not None and b["mid"] != milestone:
            continue
        for f in b["fine"]:
            t = f["text"].rstrip(".").lower()
            # SKIP steps already marked [x]. Without this the docstring's "not-yet-done" was a lie:
            # in a multi-grasp milestone ([x] grasp the apple / [~] grasp the banana) the COMPLETED
            # apple was returned and the banana never monitored. Returning None once the step is
            # done is what missing_means_done already expects, so single-grasp plans are unchanged.
            if f["mark"] == "x":
                continue
            if target_re.match(t) and not t.endswith("again") and not t.startswith("continue to"):
                return f
    return None


def _inject_after_step(task_key: str, plan: str, subgoal: str, state, milestone: str,
                       target_re, extra_fn, rule: str, why, tx_label: str,
                       phase2_gate=None, max_injections: int = 1,
                       missing_means_done: bool = False) -> dict:
    """Borrow turn(s) for an extra attempt once the target step completes. Plan NEVER modified.

    System2's own subgoal is HELD, not discarded: it is stashed on the first borrowed turn and
    executed once the injections are done. Without that, System2 would re-plan on the next turn
    having seen the clip of a segment it never asked for, so the borrowed turn would perturb its
    decision instead of merely delaying it.

    ``tx_label`` marks the borrowed turns in the record ("tx_sg_failed" for a recovery after a
    detected failure, "tx_sg_incomplete" for a continuation), so the GUI track can distinguish an
    injected turn from one System2 asked for.

    ``phase2_gate`` is checked ONLY at injection time -- never on the remember phase, where a
    condition on the previous segment's outcome is not yet meaningful.

    ``missing_means_done`` also treats the remembered step VANISHING from the plan as completion.
    System2 drops the fine steps of a milestone when it marks that milestone [x] -- observed on
    ScrubCuttingBoard, where "* [~] M2.2: scrub the cutting board" is simply absent from the next
    turn's plan rather than becoming [x]. Without this the injection can never fire for a step that
    is the last one in its milestone. OFF by default so rules that already measured with the
    mark-based transition keep their exact behaviour.
    """
    done = state.get(f"{task_key}_injected", 0)
    step = _target_step(plan, milestone, target_re)

    # Release: injections finished, so run the subgoal System2 proposed before we interrupted it.
    if done and state.get(f"{task_key}_held"):
        held = state.pop(f"{task_key}_held")
        if done >= max_injections:
            return {"subgoal": held, "subgoal_detail": held,
                    "interventions": [{"rule": rule, "kind": "tx_resume",
                                       "detail": f"borrowed {done} turn(s); resuming the subgoal "
                                                 "System2 proposed before the interruption",
                                       "before": subgoal, "after": held}]}
        state[f"{task_key}_held"] = held      # more injections to come, keep holding

    if done >= max_injections:
        return {}
    if step is None and not (missing_means_done and state.get(f"{task_key}_text")):
        return {}
    cur = current_fine_id(plan)
    # Phase 1 -- target in progress: remember it, and COUNT System2's own re-issues.
    #
    # A borrowed turn must never overlap what System2 already asked for. If System2 itself says
    # "continue to <step>" or "<step> again" while the step is still in progress, that IS the extra
    # attempt, so it consumes one of the max_injections rather than being stacked on top. Counting
    # has to happen HERE and not at injection time: during a re-issue the step is still marked "~",
    # so this phase is the only place those turns are visible.
    if step is not None and (cur == step["fid"] or step["mark"] == "~"):
        state[f"{task_key}_fid"] = step["fid"]
        base = step["text"].rstrip(".")
        state[f"{task_key}_text"] = base
        raw = (subgoal or "").strip().rstrip(".")
        if _norm(raw) == _norm(base) and raw.lower() != base.lower():
            # same instruction, re-issued -- "continue to ..." or "... again"
            state[f"{task_key}_injected"] = done + 1
            return {"interventions": [{"rule": rule, "kind": "tx_counted",
                                       "detail": f"System2 re-issued {step['fid']} itself "
                                                 f"({done + 1}/{max_injections} attempts used); "
                                                 "no borrowed turn added",
                                       "before": subgoal, "after": subgoal}]}
        return {}
    # Phase 2 -- a target we saw in progress is now done: borrow this turn. "Done" is either the
    # step marked [x], or (with missing_means_done) the step no longer present at all.
    if not state.get(f"{task_key}_text"):
        return {}
    if step is not None and step["mark"] != "x":
        return {}
    if phase2_gate is not None and not phase2_gate():
        return {}
    extra = extra_fn(state[f"{task_key}_text"])
    if _norm(subgoal) == _norm(extra):
        state[f"{task_key}_injected"] = done + 1    # System2 asked for it already
        return {}
    state.setdefault(f"{task_key}_held", subgoal)   # hold System2's output (first borrowed turn)
    state[f"{task_key}_injected"] = done + 1
    # HOLD SYSTEM2 FOR THE NEXT TURN. The subgoal System2 just proposed has NOT run -- System1 is
    # about to run the borrowed one instead -- so there is nothing for System2 to judge next turn and
    # nothing new for it to decide. Querying it anyway was a real defect: its answer had to be thrown
    # away by tx_resume, yet its <plan_update> was still applied, so the checklist advanced past a
    # step System1 had not finished and the fine step it proposed was SWALLOWED (PackIdenticalLunches
    # ep0: "search for the counter" was proposed at t28, discarded, and marked [x] at t29 without ever
    # being executed). The loop consumes this via pending_resume() and skips the S2 call entirely, so
    # the held subgoal runs against an unchanged plan and System2's next query sees the clip of THAT
    # segment. apply_rules fills in subgoal_detail/est below.
    state["_resume_pending"] = {"key": task_key, "rule": rule, "subgoal": subgoal}
    return {"subgoal": extra, "subgoal_detail": extra, "tx_label": tx_label,
            "interventions": [{"rule": rule, "kind": tx_label,
                               "detail": f"{state[f'{task_key}_fid']} completed; borrowed turn "
                                         f"{done + 1}/{max_injections} "
                                         f"({why() if callable(why) else why}). Plan untouched; "
                                         "System2's subgoal is held and resumes after.",
                               "before": subgoal, "after": extra}]}


def _rule_drawer_regrasp_recovery(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PickPlaceDrawerToCounter: one injected "grasp X again" turn after the M2 grasp completes."""
    if task != DRAWER:
        return {}
    # RECOVERY ONLY: inject when the grasp closed on nothing. state["grip_width"] is the width left by
    # the PREVIOUS segment (plumbed by combined_eval) -- at injection time that is the grasp being
    # judged. Passed as a phase-2 gate, NOT checked up front: during the grasp turn the recorded width
    # is still the open-gripper value from the reach, so an up-front check would block the remember
    # phase and the rule could never fire.
    def _missed():
        w = state.get("grip_width")
        return w is not None and w < DRAWER_MISS_WIDTH

    return _inject_after_step(
        "drawer", plan, subgoal, state, "M2", _DRAWER_TARGET_RE,
        extra_fn=lambda t: f"{t} again", rule="drawer_regrasp_recovery",
        tx_label="tx_sg_failed", phase2_gate=_missed,
        why=lambda: f"gripper width {state.get('grip_width'):.4f} < {DRAWER_MISS_WIDTH} -- the "
                    "fingers closed on nothing, so the grasp missed")


# GRADUATED: WashFruitColander -- drop a redundant carry CONTINUATION and run the next step.
#
#   arm (same 20 episode ids)                       success   total turns
#   baseline, no rules                                9/20        438
#   verified rules only                               5/20        485
#   verified + this rule, run a                      12/20        380
#   verified + this rule, run b                      13/20        375
#   pooled with the rule                             25/40 = 62.5%
#
#   vs the verified arm    z=+2.74  p=0.006     <- the paired control: identical official-rule
#                                                  profile (sink_faucet_est / est_resolve /
#                                                  repeat_cap all fire at the same rates), only
#                                                  this rule added
#   vs the no-rules arm    z=+1.29  p=0.20
#   P(both runs >= 12 by chance): 1e-5 at the verified rate, 0.017 at the baseline rate.
#   The rule fired 14x / 12x across 20 episodes, and total turns fell 22% -- episodes end in
#   success instead of grinding to max_turns.
#
# READ THE GAIN HONESTLY: the official rules HURT this task (9/20 -> 5/20), so most of this is
# recovering a self-inflicted loss -- 62.5% is only modestly above the 45% no-rules baseline, and
# the comparison against it is not significant. _rule_sink_faucet_est is the prime suspect: it is
# gated on SUBGOAL TEXT, not on task, so it fires here (173 est proposals in 20 episodes) even
# though it was written for TurnOnSinkFaucet, where it measured 0/3. Scoping it to its own task is
# the open follow-up and may capture much of the same gain on other tasks too.
#
# WHY IT WORKS -- not the reason predicted. The turn saving is trivial (repeat_cap already caps the
# carry continuation at 3, so at most 2 turns on 8 episodes) and cannot convert an episode, because
# every failure then burns 6-19 turns in a faucet-handle loop that this rule never touches. The
# effective channel is PHYSICAL: releasing the colander after ONE carry segment instead of up to
# four changes where it lands in the basin, and correct placement is a precondition for the
# fruit-washing and faucet steps that follow.
WFC = "WashFruitColander"
# RAW-subgoal match. _norm() strips the "continue to " prefix, so a normalised match could not tell
# the FIRST issue of the carry from a re-issue -- and only the re-issue is redundant. Requires both
# "colander" and "sink" so the fruit carries ("continue to carry the tangerine to the colander",
# which also occur) can never match.
_WFC_CONT_CARRY_RE = re.compile(r"^\s*continue\s+to\s+(carry|move|bring|lift\s+and\s+carry)\b",
                                 re.IGNORECASE)
_WFC_CARRY_STEP_RE = re.compile(r"\b(carry|move|bring)\b", re.IGNORECASE)
_WFC_RELEASE_STEP_RE = re.compile(r"\b(lower|release|place)\b", re.IGNORECASE)
WFC_MAX_FIRES = int(os.environ.get("SYS2_RULES_WFC_MAX_FIRES", "1"))
_WFC_STATE_KEY = "wfc_carry_skip_n"


def _rule_wfc_skip_carry_continue(task: str, plan: str, subgoal: str, est, state) -> dict:
    """WashFruitColander: "continue to carry the colander to the sink" -> run the NEXT step."""
    if task != WFC:
        return {}
    raw = subgoal or ""
    low = raw.lower()
    if not _WFC_CONT_CARRY_RE.match(raw) or "colander" not in low or "sink" not in low:
        return {}
    n = state.get(_WFC_STATE_KEY, 0)
    if n >= WFC_MAX_FIRES:
        return {"interventions": [{"rule": "wfc_skip_carry_continue", "kind": "cap_declined",
                                   "detail": f"already fired {n}x this episode "
                                             f"(WFC_MAX_FIRES={WFC_MAX_FIRES}) -- leave the rest "
                                             "to repeat_cap",
                                   "before": raw, "after": raw}]}
    # Only advance off the CARRY step. If the current step is already the lower/release one
    # (repeat_cap got there first), advancing would skip the release and strand the colander in the
    # gripper -- which is unrecoverable, so this branch declines rather than guessing.
    cur = current_fine_id(plan)
    cur_text = next((f["text"] for b in _blocks(plan) for f in b["fine"] if f["fid"] == cur), "")
    ct = cur_text.lower()
    if not cur_text or "colander" not in ct or not _WFC_CARRY_STEP_RE.search(ct) \
            or _WFC_RELEASE_STEP_RE.search(ct):
        return {"interventions": [{"rule": "wfc_skip_carry_continue", "kind": "advance_declined",
                                   "detail": f"current fine step {cur} is not the colander carry "
                                             f"({cur_text!r}) -- advancing would skip the release",
                                   "before": raw, "after": raw}]}
    out = _advance_current_step(
        plan, raw, "wfc_skip_carry_continue",
        "System2 judged the colander carry incomplete; give it no second segment -- mark the carry "
        "done and run the release instead")
    if any(i.get("kind") == "plan_revised" for i in out.get("interventions", [])):
        state[_WFC_STATE_KEY] = n + 1
    return out


# GRADUATED: GetToastedBread -- a WAIT subgoal runs a forced segment; the slot approach gets an est
# floor. Authored and measured on a second machine; the numbers below are read from the shared
# results mount.
#
#   arm (same 20 episode ids)                        success
#   verified rules only (estbump sweep)               15/20
#   debug-rules-v4-estbump / debug-cap800             16/20
#   verified + these rules (debug-...-wait1200)       17/20
#
# THE SUCCESS DELTA IS NOT THE EVIDENCE: 17/20 vs 15/20 is p=0.43, and this task has sat at 15-17/20
# across four recent runs -- squarely inside the n=20 band. The rule's own author says a 1-2 episode
# outcome must not be read as a win or a loss, and that is respected here.
#
# WHAT JUSTIFIES GRADUATION is a STRUCTURAL DEFECT the rule fixes, measured directly. The stop rule
# is "progress at threshold AND the arm has gone quiescent". While waiting for a toaster the arm
# commands no motion BY DEFINITION, so quiescence is satisfied on the first window and the segment
# ends after a few dozen steps with the toaster still running. In the recorded runs the wait segments
# ended on the STOP RULE or lookahead -- voluntarily, not out of budget -- at a mean of 425 steps,
# and System2 then re-issued "continue to wait" 14 times at a mean of 136 steps each. That re-issue
# storm is the symptom. Forcing the segment length is the fix; no est bump can reach it, because est
# only raises the budget and the budget was never what cut the wait short.
# Confirmed in the run: 13 forced segments, all exactly 1200 steps.
#
# NOT ACTUALLY TASK-SPECIFIC -- the open follow-up. Any task whose plan contains a wait/hold has this
# same defect, so scoping the fix to one task leaves the others broken. sys2_client.is_wait_subgoal
# already recognises the family (hold/pause/stay/settle/remain) and SYS2_GTB_WAIT_ANY=1 selects it
# here for measurement. The right end state is probably a task-agnostic wait rule, or a stop rule
# that does not treat a deliberately still arm as converged.
#
# force_steps BYPASSES --max-steps-cap by design (see run_s1_segment), so the wait gets its full
# length whatever the global cap is. The dense _check_success() break is NOT suppressed: if the bread
# pops mid-wait the episode ends there, which is the entire point of waiting.
GTB = "GetToastedBread"
# REVISED 1200 -> 800 to hold total executed steps down: on the recorded runs "wait for the bread
# to pop up" ran a mean of 958 steps (44 turns, max 1200) and the slot+wait family sat at the full
# 1200, so the forced window was the single largest consumer of steps on this task.
#
# TWO WINDOWS, because System2 emits two shapes of wait and they are not the same job:
#   PURE wait   "wait for the bread to pop up" (44 turns)          -> GTB_WAIT_FORCE_STEPS  800
#   COMPOUND    "reach to the toaster slot and wait" (6),          -> GTB_WAIT_COMPOUND_STEPS 950
#               "move above the toaster and wait for the bread" (5),
#               "move over the toaster and wait for the bread" (4),
#               "reach to the bread and wait" (4), "move to the toaster slot and wait" (2)
# A compound subgoal is TWO subgoals merged -- travel to the slot AND then wait there -- so it
# needs the approach time on top of the wait itself. All of these ran to the full 1200 under the
# old single window, i.e. they were the ones actually using it.
GTB_WAIT_FORCE_STEPS = int(os.environ.get("SYS2_GTB_WAIT_STEPS", "800"))
GTB_WAIT_COMPOUND_STEPS = int(os.environ.get("SYS2_GTB_WAIT_COMPOUND_STEPS", "950"))
# A wait PRECEDED by a motion verb: the subgoal has to get somewhere before it can wait.
_GTB_MOTION_THEN_WAIT_RE = re.compile(r"^(?:continue\s+to\s+)?(?:reach|move|go)\b")
# REVISED from a FLOOR of 200 to a CAP of 150, same motive: System2 asks 500-600 for a short
# positioning move (observed est 75/100/125/150/200/600 on "move the gripper over the toaster
# slot", executing a mean of 216 steps), so the floor was pushing budget up on a segment that did
# not need it. A cap needs est_assign, since est_proposal resolves as the largest and can only
# raise.
GTB_SLOT_EST_CAP = int(os.environ.get("SYS2_GTB_SLOT_EST", "150"))
# 1 = widen the wait family to sys2_client.is_wait_subgoal (hold/pause/stay/settle/remain anywhere).
# Default 0: only "wait" was measured, and forcing 1200 steps of the wrong subgoal is expensive.
GTB_WAIT_ANY = bool(os.environ.get("SYS2_GTB_WAIT_ANY"))

# ANY occurrence of "wait" arms the rule -- the waiting is what needs the forced budget even as the
# tail of a compound subgoal ("reach to the toaster slot and wait"). \bwait also covers waits/waiting.
_GTB_WAIT_ANYWHERE_RE = re.compile(r"\bwait")
# ...EXCEPT a subgoal OPENING with "continue to wait", which is System2 re-issuing a wait that has
# already had its forced segment. Matched on the RAW text: _norm() strips "continue to ", so a
# normalised match could not tell a fresh wait from a re-issue.
_GTB_CONT_WAIT_RE = re.compile(r"^continue\s+to\s+wait\b")
# "move the gripper over/above/onto ... slot" -- loose enough to survive rewording, but the slot must
# be the target so it cannot collide with the lever steps that action_overrides owns.
# Covers every slot-positioning phrasing seen: "move the gripper over/above/onto ... slot",
# "move over the toaster slot", "move/reach to the toaster slot".
_GTB_SLOT_RE = re.compile(
    r"^(?:move|reach)\s+(?:the\s+gripper\s+)?(?:over|above|onto|to|towards?)\b.*\bslot\b")


def _gtb_is_wait(subgoal: str | None) -> bool:
    """True for a wait subgoal that has not already been re-issued as "continue to wait ..."."""
    raw = re.sub(r"\s+", " ", (subgoal or "").strip().lower().rstrip("."))
    if _GTB_CONT_WAIT_RE.match(raw):
        return False
    if GTB_WAIT_ANY:
        # Imported lazily so this module keeps a stdlib-only import footprint (it is unit-tested on
        # its own, and sys2_client pulls in the media/encoding stack).
        from sys2_client import is_wait_subgoal
        return bool(is_wait_subgoal(raw))
    return bool(_GTB_WAIT_ANYWHERE_RE.search(raw))
# ---- the bread grasp becomes a reach-and-grasp ---------------------------------------------------
# After the toaster wait the gripper is parked ABOVE the slot, not around the bread, so a bare
# "grasp ..." asks System1 to close the fingers from wherever it happens to be. Making it an explicit
# reach-and-grasp gives the approach back. System2 nearly always says the bare form -- pooled over four
# recorded runs: "grasp the bread" 97, "grasp the sandwich bread" 15, "reach and grasp the bread" only
# 2 -- so this is a rewrite, not a preference between two things it already does.
GTB_REACH_GRASP = os.environ.get("SYS2_GTB_REACH_GRASP", "1") not in ("", "0")
# A BARE grasp (no "reach and" already, and not a "... again" re-issue, which other rules own).
_GTB_BARE_GRASP_RE = re.compile(r"^(?:continue\s+to\s+)?grasp\b(?!.*\bagain\b)", re.IGNORECASE)


def _rule_gtb_reach_grasp(task: str, plan: str, subgoal: str, est, state) -> dict:
    """GetToastedBread: "grasp the bread" -> "reach and grasp the bread"."""
    if task != GTB or not GTB_REACH_GRASP:
        return {}
    raw = (subgoal or "").strip()
    low = raw.lower()
    if "reach and grasp" in low or not _GTB_BARE_GRASP_RE.match(low):
        return {}
    # Rewrite the leading verb, preserving any "continue to " prefix and the object wording.
    new_sg = re.sub(r"^(\s*(?:continue\s+to\s+)?)grasp\b", r"\1reach and grasp", raw, count=1,
                    flags=re.IGNORECASE)
    if new_sg == raw:
        return {}
    # Keep the checklist in step with what System1 is told, so System2 conditions on the same text.
    blocks = _blocks(plan)
    cur = current_fine_id(plan)
    changed = False
    for b in blocks:
        for f in b["fine"]:
            if f["fid"] == cur and f["text"].strip().lower() == low:
                f["text"] = new_sg
                changed = True
    out = {"subgoal": new_sg, "subgoal_detail": new_sg,
           "interventions": [{"rule": "gtb_reach_grasp", "kind": "subgoal_override",
                              "detail": "after the wait the gripper is parked above the slot, not "
                                        "around the bread; ask for the approach explicitly",
                              "before": raw, "after": new_sg}]}
    if changed:
        out["plan"] = _render(blocks)
    return out


# ---- a wait CONTINUATION is redundant ------------------------------------------------------------
# The forced window is already long (800 pure / 950 compound) and the dense _check_success() ends the
# segment the moment the task completes, so a second forced window on the SAME wait buys nothing --
# it just spends another 950 steps. Measured on debug-gtb-addwait: every "continue to ... wait" turn
# ran the full 950, and the plan underneath was always the same shape, with the next step ready:
#     * [~] M2.1 move over the toaster slot and wait   <-- the continuation re-runs THIS
#     * [ ] M2.2 grasp the bread                       <-- ...instead of this
#     * [ ] M2.3 lift the bread
# So: mark the wait done and hand System1 the next fine step, which is "grasp the bread" every time.
#
# Matches BOTH "continue to wait ..." and "continue to <motion> ... and wait" -- the first was already
# exempt from forcing (it fell back to the ordinary stop rule) but still consumed a turn re-issuing a
# finished wait; the second was being forced a second time.
# Default OFF: measured 22/30 raw vs 27/30 for addwait alone, with task_finish 3 -> 8. Those
# continuations were NOT waste -- the bread genuinely had not popped, so advancing to the grasp closed
# the fingers on an empty slot. Kept because the detection is useful and the trade may differ once the
# forced window is retuned; SYS2_GTB_SKIP_WAIT_CONT=1 enables it.
GTB_SKIP_WAIT_CONT = os.environ.get("SYS2_GTB_SKIP_WAIT_CONT", "0") not in ("", "0")
# TWO INDEPENDENT term searches rather than one anchored pattern: "continue to" is not always the very
# first token, and the wait can sit anywhere ("continue to move the gripper over the toaster and wait
# for the bread"). Anchoring on ^continue would silently miss any prefixed variant.
_GTB_CONTINUE_RE = re.compile(r"\bcontinue\s+to\b", re.IGNORECASE)
_GTB_WAIT_WORD_RE = re.compile(r"\bwait\b", re.IGNORECASE)


def _gtb_is_wait_continuation(subgoal: str | None) -> bool:
    """A re-issue of a wait: contains BOTH "continue to" and the word "wait", in any order."""
    raw = subgoal or ""
    return bool(_GTB_CONTINUE_RE.search(raw) and _GTB_WAIT_WORD_RE.search(raw))


def _rule_gtb_skip_wait_cont(task: str, plan: str, subgoal: str, est, state) -> dict:
    """GetToastedBread: a "continue to ... wait" re-issue advances the plan instead of waiting again."""
    if task != GTB or not GTB_SKIP_WAIT_CONT:
        return {}
    if not _gtb_is_wait_continuation(subgoal):
        return {}
    # Only skip a wait that HAS already had its forced window this episode; otherwise a "continue to"
    # arriving first (System2 occasionally opens with one) would skip a wait that never ran.
    if not state.get("gtb_forced_wait"):
        return {}
    return _advance_current_step(
        plan, subgoal, "gtb_skip_wait_cont",
        "the wait already ran its forced window (800 pure / 950 compound) and dense _check_success "
        "would have ended it on completion, so re-waiting only spends another 950 steps; advance to "
        "the next step instead")


# ---- M2 unrolled with no wait at all ------------------------------------------------------------
# The toaster needs TIME. When System2 unrolls the pick-up milestone WITHOUT a wait step it goes
# straight from positioning to grasping, and grasps at a slot the bread has not popped out of yet.
# Measured over the recorded runs, episodes in which NO subgoal ever contains "wait":
#     base  5/30 episodes, success 2/5  (40%)   vs 19/30 (63%) overall
#     v2    9/30 episodes, success 5/9  (56%)   vs 25/30 (83%) overall
# and the unroll is the same every time, e.g. base ep3/ep16/ep19:
#     * M2.1 move the gripper over the toaster slot
#     * M2.2 grasp the bread
#     * M2.3 lift the bread (and retract the arm)
#
# So: the FIRST step of that milestone becomes "<text> and wait". That single edit converts it into a
# compound move-then-wait, which _rule_gtb_wait_force_steps then recognises and runs for
# GTB_WAIT_COMPOUND_STEPS (950) with the stop rule suppressed -- i.e. the arm travels to the slot and
# then holds there while the toaster finishes, instead of grabbing at nothing.
#
# Gated on the milestone being the PICK-UP one and on no wait having been seen anywhere in the episode
# so far (tracked in state), so an episode whose planner did include a wait is untouched.
GTB_ADD_WAIT = os.environ.get("SYS2_GTB_ADD_WAIT", "1") not in ("", "0")
_GTB_PICKUP_MILESTONE_RE = re.compile(r"\b(?:pick\s+up|pick|retrieve|take)\b.*\bbread\b", re.IGNORECASE)


def _rule_gtb_add_wait(task: str, plan: str, subgoal: str, est, state) -> dict:
    """GetToastedBread: if the pick-up milestone unrolls with NO wait, append " and wait" to its first
    step so the arm holds at the slot until the bread pops.
    """
    if task != GTB or not GTB_ADD_WAIT:
        return {}
    # Remember, for the whole episode, whether System2 has ever asked to wait.
    if "wait" in (subgoal or "").lower():
        state["gtb_saw_wait"] = True
        return {}
    if state.get("gtb_saw_wait") or state.get("gtb_added_wait"):
        return {}
    blocks = _blocks(plan)
    cur_id = current_fine_id(plan)
    cur_b = next((b for b in blocks if any(f["fid"] == cur_id for f in b["fine"])), None)
    if cur_b is None or not _GTB_PICKUP_MILESTONE_RE.search(cur_b["text"]):
        return {}
    if any("wait" in f["text"].lower() for f in cur_b["fine"]):
        return {}                       # the milestone already has a wait somewhere in it
    first = cur_b["fine"][0] if cur_b["fine"] else None
    # Only rewrite while the FIRST step is the one being run: once the grasp is under way the bread has
    # either popped or the episode is already lost, and appending a wait then would just burn steps.
    if first is None or first["fid"] != cur_id:
        return {}
    new_text = f"{first['text'].rstrip('.')} and wait"
    first["text"] = new_text
    state["gtb_added_wait"] = True
    return {"plan": _render(blocks), "subgoal": new_text, "subgoal_detail": new_text,
            "interventions": [
                {"rule": "gtb_add_wait", "kind": "plan_revised",
                 "detail": f"{cur_b['mid']} unrolled with no wait step, so the grasp would happen "
                           f"before the bread pops; {first['fid']} becomes a move-then-wait, which "
                           f"the wait rule then runs for {GTB_WAIT_COMPOUND_STEPS} forced steps",
                 "before": f"{first['fid']}: {new_text[:-9]}", "after": f"{first['fid']}: {new_text}"},
                {"rule": "gtb_add_wait", "kind": "subgoal_override",
                 "detail": "hand System1 the move-then-wait form",
                 "before": subgoal, "after": new_text}]}


def _rule_gtb_wait_force_steps(task: str, plan: str, subgoal: str, est, state) -> dict:
    """GetToastedBread: a wait subgoal runs a forced GTB_WAIT_FORCE_STEPS steps.

    Overrides BOTH the stop rule and --max-steps-cap; see the banner for why no est bump can do this.
    """
    if task != GTB or not _gtb_is_wait(subgoal):
        return {}
    # ONE forced window per episode. A CONTINUATION of a wait that has already had its window falls
    # back to System2's own est (budget = est * horizon_mult, capped), instead of being granted a
    # second full window. Measured on debug-gtb-v3 ep1: t3 forced 950, then System2 asked to continue
    # and t4 was forced 950 AGAIN -- 1900 steps on waiting in one episode, and it still failed. The
    # alternative already tried, SKIPPING the continuation outright, was worse (22/30 vs 27/30) because
    # the bread sometimes genuinely has not popped; letting it wait at its own est keeps the wait but
    # bounds it.
    if state.get("gtb_forced_wait"):
        return {"interventions": [{"rule": "gtb_wait_force_steps", "kind": "force_exempt",
                                   "detail": "a forced wait window already ran this episode; this "
                                             "continuation uses System2's own est instead of a "
                                             "second full window",
                                   "before": "forced window", "after": f"est {est}"}]}
    raw = re.sub(r"\s+", " ", (subgoal or "").strip().lower())
    compound = bool(_GTB_MOTION_THEN_WAIT_RE.match(raw))
    steps = GTB_WAIT_COMPOUND_STEPS if compound else GTB_WAIT_FORCE_STEPS
    state["gtb_forced_wait"] = True      # so gtb_skip_wait_cont knows a window has already run
    return {"force_steps": steps,
            "interventions": [{"rule": "gtb_wait_force_steps", "kind": "force_steps",
                               "detail": (f"{'compound move-then-wait' if compound else 'pure wait'} "
                                          f"subgoal -- run a forced {steps} steps (stop rule "
                                          "suppressed; a quiescent arm satisfies it immediately while "
                                          "the toaster is still running). Dense _check_success still "
                                          "ends the segment on completion."
                                          + (" A compound subgoal is two merged -- travel plus wait -- "
                                             "so it gets the longer window." if compound else "")),
                               "before": "stop rule + budget",
                               "after": f"forced {steps} steps"}]}


def _rule_gtb_slot_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """GetToastedBread: a slot-positioning subgoal gets est_length CAPPED at GTB_SLOT_EST_CAP.

    A CAP, not a floor (it was a floor of 200): System2 asks 500-600 for what executes ~216 steps, and
    the point of the revision is to stop the budget being inflated on a short positioning move. Uses
    est_assign because est_proposal resolves as the largest and therefore cannot lower.

    WAIT SUBGOALS ARE EXCLUDED. Several slot phrasings also end in "and wait" ("reach to the toaster
    slot and wait", "move to the toaster slot and wait"); those are the wait rule's, whose force_steps
    governs the segment length outright. Capping their est too would tell System1 "150 steps" while the
    forced window runs 800 -- a conditioning tag contradicting the actual segment.
    """
    if task != GTB or not _GTB_SLOT_RE.match(_norm(subgoal)):
        return {}
    if _gtb_is_wait(subgoal):
        return {}
    if not isinstance(est, int) or est <= GTB_SLOT_EST_CAP:
        return {}
    return {"est_assign": GTB_SLOT_EST_CAP,
            "interventions": [{"rule": "gtb_slot_est", "kind": "est_assign",
                               "detail": f"slot positioning is a short move (mean 216 executed "
                                         f"steps); cap est at {GTB_SLOT_EST_CAP} instead of the "
                                         f"planner's {est} to hold total executed steps down",
                               "before": est, "after": GTB_SLOT_EST_CAP}]}


# GRADUATED: WeighIngredients -- the extra carry as a BORROWED TURN, and M1's location carried down
# onto the M1 reach subgoal.
#
#   arm (same 20 episode ids)                       success   mean turns
#   baseline, no rules                                4/20       14.3
#   verified rules only                               2/20       14.4
#   ref control (today's code, no experiments)        1/20        -
#   the SAME extra carry as a PLAN INSERT             2/20       14.7
#   these two rules (borrowed turn + located reach)   6/20       13.2
#
# READ WITH THE CAVEAT: +4 over the verified arm sits exactly AT the n=20 resolution limit (~+-4 at
# 2SD), and this task's own near-replicates span 1-4/20, so ONE run cannot separate a real +4 from a
# lucky draw. Graduated on the session owner's decision without a confirming repeat. The two rules
# also ran TOGETHER and fired in overlapping episodes, so the gain is NOT attributed between them.
# What is solid is the mechanism trace: 9/20 episodes took a borrowed turn (tx_sg_incomplete 9,
# tx_resume 9) with System2's own re-issues consuming the budget 10 times (tx_counted), 9 reach
# subgoals were located across 6/20 episodes, and the plan stayed byte-identical throughout.
#
# BORROWED TURN vs PLAN INSERT is the informative comparison: identical targeted mechanism, 6/20
# against 2/20. The plan insert fired in 20/20 episodes and System2 executed the inserted step 51
# times, so its 2/20 was not a plumbing failure -- putting the continuation in the checklist is simply
# worse than borrowing one turn for it, which is the opposite of the PickPlaceCounterToCabinet result
# and is why that rule keeps its plan-insert form.
#
# WHY THE CARRY. _check_success needs FOUR things: gripper far, object in the digital-scale
# receptacle, object upright, cabinet closed. ep0 ends with the jam ON ITS SIDE beside the scale while
# System2's thought reads "the jam is now resting on the digital scale" -- it cannot see the miss, so
# the placement condition is never met and no amount of door-pushing could succeed. A second carry
# segment before the release is what addresses that.
#
# WHY THE LOCATION. System2 states where the object is in the milestone and usually drops it from the
# fine step, so System1 is told to "reach to the honey bottle" with no location. Roughly half the M1
# texts carry a phrase ("grasp the jam from the cabinet" 207, "grasp the canned food from the shelf"
# 60) and half carry none ("grasp the honey bottle" 255), where this is a no-op.
#
# THE UNFIXED SINK, which neither rule touches: 14 of 20 still end in max_turns, and the terminal
# "continue to push the cabinet door closed" loop accounts for 13 of 18 failures in the verified arm.
# 125 door-push segments over 20 episodes, mean 97 steps, and 106 of 125 end on the STOP RULE rather
# than on budget -- an arm stalled against a door commands no motion, so action quiescence reads
# "converged". That is the same structural defect as the graduated GetToastedBread wait, and
# force_steps is the lever for it; an est bump cannot help, since only 11 of 125 were budget-limited.
WI = "WeighIngredients"
# The carry step, matched on TEXT rather than a milestone id: System2 numbers it M2.1 in every observed
# plan, but the id is not guaranteed and the phrasing varies ("lift and carry the yogurt to the
# digital scale").
_WI_CARRY_RE = re.compile(r"\bcarry\b.*\bscale\b", re.IGNORECASE)
# M1's trailing location phrase, captured WITH its preposition so it can be appended verbatim.
_WI_M1_PHRASE_RE = re.compile(r"\b((?:from|in|out\s+of|on)\s+the\s+[a-z][a-z\s]*)$", re.IGNORECASE)
_WI_REACH_RE = re.compile(r"^(?:continue\s+to\s+)?reach\b", re.IGNORECASE)
# Door/handle reaches belong to the closing milestone, never to M1. This matters: the reach subgoals
# in this task are DOMINATED by the door ("continue to reach to the cabinet door" 143, "reach to the
# cabinet door" 59, against "reach to the honey bottle" 21), and locating those would be nonsense.
_WI_DOOR_RE = re.compile(r"\b(door|handle|drawer)\b", re.IGNORECASE)
# 1 = rewrite a leading "from" as "in" ("reach to the jam in the cabinet"), System2's own idiom for a
# located reach. Default 0: the milestone's phrase is appended verbatim, which is what was measured.
WI_PHRASE_IN = bool(os.environ.get("SYS2_WI_PHRASE_IN"))


def _rule_wi_extra_carry_turn(task: str, plan: str, subgoal: str, est, state) -> dict:
    """WeighIngredients: borrow ONE turn for "continue to carry ... to the digital scale".

    Plan-preserving: the checklist is untouched, System2's subgoal is held and resumed on the next
    turn, and System2's own "continue to ..." re-issue consumes the same single-injection budget so
    the two can never stack. missing_means_done=True is required -- System2 DROPS a milestone's fine
    steps when it marks the milestone [x], so the carry step vanishes rather than becoming [x].
    """
    if task != WI:
        return {}
    return _inject_after_step(
        "wi_carry", plan, subgoal, state, None, _WI_CARRY_RE,
        extra_fn=lambda t: f"continue to {t}", rule="wi_extra_carry_turn",
        tx_label="tx_sg_incomplete", max_injections=1, missing_means_done=True,
        why="one more carry segment before the release, so the object ends ON the digital scale "
            "rather than beside it (the placement condition the ep0-shape failures never satisfy)")


def _rule_wi_reach_locate(task: str, plan: str, subgoal: str, est, state) -> dict:
    """WeighIngredients: append M1's location phrase to the M1 reach subgoal.

    SUBGOAL ONLY -- the plan is not modified. No-op when M1 states no location, when the reach already
    names it, or when the current step is a door reach.
    """
    if task != WI:
        return {}
    if not _WI_REACH_RE.match((subgoal or "").strip()):
        return {}
    if _WI_DOOR_RE.search(subgoal or ""):
        return {}
    cur = current_fine_id(plan)
    if not cur or not cur.startswith("M1."):
        return {}
    m1 = next((b["text"] for b in _blocks(plan) if b["mid"] == "M1"), "")
    hit = _WI_M1_PHRASE_RE.search((m1 or "").strip().rstrip("."))
    if not hit:
        return {}
    phrase = re.sub(r"\s+", " ", hit.group(1).strip())
    if WI_PHRASE_IN:
        phrase = re.sub(r"^from\b", "in", phrase, flags=re.IGNORECASE)
    # Already located -> nothing to do. Compare the location NOUN, not the whole phrase: System2
    # writes "reach to the jam IN the cabinet" while M1 says "FROM the cabinet", so a full-phrase
    # match misses it and produced "reach to the jam in the cabinet from the cabinet".
    noun = re.sub(r"^(?:from|in|out\s+of|on)\s+the\s+", "", phrase, flags=re.IGNORECASE).strip()
    if noun and re.search(rf"\b{re.escape(noun)}\b", subgoal or "", re.IGNORECASE):
        return {}
    new = f"{(subgoal or '').strip().rstrip('.')} {phrase}"
    return {"subgoal": new, "subgoal_detail": new,
            "interventions": [{"rule": "wi_reach_locate", "kind": "subgoal_override",
                               "detail": f"M1 says where the object is ({m1.strip()!r}) but the reach "
                                         "step drops it; carry the phrase down so System1 is told "
                                         "where to reach. Plan untouched.",
                               "before": subgoal, "after": new}]}


# =============================================================================================
# GRADUATED: PackIdenticalLunches -- a width-gated re-grasp recovery, plus a reach continuation.
#
# MEASUREMENT (20 episodes, the manifest set, s1-progact270k / S2 ep3-11416, max_steps_cap 400)
#     baseline, no rules at all                                   4/20
#     verified rules only (no PIL rule)                           2/20
#     + re-grasp, window 3, miss<0.010, bare "grasp X again"       2/20   (debug-PIL-regrasp-v1)
#     + re-grasp, window 2, miss<0.015, far form + est 125         8/20   (debug-PIL-regrasp-v3)
# Paired vs the verified-rules arm on all 20: 6 GAINED (eps 5, 8, 11, 13, 18, 19), 0 LOST,
# exact McNemar p=0.031. task_finish 13 -> 9, so 4 of the 6 came out of the dominant failure
# bucket. Against v1: 7 gained / 1 lost, p=0.070. Zero regressions across 20 paired episodes is
# what carries this; +6 is outside the +-4.5-episode (2SD) noise band for n=20 on this task.
#
# WHY IT WORKS. The failure is not a failed grasp: 77 of 80 grasp segments end with the fingers
# around the object (width >=0.015). The object is lost ON THE LIFT -- 33 of the 35 sub-0.010
# segments are the ones AFTER the grasp ("retract the arm with X" at 0.001, then "search for the
# counter") -- so the robot navigates and "places" nothing. 11 of 20 episodes show it and neither
# of the two successes does.
#
# THE THREE SETTINGS THAT MATTERED, and why v1 -> v3 moved 2/20 to 8/20:
#   * WINDOW 2, not 3. A recovery is only worth taking before the robot starts SEARCHING; all four
#     of v1's grasp+3 injections had an intervening "search for ..." turn, displaced a "go to ..."
#     navigation subgoal, and all four failed.
#   * MISS WIDTH 0.015, not 0.010. The width distribution is bimodal with a clean gap
#     (0.0041 -> 0.0107 -> 0.0118 -> 0.0166), so 0.015 sits inside the gap: no false positives, and
#     it admits the four marginal 0.0107-0.0118 cases.
#   * FAR FORM. At offset 1 the arm is still at the object, so a bare "grasp X again" can re-close.
#     At offset 2 it has retracted, so a bare re-grasp closes from the wrong pose: the recovery
#     becomes "reach and grasp X again" with an est floor of 125 to cover the approach. This fired
#     5 times in v3 and is the most likely single cause of the gain.
#
# HONEST LIMITS, recorded so this is not re-litigated from memory:
#   * v3 changed FOUR things at once (window, threshold, far form, and the reach continuation), so
#     the credit is not apportioned between them. One arm at window 2 + 0.010 + no est-125 would
#     isolate the far form.
#   * v3 ran at max_steps_cap 400 vs the reference arm's 800. PIL has ZERO segments over 400 steps,
#     so this is believed inert, but it is not strictly single-variable.
#   * NO REPLICATE was run. A second independent 20 is the cheap confirmation and was recommended.
#   * The re-grasp is only a partial fix: in v1, 7 of 12 injected turns verifiably re-acquired the
#     object (width 0.011-0.068) and NONE of those episodes succeeded, because an episode needs all
#     four placements. task_finish is still 9/20 here.
#
# The reach continuation is graduated because it was part of the measured 8/20 arm, but note it was
# INERT on those episodes: it fired 7 times and every one was tx_counted (System2 had already issued
# "continue to reach ..." itself, so the budget was spent without borrowing a turn). It has never
# actually added a turn in a measured run.
# =============================================================================================

PIL = "PackIdenticalLunches"
# "grasp X" and "reach and grasp X"; NOT "reach for X", which is positioning with the gripper open.
_PIL_GRASP_RE = re.compile(r"^(reach\s+and\s+)?grasp\b", re.IGNORECASE)
# Fingers closed below this = closed on NOTHING (see the bimodal gap above).
PIL_MISS_WIDTH = float(os.environ.get("SYS2_RULES_PIL_MISS_WIDTH", "0.015"))
# The recovery must land immediately after the grasp (offset 1) or right after the lift (offset 2).
PIL_REGRASP_WINDOW = int(os.environ.get("SYS2_RULES_PIL_REGRASP_WINDOW", "2"))
# est floor for a FAR recovery (offset 2), which has to travel back to the object first.
PIL_FAR_EST = int(os.environ.get("SYS2_RULES_PIL_FAR_EST", "125"))
_PIL_BARE_GRASP_RE = re.compile(r"^grasp\b", re.IGNORECASE)
_PIL_REACH_GRASP_RE = re.compile(r"^reach\s+and\s+grasp\b", re.IGNORECASE)
# Pure positioning reach for the continuation rule.
_PIL_REACH_RE = re.compile(r"^reach\s+(to|for)\b", re.IGNORECASE)
PIL_REACH_EST = int(os.environ.get("SYS2_RULES_PIL_REACH_EST", "50"))
# On by default (it was part of the measured arm); SYS2_RULES_PIL_REACH_CONTINUE=0 disables it.
PIL_REACH_CONTINUE = (os.environ.get("SYS2_RULES_PIL_REACH_CONTINUE", "1") not in ("", "0"))


def _rule_pil_reach_continue(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PackIdenticalLunches: ONE borrowed "continue to reach ..." turn after a short reach.

    Scope is deliberately narrow: only ``reach to/for X`` (never "reach and grasp X", where a
    continuation would re-close the fingers), only inside a "pick up ..." milestone, and only when
    System2 budgeted the reach at the smallest bucket (est 50) -- a reach it already thinks is long
    needs no lengthening, est_length being a conditioning tag.

    Budget 1 PER REACH STEP (``pilrc_<fid>``): this task has four picks, so one fixed key would let
    the first reach spend the whole episode's budget. System2's own "continue to <step>" counts
    against that budget (tx_counted), so a borrowed turn is never stacked on a continuation it
    already asked for. The plan is never modified; System2's subgoal is held and resumes after.
    """
    if task != PIL or not PIL_REACH_CONTINUE:
        return {}
    if "pick up" not in current_milestone_text(plan):
        return {}
    step = _target_step(plan, None, _PIL_REACH_RE)
    if step is not None:
        fid = step["fid"]
        if current_fine_id(plan) == fid or step["mark"] == "~":
            state["pilrc_active_fid"] = fid
            # est here is the REACH's own estimate; at injection time `est` belongs to the subgoal
            # being displaced, so the gate has to read what was recorded on this turn.
            state[f"pilrc_{fid}_est"] = est
    else:
        fid = state.get("pilrc_active_fid")
        if not fid:
            return {}
    key = f"pilrc_{fid}"
    return _inject_after_step(
        key, plan, subgoal, state, None, _PIL_REACH_RE,
        extra_fn=lambda t: f"continue to {t}", rule="pil_reach_continue",
        tx_label="tx_sg_incomplete",
        phase2_gate=lambda: state.get(f"{key}_est") == PIL_REACH_EST,
        max_injections=1, missing_means_done=True,
        why=lambda: f"reach budgeted at est {PIL_REACH_EST} (the smallest bucket) -- one extra turn "
                    "to close the remaining distance before the grasp")


def _rule_pil_regrasp_recovery(task: str, plan: str, subgoal: str, est, state) -> dict:
    """PackIdenticalLunches: one injected re-grasp per grasp whose fingers closed on nothing.

    Keyed PER GRASP STEP (``pil_<fid>``) because this task has four picks; keyed on the fid rather
    than the step text because a plan can contain "grasp the lemon" twice (M1.x and M9.x).
    """
    if task != PIL:
        return {}
    # Own turn counter (the rule is called exactly once per turn, from apply_rules).
    state["pil_turn"] = turn = state.get("pil_turn", 0) + 1
    # At most one grasp step is ever visible: System2 lists fine steps only for the milestone in
    # progress and DROPS them when it closes that milestone.
    step = _target_step(plan, None, _PIL_GRASP_RE)
    if step is not None:
        fid = step["fid"]
        if current_fine_id(plan) == fid or step["mark"] == "~":
            state["pil_active_fid"] = fid
            state[f"pil_{fid}_gturn"] = turn        # the window starts here
    else:
        fid = state.get("pil_active_fid")
        if not fid:
            return {}
    key = f"pil_{fid}"
    gturn = state.get(f"{key}_gturn")
    offset = (turn - gturn) if gturn is not None else None
    far = offset is not None and offset > 1

    # RECOVERY ONLY, as a PHASE-2 gate: state["grip_width"] is the width left by the PREVIOUS
    # segment, so during the grasp turn itself it is still the open-gripper value from the reach and
    # a check made in the remember phase could never fire.
    def _missed() -> bool:
        w = state.get("grip_width")
        if w is None or w >= PIL_MISS_WIDTH:
            return False
        return gturn is not None and (turn - gturn) <= PIL_REGRASP_WINDOW

    def _extra(t: str) -> str:
        """offset 1 -> "<step> again"; offset 2 -> "reach and grasp X again" (must travel back)."""
        if not far or _PIL_REACH_GRASP_RE.match(t):
            return f"{t} again"
        return f"{_PIL_BARE_GRASP_RE.sub('reach and grasp', t, count=1)} again"

    r = _inject_after_step(
        key, plan, subgoal, state, None, _PIL_GRASP_RE,
        extra_fn=_extra, rule="pil_regrasp_recovery",
        tx_label="tx_sg_failed", phase2_gate=_missed, max_injections=1,
        missing_means_done=True,
        why=lambda: f"gripper width {state.get('grip_width'):.4f} < {PIL_MISS_WIDTH} -- the fingers "
                    f"are closed on nothing {turn - state[f'{key}_gturn']} turn(s) after {fid}, so "
                    "the object was never held or was dropped on the lift"
                    + (" (arm has already left the object: reach first)" if far else ""))
    # A FAR recovery needs the steps to reach back first. est is PROPOSED, so apply_rules resolves it
    # as the largest of this and System2's own value -- 125 is a FLOOR, not an assignment.
    if far and any(iv.get("kind") == "tx_sg_failed" for iv in r.get("interventions", [])):
        r["est_proposal"] = PIL_FAR_EST
        r["interventions"].append(
            {"rule": "pil_regrasp_recovery", "kind": "est_proposed",
             "detail": f"far recovery ({offset} turns after the grasp): est floor {PIL_FAR_EST} to "
                       "cover reaching back to the object before closing",
             "before": est, "after": PIL_FAR_EST})
    return r


# =============================================================================================
# GENERAL: width-gated re-grasp recovery, for EVERY task.
#
# Generalised from two graduated per-task copies (_rule_pil_regrasp_recovery, PackIdenticalLunches
# 2/20 -> 8/20; _rule_drawer_regrasp_recovery, part of PickPlaceDrawerToCounter 14/20 -> 19/20) plus a
# PreSoakPan port. The mechanism is not task-specific: if the fingers closed on nothing, the object was
# never held or was dropped, and one re-grasp is the cheapest possible recovery.
#
# WHAT IT KEYS ON. state["grip_width"] is the aperture |q[14]-q[15]| left by the PREVIOUS segment, so
# it reads the PHYSICAL outcome rather than System2's text. Below REGRASP_MISS_WIDTH the fingers are
# closed on nothing (~0.001) as opposed to closed on an object (~0.02-0.06) or open (~0.0799).
#
# THE OFFSET WINDOW IS 1-2 TURNS, and both matter: measured over the qwen35 v2 sweep, offset 0 is
# HEALTHY in the cases that fail -- the object is grasped, then lost during the NEXT segment (the lift
# or the carry). So a check at the grasp turn itself sees nothing wrong. Beyond +2 the episode has
# moved on and a re-grasp is spent from the wrong pose.
#     offset 1 -> "<step> again"                 est floor REGRASP_NEAR_EST (75)
#     offset 2 -> "reach and grasp X again"      est floor REGRASP_FAR_EST (125)
# The far form must travel back before closing, hence the higher floor. Both are floors resolved as
# max(floor, the failed grasp's OWN est, System2's est for this turn) -- the grasp est is remembered
# when the step is first seen, because "at least what this object needed the first time" is a statement
# about the object, whereas the current turn's est describes whatever System2 is doing now.
# Empirical basis (v2, n=2028 grasp / 530 reach-and-grasp segments): grasp est median 50 / p90 75,
# consuming 46 steps median; reach-and-grasp est median 100 / p90 150, consuming 100. Only 1.1-1.7% of
# these segments are budget-bound, so the floors change System1's CONDITIONING far more than its
# runtime.
#
# SKIP LIST -- objects whose CORRECT end state is a thin or open gripper, so the width test cannot
# distinguish success from failure. Measured medians at offset +1: mug 0.0109, cup 0.0078, smaller bowl
# 0.0163, spatula 0.0173, straw 0.0129, basket 0.0148, shrimp 0.0152 -- all below the bar while held.
# Door and drawer HANDLES and stove KNOBS read 0.078-0.079 for the opposite reason: the gripper
# legitimately reopens once the handle has been turned, so "closed on nothing" is their normal
# post-condition (the PreSoakPan faucet finding).
# Matched with \b...\b word boundaries, NOT substrings: "straw" must not catch strawberry (0.0419,
# thick) and "pot" must not catch potato (0.0523) or sweet potato (0.0361).
# "level" is included at the session owner's request. NOTE it matches nothing in either 1500-episode
# sweep -- no grasp subgoal mentions "level" or "lever" -- so today it is a no-op guard for objects
# this corpus has not produced. The near-neighbour that DOES occur is the kettle "lever", and only in
# PRESS subgoals ("press the kettle lever down"), which this rule never examines: it looks at grasp
# steps only.
# Measured effect over 2558 recorded grasp turns: 1258 skipped (49%), 1300 monitored, and just 5 false
# positives (0.2%) -- sponge x2, chicken drumstick, croissant, broccoli, all compressible foods that no
# name list can fix.
#
# ONE injection per grasp STEP (keyed per fid), so several grasps in one episode each get their own
# recovery rather than the first consuming the only budget. Borrowed turn: the plan is never modified,
# System2's subgoal is held and resumed, its own "grasp X again" re-issues count against the same
# budget, and the turn is labelled tx_sg_failed.
#
# WHAT THIS CANNOT DO, on the record: on ScrubCuttingBoard the per-task version recovered the sponge
# outright (0.0010 -> 0.0648) and all three episodes still died in an untouched retract loop. A
# recovery only converts an episode when the grasp is what was failing.
# =============================================================================================

_REGRASP_RE = re.compile(r"^(reach\s+and\s+)?grasp\b", re.IGNORECASE)
_REGRASP_BARE_RE = re.compile(r"^grasp\b", re.IGNORECASE)
_REGRASP_REACH_RE = re.compile(r"^reach\s+and\s+grasp\b", re.IGNORECASE)
# WIDTH LADDER, not a single bar. A binary skip list threw away real drops: PickPlaceDrawerToCounter's
# three successful re-grasps were on a pizza cutter (0.0021), a measuring cup (0.0024) and a dish brush
# (0.0043) -- and two of those objects were on the skip list, so skipping them lost the very cases the
# rule exists for. Instead every object gets a bar, and thin objects simply get a TIGHTER one:
#
#   0.015   default -- normal objects (pan, bread, knife, toast, drumstick, strawberry, potato, ...)
#   0.0075  thin-ish: held below the default bar, but never below this one
#   0.003   very thin: essentially "the jaws met with nothing between them" (measured min is ~0.0009
#           for every group, so this is the object-independent floor)
#
# Tiers assigned from the measured "held in a SUCCESSFUL episode" count below each candidate bar, over
# both 1500-episode v2 sweeps at offsets +1/+2:
#   0.003 group -- mug 18 held-OK readings below 0.0075, cup 18, knob 14, ladle 7 of 12, handle 3,
#     bowl 4. A genuinely held ladle reads 0.0057, thinner than a mug, so only 0.003 separates it.
#   0.0075 group -- 0-3 held-OK readings below that bar, so the default was simply too loose for them.
#     jar / meat / drumstick / whisk join here: each misfired 2-4 times at 0.015 and 0 times at 0.0075.
#
# IRREDUCIBLE, on the record: 15 of the 24 measured misfires survive even at 0.003 -- sponge, croissant,
# broccoli, bread, pan, onion, sink spout. Those are compressible foods, or the width was read at a
# moment the gripper had legitimately released. No ladder fixes them; 15 in 2558 grasp turns is 0.6%,
# each costing one borrowed turn.
REGRASP_MISS_WIDTH = float(os.environ.get("SYS2_RULES_REGRASP_MISS_WIDTH", "0.015"))
REGRASP_BAR_THIN = float(os.environ.get("SYS2_RULES_REGRASP_BAR_THIN", "0.0075"))
REGRASP_BAR_VERY_THIN = float(os.environ.get("SYS2_RULES_REGRASP_BAR_VERY_THIN", "0.003"))
# How many turns after the grasp a drop still counts. Offset 0 is HEALTHY in the cases that fail (the
# object is lost during the NEXT segment), and beyond +2 the arm has moved on.
REGRASP_WINDOW = int(os.environ.get("SYS2_RULES_REGRASP_WINDOW", "2"))
# est floors for the retry, from what those segments actually consume (grasp p90 75; reach-and-grasp
# p90 150, median 100). Floors, resolved as max(floor, the failed grasp est, System2 est for the turn).
REGRASP_NEAR_EST = int(os.environ.get("SYS2_RULES_REGRASP_NEAR_EST", "75"))
REGRASP_FAR_EST = int(os.environ.get("SYS2_RULES_REGRASP_FAR_EST", "125"))

# Matched with \b...\b word boundaries, NOT substrings: "straw" must not catch strawberry (0.0419,
# thick) and "pot" must not catch potato (0.0523) or sweet potato (0.0361).
# 0.003: held BELOW 0.0075 in successful episodes, so only the near-closed floor separates them.
# straw (median 0.0129 held, 38 readings under 0.0075) and ice cube / sugar cube / level are here for
# the same reason -- they are the smallest things in the suite.
_REGRASP_VERY_THIN = ("mug", "cup", "knob", "ladle", "handle", "bowl",
                      "ice cube", "sugar cube", "straw", "level", "faucet")
# "faucet" is a NO-OP on every phrasing observed so far, and is here for the one that is not: all 137
# faucet grasps across the two 1500-episode v2 sweeps are "grasp the sink faucet handle" (136) or
# "reach and grasp the sink faucet handle" (1), which already resolve to 0.003 via "handle"
# (RinseSinkBasin 22, WashFruitColander 59, PreSoakPan 55, WashLettuce 1). It covers a bare "grasp the
# faucet", where the target is the same thin lever with no "handle" in the text -- so it is reasoning
# by analogy with knob/handle, NOT a measured held-width distribution like the entries above.
# 0.0075: held below the 0.015 default but never below this.
# NOTE "drumstick" was here and is REMOVED: a chicken drumstick is a large object (median 0.0472 held),
# and the tight bar delayed detection by a turn -- PackIdenticalLunches ep0 ended t27 at 0.0118, above
# the 0.0075 bar, so the drop was not caught until t29 and had to use the expensive reach-back form
# instead of the cheap in-place one at t28. It was added on only 2 misfires; the default 0.015 is right.
_REGRASP_THIN = ("lemon", "kettle", "container", "spatula", "basket", "mushroom", "shrimp",
                 "bell pepper", "yogurt", "dish brush", "cheese stick", "spoon", "tupperware",
                 "pitcher", "colander", "chocolate", "teapot", "pot", "jar", "meat", "whisk")
def _wordset(words):
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
_REGRASP_VERY_THIN_RE = _wordset(_REGRASP_VERY_THIN)
_REGRASP_THIN_RE = _wordset(_REGRASP_THIN)


def regrasp_bar(text: str | None) -> float:
    """The width below which THIS object counts as dropped. Tightest matching tier wins."""
    t = text or ""
    if _REGRASP_VERY_THIN_RE.search(t):
        return REGRASP_BAR_VERY_THIN
    if _REGRASP_THIN_RE.search(t):
        return REGRASP_BAR_THIN
    return REGRASP_MISS_WIDTH


def _rule_regrasp_recovery(task: str, plan: str, subgoal: str, est, state) -> dict:
    """ALL tasks: one injected re-grasp per grasp step whose fingers closed on nothing."""
    state["rg_turn"] = turn = state.get("rg_turn", 0) + 1
    step = _target_step(plan, None, _REGRASP_RE)
    if step is not None:
        fid = step["fid"]
        if current_fine_id(plan) == fid or step["mark"] == "~":
            state["rg_active_fid"] = fid
            state[f"rg_{fid}_gturn"] = turn
            state[f"rg_{fid}_bar"] = regrasp_bar(step["text"])
            # Remember the est the FAILED grasp asked for: the retry should get at least that.
            if isinstance(est, int) and est > 0:
                state[f"rg_{fid}_gest"] = est
    else:
        fid = state.get("rg_active_fid")
        if not fid:
            return {}
    key = f"rg_{fid}"
    gturn = state.get(f"{key}_gturn")
    offset = (turn - gturn) if gturn is not None else None
    far = offset is not None and offset >= 2

    def _missed() -> bool:
        # PHASE-2 gate: grip_width is the previous segment's aperture, so during the grasp turn itself
        # it is still the open value from the reach and an up-front check could never fire.
        w = state.get("grip_width")
        bar = state.get(f"{key}_bar", REGRASP_MISS_WIDTH)
        if w is None or w >= bar:
            return False
        return gturn is not None and 1 <= (turn - gturn) <= REGRASP_WINDOW

    def _extra(t: str) -> str:
        """offset 1 -> "<step> again"; offset 2 -> "reach and grasp X again" (must travel back)."""
        if not far or _REGRASP_REACH_RE.match(t):
            return f"{t} again"
        return f"{_REGRASP_BARE_RE.sub('reach and grasp', t, count=1)} again"

    r = _inject_after_step(
        key, plan, subgoal, state, None, _REGRASP_RE,
        extra_fn=_extra, rule="regrasp_recovery",
        tx_label="tx_sg_failed", phase2_gate=_missed, max_injections=1,
        missing_means_done=True,
        why=lambda: f"gripper width {state.get('grip_width'):.4f} < "
                    f"{state.get(f'{key}_bar', REGRASP_MISS_WIDTH)} (its tier) -- the fingers "
                    f"are closed on nothing {turn - state[f'{key}_gturn']} turn(s) after {fid}, so the "
                    "object was never held or was dropped"
                    + (" (the arm has moved on: reach back first)" if far else ""))
    if any(iv.get("kind") == "tx_sg_failed" for iv in r.get("interventions", [])):
        # Per-turn coordination signal for later rules. This means the recovery INJECTED a turn;
        # tx_counted (System2 independently asked for the same re-grasp) is deliberately not a hit.
        state["regrasp_recovery_hit"] = True
        floor = max(REGRASP_FAR_EST if far else REGRASP_NEAR_EST, state.get(f"{key}_gest") or 0)
        r["est_proposal"] = floor
        r["interventions"].append(
            {"rule": "regrasp_recovery", "kind": "est_proposed",
             "detail": f"{'reach-back' if far else 'in-place'} recovery at offset {offset}: est floor "
                       f"{floor} = max({REGRASP_FAR_EST if far else REGRASP_NEAR_EST}, "
                       f"grasp est {state.get(f'{key}_gest')})",
             "before": est, "after": floor})
    return r


# ---- THE THREE TIERS -----------------------------------------------------------------------------
# MANDATORY: always runs, even with --task-rules OFF. repeat_cap is not an optional revision of
# System2's output -- it is the loop's own termination policy. Without it a stuck subgoal is re-issued
# until max_turns with nothing advancing, so "no rules" would mean "no way out of a repeat", which is
# not a meaningful baseline of the model. So the floor for every run is repeat_cap, and what
# --task-rules adds is the OPTIONAL tiers below.
#
# It is deliberately NOT in _GENERAL_RULES: that tier is "task-agnostic revisions you may switch on",
# and repeat_cap is not switchable. SYS2_RULES_NO_MANDATORY=1 does disable it, for the single purpose
# of reproducing the historical 1000-episode zero-rules baseline that predates this tier.
_MANDATORY_RULES = (_rule_repeat_cap,)
_MANDATORY_ON = os.environ.get("SYS2_RULES_NO_MANDATORY", "") in ("", "0")


def mandatory_on() -> bool:
    """Whether the mandatory tier is active. Public because callers record the arm they ran."""
    return _MANDATORY_ON

# OPTIONAL, and NOT gated on a task name -- that is what makes a rule general here. Two of the three
# are gated on the SUBGOAL TEXT instead, so they fire on whichever task performs that operation; the
# tier means "selected by semantics, not by task", not "fires on all 50 tasks".
#
#   regrasp_recovery   genuinely universal: any grasp step whose fingers closed on nothing gets ONE
#                      borrowed re-grasp turn. Keyed on state["grip_width"], the physical outcome, so
#                      it needs no text match. This is the generalisation of two GRADUATED per-task
#                      copies -- PackIdenticalLunches 2/20 -> 8/20 (paired 6 gained / 0 lost, exact
#                      McNemar p=0.031) and part of PickPlaceDrawerToCounter 14/20 -> 19/20 -- which
#                      were retired in its favour and then left unregistered, so both wins were dark.
#                      Instrumented over 2558 grasp turns: 49% skipped by the object ladder, 0.2%
#                      false positives, 0.6% irreducible misfires at one borrowed turn each.
#                      NOT established as a general rule: the all-tasks form has no A/B of its own,
#                      the PreSoakPan port was 13/20 -> 11/20 (p=0.688, no effect either way), and no
#                      replicate of the PIL arm was ever run.
#   microwave_again    subgoal- or task-gated on "microwave" (TurnOnMicrowave, SteamInMicrowave 145
#                      button turns, WaffleReheat 36, PrepareCoffee 11): "continue to press X" ->
#                      "press X again" for System1 only. NO A/B at all -- the support is that the
#                      "... again" form is in-distribution (9x / 2x as its own subgoal) while
#                      "continue to" is System2's wrapper. Cheapest possible rule: it changes only
#                      the string handed to System1, never the checklist, the est or the budget.
#   sink_faucet_est    subgoal-gated on the faucet ACTIVATION phrase, so ~80% of its firings are on
#                      the composite sink tasks (WashLettuce 138 faucet turns, RinseSinkBasin 99,
#                      WashFruitColander 94, PreSoakPan 89) rather than the atomic one (51). Floors
#                      est at 100 because System2 budgeted 50 (18x) or re-issued "continue to" (33x).
#                      Never isolated: it was on in the v2 arms together with est_bump, so
#                      WashFruitColander's rules-vs-base zero is the COMBINATION's zero. NOTE it
#                      can raise est on the 600-step TurnOnSinkFaucet task, while its task-specific
#                      wording rule deliberately uses no estimate lever. The combined tier keeps the
#                      floor: System2 normally predicts >=100 there, so it is usually a no-op.
#
# ORDER: rewrite text first, then let the physical recovery replace the turn when needed. The faucet
# estimate runs last and explicitly declines a recovery turn, so its estimate cannot leak from the
# held faucet action onto the injected re-grasp. est proposals are still resolved once below.
_GENERAL_RULES: tuple = (_rule_microwave_again, _rule_regrasp_recovery, _rule_sink_faucet_est)

# =============================================================================================
# GRADUATED: CloseToasterOvenDoor -- pin est to 100, and name the door HANDLE on a bare-door reach.
#
# MEASUREMENT (qwen3vl, 30 episodes, the manifest set, repeat_cap the only other rule active):
#     arm                       success   turns  steps/ep  push turns  4+ push loop
#     v2 (full rule set)         19/30     5.2      611       4.23         13
#     base (repeat_cap only)     23/30     4.1      356       3.03          9
#     + est pinned to 100        26/30     3.6      329       2.33          6
#     + est 100 AND handle       28/30     3.0      257       1.93          3
#     qwen35 base, for reference 28/30     3.1      302       2.10          4
# Monotone on ALL FIVE columns, and it lands exactly on the other planner's score with slightly
# better mechanics. max_turns deaths went 11 -> 2 -> 0.
#
# WHY THIS TASK NEEDED A RULE AT ALL. It was the largest planner gap in the 1500-episode sweeps with
# NO rule on either side (qwen35 28-29/30 vs qwen3vl 19-23/30). The failure is sharp: BOTH planners
# succeed on every episode that needs <=3 push turns (qwen35 27/27, qwen3vl 17/17); the whole gap is
# that qwen3vl reaches 4+ push turns 13 times vs 3 and converts 2/13. On the shared subgoal
# "continue to push the toaster oven door closed" it ran 182 mean steps with 42% hitting budget,
# versus qwen35's 88 mean and 8%: it pushes without latching, and System2 answers by asking again.
#
# WHY est. est_length is a POLICY CONDITIONING tag. qwen3vl's est was both rigid and wrong: a flat 75
# on every reach (qwen35 mostly 100), then COLLAPSING to a modal 50 from turn 2 while the segments it
# described grew to 111-142 executed steps. The two planners recover in opposite directions -- qwen35
# RAISES est when it needs more turns (t3 mean 111), qwen3vl LOWERS it -- and qwen35's is the one that
# wins. Pinning removes the planner's estimate from the loop: of 272 recorded turns, 53% were raised
# to 100 and 21% lowered.
#
# ASSIGN, NOT FLOOR -- and note this is the ONLY est rule here that assigns. It needs the est_assign
# channel in apply_rules because est_proposal resolves as the largest and cannot lower. The risk was
# stated before running: the lowered turns are mostly est 150 executing ~269 steps, i.e. using nearly
# all of a 300-step budget, so pinning them to 100 truncates them. It still won, but
# SYS2_CTOD_MODE=floor selects the safer raise-only variant if that ever looks load-bearing.
#
# THE HANDLE REWRITE, and an honest correction. The OBSERVATIONAL test said the word was irrelevant:
# within qwen3vl (reach est constant at 75, so wording was the only variable) naming the handle gave
# 14/20 = 70% versus 28/40 = 70% without, Fisher exact p=1.000, and equal on reach steps, push turns
# and loop rate. The pooled figure that looks decisive (89% vs 70%) is SIMPSON'S PARADOX -- qwen35
# names the handle 60/60, so the pooled split measures the planner. On that basis the rewrite was
# judged pointless. The INTERVENTION disagreed: 26 -> 28 with the loop rate halving. The lesson is
# that an observational null on a variable the PLANNER CHOOSES is not an interventional null -- the
# choice can correlate with the scene, and n=20 vs 40 had little power.
#
# STATISTICAL HONESTY. est100+handle vs base is won 7 / lost 2, p=0.180; the handle increment alone
# (vs est100-only) is won 4 / lost 2, p=0.688. So the COMBINATION is suggestive but not significant,
# and the handle half is unproven at n=30 -- kept because it is free, monotone on every mechanism
# metric, and makes qwen3vl emit what qwen35 emits. NO REPLICATE WAS RUN. The two remaining failures
# (eps 21, 25) lost in both comparisons and both end on max_cap.
#
# The rewrite only ever ADDS the word: it bails if "handle" appears anywhere in the subgoal, and the
# regex additionally requires a trailing "door", so "reach to the handle of the ... door" and
# "... door, then the handle" are both left alone. Verified 0 double-"handle" subgoals in the run.
# =============================================================================================

CTOD = "CloseToasterOvenDoor"
CTOD_EST = int(os.environ.get("SYS2_CTOD_EST", "100"))
# "assign" = pin exactly (can lower); "floor" = raise only. See the risk note above.
CTOD_MODE = os.environ.get("SYS2_CTOD_MODE", "assign").strip().lower()
CTOD_HANDLE = os.environ.get("SYS2_CTOD_HANDLE", "1") not in ("", "0")
# "reach/move to|for ... door" with nothing after "door". Anchored so a reach naming another referent
# is untouched; the "handle" in-string check below is the primary guard.
_CTOD_REACH_DOOR_RE = re.compile(
    r"^((?:continue\s+to\s+)?(?:reach|move)\s+(?:to|for)\s+.*\bdoor)\s*$", re.IGNORECASE)


def _rule_ctod_reach_handle(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CloseToasterOvenDoor: a reach naming only the door gets " handle" appended.

    Reaches ONLY. "push the toaster oven door closed" must not become "... door handle closed" -- the
    push acts on the door, and that phrasing has never been emitted by either planner.
    """
    if task != CTOD or not CTOD_HANDLE:
        return {}
    raw = (subgoal or "").strip()
    if "handle" in raw.lower():          # primary guard: never add a second one
        return {}
    m = _CTOD_REACH_DOOR_RE.match(raw)
    if not m:
        return {}
    new = f"{m.group(1)} handle"
    return {"subgoal": new, "subgoal_detail": new,
            "interventions": [{"rule": "ctod_reach_handle", "kind": "subgoal_override",
                               "detail": "reach named only the door; name the HANDLE, which is what "
                                         "the other planner does 60/60 times on this task",
                               "before": raw, "after": new}]}


def _rule_ctod_est_100(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CloseToasterOvenDoor: est_length pinned to CTOD_EST (100) for EVERY subgoal.

    Deliberately not gated on the subgoal text: the erratic values are spread across the reach (rigid
    75), the push (100) and every continuation (50 modal, sometimes 150). Gating on one phrasing would
    reintroduce the model-specific fragility that made sink_faucet_est non-transferable.
    """
    if task != CTOD:
        return {}
    if not isinstance(est, int) or est <= 0:
        # System2 omitted/garbled <estimated_step>; combined_eval substitutes its own default and
        # pinning here would silently redefine what that default means.
        return {}
    if est == CTOD_EST:
        return {}
    if CTOD_MODE == "floor":
        if est > CTOD_EST:
            return {}
        return {"est_proposal": CTOD_EST,
                "interventions": [{"rule": "ctod_est_100", "kind": "est_proposed",
                                   "detail": f"floor est at {CTOD_EST} (mode=floor): the planner's "
                                             f"{est} under-conditions a longer motion",
                                   "before": est, "after": CTOD_EST}]}
    return {"est_assign": CTOD_EST,
            "interventions": [{"rule": "ctod_est_100", "kind": "est_assign",
                               "detail": f"pin est to {CTOD_EST}: this planner's est for the task is "
                                         f"erratic (50/75/100/125/150) and uncorrelated with the "
                                         f"steps actually executed; {est} -> {CTOD_EST}",
                               "before": est, "after": CTOD_EST}]}


# RETIRED -- measured at n=30 x TWO planners and shown to earn nothing. Functions kept above for the
# record; deliberately absent from _TASK_RULES below, so re-enabling the task set does not revive them.
#
#   task                qwen35 v2 / base    qwen3vl v2 / base   net(60 eps)
#   WeighIngredients        3/30  3/30          4/30  5/30          -1
#   WashFruitColander      17/30 17/30         21/30 20/30          +1
#
# They are NOT idle -- they fire on most episodes and still move nothing:
#   wi_extra_carry_turn      40 firings / 27 of 30 eps (qwen35),  59 / 30 of 30 (qwen3vl)
#   wi_reach_locate          12 / 9,                               9 / 9
#   wfc_skip_carry_continue  42 / 21 of 30 (qwen35),                4 / 2 (qwen3vl)
#
# WHY THEY GRADUATED AND THEN DID NOT REPRODUCE -- the same measurement error twice, worth knowing
# before graduating anything else. Each was credited against the VERIFIED-RULES-ONLY arm rather than
# against the no-rules baseline, and that reference happened to be a low outlier:
#   WashFruitColander  baseline 9/20, verified-only 5/20, with rule 12/20 and 13/20  -> "+7" vs 5/20,
#                      but only ~+3 vs the 9/20 baseline
#   WeighIngredients   baseline 4/20, verified-only 2/20, with rules 6/20            -> "+4" vs 2/20,
#                      but only +2 vs the 4/20 baseline
# Both apparent gains sit inside the +-4.5-episode (2SD) band this repo documents for n=20. JUDGE A
# CANDIDATE AGAINST THE NO-RULES BASELINE, not against whatever the current rule stack happens to
# score on that task.
# Caveat kept honest: WashFruitColander also carries the text-gated sink_faucet_est (168 firings on
# qwen35), which is on in v2 and off in base too, so its zero is the COMBINATION's zero -- the two
# cannot be separated from these arms. wfc_skip_carry_continue is the only TASK-SPECIFIC rule there,
# and it is the one being retired.
_RETIRED_RULES = (_rule_wfc_skip_carry_continue,
                  _rule_wi_extra_carry_turn, _rule_wi_reach_locate)

# TASK-SPECIFIC rules -- each gated on exactly one task (by name or a single-entry tuple).
# NOTE the two per-task re-grasp copies are RETIRED: _rule_regrasp_recovery subsumes
# drawer_regrasp_recovery and pil_regrasp_recovery via the width ladder.
_TASK_RULES = (_rule_drawer_base_align, _rule_drawer_base_align_est,
               _rule_strip_retract_plan, _rule_flag_retract_emitted,
               _rule_microwave_est_bump,
               _rule_mixer_est_floor, _rule_coffee_m2_est,
               _rule_coffee_split_release, _rule_coffee_split_reach_grasp, _rule_coffee_grasp_est,
               _rule_coffee_skip_failed,
               _rule_ppc2c_skip_failed, _rule_ppc2c_extra_carry, _rule_ppc2c_grasp_est,
               _rule_gtb_reach_grasp, _rule_gtb_add_wait, _rule_gtb_skip_wait_cont,
               _rule_gtb_wait_force_steps, _rule_gtb_slot_est,
               _rule_pil_reach_continue,
               # Order: rewrite the subgoal first so the est record is against the final text.
               _rule_ctod_reach_handle, _rule_ctod_est_100)

# THE FINALISED SET: everything task-agnostic, plus every graduated per-task rule. There are exactly
# TWO execution registries and no third category -- a rule is either general or gated on one task.
#
# It used to read `_GENERAL_RULES + _COFFEE_RULES + _GTB_RULES`, where those two tuples were
# per-task SUBSETS of _TASK_RULES (the same function objects, not copies) assembled so a sweep could
# measure "general + only this task's rules" and attribute the delta to that task alone. Both were
# deleted with the experiments they served. They were also a footgun: `_GENERAL_RULES + _TASK_RULES +
# _COFFEE_RULES + _GTB_RULES` would have run those 10 rules TWICE per turn, which is harmless for
# est_proposal (resolved by max) but not for coffee_skip_failed / gtb_add_wait, which rewrite the plan
# and carry per-episode state.
#
# sys2_rules_exp*.py is appended to this AFTER the loader below, so an experiment always sees the
# checklist the verified rules produced.
#
# _RULES is the OPTIONAL set -- what --task-rules switches on. The mandatory tier is applied by
# apply_rules regardless and is NOT in here, so that `--task-rules` off still gets repeat_cap.
_RULES = _GENERAL_RULES + _TASK_RULES

# ---- PLAN-MODE rules -------------------------------------------------------------------------
# apply_rules above runs on EXECUTION turns only. A rule that must replace the checklist System2
# produces in PLAN mode has no way in from there, because the plan call happens once before the
# exec loop (combined_eval.do_plan_cold) and never passes through the rule layer.
#
# These run exactly once per episode, on the plan-mode output. During execution the plan is then
# maintained as usual -- System2's own plan_update marks progress and nothing re-forces the
# checklist, so the marks stay System2's and no rule has to reconstruct them.
# GRADUATED (plan mode): StackBowlsCabinet -- force the stack-first checklist.
#
#   arm (same 20 episode ids)                              success
#   baseline, no rules                                       7/20
#   verified rules only                                      9/20
#   with this rule, launch 1 (complete 20)                  16/20   <- see PROVENANCE below
#   with this rule, launch 2 (16 completed of 19 attempted) 11/16
#   one clean single-episode check (ep2, own label)           1/1
#
#   verifiable on disk (13/18)   vs baseline p=0.022   vs verified p=0.090
#   incl. launch 1  (27/36)      vs baseline p=0.003   vs verified p=0.025
#
# PROVENANCE, stated because it is not a clean 20/20: launch 1 and launch 2 ran under the SAME
# RUN_LABEL, and without RESUME=1 a repeated label re-runs episodes and OVERWRITES their
# episode.json -- so 19 of launch 1's files no longer exist and its 16/20 is not auditable from
# disk. Launch 2 was then truncated (3 episodes killed mid-execution). The no-rules baseline
# comparison is significant on the surviving data alone; the verified-arm comparison is not, and
# depends on the overwritten launch. A same-code -ref control for this task was queued and
# cancelled, so today's-code control is missing too.
#
# NO MECHANISM, and that is on the record: the rule's own diagnosis found 24 of 24 failures across
# both reference runs are NOT planning failures -- the plan runs to completion, both bowls land in
# the cabinet, and then 6-8 turns go to "continue to retract the arm" until max_turns while
# env_success never latches. 5 of the 11 verified failures already ran this exact stack-first plan.
# The plan-family split (stack-first 42% vs cabinet-first 27% pooled over 5 methods) is also a
# deterministic function of the scene, hence confounded with scene difficulty, and inside the paired
# run the two families are indistinguishable (44% vs 45%). So the predicted effect was ~0 and the
# measured one is larger; the retract loop remains the untouched failure mode.
#
# PLAN MODE ONLY. Runs once, before the exec loop, so System2's marks are never reconstructed: the
# checklist it is handed on turn 0 is the forced one, and execution then maintains it normally
# (verified in a real rollout -- System2 advanced M1->M4 and added its own fine steps under M4).
SBC = "StackBowlsCabinet"
# Milestones only, no fine steps -- the shape of every recorded cold plan on this task, so it is
# in-distribution for the turn-0 prompt ("Unroll the current milestone into fine steps"). A
# milestone without fine steps makes current_fine_id() -> None, so every EXECUTION rule degrades to
# a no-op (repeat_cap takes its cap_declined branch) rather than corrupting anything.
SBC_FORCED_PLAN = (
    "- [ ] M1: grasp the smaller bowl\n"
    "- [ ] M2: place the smaller bowl into the larger bowl\n"
    "- [ ] M3: grasp the larger bowl\n"
    "- [ ] M4: place the stacked bowls into the open cabinet"
)


def _plan_rule_stackbowls_force(task: str, plan: str) -> dict:
    """StackBowlsCabinet, PLAN MODE: replace the cold plan with the stack-first checklist.

    Returns nothing when System2's plan is already byte-identical, so an episode that planned this
    itself is recorded as untouched (``episode.json["plan"]["s2_plan_before_rules"]`` stays absent).
    """
    if task != SBC:
        return {}
    if (plan or "").strip() == SBC_FORCED_PLAN:
        return {}
    return {"plan": SBC_FORCED_PLAN,
            "interventions": [{"rule": "sbc_force_plan", "kind": "plan_mode_override",
                               "detail": "forced the canonical stack-first 4-milestone checklist "
                                         "(stack on the counter, then carry the stack into the "
                                         "cabinet); execution then maintains it as usual",
                               "before": (plan or "").strip(), "after": SBC_FORCED_PLAN}]}


# TASK-SPECIFIC, PLAN MODE. This is the plan-mode half of _TASK_RULES -- same contract (gated on
# exactly one task), different hook. It cannot be merged into _TASK_RULES: these take (task, plan)
# and run ONCE before the exec loop, while apply_rules calls its rules with five positional
# arguments on every turn, so a plan rule in _TASK_RULES would raise TypeError on the first turn of
# every task. The split is the call site, not the category.
_TASK_PLAN_RULES: tuple = (_plan_rule_stackbowls_force,)
_PLAN_RULES: tuple = _TASK_PLAN_RULES


def apply_plan_rules(task: str, *, plan: str) -> dict:
    """Revise the PLAN-MODE checklist. Returns {plan, interventions}.

    ``interventions`` is empty when nothing fired -- byte-identical to not calling this at all.
    """
    cur = plan
    ivs: list[dict] = []
    for fn in _PLAN_RULES + tuple(_EXP_PLAN_RULES):
        r = fn(task, cur) or {}
        if r.get("interventions"):
            ivs.extend(r["interventions"])
        if "plan" in r:
            cur = r["plan"]
    return {"plan": cur, "interventions": ivs}


# EXPERIMENTAL PATCH LAYER. Unverified per-task rules live in sys2_rules_exp*.py and are appended
# here. EVERY matching module is loaded, so parallel work on different tasks can each own a private
# file (sys2_rules_exp_ArrangeTea.py, ...) instead of several editors clobbering one shared file --
# which is how two CoffeeSetupMug rules were nearly lost. Deleting a file removes exactly its rules;
# SYS2_RULES_NO_EXP=1 ignores them all without deleting anything.
#
# Loaded LAST so an experiment sees the checklist the verified rules produced. A broken or missing
# patch file must never break a real run, so each import failure is swallowed with a warning.
_EXP_RULES: tuple = ()
_EXP_PLAN_RULES: tuple = ()
if not os.environ.get("SYS2_RULES_NO_EXP"):
    import glob as _glob
    import importlib as _importlib
    import os.path as _osp
    for _f in sorted(_glob.glob(_osp.join(_osp.dirname(_osp.abspath(__file__)), "sys2_rules_exp*.py"))):
        _mod = _osp.splitext(_osp.basename(_f))[0]
        try:
            _m = _importlib.import_module(_mod)
            _EXP_RULES = _EXP_RULES + tuple(getattr(_m, "EXP_RULES", ()))
            _EXP_PLAN_RULES = _EXP_PLAN_RULES + tuple(getattr(_m, "EXP_PLAN_RULES", ()))
        except Exception as _e:  # a bad patch file must never break a real run
            print(f"WARNING: experimental rules in {_mod} not loaded: {_e}", flush=True)
    _RULES = _RULES + tuple(_EXP_RULES)

# The per-task tier as apply_rules selects it: graduated task rules plus the experimental patch layer
# (every exp rule is gated on one task, so it belongs here and not in the general tier). _RULES stays
# the union of every OPTIONAL rule, for introspection and the --task-rules help.
_TASK_TIER_RULES: tuple = _TASK_RULES + tuple(_EXP_RULES)

# Human-readable scope shown in the --task-rules CLI help. Derived, not hand-maintained: a hardcoded
# list went stale the moment the registry changed (it still said "<all: repeat_cap>" after the whole
# task set was registered). Counts what --task-rules ADDS, per task, plus the plan-mode rules; the
# experimental patch layer is included, so a sys2_rules_exp*.py file shows up here too. The mandatory
# tier is listed separately because it runs with or without the flag.
def _tasks_with_rules() -> tuple[str, ...]:
    per: dict[str, int] = {}
    for fn in (*_RULES, *_PLAN_RULES, *_EXP_PLAN_RULES):
        if fn in _GENERAL_RULES:
            continue
        for t in _RULE_TASKS.get(fn.__name__, ()):
            per[t] = per.get(t, 0) + 1
    mand = ", ".join(f.__name__.removeprefix("_rule_") for f in _MANDATORY_RULES) or "none"
    gen = ", ".join(f.__name__.removeprefix("_rule_") for f in _GENERAL_RULES) or "none"
    return (f"<always on, no flag needed: {mand}>",
            f"<+general, task-agnostic: {gen}>",
            *(f"{t} ({n})" for t, n in sorted(per.items())))


# Which task(s) each non-general rule is gated on. Kept beside the registries so it is obvious when a
# new rule is added and forgotten here -- the self-check below fails loudly in that case.
_RULE_TASKS: dict[str, tuple[str, ...]] = {
    "_rule_drawer_base_align": _DRAWER_ALIGN_TASKS,
    "_rule_drawer_base_align_est": _DRAWER_ALIGN_TASKS,
    "_rule_strip_retract_plan": _STRIP_RETRACT_TASKS,
    "_rule_flag_retract_emitted": _STRIP_RETRACT_TASKS,
    "_rule_microwave_est_bump": ("TurnOnMicrowave",),
    "_rule_mixer_est_floor": ("OpenStandMixerHead",),
    "_rule_coffee_m2_est": (COFFEE,),
    "_rule_coffee_split_release": (COFFEE,),
    "_rule_coffee_split_reach_grasp": (COFFEE,),
    "_rule_coffee_grasp_est": (COFFEE,),
    "_rule_coffee_skip_failed": (COFFEE,),
    "_rule_ppc2c_skip_failed": (PPC2C,),
    "_rule_ppc2c_extra_carry": (PPC2C,),
    "_rule_ppc2c_grasp_est": (PPC2C,),
    "_rule_gtb_reach_grasp": (GTB,),
    "_rule_gtb_add_wait": (GTB,),
    "_rule_gtb_skip_wait_cont": (GTB,),
    "_rule_gtb_wait_force_steps": (GTB,),
    "_rule_gtb_slot_est": (GTB,),
    "_rule_pil_reach_continue": (PIL,),
    "_rule_ctod_reach_handle": (CTOD,),
    "_rule_ctod_est_100": (CTOD,),
    "_plan_rule_stackbowls_force": (SBC,),
    # Experimental patch layer (sys2_rules_exp*.py).
    "_rule_tosf_force_plan": ("TurnOnSinkFaucet",),
}

# SELF-CHECK: every registered non-general rule must be accounted for above, so the --task-rules help
# and any per-task audit cannot silently drift from the registry.
_UNMAPPED = sorted(
    fn.__name__
    for fn in (*_RULES, *_PLAN_RULES, *_EXP_PLAN_RULES)
    if fn not in _GENERAL_RULES + _MANDATORY_RULES and fn.__name__ not in _RULE_TASKS
)
if _UNMAPPED:
    print(f"WARNING: sys2_rules._RULE_TASKS is missing {len(_UNMAPPED)} registered rule(s): "
          f"{', '.join(_UNMAPPED)} -- add them so --task-rules reports the real scope", flush=True)

TASKS_WITH_RULES = _tasks_with_rules()


def rule_config(*, general: bool, task_tier: bool) -> dict:
    """Serializable description of the exact rule tiers selected for one evaluation run."""
    mandatory = bool(_MANDATORY_ON)

    def names(fns) -> list[str]:
        return [fn.__name__.removeprefix("_rule_").removeprefix("_plan_rule_") for fn in fns]

    return {
        "schema_version": 1,
        "mandatory_rules": mandatory,
        "general_rules": bool(general),
        "task_rules": bool(task_tier),
        "active_rules": {
            "mandatory": names(_MANDATORY_RULES) if mandatory else [],
            "general": names(_GENERAL_RULES) if general else [],
            "task": names(_TASK_TIER_RULES) if task_tier else [],
            "task_plan": names(_PLAN_RULES + tuple(_EXP_PLAN_RULES)) if task_tier else [],
            "task_action": ["action_override"] if task_tier and _task_rules_active() else [],
        },
    }


def apply_rules(task: str, *, plan: str, subgoal: str, subgoal_detail: str, est,
                state: dict | None = None, general: bool = True, task_tier: bool = True) -> dict:
    """Revise one turn's System2 output. Returns what the caller should actually use.

    ``general`` / ``task_tier`` select the TIERS, giving the three arms the CLI exposes:
        neither                  mandatory only (repeat_cap) -- the "no rules" arm. The loop still
                                 has a way out of a repeated subgoal, but nothing revises System2.
        general                  + the task-agnostic tier
        general and task_tier    + the per-task tier and the sys2_rules_exp*.py patch layer
    Call this UNCONDITIONALLY and pass the flags through: the mandatory tier must run either way.

    ``state`` is a per-EPISODE dict the caller threads through every turn (skip counters live
    there). Returns:
        {plan, subgoal, subgoal_detail, est, skip_s1, interventions:[...]}
    ``interventions`` is empty when nothing fired.
    """
    st = state if state is not None else {}
    # Ephemeral coordination flag: later rules may need to know that THIS turn was replaced by a
    # physical re-grasp. Reset on every application so a hit never leaks into the following turn.
    st["regrasp_recovery_hit"] = False
    # tx_label marks a turn a rule INJECTED rather than one System2 asked for -- "tx_sg_failed" for a
    # recovery after a detected failure, "tx_sg_incomplete" for a continuation. Recorded per turn so
    # the GUI track can tell the two apart.
    # force_steps: an exact segment length that overrides BOTH the stop rule and --max-steps-cap
    # (combined_eval.run_s1_segment). 0 = untouched. Unlike est, it is taken as-is rather than
    # resolved against other proposals: it is an explicit "run exactly this long", so the LAST rule
    # to set it wins and there is nothing to reconcile.
    cur = {"plan": plan, "subgoal": subgoal, "subgoal_detail": subgoal_detail, "est": est,
           "skip_s1": False, "tx_label": None, "force_steps": 0,
           "stop_episode": False, "requery_s2": False}
    ivs: list[dict] = []
    est_proposals: list[tuple] = []
    est_assign = None          # exact assignment (see the est_assign block at the end)
    est_assign_by = None
    # Mandatory tier FIRST, so repeat_cap sees System2's own text and its cap counters are identical
    # whether or not the optional layer is on -- that is what keeps a rules arm comparable with its
    # no-rules control. (_MANDATORY_ON exists only to reproduce the pre-tier zero-rules baseline.)
    active = ((_MANDATORY_RULES if _MANDATORY_ON else ())
              + (_GENERAL_RULES if general else ())
              + (_TASK_TIER_RULES if task_tier else ()))
    for fn in active:
        r = fn(task, cur["plan"], cur["subgoal"], cur["est"], st)
        if not r:
            continue
        ivs.extend(r.get("interventions", []))
        if r.get("est_assign") is not None:
            est_assign = int(r["est_assign"])
            est_assign_by = (r["interventions"][0]["rule"] if r.get("interventions") else None)
        if "est_proposal" in r:
            # Collected, NOT applied: several rules may propose an est for the same turn (a
            # task-specific one and the universal bucket bump), and applying them in sequence would
            # CHAIN -- the faucet rule's 100 would then be bumped again to 125, which is neither
            # proposal. They are resolved once, below, by taking the largest.
            est_proposals.append((r["est_proposal"], r["interventions"][0]["rule"]
                                  if r.get("interventions") else "?"))
        for k in ("plan", "subgoal", "subgoal_detail", "skip_s1", "tx_label", "force_steps",
                  "stop_episode", "requery_s2"):
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
    # EST ASSIGNMENT, applied AFTER the resolution above and therefore able to LOWER est -- which
    # est_proposal deliberately cannot do (it resolves as the largest, so a proposal can only raise).
    # A rule returns est_assign only when it means "this subgoal takes exactly N steps, whatever
    # System2 guessed", e.g. pinning a task whose planner emits an erratic est. Use sparingly: the
    # graduated est rules are all floors, because a floor can never truncate a segment that was
    # legitimately given a long budget, and an assignment can.
    # LAST WRITER WINS if two rules assign; they would be contradicting each other outright, so
    # there is nothing to reconcile and the ordering of _RULES decides.
    # Record it here ONLY if the rule did not already log its own est_assign entry -- otherwise one
    # action appears twice in the audit and every per-rule count is inflated. (est_proposal is
    # different on purpose: the rule logs the PROPOSAL and this block logs the RESOLUTION, which are
    # two distinct facts because several rules can propose.)
    _already = any(i.get("kind") == "est_assign" for i in ivs)
    if est_assign is not None and est_assign != cur["est"]:
        if not _already:
            ivs.append({"rule": est_assign_by or "est_assign", "kind": "est_assign",
                    "detail": f"est ASSIGNED to {est_assign} (overrides System2's {est} and any "
                                  "floor proposed above); this can lower est, unlike est_proposal",
                        "before": cur["est"], "after": est_assign})
        cur["est"] = est_assign
    st["consec_skips"] = (st.get("consec_skips", 0) + 1) if cur["skip_s1"] else 0
    # A borrowed turn was just injected: complete the held record with the rest of System2's output
    # for THIS turn, so the resume turn can run it without querying System2 again. est is the
    # planner's own estimate for the held subgoal, NOT the resolved one -- the resolution above
    # belongs to the borrowed subgoal that is about to run instead.
    _pend = st.get("_resume_pending")
    if _pend is not None and "est" not in _pend:
        _pend["subgoal_detail"] = subgoal_detail
        _pend["est"] = est
    cur["interventions"] = ivs
    return cur


def pending_resume(state: dict | None) -> dict | None:
    """The subgoal a rule is holding, when the NEXT turn must run it instead of querying System2.

    Called by the rollout loop at the TOP of a turn, before the System2 call. Returns
    ``{"key", "rule", "subgoal", "subgoal_detail", "est"}`` once and then forgets it, so the hold
    lasts exactly one turn. Also clears the rule's own ``<key>_held`` slot: the loop is executing the
    held subgoal now, so the rule's tx_resume branch must NOT fire a turn later and run it twice.

    Returns None when no rule borrowed the previous turn -- the ordinary path, where the loop queries
    System2 as usual.
    """
    if not state:
        return None
    pend = state.pop("_resume_pending", None)
    if not pend:
        return None
    state.pop(f"{pend['key']}_held", None)
    return pend
