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

# Max consecutive turns a rule may skip System1. Without a cap, a task whose System2 insists on
# the skipped step would burn its whole turn budget on no-op turns; on hitting the cap the
# subgoal is executed normally instead.
MAX_CONSEC_SKIPS = 2


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
    # Hand System1 the new first step this turn, with the requested budget.
    out["subgoal"] = NEW
    out["est"] = 75
    iv.append({"rule": "drawer_base_align", "kind": "subgoal_override",
               "detail": "execute the inserted base-alignment step before reaching",
               "before": subgoal, "after": NEW})
    iv.append({"rule": "drawer_base_align", "kind": "est_override",
               "detail": "base alignment budget", "before": est, "after": 75})
    return out


def _rule_drop_retract(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnMicrowave / OpenStandMixerHead: never execute a retract-the-arm subgoal.

    These tasks' success is already latched by the dense env check before the retract; spending a
    segment (and turns) pulling the arm back only risks disturbing the achieved state. The step is
    marked DONE in the checklist and System1 is skipped, so System2 sees it completed next turn
    and moves on (or judges task_finish) instead of re-issuing it.

    Deliberately NOT applied to CoffeeSetupMug, which also emits retract subgoals but where the
    retract is load-bearing (it precedes carrying the mug).
    """
    if task not in ("TurnOnMicrowave", "OpenStandMixerHead"):
        return {}
    if not _RETRACT_RE.match(_norm(subgoal)):
        return {}
    if state.get("consec_skips", 0) >= MAX_CONSEC_SKIPS:
        return {"interventions": [{"rule": "drop_retract", "kind": "skip_declined",
                                   "detail": f"hit MAX_CONSEC_SKIPS={MAX_CONSEC_SKIPS}; "
                                             "executing normally to avoid a stalled episode",
                                   "before": subgoal, "after": subgoal}]}
    blocks = _blocks(plan)
    marked = None
    for b in blocks:
        for f in b["fine"]:
            if f["mark"] in (" ", "~") and _RETRACT_RE.match(_norm(f["text"])):
                f["mark"] = "x"
                marked = f["fid"]
                break
        if marked:
            if all(f["mark"] == "x" for f in b["fine"]):
                b["mark"] = "x"
            break
    return {"plan": _render(blocks) if marked else plan,
            "skip_s1": True,
            "interventions": [{"rule": "drop_retract", "kind": "subgoal_skipped",
                               "detail": ("dropped retract-arm subgoal; marked "
                                          f"{marked or 'no matching plan step'} done"),
                               "before": subgoal, "after": None}]}


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


def _rule_coffee_carry_est(task: str, plan: str, subgoal: str, est, state) -> dict:
    """CoffeeSetupMug: M2.1 (carry the mug to the dispenser) always gets 100 steps.

    Keyed on the milestone POSITION, per the requested rule: M2.1's wording varied 6 ways across
    20 episodes ("lift and carry the mug to the coffee machine dispenser", "carry and place the
    red mug ...", ...), so text matching would be fragile. System2 gave it 50 (16x) or 75 (4x)
    and never 100.
    """
    if task != "CoffeeSetupMug":
        return {}
    if current_fine_id(plan) != "M2.1":
        return {}
    if est == 100:
        return {}
    return {"est": 100,
            "interventions": [{"rule": "coffee_carry_est", "kind": "est_override",
                               "detail": "M2.1 carry-to-dispenser needs a full segment",
                               "before": est, "after": 100}]}


# Order matters: the plan rewrite runs first so later rules see the revised checklist.
_RULES = (_rule_drawer_base_align, _rule_drop_retract, _rule_sink_faucet_est,
          _rule_coffee_carry_est)

TASKS_WITH_RULES = ("PickPlaceDrawerToCounter", "TurnOnMicrowave", "TurnOnSinkFaucet",
                    "OpenStandMixerHead", "CoffeeSetupMug")


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
    # A subgoal override must not leave a stale detail string describing the old instruction.
    if any(i["kind"] == "subgoal_override" for i in ivs):
        cur["subgoal_detail"] = cur["subgoal"]
    st["consec_skips"] = (st.get("consec_skips", 0) + 1) if cur["skip_s1"] else 0
    cur["interventions"] = ivs
    return cur
