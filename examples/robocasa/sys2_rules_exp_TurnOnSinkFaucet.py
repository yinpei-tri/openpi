"""TurnOnSinkFaucet -- give System1 the turn-on wording that works, and keep the checklist saying it.

ONE RULE, ONE SWITCH. Earlier revisions of this file carried three env toggles and two code paths
while the variants were being measured; the measurement is done (table below) and only the arm that
was graduated survives. ``SYS2_TOSF_FORCE_PLAN=0`` disables it, ``SYS2_RULES_NO_EXP=1`` drops the
whole file. Nothing else is configurable.

WHAT IT DOES, in two places, both aimed at the same string:
  1. on the FIRST execution turn, overwrite the TEXT of the fine steps System2 predicted under the
     faucet milestone with "reach to the sink faucet handle" / "turn on the sink faucet handle" --
     ids, marks, step count and every later turn's checklist left exactly as System2 wrote them;
  2. on any turn, rewrite a "push the sink faucet handle ..." SUBGOAL to "turn on the sink faucet
     handle", preserving a "continue to " prefix.
(2) is a fallback and measured 0 firings once (1) is in place: with the canonical text in its own
checklist, System2 reads it back and says it verbatim. It is kept because it costs nothing and catches
a drifting planner.

Lives in its OWN file rather than in sys2_rules.py because that file is edited from two machines on
the shared mount: the first version of this rule was written into sys2_rules.py and was silently
reverted by the other machine's write ~40 minutes later, mid-launch (the run's rulesnap recorded
repeat_cap only, so 24 wasted episodes measured nothing). sys2_rules_exp.py has the contract; this
module only has to export ``EXP_RULES``.

HISTORICAL BASELINE. At the time this task rule was measured, ``_rule_sink_faucet_est`` was defined
but unregistered and ``est_bump`` had been narrowed to TurnOnMicrowave, so the task ran on
repeat_cap alone. It is now part of the optional general tier, and ``--task-rules`` implies that
tier. This combination is intentional: the faucet estimate is only a floor and System2 usually
already predicts at least 100, so it is normally a no-op. Verified on the recorded historical runs,
the only rules that fired on TurnOnSinkFaucet were

    qwen3vl base   repeat_cap cap_declined 28 turns / max_cap  9 turns   (10/30 episodes)
    qwen3vl v2     est_bump 104, est_resolve 104, sink_faucet_est 2, repeat_cap 26
    qwen35  base   repeat_cap cap_declined 22 / max_cap 7                ( 8/30 episodes)
    qwen35  v2     est_bump  82, est_resolve  82, sink_faucet_est 28, repeat_cap 14

so base->v2 on this task is exactly "add the est layer", and it bought +5 raw on BOTH planners
(21->26, 23->28). It is not free: the est layer spends steps, and this task's official horizon is only
600, so qwen3vl v2's 26 raw is 21 refined -- 5 successes landing at 733-1188 steps. 11 of the 12
not-refined successes in the whole sink-faucet family are on this one task. Hence NO est lever here:
rewording costs nothing in steps, and steps are the currency that decides refined success.

WHAT SYSTEM2 DOES. The COLD plan is a bare milestone on both planners (30/30), "- [ ] M1: turn on the
sink faucet". It does not stay bare: System2 answers the first execution turn with a plan_update of its
own expanding M1 into two fine steps, and combined_eval applies that update BEFORE the rules run (a
first version of this rule gated on "no fine steps" therefore never fired at all). That expansion is
completely uniform -- measured over all 30 episodes of two independent qwen3vl runs: exactly two steps,
marks ~ and blank,

    M1.1  "reach to the sink faucet handle"             30/30, already canonical
    M1.2  "push the sink faucet handle to turn it on"   27/30   <-- the one string worth changing
          "turn on the sink faucet handle"               3/30

so M1.1 needs no change and the whole intervention is ONE string in 27 of 30 episodes. Same divergence
across the episode's exec turns, and qwen35 -- the planner that does better here -- is the one already
using the canonical form:

    qwen35   "turn on the sink faucet handle"             26 turns  (+42 "continue to ...")
    qwen3vl  "push the sink faucet handle to turn it on"  27 turns  (+47 "continue to ...")

Replayed against the qwen35 base run this rule would fire in only 4/30 episodes, so it targets the
planner-specific defect and leaves the good planner alone. That is a replay estimate, NOT a measured
run: a 30-episode qwen35 confirmation is still owed before a full sweep.

MEASURED (qwen3vl-17124 + progact-270k/269999, the 30-episode manifest set, max_steps_cap=400,
repeat_cap the only other active rule; "refined" = success at or before the 600-step horizon):

    arm                                       n   raw  refined  steps/ep  turns  max_cap deaths
    base (repeat_cap only)                   30    21     20       440      4.0        9
    base, independent replicate              28    19     16       468      4.2        9
    subgoal rewrite only, run 1              30    24     21       372      3.6        6
    subgoal rewrite only, run 2              30    23     22       427      3.7        7
    THIS RULE, invasive implementation       30    21     18       499      4.4        8
    THIS RULE, as it stands                  30    25     23       358      3.5        4
    v2's est layer, for reference            30    26     21       600      3.5        0

    pooled  base                  raw 40/58 = 69.0%   refined 36/58 = 62.1%   454 steps/ep
    pooled  subgoal rewrite only  raw 47/60 = 78.3%   refined 43/60 = 71.7%   400 steps/ep
    pooled  THIS RULE             raw 46/60 = 76.7%   refined 41/60 = 68.3%   429 steps/ep

    The base row is still on disk (combine/s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-base, the
    50-task sweep). The five rule/replicate arms -- ...-{baserep-partial, tosfword, tosfword2,
    tosfplan, tosfsteps} -- were DELETED after measurement, so this table is the only record of
    their episode-level outcomes. The exact code each one ran survives as
    _evallogs/rulesnap/<arm>.{sha256,exp.py}; re-running any of them means re-running the episodes.

READ THE TWO "THIS RULE" ROWS TOGETHER: 21/18 and 25/23 ARE ONE INTERVENTION MEASURED TWICE. Verified,
not assumed -- both runs fired exactly 27 times, all on turn 0, and their turn-0 EFFECTIVE plans and
subgoals are byte-identical across all 60 episodes. The invasive implementation additionally replaced
the whole fine list, reset its marks, pinned the turn-0 subgoal and renormalised the checklist every
later turn, and every one of those is a verifiable no-op here: System2's turn-0 checklist already has
exactly that shape, and it never reintroduces the push wording into a fine step afterwards. So the
4-raw / 5-refined spread between those rows is SAMPLING NOISE, the fourth independent demonstration of
the band sys2_rules_exp.py documents (base itself moved 21/20 -> 19/16 on the same episodes). An
earlier reading of this file called the invasive arm harmful on the strength of its single 21/18 run;
that is retracted.

WHY THIS ARM AND NOT THE SUBGOAL-ONLY ONE, given 46/60 vs 47/60 and 41/60 vs 43/60 are a coin flip:
  * it fixes the cause once per episode, instead of patching the symptom 2-3 times per episode forever;
  * it keeps the checklist and the executed command in agreement. Subgoal-only leaves System2 reading
    "M1.2: push the sink faucet handle to turn it on" while System1 is told "turn on the sink faucet
    handle", every turn -- so System2 judges progress against a step text that was never the command
    executed. n=30 cannot surface that; it is still the wrong thing to ship;
  * it is the cheapest arm in steps (358 vs base 440 vs v2 600) on a task gated at 600.

NOT ESTABLISHED, and graduated anyway on those grounds plus a consistent sign: pooled Fisher vs base
is p=0.41 raw / p=0.56 refined, and the best single paired result is this arm's own raw +4/-0, p=0.125.
Every one of four rule runs beat base on both metrics and all four were cheaper in steps, and across
them no episode that base solved in both its runs dropped below 2/4 while two episodes base never
solved came back 3/4 and 4/4.

REMAINING LEAK, deliberately unaddressed: in 3 of 30 episodes System2 puts the push phrasing in the
MILESTONE line itself ("- [~] M1: push the sink faucet handle to turn it on"). This rule touches fine
steps only. It is the same 3 episodes in all four runs, and normalising it is unmeasured -- so it is
the next experiment, not part of what was graduated.
"""

from __future__ import annotations

import os
import re

# Imported as a module, not as names: this file is loaded from inside sys2_rules.py's own module body,
# so the helpers are looked up at CALL time, when that module is fully built. _blocks / _render are the
# checklist helpers sys2_rules_exp.py advertises to this patch layer; they are underscore-named only
# because sys2_rules.py has no public API split, so SLF001 is noise here.
# ruff: noqa: SLF001
import sys2_rules as _sr

TOSF = "TurnOnSinkFaucet"
# The single switch. Kept under this name because all six recorded runs used it and their intervention
# records are keyed on the rule name below.
TOSF_FORCE_PLAN = os.environ.get("SYS2_TOSF_FORCE_PLAN", "1") not in ("", "0")
TOSF_M1_1 = "reach to the sink faucet handle"
TOSF_M1_2 = "turn on the sink faucet handle"
# qwen3vl's phrasing for the turn-on step, with or without a "continue to " prefix.
_TOSF_PUSH_RE = re.compile(r"^(\s*(?:continue\s+to\s+)?)push\s+the\s+sink\s+faucet\s+handle\b.*$", re.IGNORECASE)


def _rule_tosf_force_plan(task: str, plan: str, subgoal: str, est, state) -> dict:
    """TurnOnSinkFaucet: canonicalise the turn-on wording in the checklist and in the subgoal."""
    if task != TOSF or not TOSF_FORCE_PLAN:
        return {}
    blocks = _sr._blocks(plan)
    if not blocks:
        return {}
    ivs: list[dict] = []
    out: dict = {}
    # (1) FIRST execution turn only: overwrite the text of the steps System2 predicted. Marks, ids,
    # step count and any step beyond the first two are left alone, and it does NOT return early --
    # control falls through to (2) so this turn's subgoal is normalised too if System2 asked for the
    # push phrasing straight away.
    b0 = blocks[0]
    if not state.get("tosf_steps_done") and "faucet" in (b0["text"] or "").lower():
        state["tosf_steps_done"] = True
        hit = False
        for canon, f in zip((TOSF_M1_1, TOSF_M1_2), b0["fine"], strict=False):
            if (f["text"] or "").strip().lower().rstrip(".") == canon:
                continue  # already canonical -- no intervention to record
            before_t = f["text"]
            f["text"] = canon
            hit = True
            ivs.append(
                {
                    "rule": "tosf_force_plan",
                    "kind": "plan_revised",
                    "detail": f"overwrite the text of {f['fid']} that System2 predicted with the "
                    "canonical wording, on the first execution turn only; its mark, id and the "
                    "rest of the checklist are left as System2 wrote them",
                    "before": before_t,
                    "after": canon,
                }
            )
        if hit:
            out["plan"] = _sr._render(blocks)
    # (2) ANY turn: normalise the subgoal handed to System1. A fallback -- measured 0 firings once (1)
    # is in place, because System2 reads the canonical text back out of its own checklist.
    m = _TOSF_PUSH_RE.match(subgoal or "")
    if m:
        new_sg = f"{m.group(1)}{TOSF_M1_2}"
        if new_sg.strip().lower() != (subgoal or "").strip().lower():
            out["subgoal"] = out["subgoal_detail"] = new_sg
            ivs.append(
                {
                    "rule": "tosf_force_plan",
                    "kind": "subgoal_override",
                    "detail": "normalise the turn-on wording: this planner says 'push ... to turn it "
                    "on' where the better-scoring one says 'turn on the sink faucet handle'",
                    "before": subgoal,
                    "after": new_sg,
                }
            )
    return {**out, "interventions": ivs} if ivs else {}


EXP_RULES: tuple = (_rule_tosf_force_plan,)
