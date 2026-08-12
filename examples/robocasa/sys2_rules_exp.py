"""EXPERIMENTAL, UNVERIFIED per-task rules -- a deletable patch layer over sys2_rules.py.

Currently EMPTY: everything tried so far has either graduated into sys2_rules.py or been rejected.

WORKFLOW
  1. add a section here for ONE task, with every rule gated on that task name;
  2. run it (RUN_LABEL=debug-<Task>-vN) and compare against the task's baseline and the verified
     rules on the SAME episodes -- twice, since a single n=20 run cannot resolve a 1-2 episode effect;
  3. if it helps, move the rules into sys2_rules.py under a "GRADUATED" banner WITH the measurement
     table, and empty this file again;
  4. if it does not, delete the section -- nothing else in the pipeline changes.

Do NOT overwrite an in-flight section: two of the CoffeeSetupMug rules were never in git and were
nearly lost that way.

CONTRACT
  * export ``EXP_RULES``: a tuple of ``fn(task, plan, subgoal, est, state) -> dict``, same protocol
    as the verified rules -- return {} for "does not apply", else any of
    ``plan`` / ``subgoal`` / ``subgoal_detail`` / ``est_proposal`` / ``skip_s1`` / ``tx_label``
    plus ``interventions`` (each with rule, kind, detail, before, after).
  * est is PROPOSED, never assigned: apply_rules resolves the largest proposal once, so an
    experimental est cannot chain with a verified one.
  * ``state`` is a per-EPISODE dict threaded across turns -- the place for a rule's bookkeeping.
    combined_eval also puts ``grip_width`` / ``grip_width_min`` there, the gripper aperture left by
    the PREVIOUS segment, so a rule can react to a physical outcome and not only to System2's text.
  * ``tx_label`` marks a turn a rule INJECTED rather than one System2 asked for -- "tx_sg_failed"
    for a recovery after a detected failure, "tx_sg_incomplete" for a continuation.
  * EVERY rule is gated on exactly ONE task, checked as the first statement -- verify with a
    cross-contamination check before running.
  * ``SYS2_RULES_NO_EXP=1`` ignores this file without deleting it.

WHAT THE GRADUATED RULES ESTABLISHED, worth reading before designing the next experiment:

  * est_length is the most reliable lever. It is a POLICY CONDITIONING tag (rendered into System1's
    prompt as "Estimated Length"), not just a budget, so raising it makes System1 move slower and
    more precisely. It works on precision-bound tasks and does nothing -- or hurts -- on turn-bound
    ones (ArrangeTea -4, WashFruitColander -4).
  * A CONTINUATION belongs in the plan; a RECOVERY does not. Inserting a step also gives repeat_cap
    a non-last step to advance into, which is often where the real gain comes from.
  * Never implement "skip a subgoal" as ``skip_s1``: a skipped turn does not step the env and
    ``_check_success()`` is polled per executed step, so it freezes the world (4/8 -> 0/8 measured).
    Use ``_advance_current_step`` instead.
  * Reusable helpers in sys2_rules.py: ``_blocks`` / ``_render`` / ``current_fine_id`` /
    ``current_milestone_text`` / ``_norm`` / ``bump_est`` / ``_split_fine_step`` /
    ``_advance_current_step`` / ``_target_step`` / ``_inject_after_step`` (borrow turns, hold and
    resume System2's subgoal, count System2's own re-issues against the budget).
"""

from __future__ import annotations

EXP_RULES: tuple = ()
