"""EXPERIMENTAL, UNVERIFIED per-task rules -- a deletable patch layer over sys2_rules.py.

Currently EMPTY: everything tried so far has either graduated into sys2_rules.py or been dropped.

WORKFLOW
  1. add a section here for ONE task, with the rules gated on that task name;
  2. run it (RUN_LABEL=debug-<Task>-vN) and compare against the task's baseline and the verified
     rules on the SAME episodes;
  3. if it helps, move the rules into sys2_rules.py under a "GRADUATED" banner WITH the measurement
     table, and empty this file again;
  4. if it does not, delete the section -- nothing else in the pipeline changes.

Nothing here is committed until a run shows it helps, so keep a section in place until it has
either graduated or been rejected. Do NOT overwrite an in-flight section: two of the CoffeeSetupMug
rules were never in git and were nearly lost that way.

CONTRACT
  * export ``EXP_RULES``: a tuple of ``fn(task, plan, subgoal, est, state) -> dict``, same protocol
    as the verified rules -- return {} for "does not apply", else any of
    ``plan`` / ``subgoal`` / ``subgoal_detail`` / ``est_proposal`` / ``skip_s1`` plus
    ``interventions`` (each with rule, kind, detail, before, after).
  * est is PROPOSED, never assigned: apply_rules resolves the largest proposal once, so an
    experimental est cannot chain with a verified one.
  * EVERY rule is gated on exactly ONE task, checked as the first statement. Nothing in this file
    may affect any other task -- verify with a cross-contamination check before running.
  * ``SYS2_RULES_NO_EXP=1`` ignores this file without deleting it (e.g. to re-measure the verified
    baseline); deleting the file has the same effect, since the import hook is a no-op on failure.

Reusable helpers live in sys2_rules.py and can be imported here:
  ``_blocks`` / ``_render`` / ``current_fine_id`` / ``current_milestone_text`` / ``_norm`` /
  ``bump_est`` / ``_split_fine_step`` (split one fine step in two) /
  ``_advance_current_step`` (mark the current step done and hand System1 the next -- use this
  rather than ``skip_s1``, which freezes the env and blocks success detection).
"""

from __future__ import annotations

EXP_RULES: tuple = ()
