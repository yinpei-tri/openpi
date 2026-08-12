"""EXPERIMENTAL, UNVERIFIED per-task rules -- a deletable patch layer over sys2_rules.py.

Currently EMPTY: everything tried so far has either graduated into sys2_rules.py or been rejected.
One file per task -- sys2_rules_exp_<Task>.py is loaded too, so parallel work on different tasks
never shares a file. See .claude/agents/rule-experiment.md for the per-task agent protocol.

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

  * n=20 CANNOT RESOLVE A 2-EPISODE EFFECT. Measured directly: FOUR runs of the same effective code
    path on ScrubCuttingBoard (repeat_cap declines on every one, so nothing acts) gave 13/20, 9/20,
    7/20, 9/20 -- pooled 38/80 = 47.5%, observed range 7-13. At p=.475 the binomial SD is 2.23
    episodes, so +-2SD is +-4.5: that spread is PLAIN SAMPLING NOISE and needs no mechanism. Power at
    n=20 reaches only about +20pp (+4/20); detecting a true +10pp needs ~390 episodes PER ARM. So the
    famous "13/20" was never a rules achievement and does not reproduce, and any per-task n=20 delta
    of 1-3 episodes is unmeasured either way. Establish the band first (count each reference run's
    actual interventions; if none acts, the two runs are replicates and their spread is free
    variance), then judge against it, and prefer paired McNemar on the discordant pairs to comparing
    two rates. Code identity for those four runs was verified, not assumed: same S1/S2 checkpoint and
    step, same max_steps_cap=800 / max_turns=14 / horizon_mult=2.0, repeat_cap AST-identical between
    the estbump commit (8ce348d, when the official set was 9 rules) and now, apply_rules differing
    only by additive tx_label plumbing no scrub rule returns, and the cap_declined rate inside FAILING
    episodes identical at 0.42/0.42/0.43 per turn -- the raw decline counts differ 41 vs 81 only
    because a run with more failures has more failing turns. Pooling just the three cap=800 runs gives
    29/60 = 48.3%, the same answer.
  * Consequently both ScrubCuttingBoard experiments are INSIDE that band, not refuted: grasp est>=75
    gave 10/20 and 5/20, and one borrowed "continue to scrub ..." turn gave 8/20 (McNemar vs the
    13/20 run: 6 lost / 1 gained, p=0.125). The injector itself worked exactly as designed -- 20
    tx_sg_incomplete + 20 tx_resume with the plan byte-identical. What is established is only that
    neither lever produces a LARGE effect; a small one is beyond this sample size. Do not record
    either as "harmful".
  * Mean turns is NOT independent evidence. A success ends its episode early, so mean turns tracks
    the success count -- 13/20 -> 8.8, 9/20 -> 10.3/10.4, 8/20 -> 11.0, 7/20 -> 11.1 on this task.
    Quoting a success delta and a turns delta as if they corroborated each other double-counts one
    measurement.
  * ScrubCuttingBoard IS DECIDED ENTIRELY BY THE FIRST RETRACT, and the split is absolute. Retract
    turns per episode, two independent runs:
        verified run   successes 1,1,1,...  (all 13 -- mean 1.0)   failures 8,8,8,9,9,10,10
        experiment run successes 1,1,1,...  (all 11 -- mean 1.0)   failures 0,6,7,7,8,8,8,9,9
    Every success retracts ONCE; every failure loops 6-10 times to max_turns. There is no middle. So
    the task turns on whether the first retract latches env_success, and nothing after it ever
    rescues the episode. 73-75 retract segments per 20 episodes, mean 25 steps, and NOT ONE of them
    ends on budget (57 stop_rule, 5 lookahead, 13 env_success) with the arm parked at width 0.0799 --
    the third instance of the quiescence defect after the GetToastedBread wait and the WeighIngredients
    cabinet door. force_steps is the untried lever and its ceiling is +8/20, well clear of the band.
  * REJECTED on that task, all inside the band: grasp est>=75 (10/20, 5/20), one borrowed
    "continue to scrub" turn (8/20), and a borrowed "continue to press and scrub" turn at est 100
    PLUS a width-gated sponge re-grasp (11/20 together). The re-grasp is worth a note because it
    WORKED and still did not matter: it fired on 3 episodes whose width read 0.0010 after the carry
    and recovered the sponge outright (0.0010 -> 0.0648 and -> 0.0581 on two of them, the third by the
    following turn), those episodes then scrubbed and released normally -- and all three still died in
    the retract loop. A rule can be mechanically perfect and irrelevant if it is not aimed at the sink.
  * A WIDTH SIGNATURE IS A PROPERTY OF THE ROLLOUT, NOT OF THE EPISODE. The sponge-drop analysis
    predicted the re-grasp would fire on eps 0/15/18 (the episodes that dropped it in the recorded
    run); live it fired on 1/14/17. Only the RATE transfers, never the identities -- so "this rule will
    fire on these five episodes" is not a testable prediction, and the fired-on set being all
    previously-successful episodes is an artefact of comparing different rollouts, not evidence of harm.
    INDEPENDENTLY REPRODUCED on PreSoakPan: replay predicted eps 2/6/8/11/12, live fired on
    1/6/8/11/13/14/19 (3 of 5, plus 4 unpredicted), and a 4:2 pan-weighted replay came back 6:1
    sponge-weighted.

  * CHECK A WIDTH BAR AGAINST THE SUCCESS READINGS, NOT ONLY THE FAILURES -- and require a real GAP,
    not merely separation on the recorded sample. The graduated PackIdenticalLunches re-grasp works
    because that task's widths are bimodal with a clean gap (0.0041 -> 0.0107 -> 0.0118 -> 0.0166), so
    a 0.015 bar sits INSIDE the gap. Ported unchanged to PreSoakPan, whose pan and sponge are thin
    enough that a SUCCESSFUL grasp reads only 0.017-0.025 against that same bar, it went 13/20 ->
    11/20 (2 gained, 4 lost, McNemar p=0.688 -- no effect established in either direction, so this is
    NOT recorded as harm). The bar separated the 7 failures from the 13 successes on the recorded
    sample with a margin of only 0.002-0.010; that margin is smaller than the run-to-run variation in
    the signal itself, which is the actual reason it does not transfer. On a task where 9/20 die at the
    turn cap, a borrowed turn spent re-grasping an object already in the gripper is not free.
    (Rejected, not graduated. Results kept under combine/debug-PreSoakPan-regrasp-v1.)
"""

from __future__ import annotations

EXP_RULES: tuple = ()
