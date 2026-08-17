# Human-interactive System2 + System1 evaluation

The interactive evaluator is mounted at `/human-interactive` in the existing System1 evaluation
GUI. It assumes the System1 websocket policy server and System2 vLLM server are already running.
The batch evaluator and `/combine` browser are unchanged.

## Start the GUI

Run the GUI under the RoboCasa interpreter because this process owns the live simulator:

```bash
bash examples/robocasa/run_sys1_eval_gui.sh \
  --port 8092 \
  --s1-port 8060 \
  --s2-port 8100
```

This single GUI process serves the existing rollout views (including `/combine`) and the live
interactive page. Open `http://<host>:8092/`; the root page links to
`/human-interactive`. The S1 and S2 model servers still need to be running independently.

Use `--s1-host`, `--s1-port`, `--s2-host`, `--s2-port`, and `--s2-model` for the served models.
The older `--hitl-s1-*` and `--hitl-s2-*` spellings remain accepted. Other useful overrides
include `--hitl-data-root`, `--hitl-results-root`, `--hitl-general-rules`, `--hitl-task-rules`,
`--hitl-max-steps-cap`, and `--hitl-prompt-source`. Run `subtask_eval_gui.py --help` for the full
list.

## Interaction flow

1. Select one of the exact 1,500 official target episodes. The picker is generated from
   `TARGET_EVAL_EPISODES` and labels the task as `atomic_seen`, `composite_seen`, or
   `composite_unseen`.
2. System2 proposes a cold plan. A structured card editor owns milestone/fine-step IDs and
   checklist syntax. Edit only the natural-language sentences; use buttons to set each item to
   To do, Active, or Done, and to insert, delete, or reorder items.
3. System2 proposes the next execution turn. The page shows the raw model layer and the
   rule-adjusted layer separately. The editor starts from the raw System2 prediction by default;
   “Use S2 prediction” and “Use rule-adjusted” can restore either version before committing.
   The Judge & Subgoal card displays the real vLLM token stream while generation is in progress.
   As complete tags become available, the corresponding plan, judge, subgoal, detail, and
   estimated-length controls update and briefly shine so the operator can see what changed.
4. Edit the plan, judge, subgoal, or estimated length. The detail field is retained in saved model
   data but hidden from the operator. Four presets cover common interventions:
   **Continue last subgoal** reactivates the previous fine step with `subgoal_incomplete`;
   **Redo last subgoal** reactivates it with `subgoal_failed`; **Use current subgoal** restores the
   raw System2 decision; and **Step next subgoal** completes the current fine step and activates the
   next one. If a milestone has no remaining fine step, Step next marks it done and enables
   **Ask S2 for next subgoal** to unroll the next milestone.
   From the Ready state, **Intervene** opens a free-play turn without querying System2. Type any
   subgoal and estimated length, then click **Execute**. This does not add the subgoal to the plan;
   the checklist changes only when the operator edits its cards directly. Free interventions are
   explicitly labeled in storage and bypass rule-generated System1 action overrides.
5. While System1 acts, each post-action observation enters a bounded server queue. The browser
   consumes one frame at a time and keeps the most recent image when the queue is empty. Below the
   image, the panel presents a compact metadata/control row, the exact tokenized language prompt,
   three telemetry plots, and finally the numeric System1 action-chunk table with its rolling
   pointer. Progress and gripper width have separate plots; commanded motion overlays EEF position,
   EEF rotation, base motion, and total `|Δa|`. The fixed turn anchor image is available under the
   collapsed **More details** section.
6. “Revert one turn” returns to the beginning of the previous executed subgoal. The abandoned
   attempt remains saved as a superseded branch and a new attempt receives the next branch index.

The bottom reference panel retains the most recent exact System2 system prompt, user prompt,
visual input, raw response, parsed prediction, latency, and usage after the proposal is accepted or
executed.

The compact timing strip in the status bar separates browser/catalog loading, episode construction
and reset, latest System2 latency, average System1 policy latency, average simulator step time, and
turn wall time. Hover a value for call counts, latest/total timing, render timing, and
post-processing details. During a blocking reset or prediction, the status bar also shows a live
elapsed-time counter.

Rule stop/skip/action-override/forced-length switches are not exposed in the operator UI. Rule
interventions remain visible in **More details** and in the saved decision layers. Clicking
**Execute** is authoritative over a rule-requested stop or skip; an override
is recorded only when such a rule was actually active. Rule action channels and forced lengths
remain applied.

Human-interactive sessions have no episode-level step or System2-turn cap. Environment success and
`task_finish` remain visible signals but do not make the session terminal; the operator may keep
probing until they load another episode or restart the GUI. Each individual System1 subgoal still
uses its estimated-length budget and normal progress/quiescence stopping behavior.

## Storage and rollback

Sessions are written under:

```text
/home/ec2-user/data/sys1_eval_results/human_interactive/
  target__<task_split>__<task>__episode_<index>__<timestamp>__<session>/
```

SQLite in WAL mode is authoritative. `manifest.json`, `events.jsonl`, node checkpoint documents,
attempt documents, edit records, and every binary-artifact sidecar repeat the complete target
episode provenance. Each attempt preserves:

- immutable raw System2 prediction;
- rule-adjusted decision and every intervention;
- human-final decision and field-level edits;
- exact actions applied to `env.step` after rule overrides and zero-arm-in-base;
- System1 step trace, raw rollout video, and the condensed clip used by the next System2 turn;
- `execution/demo_manifest.json`, a per-turn index containing the turn number, accepted subgoal,
  estimated length, plan, intervention label, execution outcome, System1 prompts, and paths to the
  video, anchor, action array, and step trace for later demo assembly.

Rollback uses a soft controller reset, restores the official initial MuJoCo state plus captured
Python-side fixture/task latches, then replays the accepted `action_applied_raw12` prefix. A flattened
MuJoCo checkpoint is also saved for diagnosis, but it is not authoritative because appliance latches
and cooking timers live in Python.

The official target source is deliberately marked as training-contaminated: automatic training
export is blocked by default. An exporter should require an explicit target-data opt-in.

On this host, destroying an EGL simulator context can terminate the long-lived Flask process.
Closed sessions are therefore retained in memory until the GUI process exits. Restart the GUI
between long annotation batches or after switching episodes many times.
