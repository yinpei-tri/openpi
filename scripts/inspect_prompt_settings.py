"""Render the COMPLETE pi0.5 prompt for each RoboCasa System1 prompt-ablation setting.

Shows the literal text the model conditions on (task/subgoal line + optional conditioning
line + state block + gripper + Action:) for each tag setting, so you can eyeball exactly
what each ablation changes. Uses representative placeholder text/state (no data/S3 needed).

    uv run python scripts/inspect_prompt_settings.py
"""

import numpy as np

from openpi.models import tokenizer as _tok
from openpi.policies import robocasa_policy as rp

# Representative content (fake but realistic).
TASK = "make coffee"
SG_SIMPLE = "pick up the mug"
SG_RICH = "grasp the white mug by its handle and lift it off the rack"
SG_COARSE = "prepare the mug"
COND = {"quality": "good", "est_length": "42", "executed_step": "7"}
COND_NOEXEC = {"quality": "good", "est_length": "42"}  # noexec: drop only Executed Step
GRIP = "Open"

# 14-d current + 14-d anchor lean state (numbers arbitrary — just to show the layout).
CUR = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
ANC = np.linspace(-0.3, 0.3, 14, dtype=np.float32)
STATE_28 = np.concatenate([CUR, ANC])  # layout is [current, anchor]

_PT = _tok.PaligemmaTokenizer(max_len=256)


def render(*, subgoal, include_task_goal, conditioning, state, state_split, grip, include_state=True):
    """Full prompt exactly as TokenizePrompt would build it (RoboCasa flavor)."""
    line = rp.build_prompt(subgoal, TASK, include_task_goal=include_task_goal, conditioning=conditioning)
    toks, _ = _PT.tokenize(
        line,
        state,
        state_split=state_split,
        state_split_label="Initial State",
        task_state_sep="\n",
        preserve_newlines=True,
        gripper_flag=grip,
        include_state=include_state,
    )
    ids = [int(t) for t in toks if int(t) != 0]
    return _PT._tokenizer.decode(ids)  # noqa: SLF001 - decode for display only


# Each entry: tag suffix -> the kwargs that differ from default.
DEFAULT = {
    "subgoal": SG_SIMPLE,
    "include_task_goal": True,
    "conditioning": COND,
    "state": STATE_28,
    "state_split": 14,
    "grip": GRIP,
}
SETTINGS = [
    ("progcls_granfine_verbsimp  (DEFAULT)", {}),
    ("..._verbrich               (detailed subgoal)", {"subgoal": SG_RICH}),
    ("..._grancrse               (milestone/coarse subgoal)", {"subgoal": SG_COARSE}),
    ("..._notask", {"include_task_goal": False}),
    ("..._nocond                (drop the whole conditioning line)", {"conditioning": None}),
    ("..._noexec                (drop only Executed Step)", {"conditioning": COND_NOEXEC}),
    ("..._noanchorstate          (drop anchor half; single State:)", {"state": CUR, "state_split": None}),
    ("..._nostate                (drop the ENTIRE state block)", {"include_state": False}),
    ("..._nogrip", {"grip": None}),
    ("..._nostate_nogrip          (no state, no gripper)", {"include_state": False, "grip": None}),
    ("..._nocond_nostate_notask_nogrip  (minimal)",
     {"include_task_goal": False, "conditioning": None, "include_state": False, "grip": None}),
]

for name, over in SETTINGS:
    kw = {**DEFAULT, **over}
    print("=" * 92)
    print(name)
    print("-" * 92)
    print(render(**kw))

print("=" * 92)
print("NOTE: `nostate` drops the ENTIRE state block (no State:/Initial State:/Current State:);")
print("`noanchorstate` drops only the anchor half (Current State stays). Progress predictor")
print("(progact/progreg/progcls) does NOT change the prompt text — only the training target/head.")
