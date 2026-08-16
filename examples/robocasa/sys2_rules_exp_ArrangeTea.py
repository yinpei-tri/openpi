"""ArrangeTea experiment: carry the kettle to the inner side of the tray.

The recipe already asks the planner to put the kettle well inside the tray, but qwen3vl's execution
subgoal normally collapses that location back to merely "the tray". The kettle is then often left
too close to the outer edge. This task-only rule restores the more precise destination in the text
System1 actually receives.

This is deliberately a subgoal experiment, not a plan rewrite: it changes only kettle carry turns,
including their continuation variants, and does not touch the following lower/release action.
``SYS2_ARRANGE_TEA_INNER_TRAY=0`` disables this one experiment;
``SYS2_RULES_NO_EXP=1`` disables the complete experimental layer.
"""

from __future__ import annotations

import os
import re

ARRANGE_TEA = "ArrangeTea"
INNER_TRAY_ENABLED = os.environ.get("SYS2_ARRANGE_TEA_INNER_TRAY", "1") not in ("", "0")

# Observed in the 30-episode human-recipe run as four forms: carry (18), lift-and-carry (15),
# continue lift-and-carry (5), and continue carry (1). Match only the complete command so a kettle
# action aimed at another destination can never be rewritten.
_KETTLE_TO_TRAY_RE = re.compile(
    r"^(?P<continue>continue\s+to\s+)?(?:lift\s+and\s+)?carry\s+the\s+kettle\s+"
    r"to\s+the\s+tray\.?$",
    re.IGNORECASE,
)


def _rule_arrangetea_kettle_inner_tray(task: str, plan: str, subgoal: str, est, state) -> dict:
    """Tell System1 to carry the kettle to the tray's inner side."""
    if task != ARRANGE_TEA or not INNER_TRAY_ENABLED:
        return {}
    match = _KETTLE_TO_TRAY_RE.match((subgoal or "").strip())
    if not match:
        return {}
    prefix = "continue to " if match.group("continue") else ""
    revised = f"{prefix}lift and carry the kettle to the inner side of the tray"
    return {
        "subgoal": revised,
        # combined_eval normally prompts System1 with subgoal_detail. Replace it too, otherwise the
        # planner's old generic detail ("over the tray") would silently remain the executed prompt.
        "subgoal_detail": revised,
        "interventions": [
            {
                "rule": "arrangetea_kettle_inner_tray",
                "kind": "subgoal_override",
                "detail": "ArrangeTea experiment: target the inner side of the tray so the kettle "
                          "is not left near its outer edge; plan and estimate untouched",
                "before": subgoal,
                "after": revised,
            }
        ],
    }


EXP_RULES: tuple = (_rule_arrangetea_kettle_inner_tray,)
EXP_RULE_TASKS: dict[str, tuple[str, ...]] = {
    "_rule_arrangetea_kettle_inner_tray": (ARRANGE_TEA,),
}
