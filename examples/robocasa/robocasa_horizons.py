"""RoboCasa's OFFICIAL per-task rollout horizon, in env steps -- the benchmark's own step budget.

GENERATED, do not hand-edit. Source of truth is ``robocasa.utils.dataset_registry``
(``ATOMIC_TASK_DATASETS`` / ``COMPOSITE_TASK_DATASETS``, field ``horizon``, read by
``dataset_registry_utils.get_task_horizon``) at robocasa revision 26a7ba9. Copied in-tree so scoring
does not need robosuite/robocasa importable -- the openpi venv has neither, and a number the eval
tables depend on should be auditable in this repo rather than fetched from a sibling checkout.

WHY IT MATTERS. RoboCasa scores a rollout by stepping the env at most ``horizon`` steps while polling
``env._check_success()`` densely. Our hierarchical eval budgets per SUBGOAL (est_length *
horizon_mult, capped by --max-steps-cap) and per TURN (max_turns) and never counts total env steps,
so an episode can legitimately execute more than ``horizon`` steps -- which makes our raw success
rate incomparable with any number measured under the official protocol. ``scripts/extract_combine_
results.py`` therefore also reports a REFINED rate: a success counts only if the env's success check
first fired at or before ``horizon`` cumulative steps.

To regenerate (needs the robocasa env, e.g. ~/micromamba/envs/robocasa/bin/python):
    python scripts/build_robocasa_horizons.py
"""

from __future__ import annotations

# task -> official horizon in env steps. 50 tasks: the 50 in combined_eval.TARGET_EVAL_EPISODES.
TASK_HORIZON: dict[str, int] = {
    "ArrangeBreadBasket": 4350,            # composite
    "ArrangeTea": 2250,                    # composite
    "BreadSelection": 1950,                # composite
    "CategorizeCondiments": 1650,          # composite
    "CloseBlenderLid": 900,                # atomic
    "CloseFridge": 900,                    # atomic
    "CloseToasterOvenDoor": 450,           # atomic
    "CoffeeSetupMug": 600,                 # atomic
    "CuttingToolSelection": 1200,          # composite
    "DeliverStraw": 2550,                  # composite
    "GarnishPancake": 2700,                # composite
    "GatherTableware": 2250,               # composite
    "GetToastedBread": 3000,               # composite
    "HeatKebabSandwich": 2700,             # composite
    "KettleBoiling": 1500,                 # composite
    "LoadDishwasher": 1800,                # composite
    "MakeIceLemonade": 3000,               # composite
    "NavigateKitchen": 450,                # atomic
    "OpenCabinet": 1050,                   # atomic
    "OpenDrawer": 750,                     # atomic
    "OpenStandMixerHead": 450,             # atomic
    "PackIdenticalLunches": 3900,          # composite
    "PanTransfer": 1800,                   # composite
    "PickPlaceCounterToCabinet": 750,      # atomic
    "PickPlaceCounterToStove": 600,        # atomic
    "PickPlaceDrawerToCounter": 750,       # atomic
    "PickPlaceSinkToCounter": 900,         # atomic
    "PickPlaceToasterToCounter": 600,      # atomic
    "PortionHotDogs": 2250,                # composite
    "PreSoakPan": 2400,                    # composite
    "PrepareCoffee": 1800,                 # composite
    "RecycleBottlesByType": 2850,          # composite
    "RinseSinkBasin": 1350,                # composite
    "ScrubCuttingBoard": 1200,             # composite
    "SearingMeat": 4350,                   # composite
    "SeparateFreezerRack": 2400,           # composite
    "SetUpCuttingStation": 2400,           # composite
    "SlideDishwasherRack": 450,            # atomic
    "StackBowlsCabinet": 2100,             # composite
    "SteamInMicrowave": 2100,              # composite
    "StirVegetables": 2400,                # composite
    "StoreLeftoversInBowl": 2550,          # composite
    "TurnOffStove": 750,                   # atomic
    "TurnOnElectricKettle": 450,           # atomic
    "TurnOnMicrowave": 450,                # atomic
    "TurnOnSinkFaucet": 600,               # atomic
    "WaffleReheat": 4050,                  # composite
    "WashFruitColander": 3150,             # composite
    "WashLettuce": 1650,                   # composite
    "WeighIngredients": 3000,              # composite
}
