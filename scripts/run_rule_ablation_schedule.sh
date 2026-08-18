#!/usr/bin/env bash
# Four rule-only qwen3vl ablations, balanced over the eight already-running S1/S2 stacks.
# Every arm uses the historical per-task max_turn policy and the 30-episode target manifest.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

S2_PATH=/shared/data/sys2_ckpts/system2-full-0804-qwen3vl-4b-gb192-full-vitfull-lr1e5-vitlr2e6-alignerlr1e5-zero2-2n-ep3/checkpoint-17124

SINK_TASKS=PreSoakPan,RinseSinkBasin,TurnOnSinkFaucet,WashFruitColander,WashLettuce
PRESS_TASKS=TurnOnMicrowave,SteamInMicrowave,WaffleReheat,TurnOnElectricKettle,KettleBoiling,PrepareCoffee
TASK_RULE_TASKS=ArrangeTea,CloseToasterOvenDoor,CoffeeSetupMug,GetToastedBread,OpenStandMixerHead,PackIdenticalLunches,PickPlaceCounterToCabinet,PickPlaceDrawerToCounter,PickPlaceSinkToCounter,StackBowlsCabinet,TurnOnMicrowave,TurnOnSinkFaucet
REGRASP_TASKS=ArrangeBreadBasket,ArrangeTea,BreadSelection,CategorizeCondiments,CloseBlenderLid,CuttingToolSelection,DeliverStraw,GarnishPancake,GatherTableware,GetToastedBread,HeatKebabSandwich,KettleBoiling,LoadDishwasher,MakeIceLemonade,PackIdenticalLunches,PanTransfer,PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,PickPlaceToasterToCounter,PortionHotDogs,PreSoakPan,RecycleBottlesByType,RinseSinkBasin,ScrubCuttingBoard,SearingMeat,SeparateFreezerRack,SetUpCuttingStation,StackBowlsCabinet,SteamInMicrowave,StirVegetables,StoreLeftoversInBowl,TurnOffStove,WaffleReheat,WashFruitColander,WashLettuce,WeighIngredients

run_arm() {
    local label=$1 selector=$2 task_rules=$3 tasks=$4 gpus=$5 s1_base=$6 s2_base=$7
    env \
        METHOD=progact STEP=269999 \
        S2_CKPT="$S2_PATH" \
        RUN_LABEL="$label" \
        USE_EVAL_SET=1 TASK_SET=all TASKS="$tasks" \
        S1_GPUS="$gpus" S2_GPUS="$gpus" S1_BASE="$s1_base" S2_BASE="$s2_base" \
        SKIP_SERVERS=1 RESUME=1 \
        ROLLOUT_LIMIT_MODE=max_turns MAX_TURNS=20 \
        GENERAL_RULES=1 TASK_RULES="$task_rules" LAST_MILESTONE_RETRY=0 \
        SYS2_RULES_GENERAL_ALLOWLIST="$selector" \
        bash examples/robocasa/run_combine_fleet.sh
}

# Work is approximately balanced by expected episode cost: 150/180/360/1110 episodes receive
# 1/1/2/4 stacks respectively.  Port bases match the persistent server on each physical GPU.
run_arm ablation-sink-faucet sink_faucet_est 0 "$SINK_TASKS" 0 8060 8100 &
p_sink=$!
run_arm ablation-press-again press_again 0 "$PRESS_TASKS" 1 8061 8101 &
p_press=$!
run_arm ablation-task-specific-bundle none 1 "$TASK_RULE_TASKS" 2,3 8062 8102 &
p_task=$!
run_arm ablation-regrasp-recovery regrasp_recovery 0 "$REGRASP_TASKS" 4,5,6,7 8064 8104 &
p_regrasp=$!

status=0
for pid in "$p_sink" "$p_press" "$p_task" "$p_regrasp"; do
    wait "$pid" || status=1
done
exit "$status"
