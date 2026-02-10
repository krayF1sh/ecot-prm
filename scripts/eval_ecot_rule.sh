#!/bin/bash

TASK_SUITE=${1:-"libero_spatial"}
NUM_EPISODES=${2:-10}
GPUS=${3:-"0"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES=$GPUS python experiments/robot/eval_ecot_rule.py \
    --task_suite $TASK_SUITE \
    --num_episodes $NUM_EPISODES \
    --output results/ecot_rule_${TASK_SUITE}.json
