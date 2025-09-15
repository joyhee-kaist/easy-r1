#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct #rsoohyun213/Qwen2.5-VL-3B-Instruct_blocks  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=joyhee/sft_multi_v2@train \
    data.val_files=joyhee/sft_multi_v2@test \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    worker.rollout.n=5 \
    trainer.experiment_name=qwen2_5_vl_3b-sft@blocks-rl@sft_multi-grpo \
    trainer.n_gpus_per_node=8 \