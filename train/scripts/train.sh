#!/bin/bash
config_path="train/config/mistral.yml"

run_id=$((RANDOM))
echo "Run ID: $run_id"
root_folder="ma_llm/shared/models/"

python -m train.src.train \
    --run_id $run_id \
    --config_path $config_path \
    --wandb

python -m train.src.merge_peft \
    --config_path $config_path \
    --root_folder $root_folder \
    --push_to_hub
