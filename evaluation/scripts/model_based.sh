#!/usr/bin/bash
model_based() {
    output_path="$(root_folder)/ma_llm/evaluation/results/model-based/$(timestamp)"
    echo "Start Evaluation ..."
    python -m evaluation.src.model_based.Evaluator \
    --run_id=${run_id} \
    --model_name=${judge_model_name} \
    --temperature=${mbe_temperature} \
    --top_p=${mbe_top_p} \
    --max_tokens=${max_tokens} \
    --output_filepath=${output_path} \
    --sys_prompt_path=${sys_prompt_model_based_path} \
    --qa_data_path=${qa_data_path} \
    --answers1_path=${answers1_path} \
    --answers2_path=${answers2_path} \
    --dataset_name=${dataset_name} \
    --wandb
}
