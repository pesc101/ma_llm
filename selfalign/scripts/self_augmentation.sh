#!/usr/bin/bash

self_augmentation() {
    ## Save Self-augmented data Step 1
    augmented_data_filepath_teacher="${root_folder}/ma_llm/${final_folder}/${dataset}_augmented_teacher.jsonl"

    ## Save Self-augmented data Step 2
    augmented_data_filepath_answer="${root_folder}/ma_llm/${final_folder}/${dataset}_augmented_answer.jsonl"

    echo "Start Self Augmentation (1/2) ..."
    python -m selfalign.src.SelfAugmentation \
    --run_id=${run_id} \
    --model_path=${model_path} \
    --model_name="mistral" \
    --unlabelled_data_filepath=${unlabelled_data_filepath} \
    --output_filepath=${augmented_data_filepath_teacher} \
    --tensor_parallel_size=${tensors} \
    --dataset_type "teacher" \
    --dataset_factor=${dataset_factor} \
    --top_p=${top_p_teacher} \
    --temperature=${temperature_teacher} \
    --gpu_memory_utilization=${gpu_memory_utilization} \
    --wandb

    echo "Start Self Augmentation (2/2) ..."
    python -m selfalign.src.SelfAugmentation \
    --run_id=${run_id} \
    --model_path=${model_path} \
    --model_name="mistral" \
    --unlabelled_data_filepath=${augmented_data_filepath_teacher} \
    --output_filepath=${augmented_data_filepath_answer} \
    --tensor_parallel_size=${tensors} \
    --dataset_type "answer-predictor" \
    --top_p=${top_p_answer} \
    --temperature=${temperature_answer} \
    --gpu_memory_utilization=${gpu_memory_utilization} \
    --wandb
    echo "Finished Self Augmentation"
}
