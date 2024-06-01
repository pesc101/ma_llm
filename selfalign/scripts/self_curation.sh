#!/usr/bin/bash

self_curation() {

    ## Dataset after Step 3
    augmented_data_filepath="${root_folder}/ma_llm/${final_folder}/${dataset}_augmented_answer.jsonl"

    ## Dataset after Step 5
    predicted_save_filepath="${root_folder}/ma_llm/${final_folder}/${dataset}_with_curation_score.jsonl"

    ## Dataset after Step 6
    curated_save_filepath="${root_folder}/ma_llm/${final_folder}/${dataset}_with_curation_final.jsonl"

    echo "(1/2) => Predict curation results ..."
    python -m selfalign.src.SelfAugmentation\
    --run_id=${run_id} \
    --model_path=${model_path} \
    --model_name="mistral" \
    --unlabelled_data_filepath=${augmented_data_filepath} \
    --output_filepath=${predicted_save_filepath} \
    --tensor_parallel_size=${tensors} \
    --dataset_type "curation" \
    --top_p=${top_p_curation} \
    --temperature=${temperature_curation} \
    --gpu_memory_utilization=${gpu_memory_utilization} \
    --wandb

    echo "(2/2) => Curate results ..."
    python -m selfalign.src.SelfCuration \
    --run_id=${run_id} \
    --data_filepath=${predicted_save_filepath} \
    --save_filepath=${curated_save_filepath} \
    --wandb

}
