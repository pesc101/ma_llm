create_answers() {
  local model_name="$1"
  local output_path="$2"
  echo "Start Create Answers for Model: ${model_name} ..."
  python -m evaluation.src.model_based.AnswerCreator \
      --run_id=${run_id} \
      --model_name=${model_name} \
      --temperature=${answer_creator_temperature} \
      --top_p=${answer_creator_top_p} \
      --max_tokens=${max_tokens} \
      --output_filepath=${output_path} \
      --sys_prompt_path=${sys_prompt_creator_path} \
      --qa_data_path=${qa_data_path} \
      --gpu_memory_utilization=${gpu_memory_utilization} \
      --rag_flag=${rag_flag} \
      --rag_collection=${rag_collection} \
      --rag_n=${rag_n} \
      --meta_data_flag=${meta_data_flag} \
      --dataset_name=${dataset_name} 
      # --wandb 
  echo "Finished Create Answers for Model: ${model_name}"
}