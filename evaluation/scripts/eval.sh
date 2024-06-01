#!/usr/bin/bash
source evaluation/scripts/answer_creator.sh
source evaluation/scripts/model_based.sh


## ------------------------------------------- PARAMS -------------------------------------------------------
timestamp() {
  date '+%m-%d-%H:%M'
}
run_id=$((RANDOM))
echo "Run ID: $run_id"
root_folder=""
## Models

model_name_answer1="mistralai/Mistral-7B-Instruct-v0.2"
model_name_answer2="Mistral-7B-Instruct-v0.2-lbl-2x"
judge_model_name="gpt-4-turbo"

## Models Params
answer_creator_temperature=0.7
answer_creator_top_p=0.9
mbe_temperature=0.7
mbe_top_p=0.9
max_tokens=2500
gpu_memory_utilization=0.9

#Rag Parameter
rag_collection="spyder-embeddings"
rag_n=1
dataset_name="all"

## Data
# qa_data_path="$(root_folder)/ma_llm/evaluation/data/test_qa_data.jsonl"
qa_data_path="$(root_folder)/ma_llm/evaluation/data/final_eval_custom_meta.jsonl"

## Answers Path
answers1_path="$(root_folder)/ma_llm/evaluation/results/answers/$(timestamp)_${run_id}_model1.json"
answers2_path="$(root_folder)/ma_llm/evaluation/results/answers/$(timestamp)_${run_id}_model2.json"

## Sys Prompts
sys_prompt_creator_path="$(root_folder)/ma_llm/evaluation/prompts/answer_creator.txt"
sys_prompt_model_based_path="$(root_folder)/ma_llm/evaluation/prompts/model-based.txt"


## ------------------------------------- Execute ----------------------------------
# Set the default option
option=${1:-"evaluate"}

case $option in
    "evaluate")
        rag_flag=0
        meta_data_flag=0
        create_answers $model_name_answer1 $answers1_path
        echo "Answers1 Path: $answers1_path"
        ## Set Rag flag!
        rag_flag=1
        meta_data_flag=1
        create_answers $model_name_answer2 $answers2_path
        echo "Answers2 Path: $answers2_path"
        model_based
        ;;
    "create_answers")
        echo "Start Create Answers ..."
        create_answers
        ;;
    "model_based")
        echo "Start Model Based ..."
        answers1_path=$2
        answers2_path=$3
        model_based
        ;;
    *)
        echo "Invalid option. Please use 'evaluate', 'create_answers' or model_based as the first parameter."
        exit 1
        ;;
esac
