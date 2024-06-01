#!/usr/bin/bash

source selfalign/scripts/create_results_folder.sh
source selfalign/scripts/self_augmentation.sh
source selfalign/scripts/self_curation.sh
source selfalign/scripts/clear_folders.sh

## ------------------------------------------- PARAMS -------------------------------------------------------

run_id=$((RANDOM))
echo "Run ID: $run_id"
root_folder=""

## Model
model_path="mistralai/Mistral-7B-Instruct-v0.2"

## Number of GPUS
tensors=1
## Score
score=1
## Dataset Factor
dataset_factor=2

gpu_memory_utilization=0.6

## Top P and Temperature
top_p_teacher=0.3
temperature_teacher=0.3
top_p_answer=0.8
temperature_answer=0.7
top_p_curation=0.5
temperature_curation=0.5

## Dataset
dataset="spyder"
unlabelled_data_filepath="${root_folder}/ma_llm/selfalign/data/unlabelled/${dataset}_unlabelled_dataset_all.jsonl"


## ------------------------------------- Execute ----------------------------------
# Set the default option
option=${1:-"align"}

case $option in
    "align")
        echo "Self-Alignment..."
        create_folder
        self_augmentation
        self_curation
        clear_folders
        ;;
    "augment")
        echo "Executing only augmentation..."
        create_folder
        self_augmentation
        clear_folders
        ;;
    "curation")
        echo "Executing only curation..."
        final_folder=$2
        self_curation
        clear_folders
        ;;
    *)
        echo "Invalid option. Please use 'augment', 'align', or 'curation' as the first parameter."
        exit 1
        ;;
esac
