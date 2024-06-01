
root_folder=""
selfalign_dataset_path="${root_folder}/ma_llm/selfalign/results/aug/2024-04-10/spyder_with_curation_final.jsonl"
output_dataset_name="spyder-ide-lbl-all-2x-low-all"

python -m selfalign.src.Reformater \
 --selfalign_dataset_path ${selfalign_dataset_path} \
 --output_dataset_name ${output_dataset_name}
