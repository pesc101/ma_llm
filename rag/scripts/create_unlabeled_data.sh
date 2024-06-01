
root_folder=""
dataset="spyder"
input_path="${root_folder}/spyder"

output_file_path="${root_folder}/ma_llm/rag/data/${dataset}_unlabelled_dataset_all_meta_clean.jsonl"

python -m shared.src.UnlabeledDataSet \
 --repository_path ${input_path} \
 --output_file_path ${output_file_path} \
 --chunk_size 1500 \
 --chunk_overlap 200
