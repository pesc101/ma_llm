root_folder=""
dataset="spyder"
input_path="${root_folder}/spyder"



output_file_path="${root_folder}/ma_llm/selfalign/data/unlabelled/${dataset}_unlabelled_dataset.jsonl"

python -m selfalign.src.custom_datasets.create_unlabelled_data \
 --input_path ${input_path} \
 --output_file_path ${output_file_path} \
 --chunk_size 1500
