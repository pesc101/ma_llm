
root_folder=""
input_path="${root_folder}/ma_llm/rag/data/spyder_unlabelled_dataset_all_meta_clean.jsonl"

python -m rag.src.init_collection \
    --raw_data_file_path $input_path \
    --embeddings_dir ${root_folder}/ma_llm/rag/embeddings \
    --collection_name "spyder-embedding-meta-clean"
