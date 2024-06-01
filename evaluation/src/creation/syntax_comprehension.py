# %%
import argparse
import os
from datetime import datetime

import pandas as pd
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_and_split_documents(
    repository_path, glob_pattern, suffixes, parser_threshold, chunk_size, chunk_overlap
):
    loader = GenericLoader.from_filesystem(
        repository_path,
        glob=glob_pattern,
        suffixes=suffixes,
        parser=LanguageParser(
            language=Language.PYTHON, parser_threshold=parser_threshold
        ),
    )
    documents = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = python_splitter.split_documents(documents)

    return texts


# %%


def find_document_range(langchain_document):
    file_path = langchain_document.metadata.get("source")
    with open(file_path, "r") as file:
        file_content = file.readlines()

    document_lines = langchain_document.page_content.split("\n")
    if len(document_lines) <= 1:
        return -1, -1

    start_line = 1
    end_line = len(file_content)

    found_start = False

    for i, line in enumerate(file_content):
        if (
            i < len(file_content) - 1
            and document_lines[0] in line
            and file_content[i + 1].strip() == document_lines[1].strip()
        ):
            # Found the starting point
            start_line = i + 1
            found_start = True
            break

    if not found_start:
        return -1, -1

    # Find the end line based on the length of the document
    end_line = min(start_line + len(document_lines) - 1, len(file_content))

    return start_line, end_line


def create_data_frame(document):
    start_line, end_line = find_document_range(document)
    return pd.DataFrame(
        {
            "file_path": document.metadata.get("source"),
            "start_line": start_line,
            "end_line": end_line,
            "code": document.page_content,
        },
        index=[0],
    )


def save_dfs_to_csv(dfs):
    root_folder = ""
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["start_line"] != -1) & (df["end_line"] != -1)]
    file_path = f"{root_folder}/ma_llm/evaluation/results/code_images_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(file_path, index=False)


def create_folder_from_file_name(root_folder, file_name):
    folder_name = file_name.split("/")[-1].split(".")[0]
    if not os.path.exists(root_folder + folder_name):
        os.makedirs(root_folder + folder_name)
    return folder_name


# %%
def main():
    root_folder = ""
    parser = argparse.ArgumentParser(
        description="Process code files and save code images."
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        default=f"{root_folder}/ma_llm/evaluation/results/spyder/",
        help="Root folder to save the code images",
    )
    parser.add_argument(
        "--repository-path",
        type=str,
        default=f"{root_folder}/spyder",
        help="Path to the code repository",
    )
    parser.add_argument(
        "--parser-threshold", type=int, default=500, help="Parser threshold"
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")

    args = parser.parse_args()

    root_folder = args.root_folder
    repository_path = args.repository_path
    parser_threshold = args.parser_threshold
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    texts = load_and_split_documents(
        repository_path, "**/*", [".py"], parser_threshold, chunk_size, chunk_overlap
    )

    dfs = []
    for document in tqdm(texts):
        df = create_data_frame(document)
        dfs.append(df)

    save_dfs_to_csv(dfs)


if __name__ == "__main__":
    main()
