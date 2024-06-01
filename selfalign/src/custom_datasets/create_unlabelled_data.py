# %%
import argparse
import json
import os
import re
from typing import List

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter


# %%
def load_documents(input_path: str) -> List[Document]:
    loader = GenericLoader.from_filesystem(
        input_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int) -> List[str]:
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=200
    )
    return python_splitter.split_documents(documents)  # type: ignore


def document_to_json(document: Document) -> str:
    code = None
    meta_data = None
    start_line, end_line = find_document_range(document)
    doc = document.to_json()

    if doc is not None:
        code = doc.get("kwargs", {}).get("page_content")
        meta_data = process_meta_data(
            doc.get("kwargs", {}).get("metadata"), code, start_line, end_line  # type: ignore
        )

    return json.dumps({"meta_data": meta_data, "code": code})


def process_meta_data(
    meta_data: dict, code: str, start_line: int, end_line: int
) -> dict:
    path: str = meta_data.get("source") or ""
    file_name = os.path.basename(path)
    ## TODO: Insert here the root_folder to delete unnecessary information
    # module = path.replace("root_folder", "")
    module = os.path.dirname(module)
    module = module.replace("/", ".")

    contains_class = file_contains_class(code)
    contains_function = file_contains_function(code)
    file_imports = get_file_imports(path)

    return {
        "file_name": file_name,
        "module": module,
        "contains_class": contains_class,
        "contains_function": contains_function,
        "file_imports": file_imports,
        "start_line": start_line,
        "end_line": end_line,
    }


def find_document_range(langchain_document: Document):
    file_path = langchain_document.metadata.get("source")
    with open(file_path, "r") as file:  # type: ignore
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


def save_to_jsonl(json_documents: List[str], output_file_path: str) -> None:
    with open(output_file_path, "w") as jsonl_file:
        for json_doc in json_documents:
            jsonl_file.write(json_doc + "\n")


# %%
def main(args: argparse.Namespace) -> None:
    root_folder = ""
    documents = load_documents(args.input_path)
    texts = split_documents(documents, args.chunk_size)
    json_documents = map(document_to_json, texts)  # type: ignore
    save_to_jsonl(json_documents, args.output_file_path)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create JSONL file.")
    parser.add_argument("--input_path", type=str, help="Path to the python repository")
    parser.add_argument(
        "--output_file_path", type=str, help="Path to the output JSONL file."
    )
    parser.add_argument("--chunk_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
    print("Done.")
