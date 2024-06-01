# %%
import argparse
import json
import os
import re
import sys

import tqdm
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

root_folder = ""
sys.path.append(root_folder)
from shared.src.AlternativeGenericLoader import AlternativeGenericLoader


class PythonDataset:
    def __init__(
        self,
        repository_path: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        self.repository_path = repository_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __len__(self):
        return len(self.documents)

    def load_repository(self):
        python_loader = GenericLoader.from_filesystem(
            self.repository_path,
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            show_progress=True,
        )
        self.documents = python_loader.load()

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.documents = splitter.split_documents(self.documents)

    def create_documents(self):
        print("Creating dataset...")
        temp_documents = []
        for document in tqdm.tqdm(self.documents):
            code = None
            meta_data = None
            start_line, end_line = get_document_range(document)
            doc = document.to_json()

            if doc is not None:
                meta_data = self.process_meta_data(doc, start_line, end_line)
                code = doc.get("kwargs", {}).get("page_content")
                temp_dict = {"meta_data": meta_data, "code": code}
                temp_documents.append(json.dumps(temp_dict))
        self.documents = temp_documents

    def save_to_jsonl(self, output_file_path: str) -> None:
        with open(output_file_path, "w") as jsonl_file:
            for json_doc in self.documents:
                jsonl_file.write(json_doc + "\n")

    def process_meta_data(self, doc_json, start_line: int, end_line: int) -> dict:
        code = doc_json.get("kwargs", {}).get("page_content")
        meta_data = doc_json.get("kwargs", {}).get("metadata")
        path: str = meta_data.get("source") or ""

        file_name = os.path.basename(path)
        module = os.path.dirname(path)
        module = module.replace("/", ".")

        file_imports = get_file_imports(path)
        contains_class = file_contains_class(code)
        contains_function = file_contains_function(code)

        return {
            "file_name": file_name,
            "module": module,
            "contains_class": contains_class,
            "contains_function": contains_function,
            "file_imports": file_imports,
            "start_line": start_line,
            "end_line": end_line,
        }


class ConfigDataSet:
    def __init__(
        self,
        repository_path: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        self.repository_path = repository_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.md_sep = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]

    def load_repository(self):
        custom_suffixes = ["md", "yml", "yaml", "json", "jsonl", "txt"]
        loader = AlternativeGenericLoader(
            self.repository_path, glob="**/*", suffixes=custom_suffixes
        )
        self.documents.extend(loader.load())

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(separators=self.md_sep)
        temp_documents = []
        md_documents = []
        for document in self.documents:
            if not document.metadata.get("source").endswith(".md"):
                temp_documents.append(document)
            else:
                md_documents.append(document)
        self.documents = temp_documents + splitter.split_documents(md_documents)

    def create_documents(self):
        print("Creating dataset...")
        temp_documents = []
        for document in tqdm.tqdm(self.documents):
            code = None
            meta_data = None
            doc = document.to_json()

            if doc is not None:
                meta_data = self.process_meta_data(doc)
                code = doc.get("kwargs", {}).get("page_content")
                temp_dict = {"meta_data": meta_data, "code": code}
                temp_documents.append(json.dumps(temp_dict))
        self.documents = temp_documents

    def save_to_jsonl(self, output_file_path: str) -> None:
        with open(output_file_path, "w") as jsonl_file:
            for json_doc in self.documents:
                jsonl_file.write(json_doc + "\n")

    def process_meta_data(self, doc_json) -> dict:
        meta_data = doc_json.get("kwargs", {}).get("metadata")
        path: str = meta_data.get("source") or ""

        file_name = os.path.basename(path)
        module = os.path.dirname(path)
        module = module.replace("/", ".")

        return {
            "file_name": file_name,
            "module": module,
            "contains_class": False,
            "contains_function": False,
            "file_imports": [],
            "start_line": 0,
            "end_line": 0,
        }


# %%
def get_document_range(document: Document):
    file_path = document.metadata.get("source")
    with open(file_path, "r") as file:  # type: ignore
        file_content = file.readlines()

    start_line = 1
    end_line = len(file_content)
    found_start = False

    document_lines = document.page_content.split("\n")
    if len(document_lines) <= 1:
        return -1, -1

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

    end_line = min(start_line + len(document_lines) - 1, len(file_content))
    return start_line, end_line


def file_contains_class(code: str) -> bool:
    return re.search(r"class\s+\w+\s*:", code) is not None


def file_contains_function(code: str) -> bool:
    return re.search(r"def\s+\w+\(.*\)\s*:", code) is not None


def get_file_imports(path: str) -> list[str]:
    with open(path, "r") as file:
        code = file.readlines()
        import_from_lines = []
        for line in code:
            line = line.strip()
            if (
                line.startswith("#")
                or line.startswith("print")
                or line.startswith('"""')
            ):
                continue
            if "import " in line:
                import_from_lines.append(line)
        return import_from_lines


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create JSONL file.")
    parser.add_argument(
        "--repository_path", type=str, help="Path to the python repository"
    )
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--chunk_overlap", type=int)
    parser.add_argument(
        "--output_file_path", type=str, help="Path to the output JSONL file."
    )
    args = parser.parse_args()

    py_dataset = PythonDataset(
        args.repository_path,
        args.chunk_size,
        args.chunk_overlap,
    )
    config_dataset = ConfigDataSet(
        args.repository_path,
        args.chunk_size,
        args.chunk_overlap,
    )
    datasets = [py_dataset, config_dataset]
    for dataset in datasets:
        dataset.load_repository()
        dataset.split_documents()
        dataset.create_documents()

    py_dataset.documents.extend(config_dataset.documents)
    py_dataset.save_to_jsonl(args.output_file_path)
    print("Done.")
