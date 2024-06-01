# %%
import json
import re
from pathlib import Path

import markdown
import yaml
from langchain.schema import Document


class AlternativeGenericLoader:
    def __init__(
        self,
        repository_path: str,
        glob: str,
        suffixes: list[str],
    ) -> None:
        self.repository_path = Path(repository_path)
        self.glob = glob
        self.suffixes = suffixes
        self.files = []

    def __len__(self):
        return len(self.content)

    def get_all_file_paths(self):
        self.files = list(self.repository_path.glob(self.glob))

    def filter_files_by_suffixes(self):
        self.files = [
            file
            for file in self.files
            if re.search(r"\.(" + "|".join(self.suffixes) + ")$", str(file))
        ]

    def load_files(self):
        self.content = []
        for file in self.files:
            content = self.read_file(file)
            self.content.append(content)

    def remove_empty_files(self):
        for file, content in zip(self.files, self.content):
            if not content:
                self.files.remove(file)
                self.content.remove(content)

    def read_file(self, file_path: Path):
        suffix = file_path.suffix[1:]
        if (
            suffix == "txt"
            or suffix == "ini"
            or suffix == "cfg"
            or suffix == "in"
            or suffix == "gitignore"
            or suffix == "gitattributes"
            or suffix == "coveragerc"
            or suffix == "git-blame-ignore-revs"
        ):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading file: {file_path}, {e}")
        elif suffix == "md":
            try:
                with open(file_path, "r") as file:
                    return markdown.markdown(file.read())
            except Exception as e:
                print(f"Error reading file: {file_path}, {e}")
        elif suffix == "jsonl":
            with open(file_path, "r") as file:
                return [json.loads(line) for line in file]
        elif suffix == "json":
            try:
                with open(file_path, "r") as file:
                    return json.load(file)
            except Exception as e:
                print(f"Error reading file: {file_path}, {e}")
        elif suffix == "yml" or "yaml":
            try:
                with open(file_path, "r") as file:
                    return yaml.dump(yaml.safe_load(file), sort_keys=False, indent=4)
            except Exception as e:
                print(f"Error reading file: {file_path}, {e}")

    def create_hf_document(self, document: str, file_path: Path):
        if document or document != "":
            return Document(
                page_content=str(document),
                metadata={
                    "source": str(file_path),
                    "language": str(file_path.suffix[1:]),
                },
            )

    def load(self):
        self.get_all_file_paths()
        self.filter_files_by_suffixes()
        self.load_files()
        self.remove_empty_files()
        return [
            self.create_hf_document(doc, file)
            for doc, file in zip(self.content, self.files)
        ]
