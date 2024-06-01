# %%
import argparse
import ast
import glob
import os
from functools import lru_cache

import pandas as pd
from tqdm import tqdm


# %%
class FileAnalyzer:
    def __init__(self, file_path, project_directory, artifact_type=True):
        self.file_path = file_path
        self.project_directory = project_directory
        self.dependencies = []
        self.artifact_type = artifact_type

    def analyze(self):
        code = ""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                code = file.read()
        except IOError as e:
            print(f"Unable to open file {self.file_path}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.process_node(node, "direct")
            elif isinstance(node, ast.ImportFrom):
                self.process_node(node, "from")

        return self.dependencies

    def process_node(self, node, import_type):
        library_name = (
            node.names[0].name if isinstance(node, ast.Import) else node.module
        )
        library_name = "" if library_name is None else library_name
        import_category = (
            "file_import" if self.is_file_import(library_name) else "library_import"
        )
        self.process_imports(node, f"{import_category}_{import_type}", library_name)

    @lru_cache(maxsize=10000)
    def process_imports(self, node, category, library_name):
        for alias in node.names:
            imported_file_name = self.get_imported_file_name(library_name)
            artifact = alias.name
            artifact_type = self.get_imported_artifact_type(
                artifact, imported_file_name
            )
            self.dependencies.append(
                [
                    category,
                    self.file_path,
                    library_name,
                    imported_file_name,
                    artifact,
                    artifact_type,
                ]
            )

    def is_file_import(self, module_name):
        if not module_name:
            return False
        if module_name.startswith("."):
            module_path = module_name.replace(".", os.path.sep) + ".py"
            full_path = os.path.abspath(
                os.path.join(os.path.dirname(self.file_path), module_path)
            )
            return os.path.exists(full_path)

        search_pattern = os.path.join(
            self.project_directory, "**", module_name.replace(".", os.path.sep) + ".py"
        )
        matching_files = glob.glob(search_pattern, recursive=True)

        if matching_files:
            return True
        return False

    def get_imported_file_name(self, library_name):
        search_pattern = os.path.join(
            self.project_directory, "**", library_name.replace(".", os.path.sep) + ".py"
        )
        matching_files = glob.glob(search_pattern, recursive=True)
        return matching_files[0] if matching_files else None

    def get_imported_artifact_type(self, artifact_name, imported_file_name=""):
        code = ""

        if not self.artifact_type:
            return "unknown"

        if artifact_name == "*" or imported_file_name is None:
            return "unknown"

        try:
            with open(imported_file_name, "r", encoding="utf-8") as file:
                code = file.read()
        except UnicodeDecodeError as e:
            print(f"Unable to read file {imported_file_name}: {e}")

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == artifact_name:
                return "function"
            elif isinstance(node, ast.ClassDef) and node.name == artifact_name:
                return "class"
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == artifact_name:
                        return "variable"

        return "unknown"


class DirectoryAnalyzer:
    def __init__(self, directory, exclude_folders=None, artifact_type=True):
        self.directory = directory
        self.exclude_folders = exclude_folders
        self.artifact_type = artifact_type

    def analyze(self):
        all_dependencies = []
        total_files = sum(len(files) for _, _, files in os.walk(self.directory))
        for root, dirs, files in tqdm(os.walk(self.directory), total=total_files):
            # Exclude specified folders
            if self.exclude_folders:
                dirs[:] = [d for d in dirs if d not in self.exclude_folders]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_analyzer = FileAnalyzer(
                        file_path, self.directory, self.artifact_type
                    )
                    dependencies = file_analyzer.analyze()
                    all_dependencies.extend(dependencies)
        return all_dependencies


class DataFrameCreator:
    def __init__(self, dependencies, directory_path):
        self.dependencies = dependencies
        self.directory_path = directory_path
        self.df = pd.DataFrame()

    def create(self):
        columns = [
            "category",
            "target_file",
            "library_name",
            "file_name",
            "artifact",
            "artifact_type",
        ]
        df = pd.DataFrame(self.dependencies, columns=columns)
        df["target_file"] = df.apply(
            lambda x: os.path.relpath(x["target_file"], self.directory_path)
            if x["target_file"]
            else None,
            axis=1,
        )
        df["file_name"] = df.apply(
            lambda x: os.path.relpath(x["file_name"], self.directory_path)
            if x["file_name"]
            else None,
            axis=1,
        )
        df.loc[df["category"] == "file_import_direct", "artifact_type"] = "file"
        self.df = df

    def export_to_excel(self, file_name):
        self.df.to_excel(file_name, index=False)


def main(args):
    directory_analyzer = DirectoryAnalyzer(
        args.directory_path, args.exclude_folders, args.artifact_type
    )
    dependencies_list = directory_analyzer.analyze()
    data_frame_creator = DataFrameCreator(dependencies_list, args.directory_path)
    data_frame_creator.create()
    data_frame_creator.export_to_excel(args.output_file_name)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a directory structure.")
    parser.add_argument(
        "--directory_path", help="The path of the directory to analyze."
    )
    parser.add_argument(
        "--exclude_folders",
        nargs="*",
        default=[".venv"],
        help="A list of folders to exclude.",
    )
    parser.add_argument(
        "--artifact_type",
        action="store_true",
        help="Whether to include the artifact type in the output.",
    )
    parser.add_argument(
        "--output_file_name",
        default="dependencies.xlsx",
        help="The name of the output file.",
    )
    args = parser.parse_args()

    main(args)
