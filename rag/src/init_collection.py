import argparse
import sys

root_folder = ""
sys.path.append(root_folder)
import uuid

from rag.src.client import RAGClient
from shared.src.utils.io import load_jsonlines


class RAGInit:
    def __init__(self, embeddings_dir: str, raw_data_file_path: str) -> None:
        self.raw_data_file_path = raw_data_file_path
        self.client = RAGClient(embeddings_dir)

    def load_data(self):
        self.documents = load_jsonlines(self.raw_data_file_path)

    def reformat_data(self):
        for doc in self.documents:
            doc["meta_data"]["file_imports"] = "\n ".join(
                doc.get("meta_data").get("file_imports")
            )
        self.codes = [doc.get("code") for doc in self.documents]
        self.metadatas = [doc.get("meta_data") for doc in self.documents]
        self.ids = [str(uuid.uuid4()) for _ in range(len(self.documents))]

    def init_db(self, collection_name: str, overwrite: bool = False):
        self.client.add(
            collection_name=collection_name,
            codes=self.codes,
            metadatas=self.metadatas,
            ids=self.ids,
            overwrite=overwrite,
        )


# %%
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str)
    parser.add_argument("--collection_name", type=str)
    parser.add_argument("--raw_data_file_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    rag_init = RAGInit(
        embeddings_dir=args.embeddings_dir, raw_data_file_path=args.raw_data_file_path
    )
    rag_init.load_data()
    rag_init.reformat_data()
    rag_init.init_db(collection_name=args.collection_name, overwrite=True)
