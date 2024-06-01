import chromadb
import pretty_errors
from chromadb.utils import embedding_functions


class RAGClient:
    def __init__(
        self,
        embeddings_dir: str,
        embedding_model: str = "hkunlp/instructor-base",
    ) -> None:
        self.client = chromadb.PersistentClient(path=embeddings_dir)
        self.instructor_ef = embedding_functions.InstructorEmbeddingFunction(
            model_name=embedding_model
        )

    def add(
        self,
        collection_name: str,
        codes: list[str],
        metadatas: list[dict],
        ids: list[str],
        overwrite: bool = False,
    ):
        try:
            self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.instructor_ef)  # type: ignore
            exists = True
        except Exception as e:
            print(e)
            exists = False

        if exists and not overwrite:
            raise ValueError(
                f"Collection {collection_name} already exists. Set overwrite=True to overwrite it."
            )

        if not exists or overwrite:
            self.collection.add(documents=codes, metadatas=metadatas, ids=ids)  # type: ignore

    def query(self, collection_name: str, query: str, meta_data, n: int = 10, **kwargs):
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.instructor_ef)  # type: ignore
        if meta_data is not {}:
            return self.collection.query(
                query_texts=query, where=meta_data, n_results=n, **kwargs
            )
        return self.collection.query(query_texts=query, n_results=n, **kwargs)
