from typing import List
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

class CustomFastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # fastembed returns a generator of numpy arrays, convert to list of lists
        embeddings = self.model.embed(texts)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.embed([text])
        return list(embeddings)[0].tolist()
