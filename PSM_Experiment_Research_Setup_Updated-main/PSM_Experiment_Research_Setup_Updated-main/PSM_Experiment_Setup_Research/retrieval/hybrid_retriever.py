import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from embedding.Embedding import EmbeddingModel

class HybridRetriever:
    def __init__(self, ef_construction=200, M=32, ef_search=50, embedding_model=None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = None
        self.documents = []
        self.bm25 = None
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search

    def build_index(self, documents):
        self.documents = documents

        # Sparse BM25
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Dense HNSW
        embeddings = self.embedding_model.encode_documents(documents)
        dim = embeddings.shape[1]

        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(embeddings)

    def retrieve(self, query, k=3, bm25_weight=0.5, dense_weight=0.5):
        # Guard: index must be built first
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Call build_index(documents) before retrieve().")

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        # Dense scores
        query_embedding = self.embedding_model.encode_query(query)
        distances, indices = self.index.search(query_embedding, k * 2)

        dense_scores = np.zeros(len(self.documents))
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                dense_scores[idx] = 1 / (1 + dist)

        dense_max = dense_scores.max()
        if dense_max > 0:
            dense_scores = dense_scores / dense_max

        # Combine
        hybrid_scores = (bm25_weight * bm25_scores) + (dense_weight * dense_scores)
        top_k_indices = np.argsort(hybrid_scores)[::-1][:k]
        return [self.documents[i] for i in top_k_indices]