from __future__ import annotations

import numpy as np

from embedding.Embedding import EmbeddingModel
from memory.memory_index import MemoryIndex
from memory.memory_store import MemoryStore

class MemoryRetriever:
    def __init__(self, threshold=0.55, memory_file='memory_store.json', index_file='memory_index.faiss', embedding_model=None): #confidence score threshold kept to 0.55
        self.embedding_model = embedding_model or EmbeddingModel()
        self.memory_index = MemoryIndex(index_file=index_file, memory_file=memory_file, embedding_model=self.embedding_model)
        self.memory_store = MemoryStore(memory_file=memory_file)
        self.threshold = threshold

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def retrieve(self, query):
        memory_id, similarity = self.memory_index.search_memory(query)

        # If no memory exists
        if memory_id is None:
            return {
                "use_memory": False,
                "confidence": 0,
                "sim_query": None,
                "sim_answer": None,
                "sim_docs": None,
                "retrieved_docs": None,
                "answer": None,
                "memory_id": None
            }

        # Compute similarity breakdowns for analysis (even if below threshold).
        memory_entry = self.memory_store.get_memory_by_id(memory_id)
        query_embedding = self.embedding_model.encode_query(query)[0]

        sim_query = float(similarity)

        sim_answer = None
        answer_emb = memory_entry.get("answer_embedding")
        if answer_emb is not None:
            sim_answer = self._cosine(query_embedding, np.asarray(answer_emb, dtype=np.float32))

        sim_docs = None
        docs = memory_entry.get("retrieved_docs") or []
        if len(docs) > 0:
            doc_embs = self.embedding_model.encode_documents(docs)  # (n, d)
            # Mean cosine similarity between query and each doc.
            sims = [self._cosine(query_embedding, doc_embs[i]) for i in range(doc_embs.shape[0])]
            sim_docs = float(np.mean(sims)) if len(sims) > 0 else None

        # If confidence high → use memory
        if similarity >= self.threshold:
            return {
                "use_memory": True,
                "confidence": similarity,
                "sim_query": sim_query,
                "sim_answer": sim_answer,
                "sim_docs": sim_docs,
                "retrieved_docs": memory_entry["retrieved_docs"],
                "answer": memory_entry["answer"],
                "memory_id": memory_id
            }

        # If confidence low → do retrieval
        else:
            return {
                "use_memory": False,
                "confidence": similarity,
                "sim_query": sim_query,
                "sim_answer": sim_answer,
                "sim_docs": sim_docs,
                "retrieved_docs": None,
                "answer": None,
                "memory_id": memory_id
            }