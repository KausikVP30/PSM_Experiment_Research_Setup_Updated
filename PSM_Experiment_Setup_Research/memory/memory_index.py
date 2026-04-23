import faiss
import numpy as np
import os
from embedding.Embedding import EmbeddingModel
from memory.memory_store import MemoryStore

class MemoryIndex:
    def __init__(self, index_file='memory_index.faiss', memory_file='memory_store.json', embedding_model=None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.memory_store = MemoryStore(memory_file=memory_file)
        self.index_file = index_file
        self.index = None

        # Infer dimension from model instead of hardcoding
        dummy = self.embedding_model.encode_query("test")
        self.dimension = dummy.shape[1]

        self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.rebuild_index()

    def rebuild_index(self):
        embeddings = self.memory_store.get_all_embeddings()
        if len(embeddings) == 0:
            return

        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_file)

    def add_memory_embedding(self, embedding):
        embedding = np.array(embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        faiss.write_index(self.index, self.index_file)

    def search_memory(self, query, top_k=1):
        # If this instance was created before memories were added, its in-memory
        # FAISS index may be stale. Try reloading from disk once.
        if self.index is None:
            self.load_or_create_index()
        if self.index.ntotal == 0 and os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception:
                # Fall back to empty index behavior.
                pass
        if self.index.ntotal == 0:
            return None, 0

        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.index.search(query_embedding, top_k)
        memory_id = indices[0][0]
        similarity_score = float(similarities[0][0])

        return memory_id, similarity_score