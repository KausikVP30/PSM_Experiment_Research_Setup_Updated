from sentence_transformers import SentenceTransformer
import os
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name = "all-MiniLM-L6-v2", device=None):
        self.device = device or os.getenv("PSM_EMBEDDING_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy = True, show_progress_bar = False)
        return embeddings

    def encode_query(self, query):
        embeddings = self.model.encode([query], convert_to_numpy = True, show_progress_bar = False)
        return embeddings

    