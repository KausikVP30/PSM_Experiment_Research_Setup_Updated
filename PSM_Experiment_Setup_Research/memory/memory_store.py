import json
import os
from datetime import datetime

class MemoryStore:
    def __init__(self, memory_file='memory_store.json'):
        self.memory_file = memory_file
        self.memory = []
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = []

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def add_memory(self, query, query_embedding, retrieved_docs, answer, answer_embedding):
        memory_entry = {
            "id": len(self.memory),
            "query": query,
            "query_embedding": query_embedding.tolist(),
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "answer_embedding": answer_embedding.tolist(),
            "timestamp": str(datetime.now())
        }

        self.memory.append(memory_entry)
        self.save_memory()

    def get_memory_by_id(self, memory_id):
        return self.memory[memory_id]

    def get_all_embeddings(self):
        return [entry["query_embedding"] for entry in self.memory]

    def size(self):
        return len(self.memory)