from memory.memory_retriever import MemoryRetriever
from retrieval.hybrid_retriever import HybridRetriever
from prompt_template import build_prompt       # was missing
from llm.llm_interface import get_llm              # selects Ollama or Kaggle backend
import time
from logs.logger import Logger
from embedding.Embedding import EmbeddingModel


class Router:
    def __init__(
        self,
        documents,
        threshold=0.55,
        memory_file='memory_store.json',
        index_file='memory_index.faiss',
        log_file='logs/experiment_log_v2.csv',
    ):
        self.embedding_model = EmbeddingModel()
        self.memory_retriever = MemoryRetriever(
            threshold=threshold,
            memory_file=memory_file,
            index_file=index_file,
            embedding_model=self.embedding_model,
        )
        self.hybrid_retriever = HybridRetriever(embedding_model=self.embedding_model)
        self.hybrid_retriever.build_index(documents)   # was never called
        # Use the same store/index instances as the MemoryRetriever
        # so new memories immediately affect confidence.
        self.memory_store = self.memory_retriever.memory_store
        self.memory_index = self.memory_retriever.memory_index
        self.llm = get_llm()                          # selects backend via PSM_LLM_BACKEND env var
        self.logger = Logger(log_file=log_file)
        self.retrieval_count = 0                     # was missing
        self.memory_count = 0                         # was missing

    def route(self, query):
        memory_result = self.memory_retriever.retrieve(query)

        if memory_result["use_memory"]:
            print("Using Memory...")
            return {
                "source": "memory",
                "docs": memory_result["retrieved_docs"],
                "answer": memory_result["answer"],
                "confidence": memory_result["confidence"]
            }
        else:
            print("Using Hybrid Retrieval...")
            docs = self.hybrid_retriever.retrieve(query)
            return {
                "source": "retrieval",
                "docs": docs,
                "answer": None,
                "confidence": memory_result["confidence"]
            }

    def store_memory(self, query, retrieved_docs, answer):
        query_embedding = self.embedding_model.encode_query(query)
        answer_embedding = self.embedding_model.encode_query(answer)

        self.memory_store.add_memory(
            query,
            query_embedding[0],
            retrieved_docs,
            answer,
            answer_embedding[0]
        )
        self.memory_index.add_memory_embedding(query_embedding[0])

    def process_query(self, query):
        query = (query or "").strip()
        if not query or not any(ch.isalnum() for ch in query):
            raise ValueError("Query must contain meaningful text.")

        start_time = time.time()

        memory_result = self.memory_retriever.retrieve(query)
        confidence = memory_result["confidence"]
        memory_id = memory_result.get("memory_id")

        if memory_result["use_memory"]:
            print("Using Memory")
            docs = memory_result["retrieved_docs"]
            past_answer = memory_result["answer"]
            source = "memory"
            self.memory_count += 1
        else:
            print("Using Hybrid Retrieval")
            docs = self.hybrid_retriever.retrieve(query)
            past_answer = None
            source = "retrieval"                       # was missing
            self.retrieval_count += 1

        if source == "memory" and past_answer:
            answer = past_answer
        else:
            prompt = build_prompt(query, docs, past_answer)          # was missing
            answer = self.llm.generate(prompt)                       # was missing
        # Store memory if retrieval used
        if source == "retrieval":                              # was missing
            self.store_memory(query, docs, answer)

        end_time = time.time()
        latency = end_time - start_time

        sim_q = memory_result.get("sim_query")
        sim_a = memory_result.get("sim_answer")
        sim_d = memory_result.get("sim_docs")

        # Log data
        self.logger.log(
            query=query,
            confidence=confidence,
            memory_id=memory_id,
            sim_q=sim_q,
            sim_a=sim_a,
            sim_d=sim_d,
            source=source,
            latency=latency,
            memory_size=self.memory_store.size(),              # was missing
            retrieval_count=self.retrieval_count,               # was missing
            memory_count=self.memory_count,
        )

        # Performance summary (print)
        def _fmt(x):
            return "NA" if x is None else f"{float(x):.3f}"

        print(
            "\n--- run metrics ---\n"
            f"source         : {source}\n"
            f"confidence     : {confidence:.3f}\n"
            f"memory_id      : {memory_id}\n"
            f"sim_query      : {_fmt(sim_q)}\n"
            f"sim_answer     : {_fmt(sim_a)}\n"
            f"sim_docs       : {_fmt(sim_d)}\n"
            f"latency_s      : {latency:.3f}\n"
            f"memory_size    : {self.memory_store.size()}\n"
            f"retrieval_cnt  : {self.retrieval_count}\n"
            f"memory_cnt     : {self.memory_count}\n"
        )

        return answer, confidence