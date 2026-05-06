#!/usr/bin/env python3
"""
PSM TriviaQA Smoke Run - Fast version with synthetic fallback
"""

import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from router.router import Router


# Synthetic dataset for fast smoke testing
SYNTHETIC_QUESTIONS = [
    {
        "question": "What is a linked list?",
        "answer": "A linear data structure where elements are stored in nodes with pointers",
    },
    {
        "question": "What is binary search tree?",
        "answer": "A tree data structure that allows fast lookup, insertion, and deletion",
    },
    {
        "question": "What principle does a stack follow?",
        "answer": "Last In First Out (LIFO)",
    },
    {
        "question": "What principle does a queue follow?",
        "answer": "First In First Out (FIFO)",
    },
    {
        "question": "What is a hash table?",
        "answer": "A data structure that uses a hash function to map keys to values for fast retrieval",
    },
    {
        "question": "What are graph traversal algorithms?",
        "answer": "Algorithms like BFS and DFS for visiting all nodes in a graph",
    },
    {
        "question": "What is dynamic programming?",
        "answer": "An approach that breaks problems into overlapping subproblems",
    },
    {
        "question": "What is bubble sort?",
        "answer": "A sorting algorithm that repeatedly swaps adjacent elements if they are in wrong order",
    },
    {
        "question": "Explain linked lists",
        "answer": "Linear structures with nodes containing data and pointers",
    },
    {
        "question": "What are doubly linked lists?",
        "answer": "Lists with pointers to both next and previous nodes",
    },
    {
        "question": "How do stacks work?",
        "answer": "Using LIFO principle for storing and retrieving data",
    },
    {
        "question": "How do queues differ from stacks?",
        "answer": "Queues follow FIFO while stacks follow LIFO",
    },
    {
        "question": "What are search trees?",
        "answer": "Trees that enable efficient searching of data",
    },
    {
        "question": "What is recursion?",
        "answer": "A function calling itself to solve problems",
    },
    {
        "question": "What is memoization?",
        "answer": "A technique to cache results for faster computation",
    },
    {
        "question": "What is NLP?",
        "answer": "Natural Language Processing deals with text and language",
    },
    {
        "question": "What is machine learning?",
        "answer": "A subset of AI that learns from data",
    },
    {
        "question": "What are neural networks?",
        "answer": "Computing systems inspired by biological neural networks",
    },
    {
        "question": "What is deep learning?",
        "answer": "Machine learning using neural networks with multiple layers",
    },
    {
        "question": "What are transformers?",
        "answer": "Models used in modern NLP based on attention mechanisms",
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Learning using rewards and punishments",
    },
    {
        "question": "What is supervised learning?",
        "answer": "Learning from labeled training data",
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Learning from unlabeled data to discover patterns",
    },
    {
        "question": "What is clustering?",
        "answer": "Grouping similar data points together",
    },
    {
        "question": "What is classification?",
        "answer": "Assigning data points to predefined categories",
    },
    {
        "question": "What is regression?",
        "answer": "Predicting continuous values based on input features",
    },
    {
        "question": "What are decision trees?",
        "answer": "Tree-like models for classification and regression",
    },
    {
        "question": "What is random forest?",
        "answer": "An ensemble method using multiple decision trees",
    },
    {
        "question": "What is KNN?",
        "answer": "K-Nearest Neighbors algorithm for classification",
    },
    {
        "question": "What is SVM?",
        "answer": "Support Vector Machine for classification",
    },
]


class PSMSmokeRunFast:
    """Fast smoke run with synthetic questions."""
    
    def __init__(self, smoke_size: int = 20, run_id: str = "psm_smoke_fast"):
        self.smoke_size = min(smoke_size, len(SYNTHETIC_QUESTIONS))
        self.run_id = run_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.output_dir = Path("smoke_results") / self.run_id / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.queries_manifest_path = self.output_dir / "planned_queries.csv"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.metrics_path = self.output_dir / "summary_metrics.json"
        self.run_config_path = self.output_dir / "run_config.json"
        self.log_file_path = self.output_dir / "experiment_log_v2.csv"
        
        self.router = None
        self.test_data = SYNTHETIC_QUESTIONS[:self.smoke_size]
        self.predictions = []
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for EM/F1."""
        import re
        
        def remove_articles(s):
            return re.sub(r'\b(a|an|the)\b', ' ', s)
        
        def white_space_fix(s):
            return ' '.join(s.split())
        
        def remove_punc(s):
            return re.sub(r'[^\w\s]', ' ', s)
        
        def lower(s):
            return s.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
    def metric_exact_match(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match."""
        norm_pred = self.normalize_answer(prediction)
        norm_gt = self.normalize_answer(ground_truth)
        return 1.0 if norm_pred == norm_gt else 0.0
    
    def metric_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1."""
        from collections import Counter
        
        norm_pred = self.normalize_answer(prediction)
        norm_gt = self.normalize_answer(ground_truth)
        
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def initialize_router(self):
        """Initialize PSM router."""
        print("\n[INFO] Initializing PSM router with synthetic corpus...")
        
        # Create a simple corpus from the test data
        corpus = [
            "A linked list is a linear data structure where elements are stored in nodes.",
            "Each node in a linked list contains data and a pointer to the next node.",
            "A doubly linked list has pointers to both the next and previous nodes.",
            "Binary search trees allow fast lookup, insertion, and deletion of elements.",
            "Stacks follow Last In First Out (LIFO) principle for data storage.",
            "Queues follow First In First Out (FIFO) principle for data storage.",
            "Hash tables use a hash function to map keys to values for fast retrieval.",
            "Graph traversal algorithms include BFS and DFS for visiting all nodes.",
            "Dynamic programming breaks problems into overlapping subproblems.",
            "Bubble sort repeatedly swaps adjacent elements if they are in wrong order.",
            "Machine learning is a subset of AI that learns from data.",
            "Neural networks are computing systems inspired by biological networks.",
            "Natural language processing deals with text and language.",
            "Reinforcement learning learns using rewards and punishments.",
            "Trees are hierarchical data structures with nodes and edges.",
        ]
        
        self.router = Router(
            documents=corpus,
            threshold=0.55,
            log_file=str(self.log_file_path),
        )
        self.router.hybrid_retriever.build_index(corpus)
        print("[INFO] ✓ Router initialized.")
    
    def run_evaluation(self):
        """Run smoke evaluation with persistent memory."""
        print("\n" + "="*70)
        print("PSM SMOKE RUN (Persistent Memory Mode)")
        print("="*70)
        print(f"Questions: {len(self.test_data)}")
        print(f"Memory mode: PERSISTENT (learning across questions)")
        print("="*70 + "\n")
        
        results = []
        memory_count = 0
        retrieval_count = 0
        
        for idx, test_q in enumerate(tqdm(self.test_data, desc="Evaluating")):
            question = test_q["question"]
            correct_answer = test_q["answer"]
            query_id = f"synthetic_{idx:03d}"
            
            try:
                # Process through PSM router
                generated_answer, confidence = self.router.process_query(question)
                if isinstance(generated_answer, str) and generated_answer.lstrip().startswith("[LLM_ERROR]"):
                    raise RuntimeError(f"LLM backend error: {generated_answer}")
                
                # Get source
                last_log = self.router.logger.get_metrics_for_query(question)
                source = last_log["latest"]["source"] if last_log else "unknown"
                
                if source == "memory":
                    memory_count += 1
                elif source == "retrieval":
                    retrieval_count += 1
                
                # Compute metrics
                em = self.metric_exact_match(generated_answer, correct_answer)
                f1 = self.metric_f1(generated_answer, correct_answer)
                
                result = {
                    "query_id": query_id,
                    "question": question,
                    "generated_answer": generated_answer,
                    "correct_answer": correct_answer,
                    "confidence": float(confidence),
                    "source": source,
                    "em": float(em),
                    "f1": float(f1),
                    "memory_size": self.router.memory_store.size(),
                    "retrieval_count": self.router.retrieval_count,
                    "memory_count": self.router.memory_count,
                }
                
                results.append(result)
                self.predictions.append(result)
                
            except Exception as e:
                print(f"\n[ERROR] Query {query_id} failed: {str(e)[:100]}")
                results.append({
                    "query_id": query_id,
                    "question": question,
                    "error": str(e)[:100]
                })
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        return results, memory_count, retrieval_count
    
    def save_outputs(self, results: List[Dict], memory_count: int, retrieval_count: int):
        """Save all evaluation outputs."""
        
        # Save queries manifest
        print(f"\n[INFO] Saving queries manifest...")
        with open(self.queries_manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "question", "correct_answer"])
            writer.writeheader()
            for idx, q in enumerate(self.test_data):
                writer.writerow({
                    "query_id": f"synthetic_{idx:03d}",
                    "question": q["question"],
                    "correct_answer": q["answer"],
                })
        
        # Save predictions
        print(f"[INFO] Saving predictions...")
        with open(self.predictions_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Compute and save metrics
        valid_results = [r for r in results if "em" in r]
        
        if valid_results:
            em_scores = [r["em"] for r in valid_results]
            f1_scores = [r["f1"] for r in valid_results]
            confidences = [r["confidence"] for r in valid_results]
            
            memory_results = [r for r in valid_results if r["source"] == "memory"]
            retrieval_results = [r for r in valid_results if r["source"] == "retrieval"]
            
            metrics = {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "dataset": "Synthetic Questions",
                "smoke_size": len(self.test_data),
                "mode": "persistent_memory",
                "model": "llama3 (8B)",
                "threshold": 0.55,
                "total_queries": len(valid_results),
                "avg_em": float(np.mean(em_scores)),
                "avg_f1": float(np.mean(f1_scores)),
                "avg_confidence": float(np.mean(confidences)),
                "memory_path_count": memory_count,
                "retrieval_path_count": retrieval_count,
                "memory_path_rate": float(memory_count / len(valid_results)) if valid_results else 0.0,
                "retrieval_path_rate": float(retrieval_count / len(valid_results)) if valid_results else 0.0,
                "memory_path": {
                    "count": len(memory_results),
                    "avg_em": float(np.mean([r["em"] for r in memory_results])) if memory_results else 0.0,
                    "avg_f1": float(np.mean([r["f1"] for r in memory_results])) if memory_results else 0.0,
                    "avg_confidence": float(np.mean([r["confidence"] for r in memory_results])) if memory_results else 0.0,
                },
                "retrieval_path": {
                    "count": len(retrieval_results),
                    "avg_em": float(np.mean([r["em"] for r in retrieval_results])) if retrieval_results else 0.0,
                    "avg_f1": float(np.mean([r["f1"] for r in retrieval_results])) if retrieval_results else 0.0,
                    "avg_confidence": float(np.mean([r["confidence"] for r in retrieval_results])) if retrieval_results else 0.0,
                },
                "final_memory_size": self.router.memory_store.size(),
            }
            
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save run config
            config = {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "dataset": "Synthetic (fast smoke test)",
                "smoke_size": len(self.test_data),
                "mode": "persistent_memory",
                "memory_learning": True,
                "threshold": 0.55,
                "model": "llama3 (Ollama local, 8B)",
                "embedding_model": "all-MiniLM-L6-v2",
            }
            
            with open(self.run_config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Print summary
            self._print_summary(metrics)
            
            return metrics
        
        return None
    
    def _print_summary(self, metrics: Dict):
        """Print summary."""
        print("\n" + "="*70)
        print("SMOKE RUN SUMMARY")
        print("="*70)
        print(f"Run ID:             {metrics['run_id']}")
        print(f"Total Questions:    {metrics['total_queries']}")
        print(f"Avg EM:             {metrics['avg_em']:.3f}")
        print(f"Avg F1:             {metrics['avg_f1']:.3f}")
        print(f"Avg Confidence:     {metrics['avg_confidence']:.3f}")
        print(f"\nPath Statistics:")
        print(f"  Memory Rate:      {metrics['memory_path_rate']:.1%}")
        print(f"  Retrieval Rate:   {metrics['retrieval_path_rate']:.1%}")
        print(f"\nMemory Path Performance:")
        print(f"  Count:            {metrics['memory_path']['count']}")
        print(f"  EM:               {metrics['memory_path']['avg_em']:.3f}")
        print(f"  F1:               {metrics['memory_path']['avg_f1']:.3f}")
        print(f"  Confidence:       {metrics['memory_path']['avg_confidence']:.3f}")
        print(f"\nRetrieval Path Performance:")
        print(f"  Count:            {metrics['retrieval_path']['count']}")
        print(f"  EM:               {metrics['retrieval_path']['avg_em']:.3f}")
        print(f"  F1:               {metrics['retrieval_path']['avg_f1']:.3f}")
        print(f"  Confidence:       {metrics['retrieval_path']['avg_confidence']:.3f}")
        print(f"\nFinal Memory Size:  {metrics['final_memory_size']} entries")
        print(f"Output Directory:   {self.output_dir}")
        print("="*70 + "\n")
    
    def run(self):
        """Execute."""
        self.initialize_router()
        results, memory_count, retrieval_count = self.run_evaluation()
        metrics = self.save_outputs(results, memory_count, retrieval_count)
        return metrics


if __name__ == "__main__":
    evaluator = PSMSmokeRunFast(smoke_size=20, run_id="psm_smoke_persistent_final")
    evaluator.run()
