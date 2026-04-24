#!/usr/bin/env python3
"""
Predictive Semantic Memory (PSM) - TriviaQA Smoke Run Evaluator
Persistent memory mode (learning across questions)
For experimental evaluation only.
"""

import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from router.router import Router
from logs.logger import Logger


class TriviaQAEvaluator:
    """Smoke run evaluator for PSM on TriviaQA Wikipedia dev."""
    
    def __init__(self, smoke_size: int = 50, run_id: str = "psm_triviaqa_smoke_01"):
        self.smoke_size = smoke_size
        self.run_id = run_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Output directories
        self.output_dir = Path("triviaqa_results") / self.run_id / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log paths
        self.queries_manifest_path = self.output_dir / "planned_queries.csv"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.metrics_path = self.output_dir / "summary_metrics.json"
        self.run_config_path = self.output_dir / "run_config.json"
        self.log_file_path = self.output_dir / "experiment_log_v2.csv"
        
        self.router = None
        self.dataset = None
        self.queries = []
        self.predictions = []
        
    def load_triviaqa_dataset(self):
        """Load TriviaQA Wikipedia dev split."""
        print("\n[INFO] Loading TriviaQA Wikipedia dev split...")
        try:
            self.dataset = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
            print(f"[INFO] ✓ Loaded {len(self.dataset)} questions from TriviaQA Wikipedia dev.")
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise
    
    def prepare_smoke_subset(self):
        """Sample first N questions for smoke test."""
        print(f"\n[INFO] Preparing smoke subset (size={self.smoke_size})...")
        
        # Take first N samples
        smoke_data = self.dataset.select(range(min(self.smoke_size, len(self.dataset))))
        
        for idx, sample in enumerate(smoke_data):
            query_id = f"triviaqa_wiki_{idx:04d}"
            question = sample["question"]
            
            # Store expected answers (aliases)
            answers = sample.get("answer", {})
            if isinstance(answers, dict):
                answer_aliases = answers.get("aliases", [])
                answer_value = answers.get("value", "")
            else:
                answer_aliases = [answers]
                answer_value = answers
            
            self.queries.append({
                "query_id": query_id,
                "question": question,
                "answer_value": answer_value,
                "answer_aliases": answer_aliases,
                "evidence": sample.get("evidence", {}).get("wikipedia", []) if sample.get("evidence") else []
            })
        
        print(f"[INFO] ✓ Prepared {len(self.queries)} questions for smoke run.")
        self._save_queries_manifest()
    
    def _save_queries_manifest(self):
        """Save planned queries manifest."""
        print(f"\n[INFO] Saving planned queries manifest to {self.queries_manifest_path}...")
        
        with open(self.queries_manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "question", "answer_value", "num_aliases", "num_evidence"])
            writer.writeheader()
            
            for q in self.queries:
                writer.writerow({
                    "query_id": q["query_id"],
                    "question": q["question"],
                    "answer_value": q["answer_value"],
                    "num_aliases": len(q["answer_aliases"]),
                    "num_evidence": len(q["evidence"])
                })
        
        print(f"[INFO] ✓ Queries manifest saved ({len(self.queries)} rows).")
    
    def initialize_router(self):
        """Initialize PSM router with persistent memory."""
        print("\n[INFO] Initializing PSM router (persistent memory mode)...")
        
        # Use evidence docs as corpus for retrieval
        all_docs = []
        for q in self.queries:
            for evidence in q["evidence"]:
                if isinstance(evidence, dict):
                    text = evidence.get("text", "")
                elif isinstance(evidence, str):
                    text = evidence
                else:
                    text = str(evidence)
                
                if text and text not in all_docs:
                    all_docs.append(text)
        
        # Ensure we have docs
        if len(all_docs) < 10:
            print(f"[WARNING] Only {len(all_docs)} unique evidence docs. Using fallback corpus.")
            all_docs = [f"Document {i}: Context about general knowledge" for i in range(20)]
        
        print(f"[INFO] Using {len(all_docs)} evidence documents as retrieval corpus.")
        
        # Initialize router with persistent memory
        self.router = Router(
            documents=all_docs,
            threshold=0.55,
            log_file=str(self.log_file_path),
        )
        self.router.hybrid_retriever.build_index(all_docs)
        
        print("[INFO] ✓ Router initialized with persistent memory enabled.")
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for EM/F1 computation."""
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
    
    def metric_exact_match(self, prediction: str, ground_truths: List[str]) -> float:
        """Compute exact match (1 if prediction matches any alias, 0 otherwise)."""
        norm_pred = self.normalize_answer(prediction)
        for gt in ground_truths:
            norm_gt = self.normalize_answer(gt)
            if norm_pred == norm_gt:
                return 1.0
        return 0.0
    
    def metric_f1(self, prediction: str, ground_truths: List[str]) -> float:
        """Compute token-level F1 score."""
        from collections import Counter
        
        norm_pred = self.normalize_answer(prediction)
        pred_tokens = norm_pred.split()
        
        if not pred_tokens:
            return 1.0 if not ground_truths else 0.0
        
        max_f1 = 0.0
        for gt in ground_truths:
            norm_gt = self.normalize_answer(gt)
            gt_tokens = norm_gt.split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1 = 0.0
            else:
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
            
            max_f1 = max(f1, max_f1)
        
        return max_f1
    
    def run_evaluation(self):
        """Run smoke evaluation loop with persistent memory."""
        print("\n" + "="*70)
        print("STARTING PSM TRIVIAQA SMOKE RUN (Persistent Memory Mode)")
        print("="*70)
        
        results = []
        memory_queries_seen = 0
        retrieval_queries_seen = 0
        
        for idx, query_data in enumerate(tqdm(self.queries, desc="Evaluating queries")):
            query_id = query_data["query_id"]
            question = query_data["question"]
            answer_aliases = query_data["answer_aliases"]
            answer_value = query_data["answer_value"]
            
            try:
                # Process query through router (persistent memory across iterations)
                generated_answer, confidence = self.router.process_query(question)
                
                # Get source from logger (last entry)
                last_log = self.router.logger.get_metrics_for_query(question)
                source = last_log["latest"]["source"] if last_log else "unknown"
                
                if source == "memory":
                    memory_queries_seen += 1
                elif source == "retrieval":
                    retrieval_queries_seen += 1
                
                # Compute metrics
                em_score = self.metric_exact_match(generated_answer, answer_aliases)
                f1_score = self.metric_f1(generated_answer, answer_aliases)
                
                result = {
                    "query_id": query_id,
                    "question": question,
                    "generated_answer": generated_answer,
                    "correct_answer": answer_value,
                    "confidence": float(confidence),
                    "source": source,
                    "em": float(em_score),
                    "f1": float(f1_score),
                    "memory_size": self.router.memory_store.size(),
                    "retrieval_count": self.router.retrieval_count,
                    "memory_count": self.router.memory_count
                }
                
                results.append(result)
                self.predictions.append(result)
                
            except Exception as e:
                print(f"\n[ERROR] Query {query_id} failed: {e}")
                results.append({
                    "query_id": query_id,
                    "question": question,
                    "error": str(e)
                })
        
        print("\n" + "="*70)
        print("SMOKE RUN COMPLETED")
        print("="*70)
        
        return results, memory_queries_seen, retrieval_queries_seen
    
    def save_predictions(self, results: List[Dict]):
        """Save per-query predictions."""
        print(f"\n[INFO] Saving predictions to {self.predictions_path}...")
        
        with open(self.predictions_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(f"[INFO] ✓ Saved {len(results)} predictions.")
    
    def compute_summary_metrics(self, results: List[Dict], memory_count: int, retrieval_count: int):
        """Compute and save summary metrics."""
        print(f"\n[INFO] Computing summary metrics...")
        
        valid_results = [r for r in results if "em" in r]
        
        if not valid_results:
            print("[WARNING] No valid results to compute metrics.")
            return {}
        
        em_scores = [r["em"] for r in valid_results]
        f1_scores = [r["f1"] for r in valid_results]
        confidences = [r["confidence"] for r in valid_results]
        
        memory_results = [r for r in valid_results if r["source"] == "memory"]
        retrieval_results = [r for r in valid_results if r["source"] == "retrieval"]
        
        metrics = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "smoke_size": self.smoke_size,
            "mode": "persistent_memory",
            "model": "llama3 (local)",
            "threshold": 0.55,
            
            # Overall metrics
            "total_queries": len(valid_results),
            "avg_em": float(np.mean(em_scores)),
            "avg_f1": float(np.mean(f1_scores)),
            "avg_confidence": float(np.mean(confidences)),
            
            # Path split
            "memory_path_count": memory_count,
            "retrieval_path_count": retrieval_count,
            "memory_path_rate": float(memory_count / len(valid_results)) if valid_results else 0.0,
            "retrieval_path_rate": float(retrieval_count / len(valid_results)) if valid_results else 0.0,
            
            # Memory-specific metrics
            "memory_path": {
                "count": len(memory_results),
                "avg_em": float(np.mean([r["em"] for r in memory_results])) if memory_results else 0.0,
                "avg_f1": float(np.mean([r["f1"] for r in memory_results])) if memory_results else 0.0,
                "avg_confidence": float(np.mean([r["confidence"] for r in memory_results])) if memory_results else 0.0,
            },
            
            # Retrieval-specific metrics
            "retrieval_path": {
                "count": len(retrieval_results),
                "avg_em": float(np.mean([r["em"] for r in retrieval_results])) if retrieval_results else 0.0,
                "avg_f1": float(np.mean([r["f1"] for r in retrieval_results])) if retrieval_results else 0.0,
                "avg_confidence": float(np.mean([r["confidence"] for r in retrieval_results])) if retrieval_results else 0.0,
            },
            
            # Growth
            "final_memory_size": self.router.memory_store.size() if self.router else 0,
        }
        
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[INFO] ✓ Metrics saved to {self.metrics_path}")
        
        return metrics
    
    def save_run_config(self):
        """Save run configuration for reproducibility."""
        config = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset": "TriviaQA Wikipedia",
            "split": "validation (dev)",
            "smoke_size": self.smoke_size,
            "mode": "persistent_memory",
            "memory_learning": True,
            "threshold": 0.55,
            "model": "llama3 (Ollama local)",
            "embedding_model": "all-MiniLM-L6-v2",
            "seed": 42,
            "version": "PSM v1.0",
        }
        
        with open(self.run_config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] ✓ Run config saved to {self.run_config_path}")
    
    def print_summary(self, metrics: Dict):
        """Print human-readable summary."""
        print("\n" + "="*70)
        print("SMOKE RUN SUMMARY")
        print("="*70)
        print(f"Run ID:             {metrics.get('run_id')}")
        print(f"Total Questions:    {metrics.get('total_queries')}")
        print(f"Avg EM:             {metrics.get('avg_em', 0):.3f}")
        print(f"Avg F1:             {metrics.get('avg_f1', 0):.3f}")
        print(f"Memory Path Rate:   {metrics.get('memory_path_rate', 0):.1%}")
        print(f"Retrieval Rate:     {metrics.get('retrieval_path_rate', 0):.1%}")
        print(f"\nMemory Path:")
        print(f"  Count: {metrics.get('memory_path', {}).get('count', 0)}")
        print(f"  EM:    {metrics.get('memory_path', {}).get('avg_em', 0):.3f}")
        print(f"  F1:    {metrics.get('memory_path', {}).get('avg_f1', 0):.3f}")
        print(f"\nRetrieval Path:")
        print(f"  Count: {metrics.get('retrieval_path', {}).get('count', 0)}")
        print(f"  EM:    {metrics.get('retrieval_path', {}).get('avg_em', 0):.3f}")
        print(f"  F1:    {metrics.get('retrieval_path', {}).get('avg_f1', 0):.3f}")
        print(f"\nFinal Memory Size:  {metrics.get('final_memory_size', 0)} entries")
        print(f"\nOutput Directory:   {self.output_dir}")
        print("="*70)
    
    def run(self):
        """Execute full smoke run."""
        self.load_triviaqa_dataset()
        self.prepare_smoke_subset()
        self.initialize_router()
        self.save_run_config()
        
        results, memory_count, retrieval_count = self.run_evaluation()
        
        self.save_predictions(results)
        metrics = self.compute_summary_metrics(results, memory_count, retrieval_count)
        
        self.print_summary(metrics)
        
        return metrics


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PSM TriviaQA Smoke Run")
    parser.add_argument("--smoke-size", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--run-id", type=str, default="psm_triviaqa_smoke_01", help="Run identifier")
    
    args = parser.parse_args()
    
    evaluator = TriviaQAEvaluator(smoke_size=args.smoke_size, run_id=args.run_id)
    evaluator.run()


if __name__ == "__main__":
    main()
