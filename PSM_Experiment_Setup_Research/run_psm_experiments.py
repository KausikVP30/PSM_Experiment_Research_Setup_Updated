#!/usr/bin/env python3
"""Run PSM experiments on real TriviaQA data with isolated per-run memory."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from evaluation.aggregate_runs import aggregate_runs
from evaluation.metrics import bleu, contains_any_answer, exact_match, rouge_l, token_f1
from evaluation.report_writer import write_run_readable_report
from ingestion.corpus_ingestor import build_corpus_from_samples
from router.router import Router


@dataclass
class ExperimentProfile:
    name: str
    num_questions: int
    corpus_pool_size: int
    max_chunks: int


def extract_answers(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    answer = sample.get("answer", {})
    if isinstance(answer, dict):
        value = answer.get("value", "")
        aliases = answer.get("aliases", [])
    else:
        value = str(answer)
        aliases = []

    if value and value not in aliases:
        aliases = [value] + aliases

    aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
    return value, aliases or ([value] if value else [])


def sample_id(sample: Dict[str, Any], index: int) -> str:
    return str(sample.get("question_id") or sample.get("qid") or sample.get("id") or f"triviaqa_{index:05d}")


class PSMExperimentRunner:
    def __init__(self, base_results_dir: str = "experiment_runs", threshold: float = 0.55):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def run_profile(self, profile: ExperimentProfile, dataset_split: str = "validation") -> Dict[str, Any]:
        run_id = f"{profile.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.base_results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        memory_file = run_dir / "memory_store.json"
        index_file = run_dir / "memory_index.faiss"
        log_file = run_dir / "experiment_log.csv"

        print("\n" + "=" * 80)
        print(f"Running profile: {profile.name}")
        print(f"Run directory: {run_dir}")
        print("=" * 80)

        # 1) Load dataset pool.
        ds = load_dataset("trivia_qa", "rc.wikipedia", split=f"{dataset_split}[:{profile.corpus_pool_size}]")
        samples = [dict(x) for x in ds]

        # 2) Build corpus from real evidence.
        corpus_result = build_corpus_from_samples(
            samples,
            max_docs_before_chunking=profile.corpus_pool_size,
            max_chunks=profile.max_chunks,
            chunk_words=120,
            overlap_words=30,
        )
        corpus = corpus_result.documents

        # 3) Select query set.
        queries: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            question = str(sample.get("question", "")).strip()
            if not question:
                continue
            answer_value, aliases = extract_answers(sample)
            if not answer_value:
                continue
            queries.append(
                {
                    "query_id": sample_id(sample, i),
                    "question": question,
                    "answer_value": answer_value,
                    "answer_aliases": aliases,
                }
            )
            if len(queries) >= profile.num_questions:
                break

        # 4) Save planned queries.
        planned_csv = run_dir / "planned_queries.csv"
        with planned_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "question", "answer_value", "num_aliases"])
            writer.writeheader()
            for q in queries:
                writer.writerow(
                    {
                        "query_id": q["query_id"],
                        "question": q["question"],
                        "answer_value": q["answer_value"],
                        "num_aliases": len(q["answer_aliases"]),
                    }
                )

        # 5) Initialize isolated router.
        router = Router(
            documents=corpus,
            threshold=self.threshold,
            memory_file=str(memory_file),
            index_file=str(index_file),
            log_file=str(log_file),
        )

        # 6) Execute evaluation.
        results: List[Dict[str, Any]] = []
        retrieval_calls = 0
        memory_calls = 0
        latencies = []
        hallucinations = 0

        for q in tqdm(queries, desc=f"{profile.name}: evaluating"):
            t0 = time.time()
            answer, confidence = router.process_query(q["question"])
            latency_s = time.time() - t0
            latencies.append(latency_s)

            latest = router.logger.get_metrics_for_query(q["question"])
            source = latest["latest"]["source"] if latest else "unknown"
            if source == "memory":
                memory_calls += 1
            elif source == "retrieval":
                retrieval_calls += 1

            em = exact_match(answer, q["answer_aliases"])
            f1 = token_f1(answer, q["answer_aliases"])
            rl = rouge_l(answer, q["answer_aliases"])
            bl = bleu(answer, q["answer_aliases"])
            contains = contains_any_answer(answer, q["answer_aliases"])
            if not contains:
                hallucinations += 1

            results.append(
                {
                    "query_id": q["query_id"],
                    "question": q["question"],
                    "generated_answer": answer,
                    "correct_answer": q["answer_value"],
                    "confidence": float(confidence),
                    "source": source,
                    "em": float(em),
                    "f1": float(f1),
                    "rouge_l": float(rl),
                    "bleu": float(bl),
                    "answer_contains_gold": contains,
                    "latency_s": float(latency_s),
                    "memory_size": router.memory_store.size(),
                    "retrieval_count": router.retrieval_count,
                    "memory_count": router.memory_count,
                }
            )

        # 7) Write predictions files.
        pred_jsonl = run_dir / "predictions.jsonl"
        with pred_jsonl.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        pred_csv = run_dir / "predictions.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "query_id",
                "question",
                "generated_answer",
                "correct_answer",
                "confidence",
                "source",
                "em",
                "f1",
                "rouge_l",
                "bleu",
                "answer_contains_gold",
                "latency_s",
                "memory_size",
                "retrieval_count",
                "memory_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in fieldnames})

        # 8) Summaries.
        def avg(key: str) -> float:
            return float(np.mean([r[key] for r in results])) if results else 0.0

        memory_rows = [r for r in results if r["source"] == "memory"]
        retrieval_rows = [r for r in results if r["source"] == "retrieval"]

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "dataset": "TriviaQA rc.wikipedia",
            "split": dataset_split,
            "num_questions": len(results),
            "corpus_size": len(corpus),
            "documents_before_chunking": corpus_result.document_count_before_chunking,
            "mode": "persistent_memory",
            "memory_mode": "isolated",
            "model": "llama3 (8B)",
            "embedding_model": "all-MiniLM-L6-v2",
            "threshold": self.threshold,
            "avg_em": avg("em"),
            "avg_f1": avg("f1"),
            "avg_rouge_l": avg("rouge_l"),
            "avg_bleu": avg("bleu"),
            "avg_confidence": avg("confidence"),
            "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
            "memory_path_count": memory_calls,
            "retrieval_path_count": retrieval_calls,
            "memory_path_rate": float(memory_calls / len(results)) if results else 0.0,
            "retrieval_path_rate": float(retrieval_calls / len(results)) if results else 0.0,
            "retrieval_call_rate": float(retrieval_calls / len(results)) if results else 0.0,
            "hallucination_rate": float(hallucinations / len(results)) if results else 0.0,
            "memory_path": {
                "count": len(memory_rows),
                "avg_em": float(np.mean([r["em"] for r in memory_rows])) if memory_rows else 0.0,
                "avg_f1": float(np.mean([r["f1"] for r in memory_rows])) if memory_rows else 0.0,
                "avg_confidence": float(np.mean([r["confidence"] for r in memory_rows])) if memory_rows else 0.0,
            },
            "retrieval_path": {
                "count": len(retrieval_rows),
                "avg_em": float(np.mean([r["em"] for r in retrieval_rows])) if retrieval_rows else 0.0,
                "avg_f1": float(np.mean([r["f1"] for r in retrieval_rows])) if retrieval_rows else 0.0,
                "avg_confidence": float(np.mean([r["confidence"] for r in retrieval_rows])) if retrieval_rows else 0.0,
            },
            "final_memory_size": router.memory_store.size(),
            "memory_file": str(memory_file),
            "index_file": str(index_file),
            "log_file": str(log_file),
        }

        summary_path = run_dir / "summary_metrics.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        config = {
            "run_id": run_id,
            "timestamp": summary["timestamp"],
            "dataset": summary["dataset"],
            "split": dataset_split,
            "num_questions": profile.num_questions,
            "corpus_pool_size": profile.corpus_pool_size,
            "max_chunks": profile.max_chunks,
            "threshold": self.threshold,
            "memory_mode": "isolated",
            "model": summary["model"],
            "embedding_model": summary["embedding_model"],
            "output_dir": str(run_dir),
        }
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        with (run_dir / "corpus_info.txt").open("w", encoding="utf-8") as f:
            f.write(f"documents_before_chunking={corpus_result.document_count_before_chunking}\n")
            f.write(f"chunk_count={corpus_result.chunk_count}\n")
            for i, chunk in enumerate(corpus[:10]):
                f.write(f"[{i}] {chunk[:240]}\n")

        notes = []
        if summary["memory_path_rate"] < 0.1:
            notes.append("Memory path triggered rarely; corpus diversity or threshold may be limiting early memory reuse.")
        if summary["avg_f1"] < 0.1:
            notes.append("Answer quality is currently low; retrieval context quality and prompting need improvement.")
        if summary["avg_latency_s"] > 20:
            notes.append("Latency is high; consider smaller prompt context or generation limits.")
        if not notes:
            notes.append("Run is stable with balanced routing and measurable answer quality.")

        write_run_readable_report(run_dir, config, summary, notes)

        print(f"\n[INFO] Completed run {run_id}")
        print(json.dumps(summary, indent=2))
        return summary

    def execute(self) -> Dict[str, Any]:
        profiles = [
            ExperimentProfile(name="smoke", num_questions=5, corpus_pool_size=140, max_chunks=500),
            ExperimentProfile(name="pilot", num_questions=8, corpus_pool_size=260, max_chunks=900),
        ]

        summaries = []
        for p in profiles:
            summaries.append(self.run_profile(p))

        # create master aggregate artifacts
        agg = aggregate_runs(self.base_results_dir)
        print("\n[INFO] Master aggregation complete:")
        print(json.dumps(agg, indent=2))
        return {"runs": summaries, "aggregate": agg}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full PSM experiment suite")
    parser.add_argument("--results-dir", type=str, default="experiment_runs")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    runner = PSMExperimentRunner(base_results_dir=args.results_dir, threshold=args.threshold)
    runner.execute()
