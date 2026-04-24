#!/usr/bin/env python3
"""
Phase 2 TriviaQA evaluation for Predictive Semantic Memory (PSM).

This runner:
- Loads TriviaQA rc.wikipedia validation data
- Builds a real retrieval corpus from dataset evidence-like fields
- Runs the existing Router in persistent memory mode
- Writes a new CSV log for this phase plus JSON/JSONL artifacts
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from router.router import Router


def normalize_answer(text: str) -> str:
    """Normalize answer text for TriviaQA-style EM/F1."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Max token F1 over gold aliases."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    for gold in gold_answers:
        if pred == normalize_answer(gold):
            return 1.0
    return 0.0


def flatten_text(value: Any) -> Iterable[str]:
    """Recursively collect text-like strings from nested dataset objects."""
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for key in ("text", "sentence", "content", "description", "title", "passage"):
            if key in value:
                yield from flatten_text(value[key])
        for nested_value in value.values():
            if isinstance(nested_value, (dict, list, tuple, set)):
                yield from flatten_text(nested_value)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten_text(item)


def extract_gold_answers(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Extract answer value and aliases from TriviaQA sample in a schema-safe way."""
    answer = sample.get("answer", {})
    if isinstance(answer, dict):
        value = answer.get("value") or answer.get("text") or ""
        aliases = answer.get("aliases") or []
    else:
        value = str(answer) if answer is not None else ""
        aliases = []

    if value and value not in aliases:
        aliases = [value] + list(aliases)

    aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
    return value, aliases or ([value] if value else [])


def sample_id(sample: Dict[str, Any], fallback_index: int) -> str:
    qid = sample.get("question_id") or sample.get("qid") or sample.get("id")
    return str(qid) if qid is not None else f"triviaqa_{fallback_index:05d}"


@dataclass
class Example:
    query_id: str
    question: str
    answer_value: str
    answer_aliases: List[str]


class Phase2TriviaQARunner:
    def __init__(self, num_questions: int = 20, corpus_pool_size: int = 250, run_id: str = "psm_phase2_triviaqa"):
        self.num_questions = num_questions
        self.corpus_pool_size = corpus_pool_size
        self.run_id = run_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.output_dir = Path(__file__).parent / "phase2_results" / run_id / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_csv = self.output_dir / "planned_queries.csv"
        self.predictions_csv = self.output_dir / "predictions.csv"
        self.predictions_jsonl = self.output_dir / "predictions.jsonl"
        self.summary_json = self.output_dir / "summary_metrics.json"
        self.run_config_json = self.output_dir / "run_config.json"
        self.corpus_info_txt = self.output_dir / "corpus_info.txt"

        self.dataset = None
        self.corpus: List[str] = []
        self.examples: List[Example] = []
        self.router: Optional[Router] = None

    def load_dataset(self) -> None:
        print("\n[INFO] Loading TriviaQA rc.wikipedia validation split...")
        self.dataset = load_dataset("trivia_qa", "rc.wikipedia", split=f"validation[:{self.corpus_pool_size}]")
        print(f"[INFO] Loaded {len(self.dataset)} samples for corpus/question pool.")

    def build_corpus(self) -> None:
        print("[INFO] Building evidence corpus from TriviaQA sample fields...")
        corpus_set = []
        seen = set()

        for sample in self.dataset:
            for field_name in ("evidence", "entity_pages", "search_results", "web_pages", "context"):
                if field_name not in sample:
                    continue
                for text in flatten_text(sample[field_name]):
                    if len(text) < 20:
                        continue
                    if text not in seen:
                        seen.add(text)
                        corpus_set.append(text)

        # Fallback in case this split exposes fewer evidence-like fields than expected.
        if len(corpus_set) < 20:
            for sample in self.dataset:
                question = sample.get("question", "").strip()
                if question and question not in seen:
                    seen.add(question)
                    corpus_set.append(question)

        self.corpus = corpus_set[:1000]
        with open(self.corpus_info_txt, "w", encoding="utf-8") as handle:
            handle.write(f"Corpus size: {len(self.corpus)}\n")
            handle.write("Sample passages:\n")
            for idx, passage in enumerate(self.corpus[:10]):
                handle.write(f"[{idx}] {passage[:250]}\n")

        print(f"[INFO] ✓ Built corpus with {len(self.corpus)} passages.")

    def select_questions(self) -> None:
        print(f"[INFO] Selecting {self.num_questions} TriviaQA questions...")
        examples: List[Example] = []
        for idx, sample in enumerate(self.dataset):
            question = (sample.get("question") or "").strip()
            if not question:
                continue
            answer_value, aliases = extract_gold_answers(sample)
            if not answer_value:
                continue
            examples.append(
                Example(
                    query_id=sample_id(sample, idx),
                    question=question,
                    answer_value=answer_value,
                    answer_aliases=aliases,
                )
            )
            if len(examples) >= self.num_questions:
                break

        self.examples = examples
        print(f"[INFO] ✓ Selected {len(self.examples)} evaluation questions.")

    def initialize_router(self) -> None:
        print("[INFO] Initializing router with persistent memory mode...")
        memory_file = self.output_dir / "memory_store.json"
        index_file = self.output_dir / "memory_index.faiss"
        log_file = self.output_dir / "experiment_log.csv"
        self.router = Router(
            documents=self.corpus,
            threshold=0.55,
            memory_file=str(memory_file),
            index_file=str(index_file),
            log_file=str(log_file),
        )
        self.router.hybrid_retriever.build_index(self.corpus)
        print("[INFO] ✓ Router ready.")

    def save_manifest(self) -> None:
        with open(self.manifest_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query_id", "question", "answer_value", "num_aliases"])
            writer.writeheader()
            for ex in self.examples:
                writer.writerow(
                    {
                        "query_id": ex.query_id,
                        "question": ex.question,
                        "answer_value": ex.answer_value,
                        "num_aliases": len(ex.answer_aliases),
                    }
                )

    def run(self) -> List[Dict[str, Any]]:
        assert self.router is not None
        results: List[Dict[str, Any]] = []

        memory_hits = 0
        retrieval_hits = 0

        print("\n" + "=" * 72)
        print("PHASE 2 TRIVIAQA EVALUATION - PERSISTENT MEMORY MODE")
        print("=" * 72)
        print(f"Questions: {len(self.examples)}")
        print(f"Corpus passages: {len(self.corpus)}")
        print(f"Model: llama3 (8B) via local Ollama")
        print(f"Threshold: 0.55")
        print("=" * 72)

        for ex in tqdm(self.examples, desc="Evaluating TriviaQA"):
            try:
                answer, confidence = self.router.process_query(ex.question)
                latest = self.router.logger.get_metrics_for_query(ex.question)
                source = latest["latest"]["source"] if latest else "unknown"

                if source == "memory":
                    memory_hits += 1
                elif source == "retrieval":
                    retrieval_hits += 1

                em = exact_match(answer, ex.answer_aliases)
                f1 = token_f1(answer, ex.answer_aliases)

                result = {
                    "query_id": ex.query_id,
                    "question": ex.question,
                    "generated_answer": answer,
                    "correct_answer": ex.answer_value,
                    "aliases": " | ".join(ex.answer_aliases),
                    "confidence": float(confidence),
                    "source": source,
                    "em": float(em),
                    "f1": float(f1),
                    "memory_size": self.router.memory_store.size(),
                    "retrieval_count": self.router.retrieval_count,
                    "memory_count": self.router.memory_count,
                }
                results.append(result)

            except Exception as exc:
                print(f"[ERROR] {ex.query_id}: {exc}")
                results.append(
                    {
                        "query_id": ex.query_id,
                        "question": ex.question,
                        "error": str(exc),
                        "correct_answer": ex.answer_value,
                    }
                )

        self._write_outputs(results, memory_hits, retrieval_hits)
        return results

    def _write_outputs(self, results: List[Dict[str, Any]], memory_hits: int, retrieval_hits: int) -> None:
        valid = [row for row in results if "em" in row]

        with open(self.predictions_jsonl, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(self.predictions_csv, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "query_id",
                "question",
                "generated_answer",
                "correct_answer",
                "confidence",
                "source",
                "em",
                "f1",
                "memory_size",
                "retrieval_count",
                "memory_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in valid:
                writer.writerow({k: row.get(k) for k in fieldnames})

        avg_em = float(np.mean([r["em"] for r in valid])) if valid else 0.0
        avg_f1 = float(np.mean([r["f1"] for r in valid])) if valid else 0.0
        avg_conf = float(np.mean([r["confidence"] for r in valid])) if valid else 0.0

        memory_rows = [r for r in valid if r.get("source") == "memory"]
        retrieval_rows = [r for r in valid if r.get("source") == "retrieval"]

        metrics = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "phase": "phase2",
            "dataset": "TriviaQA rc.wikipedia",
            "num_questions": len(valid),
            "corpus_size": len(self.corpus),
            "mode": "persistent_memory",
            "model": "llama3 (8B)",
            "threshold": 0.55,
            "avg_em": avg_em,
            "avg_f1": avg_f1,
            "avg_confidence": avg_conf,
            "memory_path_count": memory_hits,
            "retrieval_path_count": retrieval_hits,
            "memory_path_rate": float(memory_hits / len(valid)) if valid else 0.0,
            "retrieval_path_rate": float(retrieval_hits / len(valid)) if valid else 0.0,
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
            "final_memory_size": self.router.memory_store.size() if self.router else 0,
        }

        run_config = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset": "TriviaQA rc.wikipedia",
            "num_questions": len(self.examples),
            "corpus_size": len(self.corpus),
            "mode": "persistent_memory",
            "memory_learning": True,
            "threshold": 0.55,
            "model": "llama3 (8B)",
            "embedding_model": "all-MiniLM-L6-v2",
            "python": sys.version,
        }

        with open(self.summary_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        with open(self.run_config_json, "w", encoding="utf-8") as handle:
            json.dump(run_config, handle, indent=2)

        print("\n[INFO] Phase 2 summary:")
        print(json.dumps(metrics, indent=2))
        print(f"[INFO] Results saved under: {self.output_dir}")

    def execute(self) -> None:
        self.load_dataset()
        self.build_corpus()
        self.select_questions()
        self.save_manifest()
        self.initialize_router()
        self.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 TriviaQA PSM evaluation")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--corpus-pool-size", type=int, default=250)
    parser.add_argument("--run-id", type=str, default="psm_phase2_triviaqa")
    args = parser.parse_args()

    runner = Phase2TriviaQARunner(
        num_questions=args.num_questions,
        corpus_pool_size=args.corpus_pool_size,
        run_id=args.run_id,
    )
    runner.execute()


if __name__ == "__main__":
    main()