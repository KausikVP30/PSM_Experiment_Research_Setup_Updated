from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _pct(x: float) -> str:
    return f"{(x or 0.0) * 100:.1f}%"


def write_run_readable_report(output_dir: Path, config: Dict, summary: Dict, notes: List[str]) -> Path:
    path = output_dir / "run_readable_report.txt"
    lines = []
    lines.append("PSM Run Readable Report")
    lines.append("=" * 80)
    lines.append(f"Run ID: {config.get('run_id')}")
    lines.append(f"Timestamp: {config.get('timestamp')}")
    lines.append(f"Dataset: {config.get('dataset')} | Split: {config.get('split')}")
    lines.append(f"Model: {config.get('model')} | Embedding: {config.get('embedding_model')}")
    lines.append(f"Threshold: {config.get('threshold')} | Memory Mode: {config.get('memory_mode')}")
    lines.append(f"Questions: {summary.get('num_questions')} | Corpus Chunks: {summary.get('corpus_size')}")
    lines.append("")

    lines.append("Main Metrics")
    lines.append("-" * 80)
    lines.append(f"EM: {summary.get('avg_em', 0.0):.4f}")
    lines.append(f"F1: {summary.get('avg_f1', 0.0):.4f}")
    lines.append(f"ROUGE-L: {summary.get('avg_rouge_l', 0.0):.4f}")
    lines.append(f"BLEU: {summary.get('avg_bleu', 0.0):.4f}")
    lines.append(f"Hallucination Rate (answer-not-contained proxy): {_pct(summary.get('hallucination_rate', 0.0))}")
    lines.append("")

    lines.append("Routing Behavior")
    lines.append("-" * 80)
    lines.append(f"Memory path count: {summary.get('memory_path_count', 0)} ({_pct(summary.get('memory_path_rate', 0.0))})")
    lines.append(f"Retrieval path count: {summary.get('retrieval_path_count', 0)} ({_pct(summary.get('retrieval_path_rate', 0.0))})")
    lines.append(f"Avg confidence: {summary.get('avg_confidence', 0.0):.4f}")
    lines.append(f"Avg latency (s): {summary.get('avg_latency_s', 0.0):.3f}")
    lines.append(f"Final memory size: {summary.get('final_memory_size', 0)}")
    lines.append("")

    lines.append("Interpretation")
    lines.append("-" * 80)
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append("- No interpretation notes available.")

    lines.append("")
    lines.append("Raw summary JSON")
    lines.append("-" * 80)
    lines.append(json.dumps(summary, indent=2))

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_master_readable_report(base_dir: Path, all_summaries: List[Dict]) -> Path:
    path = base_dir / "experiment_master_report.txt"
    lines = []
    lines.append("PSM Experiment Master Report")
    lines.append("=" * 80)

    if not all_summaries:
        lines.append("No runs found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    lines.append(f"Total runs: {len(all_summaries)}")
    lines.append("")

    for idx, s in enumerate(all_summaries, 1):
        lines.append(f"Run {idx}: {s.get('run_id')} ({s.get('timestamp')})")
        lines.append(f"  Dataset: {s.get('dataset')} | Questions: {s.get('num_questions')} | Corpus: {s.get('corpus_size')}")
        lines.append(f"  EM/F1: {s.get('avg_em', 0.0):.4f} / {s.get('avg_f1', 0.0):.4f}")
        lines.append(f"  ROUGE-L/BLEU: {s.get('avg_rouge_l', 0.0):.4f} / {s.get('avg_bleu', 0.0):.4f}")
        lines.append(f"  Memory/Retrieval: {_pct(s.get('memory_path_rate', 0.0))} / {_pct(s.get('retrieval_path_rate', 0.0))}")
        lines.append(f"  Avg latency: {s.get('avg_latency_s', 0.0):.3f}s")
        lines.append("")

    # best-by-f1 quick summary
    best = max(all_summaries, key=lambda x: x.get("avg_f1", 0.0))
    lines.append("Best run by F1")
    lines.append("-" * 80)
    lines.append(f"Run ID: {best.get('run_id')}")
    lines.append(f"F1: {best.get('avg_f1', 0.0):.4f}")
    lines.append(f"Memory rate: {_pct(best.get('memory_path_rate', 0.0))}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
