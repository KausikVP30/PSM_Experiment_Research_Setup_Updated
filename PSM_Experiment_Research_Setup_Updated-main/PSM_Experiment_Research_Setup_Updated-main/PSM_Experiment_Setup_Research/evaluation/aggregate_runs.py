from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from evaluation.report_writer import write_master_readable_report


def _find_summary_files(base_dir: Path) -> List[Path]:
    return sorted(base_dir.glob("*/**/summary_metrics.json"))


def aggregate_runs(results_base_dir: Path) -> Dict:
    summary_files = _find_summary_files(results_base_dir)
    summaries: List[Dict] = []

    for summary_file in summary_files:
        try:
            summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
        except Exception:
            continue

    table_path = results_base_dir / "all_runs_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_id",
            "timestamp",
            "dataset",
            "num_questions",
            "corpus_size",
            "avg_em",
            "avg_f1",
            "avg_rouge_l",
            "avg_bleu",
            "avg_confidence",
            "avg_latency_s",
            "memory_path_rate",
            "retrieval_path_rate",
            "final_memory_size",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary.get(k) for k in fieldnames})

    json_path = results_base_dir / "all_runs_metrics.json"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    master_report_path = write_master_readable_report(results_base_dir, summaries)

    return {
        "runs": len(summaries),
        "csv": str(table_path),
        "json": str(json_path),
        "report": str(master_report_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate PSM run metrics")
    parser.add_argument("--results-dir", type=str, default="experiment_runs")
    args = parser.parse_args()

    info = aggregate_runs(Path(args.results_dir))
    print(json.dumps(info, indent=2))
