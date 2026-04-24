import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, log_file='logs/experiment_log_v2.csv'):
        self.log_file = log_file

        # Ensure parent directory exists (handles changed working directories)
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Create file with header if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "query",
                    "confidence",
                    "memory_id",
                    "sim_query",
                    "sim_answer",
                    "sim_docs",
                    "source",
                    "latency",
                    "memory_size",
                    "retrieval_count",
                    "memory_count",
                ])

    def log(self, query, confidence, memory_id, sim_q, sim_a, sim_d,
            source, latency, memory_size, retrieval_count, memory_count):
        query = (query or "").strip()
        if not query:
            return

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                query,
                confidence,
                memory_id,
                sim_q,
                sim_a,
                sim_d,
                source,
                latency,
                memory_size,
                retrieval_count,
                memory_count,
            ])

    @staticmethod
    def _to_float(value):
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_metrics_for_query(self, query_text):
        if not query_text or not os.path.exists(self.log_file):
            return None

        normalized = query_text.strip().lower()
        if not normalized:
            return None

        matched_rows = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_query = (row.get("query") or "").strip()
                if not row_query:
                    continue
                row_normalized = row_query.lower()

                # Match exact query first, then allow substring match.
                if row_normalized == normalized or normalized in row_normalized:
                    matched_rows.append(row)

        if not matched_rows:
            return None

        confidences = [
            self._to_float(r.get("confidence"))
            for r in matched_rows
            if self._to_float(r.get("confidence")) is not None
        ]
        latencies = [
            self._to_float(r.get("latency"))
            for r in matched_rows
            if self._to_float(r.get("latency")) is not None
        ]

        source_counts = {"memory": 0, "retrieval": 0, "other": 0}
        for row in matched_rows:
            source = (row.get("source") or "").strip().lower()
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts["other"] += 1

        latest_row = matched_rows[-1]
        return {
            "query": query_text,
            "matches": len(matched_rows),
            "avg_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
            "avg_latency": (sum(latencies) / len(latencies)) if latencies else None,
            "min_latency": min(latencies) if latencies else None,
            "source_counts": source_counts,
            "latest": {
                "timestamp": latest_row.get("timestamp"),
                "source": latest_row.get("source"),
                "confidence": self._to_float(latest_row.get("confidence")),
                "latency": self._to_float(latest_row.get("latency")),
                "memory_size": latest_row.get("memory_size"),
                "retrieval_count": latest_row.get("retrieval_count"),
                "memory_count": latest_row.get("memory_count"),
            },
        }