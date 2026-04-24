#!/usr/bin/env python3
"""Automatically pick the most available GPU server and run an experiment.

Usage example:
    python auto_gpu_experiment_runner.py \
        --config gpu_servers.example.json \
        --command "python test_retrievers.py" \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ServerConfig:
    name: str
    host: str
    user: str
    port: int = 22
    workdir: str = ""
    python_command: str = "python"


@dataclass
class GpuSnapshot:
    server: ServerConfig
    gpu_index: int
    memory_free_mb: int
    memory_total_mb: int
    utilization_pct: int

    @property
    def free_ratio(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return self.memory_free_mb / self.memory_total_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick the best free GPU server and run an experiment command via SSH.",
    )
    parser.add_argument(
        "--config",
        default="gpu_servers.json",
        help="Path to JSON config with server definitions (default: gpu_servers.json)",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Experiment command to run remotely (example: 'python test_retrievers.py')",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=4096,
        help="Minimum free memory (MB) to treat a GPU as available (default: 4096)",
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=40,
        help="Maximum GPU utilization percent to treat a GPU as available (default: 40)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=6,
        help="SSH connection timeout in seconds (default: 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected server/GPU and command without executing",
    )
    return parser.parse_args()


def load_servers(config_path: Path) -> List[ServerConfig]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Create it from gpu_servers.example.json"
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_servers = data.get("servers", [])
    servers: List[ServerConfig] = []

    for item in raw_servers:
        servers.append(
            ServerConfig(
                name=item["name"],
                host=item["host"],
                user=item["user"],
                port=int(item.get("port", 22)),
                workdir=item.get("workdir", ""),
                python_command=item.get("python_command", "python"),
            )
        )

    if not servers:
        raise ValueError("No servers defined under 'servers' in config file.")

    return servers


def ssh_base_command(server: ServerConfig, connect_timeout: int) -> List[str]:
    return [
        "ssh",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "BatchMode=yes",
        "-p",
        str(server.port),
        f"{server.user}@{server.host}",
    ]


def probe_server_gpus(server: ServerConfig, connect_timeout: int) -> List[GpuSnapshot]:
    query_cmd = (
        "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits"
    )

    completed = subprocess.run(
        ssh_base_command(server, connect_timeout) + [query_cmd],
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "No error details"
        print(f"[WARN] {server.name}: probe failed ({stderr})", file=sys.stderr)
        return []

    snapshots: List[GpuSnapshot] = []
    for line in completed.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu_index, free_mb, total_mb, util_pct = map(int, parts)
        except ValueError:
            continue
        snapshots.append(
            GpuSnapshot(
                server=server,
                gpu_index=gpu_index,
                memory_free_mb=free_mb,
                memory_total_mb=total_mb,
                utilization_pct=util_pct,
            )
        )
    return snapshots


def pick_best_gpu(
    snapshots: List[GpuSnapshot],
    min_free_mb: int,
    max_utilization: int,
) -> Optional[GpuSnapshot]:
    if not snapshots:
        return None

    available = [
        s
        for s in snapshots
        if s.memory_free_mb >= min_free_mb and s.utilization_pct <= max_utilization
    ]

    pool = available if available else snapshots
    return max(
        pool,
        key=lambda s: (s.memory_free_mb, s.free_ratio, -s.utilization_pct),
    )


def build_remote_command(selected: GpuSnapshot, experiment_command: str) -> str:
    # Quote command pieces once and execute via remote shell.
    user_cmd = experiment_command
    if selected.server.python_command != "python":
        tokens = shlex.split(user_cmd)
        if tokens and tokens[0] in {"python", "python3"}:
            tokens[0] = selected.server.python_command
            user_cmd = shlex.join(tokens)

    parts = [f"export CUDA_VISIBLE_DEVICES={selected.gpu_index}"]
    if selected.server.workdir:
        parts.append(f"cd {shlex.quote(selected.server.workdir)}")
    parts.append(user_cmd)
    return " && ".join(parts)


def run_remote(selected: GpuSnapshot, remote_command: str, connect_timeout: int) -> int:
    print(
        f"[INFO] Running on {selected.server.name} ({selected.server.host}) "
        f"GPU {selected.gpu_index} | free={selected.memory_free_mb}MB "
        f"util={selected.utilization_pct}%"
    )

    cmd = ssh_base_command(selected.server, connect_timeout) + [remote_command]
    return subprocess.call(cmd)


def main() -> int:
    args = parse_args()
    servers = load_servers(Path(args.config))

    all_snapshots: List[GpuSnapshot] = []
    for server in servers:
        snapshots = probe_server_gpus(server, connect_timeout=args.connect_timeout)
        if snapshots:
            print(f"[INFO] {server.name}: found {len(snapshots)} GPU(s)")
        all_snapshots.extend(snapshots)

    selected = pick_best_gpu(
        all_snapshots,
        min_free_mb=args.min_free_mb,
        max_utilization=args.max_utilization,
    )
    if selected is None:
        print("[ERROR] No reachable GPU servers found.", file=sys.stderr)
        return 2

    remote_command = build_remote_command(selected, args.command)
    print(f"[INFO] Selected server={selected.server.name}, gpu={selected.gpu_index}")
    print(f"[INFO] Remote command: {remote_command}")

    if args.dry_run:
        return 0

    return run_remote(selected, remote_command, connect_timeout=args.connect_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
