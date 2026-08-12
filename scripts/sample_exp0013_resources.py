#!/usr/bin/env python3
"""Low-frequency sidecar sampler for EXP-0013 resource accounting."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


CGROUP = Path("/sys/fs/cgroup")


def read_int(path: Path, default: int = 0) -> int:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return default if value == "max" else int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return default


def read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError):
        return result
    for line in lines:
        key, value = line.split()
        result[key] = int(value)
    return result


def process_table() -> dict[int, dict[str, int | str]]:
    table: dict[int, dict[str, int | str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            close = stat.rfind(")")
            comm = stat[stat.find("(") + 1 : close]
            fields = stat[close + 2 :].split()
            ppid = int(fields[1])
            status = (entry / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        rss_kib = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
        table[pid] = {"ppid": ppid, "comm": comm, "rss_bytes": rss_kib * 1024}
    return table


def descendants(table: dict[int, dict[str, int | str]], root: int) -> set[int]:
    selected = {root}
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid not in selected and int(row["ppid"]) in selected:
                selected.add(pid)
                changed = True
    return selected


def disk_used(path: Path) -> int:
    stat = os.statvfs(path)
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


def gpu_row() -> list[str] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        return None
    return [item.strip() for item in result.stdout.splitlines()[0].split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watched-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    temp_root = args.temp_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    system_path = output / "resource_system.csv"
    process_path = output / "resource_processes.csv"
    gpu_path = output / "resource_gpu.csv"
    system_fields = [
        "timestamp_epoch",
        "cgroup_memory_current_bytes",
        "cgroup_pids_current",
        "cpu_usage_usec",
        "cpu_nr_periods",
        "cpu_nr_throttled",
        "cpu_throttled_usec",
        "memory_events_oom",
        "memory_events_oom_kill",
        "tree_process_count",
        "tree_rss_bytes",
        "java_count",
        "java_rss_bytes",
        "temp_dir_count",
        "system_disk_used_bytes",
        "data_disk_used_bytes",
    ]
    process_fields = ["timestamp_epoch", "pid", "ppid", "comm", "rss_bytes"]
    gpu_fields = [
        "timestamp_epoch",
        "memory_used_mib",
        "gpu_util_percent",
        "power_watts",
        "temperature_c",
    ]
    with (
        system_path.open("w", newline="", encoding="utf-8") as system_file,
        process_path.open("w", newline="", encoding="utf-8") as process_file,
        gpu_path.open("w", newline="", encoding="utf-8") as gpu_file,
    ):
        system_writer = csv.DictWriter(system_file, fieldnames=system_fields)
        process_writer = csv.DictWriter(process_file, fieldnames=process_fields)
        gpu_writer = csv.DictWriter(gpu_file, fieldnames=gpu_fields)
        system_writer.writeheader()
        process_writer.writeheader()
        gpu_writer.writeheader()
        while Path(f"/proc/{args.watched_pid}").exists():
            now = time.time()
            table = process_table()
            selected = descendants(table, args.watched_pid)
            rows = [table[pid] | {"pid": pid} for pid in selected if pid in table]
            java_rows = [row for row in rows if row["comm"] == "java"]
            cpu = read_key_values(CGROUP / "cpu.stat")
            memory_events = read_key_values(CGROUP / "memory.events")
            system_writer.writerow(
                {
                    "timestamp_epoch": f"{now:.6f}",
                    "cgroup_memory_current_bytes": read_int(CGROUP / "memory.current"),
                    "cgroup_pids_current": read_int(CGROUP / "pids.current"),
                    "cpu_usage_usec": cpu.get("usage_usec", 0),
                    "cpu_nr_periods": cpu.get("nr_periods", 0),
                    "cpu_nr_throttled": cpu.get("nr_throttled", 0),
                    "cpu_throttled_usec": cpu.get("throttled_usec", 0),
                    "memory_events_oom": memory_events.get("oom", 0),
                    "memory_events_oom_kill": memory_events.get("oom_kill", 0),
                    "tree_process_count": len(rows),
                    "tree_rss_bytes": sum(int(row["rss_bytes"]) for row in rows),
                    "java_count": len(java_rows),
                    "java_rss_bytes": sum(int(row["rss_bytes"]) for row in java_rows),
                    "temp_dir_count": sum(
                        1 for path in temp_root.iterdir() if path.is_dir()
                    )
                    if temp_root.exists()
                    else 0,
                    "system_disk_used_bytes": disk_used(Path("/")),
                    "data_disk_used_bytes": disk_used(Path("/root/autodl-tmp")),
                }
            )
            for row in rows:
                process_writer.writerow(
                    {
                        "timestamp_epoch": f"{now:.6f}",
                        "pid": row["pid"],
                        "ppid": row["ppid"],
                        "comm": row["comm"],
                        "rss_bytes": row["rss_bytes"],
                    }
                )
            gpu = gpu_row()
            if gpu:
                gpu_writer.writerow(
                    dict(zip(gpu_fields, [f"{now:.6f}", *gpu], strict=True))
                )
            system_file.flush()
            process_file.flush()
            gpu_file.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
