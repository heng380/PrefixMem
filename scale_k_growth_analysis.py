#!/usr/bin/env python3
"""Cost-only sensitivity for a fixed or population-dependent group count.

Measured K runs anchor the payload and Trainer-runtime accounting. Larger
populations and K > 10 are projections, not new training experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scale_cost_analysis import DEFAULT_N, prefix_artifact_bytes, parse_prefix_training


ROOT = Path(__file__).resolve().parent
TASKS = ("movie_tagging", "scholarly_title")
MEASURED_K = (3, 5, 7, 10)
PREFIX_BYTES_PER_USER = 8 * 4096 * 4  # Eight fp32 Llama-3.1-8B hidden vectors.


def growth_k(users: int, baseline_users: int, baseline_k: int = 5) -> int:
    if users < 1 or baseline_users < 1:
        raise ValueError("User counts must be positive")
    return math.ceil(baseline_k * math.sqrt(users / baseline_users))


def read_anchor(run_root: Path, task: str, k: int) -> dict:
    run = run_root / task / f"k{k}"
    data_path = run / "data" / task / "user_top_100_history.json"
    summary_path = run / "summary.json"
    checkpoint = run / "ckpt" / task
    if not data_path.exists() or not summary_path.exists() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Incomplete K={k} run: {run}")
    users = len(json.loads(data_path.read_text()))
    summary = json.loads(summary_path.read_text())
    nproc = int(summary["nproc"])
    if nproc < 1:
        raise ValueError(f"Invalid GPU count: {run}")
    timing = parse_prefix_training(summary_path, users)
    fixed, memory, breakdown = prefix_artifact_bytes(checkpoint)
    if not users or breakdown["global_task_adapter_bytes"] == 0:
        raise ValueError(f"Invalid user count or global adapter: {run}")
    if len(list(checkpoint.glob("*-group*-task_LoRA_ckpt/adapter_model.bin"))) != k:
        raise ValueError(f"Expected {k} group adapters: {checkpoint}")
    if len(list(checkpoint.glob("*-group*-local-memory-lora/adapter_model.bin"))) != k:
        raise ValueError(f"Expected {k} local mediators: {checkpoint}")
    if len(list(checkpoint.glob("*.pt"))) != k:
        raise ValueError(f"Expected {k} local-memory files: {checkpoint}")
    return {"task": task, "k": k, "users": users, "nproc": nproc,
            "run": str(run),
            "timing": timing, "fixed_bytes": fixed, "memory_bytes": memory,
            "breakdown": breakdown}


def projected_storage(users: int, k: int, anchor: dict) -> float:
    measured = anchor["breakdown"]
    per_group = (measured["group_task_adapter_bytes"]
                 + measured["local_mediator_adapter_bytes"]) / anchor["k"]
    serialization_per_group = (
        anchor["memory_bytes"] - PREFIX_BYTES_PER_USER * anchor["users"]
    ) / anchor["k"]
    if serialization_per_group < 0:
        raise ValueError("Measured local-memory payload is smaller than its parameters")
    return (measured["global_task_adapter_bytes"] + per_group * k
            + PREFIX_BYTES_PER_USER * users + serialization_per_group * k)


def make_rows(anchors: list[dict], populations: list[int]) -> list[dict]:
    by_k = {item["k"]: item for item in anchors}
    baseline = by_k[5]
    rows = []
    for users in populations:
        for scenario, k in (("fixed_k5", 5),
                            ("sqrt_growth", growth_k(users, baseline["users"]))):
            # Training assumes the same total stage-1/2 corpus and history
            # distribution per user. K-specific setup/loading time is excluded.
            seconds = (baseline["timing"]["shared_seconds"]
                       + baseline["timing"]["local_seconds_per_user"] * users)
            rows.append({
                "task": baseline["task"], "scenario": scenario, "N": users, "K": k,
                "update_gpu_hours": seconds * baseline["nproc"] / 3600,
                "storage_gib": projected_storage(users, k, baseline) / 2**30,
                "storage_bytes": round(projected_storage(users, k, baseline)),
                "raw_history_records_per_query": 0,
                "active_group_adapters_per_query": 1,
                "k_beyond_measured_range": k > max(by_k),
            })
    return rows


def render_report(anchors_by_task: dict[str, list[dict]], rows: list[dict]) -> str:
    lines = [
        "# K-Growth Cost Sensitivity",
        "",
        "This is a cost projection, not a new population-scaling training run. "
        "The anchors are completed long-history K=3/5/7/10 runs for Movie "
        "Tagging and LaMP-5. The focal populations are 100 and 101 users, "
        "respectively. All N>baseline points are extrapolated; K>10 is "
        "outside the measured K range.",
        "",
        "## Measured K anchors",
        "",
        "Training GPU-hours multiply the sum of recorded distributed Trainer "
        "runtimes by the number of GPUs. They exclude loading and other "
        "non-Trainer elapsed time. Payload is adapter and user-memory model state, "
        "excluding small config files and optimizer state.",
        "",
        "| Task | Users | K | GPUs | Shared train (GPU-h) | Local train (GPU-h) | Payload (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, anchors in anchors_by_task.items():
        for item in anchors:
            timing = item["timing"]
            lines.append(
                f"| {task} | {item['users']} | {item['k']} | {item['nproc']} | "
                f"{timing['shared_seconds']*item['nproc']/3600:.3f} | "
                f"{timing['local_seconds_at_baseline']*item['nproc']/3600:.3f} | "
                f"{(item['fixed_bytes']+item['memory_bytes'])/2**30:.3f} |"
            )
    lines += [
        "",
        "## Population-dependent K projection",
        "",
        "Fixed: K=5. Growth: K(N)=ceil(5 sqrt(N/N0)), where N0 is the "
        "measured focal-user count for that task. This square-root rule is "
        "an illustrative stress test, not a tuned or accuracy-validated K "
        "selection policy. An asterisk marks K>10. LaMP-5 has N0=101, so "
        "even its N=100 point is a one-user extrapolation.",
        "",
        "| Task | N | Growth K | Update cost (GPU-h; both) | Fixed K storage (GiB) | Growth K storage (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in anchors_by_task:
        task_rows = [row for row in rows if row["task"] == task]
        by_n = {}
        for row in task_rows:
            by_n.setdefault(row["N"], {})[row["scenario"]] = row
        for users, scenarios in by_n.items():
            fixed, growth = scenarios["fixed_k5"], scenarios["sqrt_growth"]
            marker = "*" if growth["k_beyond_measured_range"] else ""
            lines.append(
                f"| {task} | {users:,} | {growth['K']}{marker} | "
                f"{fixed['update_gpu_hours']:.3f} | "
                f"{fixed['storage_gib']:.3f} | {growth['storage_gib']:.3f} |"
            )
    lines += [
        "",
        "## Model and limits",
        "",
        "- Storage = global adapter + K x (group adapter + local mediator + "
        "per-group serialization overhead) + N x 131,072 bytes of fp32 prefix. "
        "The K=5 checkpoint calibrates the fixed payload and serialization "
        "overhead. Other measured K checkpoints check this linear formula.",
        "- Update GPU-hours = GPU count x [measured K=5 global/group Trainer "
        "time + N x (measured K=5 local Trainer time / N0)] / 3600. "
        "The formula assumes a fixed "
        "shared training corpus and constant per-user history distribution. "
        "It is K-invariant by assumption, not evidence that K has no runtime "
        "cost. The measured K=3/5/7/10 timings above show its observed "
        "variation at N0.",
        "- Checkpoint loading, repeated initialization, K-means, routing, "
        "optimizer state, and allocator overhead are excluded. They may "
        "increase with K. There is no measured end-to-end latency scaling "
        "for K>5; hard routing still activates one group adapter/query.",
        "- Standard PrefixMem inference inserts zero raw user-history "
        "records for either scenario. This is an exposure count, not a "
        "privacy guarantee.",
        "- These estimates cannot establish quality or efficiency at "
        "10,000 real users. Such claims require new cohorts and timing runs.",
        "",
        "Machine-readable rows and anchors: `runs/scale-k-growth/`.",
        "Reproduce with `python3 scale_k_growth_analysis.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/k-sensitivity")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/scale-k-growth")
    parser.add_argument("--report", type=Path, default=ROOT / "k_growth_cost_sensitivity.md")
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N))
    args = parser.parse_args()
    if any(n < 1 for n in args.n):
        parser.error("All population sizes must be positive")
    anchors = {task: [read_anchor(args.run_root, task, k) for k in MEASURED_K]
               for task in TASKS}
    rows = [row for task in TASKS for row in make_rows(anchors[task], args.n)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scaling_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "measured_anchors.json").write_text(
        json.dumps(anchors, indent=2) + "\n")
    args.report.write_text(render_report(anchors, rows))
    print(f"Wrote {args.report} and {args.output_dir / 'scaling_table.csv'}")


if __name__ == "__main__":
    main()
