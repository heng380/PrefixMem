#!/usr/bin/env python3
"""Estimate PrefixMem/OPPU cost as the number of personalized users grows.

The script deliberately separates measured quantities (checkpoint bytes and
Trainer runtimes) from the linear extrapolation to larger user populations.
It does not train synthetic 500--10,000-user cohorts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


DEFAULT_N = (100, 500, 1_000, 5_000, 10_000)


def file_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def prefix_artifact_bytes(checkpoint_dir: Path) -> tuple[int, int, dict[str, int]]:
    """Return fixed bytes, local-memory bytes, and an auditable breakdown.

    Adapter payloads for the global task, five group tasks, and five local
    mediators are shared with respect to N. The group .pt files contain the
    user-indexed local memory and are modeled as the N-dependent component.
    Metadata files are excluded because they are negligible and do not carry
    model state; the report separately records the full directory size.
    """

    bins = list(checkpoint_dir.glob("*/adapter_model.bin"))
    group_task = [p for p in bins if "-group" in p.parent.name and "-task_LoRA_ckpt" in p.parent.name]
    local_mediator = [p for p in bins if "-local-memory-lora" in p.parent.name]
    global_task = [p for p in bins if "-group" not in p.parent.name and "-task_LoRA_ckpt" in p.parent.name]
    memories = list(checkpoint_dir.glob("*.pt"))

    breakdown = {
        "global_task_adapter_bytes": file_bytes(global_task),
        "group_task_adapter_bytes": file_bytes(group_task),
        "local_mediator_adapter_bytes": file_bytes(local_mediator),
        "local_memory_payload_bytes": file_bytes(memories),
        "all_files_bytes": file_bytes([p for p in checkpoint_dir.rglob("*") if p.is_file()]),
    }
    fixed = (
        breakdown["global_task_adapter_bytes"]
        + breakdown["group_task_adapter_bytes"]
        + breakdown["local_mediator_adapter_bytes"]
    )
    return fixed, breakdown["local_memory_payload_bytes"], breakdown


def parse_prefix_training(summary_path: Path, baseline_users: int) -> dict[str, float]:
    data = json.loads(summary_path.read_text())
    training = data["training"]
    global_seconds = float(training["global_train_results"]["train_runtime"])
    group_seconds = sum(
        float(value["train_runtime"])
        for key, value in training.items()
        if re.fullmatch(r"group\d+_train_results", key)
    )
    local_seconds = sum(
        float(value["train_runtime"])
        for key, value in training.items()
        if re.fullmatch(r"local_group\d+_train_results", key)
    )
    return {
        "global_seconds": global_seconds,
        "group_seconds": group_seconds,
        "local_seconds_at_baseline": local_seconds,
        "shared_seconds": global_seconds + group_seconds,
        "local_seconds_per_user": local_seconds / baseline_users,
        "baseline_users": float(baseline_users),
    }


def parse_oppu_log(log_path: Path, baseline_users: int) -> dict[str, float]:
    text = log_path.read_text(errors="replace")
    runtimes = [float(value) for value in re.findall(r"'train_runtime':\s*([0-9.]+)", text)]
    if not runtimes:
        raise ValueError(f"No per-user train_runtime records found in {log_path}")
    return {
        "records": float(len(runtimes)),
        "runtime_sum_seconds": sum(runtimes),
        "runtime_mean_seconds": mean(runtimes),
        "runtime_std_seconds": stdev(runtimes) if len(runtimes) > 1 else 0.0,
        "runtime_min_seconds": min(runtimes),
        "runtime_max_seconds": max(runtimes),
        "seconds_per_user": sum(runtimes) / baseline_users,
        "baseline_users": float(baseline_users),
    }


def adapter_storage_bytes(adapter_dir: Path) -> tuple[int, float, int]:
    bins = list(adapter_dir.glob("*/adapter_model.bin"))
    if not bins:
        raise ValueError(f"No adapter_model.bin files found in {adapter_dir}")
    sizes = [path.stat().st_size for path in bins]
    return sum(sizes), mean(sizes), len(sizes)


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-summary", type=Path, default=Path("runs/movie-history-cohorts/long/summary.json"))
    parser.add_argument("--prefix-checkpoint-dir", type=Path, default=Path("runs/movie-history-cohorts/long/ckpt/movie_tagging"))
    parser.add_argument("--oppu-log", type=Path, default=Path("runs/movie-history-cohorts/short/oppu_base_b32.log"))
    parser.add_argument("--oppu-adapter-dir", type=Path, default=Path("runs/movie-history-cohorts/short/oppu_adapters_base/movie_tagging"))
    parser.add_argument("--baseline-users", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scale-cost"))
    parser.add_argument("--n", type=int, nargs="*", default=list(DEFAULT_N), help="User counts to extrapolate")
    parser.add_argument("--plot", action="store_true", help="Also write a two-panel PNG for training and storage")
    args = parser.parse_args()

    prefix = parse_prefix_training(args.prefix_summary, args.baseline_users)
    oppu = parse_oppu_log(args.oppu_log, args.baseline_users)
    fixed_bytes, memory_bytes_at_baseline, breakdown = prefix_artifact_bytes(args.prefix_checkpoint_dir)
    oppu_total_bytes, oppu_user_bytes, oppu_adapter_count = adapter_storage_bytes(args.oppu_adapter_dir)
    prefix_memory_per_user = memory_bytes_at_baseline / args.baseline_users

    rows = []
    for users in args.n:
        rows.append(
            {
                "N": users,
                "prefixmem_training_update_seconds": prefix["shared_seconds"] + prefix["local_seconds_per_user"] * users,
                "oppu_training_update_seconds": oppu["seconds_per_user"] * users,
                "prefixmem_storage_bytes": fixed_bytes + prefix_memory_per_user * users,
                "oppu_storage_bytes": oppu_user_bytes * users,
                "prefixmem_inference_raw_history_records": 0,
                "oppu_inference_raw_history_records": 0,
                "rag_inference_raw_history_records": "k",
                "pag_inference_raw_history_records": "profile/history text",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scaling_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "prefix_summary": str(args.prefix_summary),
        "prefix_checkpoint_dir": str(args.prefix_checkpoint_dir),
        "oppu_log": str(args.oppu_log),
        "oppu_adapter_dir": str(args.oppu_adapter_dir),
        "baseline_users": args.baseline_users,
        "n_values": args.n,
        "prefix_training": prefix,
        "oppu_training": oppu,
        "prefix_storage_breakdown_bytes": breakdown,
        "prefix_fixed_storage_bytes": fixed_bytes,
        "prefix_local_memory_bytes_at_baseline": memory_bytes_at_baseline,
        "prefix_local_memory_bytes_per_user": prefix_memory_per_user,
        "oppu_adapter_count": oppu_adapter_count,
        "oppu_adapter_total_bytes_at_baseline": oppu_total_bytes,
        "oppu_adapter_bytes_per_user": oppu_user_bytes,
        "rows": rows,
    }
    (args.output_dir / "scaling_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("--plot requires matplotlib") from exc
        x = [row["N"] for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(x, [row["prefixmem_training_update_seconds"] / 3600 for row in rows], marker="o", label="PrefixMem")
        axes[0].plot(x, [row["oppu_training_update_seconds"] / 3600 for row in rows], marker="o", label="OPPU")
        axes[0].set_xlabel("Number of users (N)")
        axes[0].set_ylabel("Summed Trainer runtime (hours)")
        axes[0].set_title("Training/update cost")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)
        axes[1].plot(x, [row["prefixmem_storage_bytes"] / 2**30 for row in rows], marker="o", label="PrefixMem")
        axes[1].plot(x, [row["oppu_storage_bytes"] / 2**30 for row in rows], marker="o", label="OPPU")
        axes[1].set_xlabel("Number of users (N)")
        axes[1].set_ylabel("Adapter/memory storage (GiB)")
        axes[1].set_title("Checkpoint storage")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.25, which="both")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output_dir / "scaling_cost.png", dpi=220)
        plt.close(fig)
        print(f"Wrote {args.output_dir / 'scaling_cost.png'}")

    print(f"Wrote {csv_path}")
    print(f"PrefixMem shared training: {prefix['shared_seconds']:.4f} s")
    print(f"PrefixMem local update: {prefix['local_seconds_per_user']:.6f} s/user")
    print(f"OPPU update: {oppu['seconds_per_user']:.6f} s/user from {int(oppu['records'])} records")
    print(f"PrefixMem fixed storage: {format_bytes(fixed_bytes)}; local memory/user: {format_bytes(prefix_memory_per_user)}")
    print(f"OPPU adapter/user: {format_bytes(oppu_user_bytes)}")
    for row in rows:
        print(
            f"N={row['N']}: PrefixMem {row['prefixmem_training_update_seconds']:.2f}s, "
            f"OPPU {row['oppu_training_update_seconds']:.2f}s, "
            f"storage {format_bytes(row['prefixmem_storage_bytes'])} vs {format_bytes(row['oppu_storage_bytes'])}"
        )


if __name__ == "__main__":
    main()
