"""Queue K-sensitivity runs and the K=5 hard/soft routing latency study."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("PREFIXMEM_PYTHON", str(ROOT.parent / "perpcs-rebuilt-20261003/bin/python")))
TASK_SOURCES = {
    "movie_tagging": ROOT / "runs/movie-history-cohorts/long/data",
    "scholarly_title": ROOT / "runs/lamp5-history-cohorts/long/data",
}


def run_logged(command, cwd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(map(str, command))}", flush=True)
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"[failed] {log_path}", flush=True)
        return False
    print(f"[done] {log_path.parent.name}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SOURCES), default=list(TASK_SOURCES))
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_root", type=Path, default=ROOT / "runs/k-sensitivity")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("WANDB_DISABLED", "true")
    failures = []
    for task in args.tasks:
        for k in args.ks:
            run_dir = args.run_root / task / f"k{k}"
            summary = run_dir / "summary.json"
            if not summary.exists():
                command = [
                    str(PYTHON), str(ROOT / "run_pipeline.py"),
                    "--task_name", task,
                    "--run_dir", str(run_dir),
                    "--nproc", str(args.nproc),
                    "--batch_size", "1",
                    "--max_epoch", "3",
                    "--max_new_tokens", "100",
                    "--seed", str(args.seed),
                    "--num_groups", str(k),
                    "--recluster",
                    "--data_source", str(TASK_SOURCES[task]),
                ]
                if not run_logged(command, ROOT.parent, run_dir / "queue.log", env):
                    failures.append(f"{task}/k{k}")
                    continue
            else:
                print(f"[skip] {task}/k{k}: summary exists", flush=True)

            # Soft routing requires the centroid and target embedding metadata
            # emitted by the reclustering step, so run it only after K=5 is
            # complete.  Hard routing uses the same evaluator for comparable
            # latency accounting and is not a second training run.
            if k == 5:
                for mode in ("hard", "soft"):
                    output_dir = run_dir / "routing"
                    routing_summary = output_dir / task / mode / "summary.json"
                    if routing_summary.exists():
                        print(f"[skip] {task}/k5/{mode}: summary exists", flush=True)
                        continue
                    command = [
                        str(PYTHON), str(ROOT / "eval_soft_routing.py"),
                        "--task_name", task,
                        "--mode", mode,
                        "--num_groups", "5",
                        "--batch_size", "4",
                        "--max_new_tokens", "100",
                        "--output_dir", str(output_dir),
                    ]
                    if not run_logged(command, run_dir, output_dir / f"{mode}.log", env):
                        failures.append(f"{task}/k5/{mode}")

    if failures:
        raise SystemExit("Failed runs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
