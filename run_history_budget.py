"""Train only Stage 3 on nested focal-history budgets using fixed K=5 bases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("PREFIXMEM_PYTHON", str(ROOT.parent / "perpcs-rebuilt-20261003/bin/python")))
BASE_RUNS = {
    "movie_tagging": ROOT / "runs/k-sensitivity/movie_tagging/k5",
    "scholarly_title": ROOT / "runs/k-sensitivity/scholarly_title/k5",
}


def link_once(source: Path, target: Path) -> None:
    resolved = source.resolve(strict=True)
    if target.is_symlink():
        if target.resolve() != resolved:
            raise ValueError(f"Existing link has a different target: {target}")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    target.symlink_to(resolved, target_is_directory=resolved.is_dir())


def run_one(task: str, percent: int, run_root: Path, seed: int, nproc: int) -> None:
    base = BASE_RUNS[task]
    baseline = json.loads((base / "summary.json").read_text())
    if baseline.get("stage3_train_config") is None or baseline.get("num_groups") != 5:
        raise ValueError(f"Base run has no auditable Stage-3 config or K=5: {base}")
    run_dir = run_root / task / f"p{percent}"
    manifest = run_dir / "data/history_budget.json"
    summary = run_dir / "summary.json"
    if summary.exists():
        if not manifest.exists() or json.loads(manifest.read_text())["fraction"] != percent / 100:
            raise ValueError(f"Completed run has incompatible manifest: {run_dir}")
        print(f"[skip] {task} {percent}% already complete", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("WANDB_DISABLED", "true")

    if not manifest.exists():
        command = [str(PYTHON), str(ROOT / "prepare_history_budget.py"),
                   "--source", str(base / "data"), "--output", str(run_dir / "data"),
                   "--task_name", task, "--fraction", str(percent / 100),
                   "--sampling", "nested_random", "--seed", str(seed)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    else:
        existing = json.loads(manifest.read_text())
        if (existing["fraction"] != percent / 100 or
                existing.get("sampling") != "nested_random" or
                existing.get("sampling_seed") != seed):
            raise ValueError(f"Existing history budget differs: {manifest}")
    link_once(ROOT / "prompt", run_dir / "prompt")
    link_once(base / "output", run_dir / "output")
    link_once(base / "output_group", run_dir / "output_group")
    ckpt_dir = run_dir / "ckpt" / task
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    source_ckpt = base / "ckpt" / task
    for name in [f"k0-{task}-llama3.1-8B-task_LoRA_ckpt"] + [
        f"k0-{task}-llama3.1-8B-group{group}-task_LoRA_ckpt" for group in range(5)
    ]:
        link_once(source_ckpt / name, ckpt_dir / name)
    local_complete = all(
        (ckpt_dir / f"k0-{task}-llama3.1-8B-group{group}-local-memory.pt").exists()
        and (ckpt_dir / f"k0-{task}-llama3.1-8B-group{group}-local-memory-lora/adapter_model.bin").exists()
        and (run_dir / "outputs" / f"local_group{group}_train_results.json").exists()
        for group in range(5)
    )
    command = [str(PYTHON), str(ROOT / "run_pipeline.py"),
               "--task_name", task, "--run_dir", str(run_dir),
               "--nproc", str(nproc), "--seed", str(seed),
               "--num_groups", "5", "--max_epoch", "3",
               "--batch_size", "1", "--max_new_tokens", "100",
               "--start_at", "31_local_eval" if local_complete else "30_local_train"]
    log = run_dir / "budget_pipeline.log"
    print(f"[run] {task} {percent}%: {run_dir}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{task} {percent}% failed ({result.returncode}); see {log}")
    print(f"[done] {task} {percent}%: {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(BASE_RUNS), default=list(BASE_RUNS))
    parser.add_argument("--percents", nargs="+", type=int, default=[25, 50, 75])
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/history-budget-nested-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()
    if any(percent < 1 or percent >= 100 for percent in args.percents):
        parser.error("Run only budgets from 1% to 99%; the 100% base is reused")
    if args.seed != 0:
        parser.error("These fixed K=5 baseline checkpoints were trained with seed 0")
    if args.nproc != 8:
        parser.error("Use the baseline's 8-GPU setup for controlled comparison")
    for task in args.tasks:
        for percent in args.percents:
            run_one(task, percent, args.run_root, args.seed, args.nproc)


if __name__ == "__main__":
    main()
