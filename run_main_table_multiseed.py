"""Queue three-seed Long-history PrefixMem runs for all main-table tasks."""
import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PYTHON = Path(os.environ.get("PREFIXMEM_PYTHON", str(ROOT.parent / "perpcs-rebuilt-20261003/bin/python")))
TASKS = ["citation", "movie_tagging", "news_categorize", "news_headline",
         "product_rating", "scholarly_title", "tweet_paraphrase"]


def source_parent(task):
    if task == "movie_tagging":
        return ROOT / "runs/movie-history-cohorts/long/data"
    if task == "scholarly_title":
        return ROOT / "runs/lamp5-history-cohorts/long/data"
    return ROOT / "data"


def run_path(task, seed, run_root, use_legacy=True):
    legacy = {
        ("movie_tagging", 0): ROOT / "runs/movie-history-cohorts/long",
        ("movie_tagging", 1): ROOT / "runs/movie-history-cohorts/long-seed1",
        ("movie_tagging", 2): ROOT / "runs/movie-history-cohorts/long-seed2",
        ("scholarly_title", 0): ROOT / "runs/lamp5-history-cohorts/long",
        ("scholarly_title", 1): ROOT / "runs/lamp5-history-cohorts/long-seed1",
        ("scholarly_title", 2): ROOT / "runs/lamp5-history-cohorts/long-seed2",
    }
    if use_legacy:
        return legacy.get((task, seed), run_root / task / f"seed{seed}")
    return run_root / task / f"seed{seed}"


def run_one(task, seed, run_dir, nproc):
    summary = run_dir / "summary.json"
    if summary.exists():
        print(f"[skip] {task} seed={seed}: {summary}", flush=True)
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    data_link = run_dir / "data"
    prompt_link = run_dir / "prompt"
    if not data_link.exists():
        data_link.symlink_to(source_parent(task), target_is_directory=True)
    if not prompt_link.exists():
        prompt_link.symlink_to(ROOT / "prompt", target_is_directory=True)
    log = run_dir / "pipeline.log"
    command = [str(ENV_PYTHON), str(ROOT / "run_pipeline.py"),
               "--task_name", task, "--run_dir", str(run_dir),
               "--nproc", str(nproc), "--batch_size", "1", "--max_epoch", "3",
               "--max_new_tokens", "100", "--seed", str(seed)]
    # If a long run was interrupted after producing checkpoints, resume at the
    # latest safe stage rather than failing on run_pipeline's checkpoint guard.
    if (run_dir / "ckpt").exists():
        if (run_dir / "output_local").exists():
            command += ["--start_at", "31_local_eval"]
        elif (run_dir / "output_group").exists():
            command += ["--start_at", "21_group_eval"]
        elif (run_dir / "output").exists():
            command += ["--start_at", "11_global_eval"]
        elif (run_dir / "ckpt" / task).exists():
            # Global training writes its task adapter before the first eval;
            # this covers an interruption during global evaluation.
            command += ["--start_at", "11_global_eval"]
        else:
            command += ["--start_at", "10_global_train"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    print(f"[run] {task} seed={seed}: {' '.join(command)}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT.parent, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"[failed] {task} seed={seed}; see {log}", flush=True)
        return False
    print(f"[done] {task} seed={seed}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--run_root", type=Path, default=ROOT / "runs/main-table-3seed")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--no_legacy", action="store_true", help="Always place results under --run_root")
    args = parser.parse_args()
    failures = []
    for task in args.tasks:
        for seed in args.seeds:
            run_dir = run_path(task, seed, args.run_root, use_legacy=not args.no_legacy)
            if not run_one(task, seed, run_dir, args.nproc):
                failures.append(f"{task}/seed{seed}")
    if failures:
        raise SystemExit("Failed runs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
