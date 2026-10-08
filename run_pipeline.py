"""Run all three PrefixMem stages and evaluations, with an isolated smoke mode."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ("10_global_train", "11_global_eval", "20_group_train", "21_group_eval", "30_local_train", "31_local_eval")
STAGE3_TRAIN_OVERRIDES = {
    "movie_tagging": {"batch_size": 1, "learning_rate": 1e-4},
    "citation": {"batch_size": 4, "learning_rate": 2e-4},
    "news_categorize": {"batch_size": 2, "learning_rate": 1e-4},
    "product_rating": {"batch_size": 4, "learning_rate": 2e-4},
    "news_headline": {"batch_size": 4, "learning_rate": 2e-4},
    "scholarly_title": {"batch_size": 2, "learning_rate": 1e-4},
    "tweet_paraphrase": {"batch_size": 4, "learning_rate": 1e-5},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", default="movie_tagging")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "/cfs/models/llama/llama3.1-8B"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cut_off", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recluster", action="store_true", help="Recompute groups in run_dir using the optional Qwen3 overlay")
    parser.add_argument("--data_source", type=Path, default=ROOT / "data",
                        help="Parent data directory used when preparing an isolated run")
    parser.add_argument("--num_groups", type=int, default=5,
                        help="Number of K-means segments used by stages 2 and 3")
    parser.add_argument("--start_at", choices=STAGES, help="Resume an existing run at the specified stage")
    args = parser.parse_args()
    if min(args.nproc, args.batch_size, args.max_epoch, args.max_new_tokens, args.num_groups) < 1 or args.cut_off < 4:
        parser.error("nproc, batch_size, max_epoch and max_new_tokens must be positive; cut_off must be >= 4")
    if args.max_step == 0 or args.max_step < -1:
        parser.error("max_step must be -1 or positive")
    run_dir = (args.run_dir or ROOT / "runs" / (f"smoke-{args.task_name}" if args.smoke else args.task_name)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_dir == ROOT:
        parser.error("Choose an isolated run_dir, not the repository root")
    if args.recluster and (run_dir / "data").resolve() == (ROOT / "data").resolve():
        parser.error("Reclustering requires an independent data copy; choose a fresh run_dir")
    if args.start_at and args.recluster:
        parser.error("Cannot recluster while resuming existing group checkpoints")
    if (run_dir / "ckpt").exists() and not args.start_at:
        parser.error(f"Checkpoints already exist in {run_dir}; select a fresh --run_dir")
    env = os.environ.copy()
    runtime = ROOT / ".deps" / "runtime"
    # The repository runtime overlay contains a patched torch build for the
    # original conda environment. In environments with a working native torch,
    # keeping it out of PYTHONPATH avoids shadowing torch C extensions.
    if runtime.is_dir() and os.environ.get("PREFIXMEM_USE_RUNTIME", "0") == "1":
        env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0" if args.nproc == 1 else ",".join(map(str, range(args.nproc))))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("WANDB_DISABLED", "true")
    env["PREFIXMEM_SEED"] = str(args.seed)

    def run(name, script, extra, distributed=False, overlay=False):
        if args.start_at and name in STAGES and STAGES.index(name) < STAGES.index(args.start_at):
            print(f"[{name}] reuse existing artifacts", flush=True)
            return
        command = [sys.executable]
        if distributed:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc}"]
        command += ["-u", str(ROOT / script)] if not distributed else [str(ROOT / script)]
        command += list(map(str, extra))
        child_env = env.copy()
        if overlay:
            child_env["PYTHONPATH"] = str(ROOT / ".deps" / "cluster") + os.pathsep + child_env.get("PYTHONPATH", "")
        log = run_dir / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"[{name}] {' '.join(command)}\nLog: {log}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=run_dir, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-10000:], file=sys.stderr)
            raise SystemExit(f"{name} failed ({result.returncode}); see {log}")
        print(f"[{name}] passed", flush=True)

    if args.smoke:
        run("00_data", "prepare_data.py", ["--source", ROOT / "data", "--output", run_dir / "data", "--tasks", args.task_name, "--smoke"])
    elif not (run_dir / "data").exists():
        if args.recluster:
            # Copy before reclustering so the imported source mappings remain intact.
            run("00_data", "prepare_data.py", ["--source", args.data_source, "--output", run_dir / "data", "--tasks", args.task_name])
        else:
            (run_dir / "data").symlink_to(ROOT / "data", target_is_directory=True)
    if not (run_dir / "prompt").exists():
        (run_dir / "prompt").symlink_to(ROOT / "prompt", target_is_directory=True)
    if args.recluster:
        task_data = run_dir / "data" / args.task_name
        run("01_cluster", "cluster_profiles.py", [
            "--task_name", args.task_name, "--batch_size", "4",
            "--n_clusters", str(args.num_groups), "--seed", str(args.seed),
            "--train_profile", task_data / "profile_user_others.json",
            "--infer_profile", task_data / "profile_user_100.json",
            "--train_output_json", task_data / "group_mapping_others.json",
            "--infer_output_json", task_data / "group_mapping_100.json",
        ], overlay=True)
    steps = 2 if args.smoke and args.max_step < 0 else args.max_step
    cutoff = min(args.cut_off, 384) if args.smoke else args.cut_off
    new_tokens = min(args.max_new_tokens, 16) if args.smoke else args.max_new_tokens
    common = ["--task_name", args.task_name, "--model_name", args.model_name]
    training = common + ["--batch_size", args.batch_size, "--max_step", steps, "--max_epoch", args.max_epoch, "--cut_off", cutoff, "--seed", args.seed]
    # Evaluation is inference-only, so use the task-specific throughput batch
    # size without changing Stage 1/2 training hyperparameters.
    evaluation_batch_size = STAGE3_TRAIN_OVERRIDES.get(
        args.task_name, {"batch_size": args.batch_size}
    )["batch_size"]
    evaluation = common + ["--batch_size", evaluation_batch_size, "--max_new_tokens", new_tokens, "--cut_off", cutoff]
    run("10_global_train", "task_LoRA.py", training, distributed=True)
    run("11_global_eval", "eval.py", evaluation)
    grouped_training = training + ["--num_groups", str(args.num_groups)]
    grouped_evaluation = evaluation + ["--num_groups", str(args.num_groups)]
    run("20_group_train", "task_LoRA_group.py", grouped_training, distributed=True)
    run("21_group_eval", "eval_group.py", grouped_evaluation)
    run("30_local_train", "task_LoRA_local_memory.py",
        grouped_training + ["--group_mode", "1", "--map_num_proc", "1"], distributed=True)
    run("31_local_eval", "eval_local.py", grouped_evaluation + ["--group_mode", "1"])

    sys.path.insert(0, str(ROOT))
    from eval.evaluation import LaMPEvaluation
    from utils import name2taskid
    golds_path = run_dir / "data" / args.task_name / "user_top_100_history_label.json"
    evaluator = LaMPEvaluation(single_gold_json_file_addr=str(golds_path))
    summary = {"task": args.task_name, "model": args.model_name, "smoke": args.smoke,
               "steps_per_group": steps, "nproc": args.nproc,
               "num_groups": args.num_groups,
               "stage3_train_config": STAGE3_TRAIN_OVERRIDES.get(
                   args.task_name, {"batch_size": args.batch_size, "learning_rate": 1e-5}
               ),
               "predictions": {}, "training": {}}
    for stage, directory in [("global", "output"), ("group", "output_group"), ("local", "output_local")]:
        paths = list((run_dir / directory).glob("*/preds/*.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one prediction file for {stage}: {paths}")
        path = paths[0]
        predictions = json.loads(path.read_text())
        metrics = evaluator.evaluate_task(str(path), name2taskid[args.task_name])
        summary["predictions"][stage] = {"path": str(path), "count": len(predictions["golds"]), "metrics": metrics}
    for path in sorted((run_dir / "outputs").glob("*_train_results.json")):
        metrics = json.loads(path.read_text())
        loss = metrics.get("train_loss")
        if loss is None or not math.isfinite(loss) or loss <= 0:
            raise RuntimeError(f"Invalid training loss in {path}: {loss}")
        if args.smoke and steps >= 2 and path.name.startswith("local_"):
            for key in ("prefix_max_update", "mediator_max_abs_B"):
                if not math.isfinite(metrics.get(key, 0)) or metrics.get(key, 0) <= 0:
                    raise RuntimeError(f"No finite {key} in {path}")
        summary["training"][path.stem] = metrics
    if not summary["training"]:
        raise RuntimeError("No training metrics found")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
