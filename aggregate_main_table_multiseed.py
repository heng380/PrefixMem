"""Aggregate completed three-seed Long-history summaries into one Markdown table."""
import json
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ["citation", "movie_tagging", "news_categorize", "news_headline",
         "product_rating", "scholarly_title", "tweet_paraphrase"]
LEGACY = {
    ("movie_tagging", 0): ROOT / "runs/movie-history-cohorts/long",
    ("movie_tagging", 1): ROOT / "runs/movie-history-cohorts/long-seed1",
    ("movie_tagging", 2): ROOT / "runs/movie-history-cohorts/long-seed2",
    ("scholarly_title", 0): ROOT / "runs/lamp5-history-cohorts/long",
    ("scholarly_title", 1): ROOT / "runs/lamp5-history-cohorts/long-seed1",
    ("scholarly_title", 2): ROOT / "runs/lamp5-history-cohorts/long-seed2",
}
COMPLETE_ROOTS = {
    "citation": ROOT / "runs/main-table-3seed/citation",
    "movie_tagging": ROOT / "runs/movie-history-cohorts",
    "news_categorize": ROOT / "runs/main-table-3seed/news_categorize",
    "news_headline": ROOT / "runs/main-table-3seed-fast/news_headline",
    "product_rating": ROOT / "runs/main-table-3seed-fast/product_rating",
    "scholarly_title": ROOT / "runs/main-table-3seed-fast/scholarly_title",
    "tweet_paraphrase": ROOT / "runs/main-table-3seed-fast/tweet_paraphrase",
}
METRICS = {
    "citation": ("accuracy", "f1"), "movie_tagging": ("accuracy", "f1"),
    "news_categorize": ("accuracy", "f1"), "news_headline": ("rouge-1", "rouge-L"),
    "product_rating": ("MAE", "RMSE"), "scholarly_title": ("rouge-1", "rouge-L"),
    "tweet_paraphrase": ("rouge-1", "rouge-L"),
}


def path_for(task, seed, run_root, use_legacy):
    if use_legacy:
        if task == "movie_tagging":
            return LEGACY[(task, seed)]
        return COMPLETE_ROOTS[task] / f"seed{seed}"
    return run_root / task / f"seed{seed}"


def fmt(values):
    if len(values) != 3:
        return "pending"
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, default=ROOT / "runs/main-table-3seed")
    parser.add_argument("--no_legacy", action="store_true", help="Read every task from --run_root")
    parser.add_argument("--output", type=Path, default=ROOT / "main_table_long_3seed_complete.md")
    args = parser.parse_args()
    lines = ["# Long-history main-table results across three training seeds", "",
             "Values are mean +/- sample standard deviation over seeds 0, 1, and 2.",
             "Each cell reports the task's two metrics in the order shown in the metric column.", ""]
    lines.append("| Task | Metrics | Stage 1 / Global | Stage 2 / Group | Stage 3 / Local |")
    lines.append("|---|---|---:|---:|---:|")
    provenance = []
    for task in TASKS:
        keys = METRICS[task]
        by_stage = {stage: {key: [] for key in keys} for stage in ("global", "group", "local")}
        counts = set()
        configs = set()
        for seed in range(3):
            path = path_for(task, seed, args.run_root, use_legacy=not args.no_legacy) / "summary.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing {task} seed {seed}: {path}")
            data = json.loads(path.read_text())
            if data.get("task") != task:
                raise ValueError(f"Task mismatch in {path}")
            counts.add(tuple(data["predictions"][stage]["count"] for stage in by_stage))
            config = data.get("stage3_train_config")
            if config is not None:
                configs.add(json.dumps(config, sort_keys=True))
            for stage in by_stage:
                metrics = data.get("predictions", {}).get(stage, {}).get("metrics", {})
                for key in keys:
                    if key not in metrics:
                        raise ValueError(f"Missing {stage}/{key} in {path}")
                    by_stage[stage][key].append(float(metrics[key]))
            provenance.append((task, seed, path))
        if len(counts) != 1 or len(configs) > 1:
            raise ValueError(f"Inconsistent evaluation counts or stage-3 config for {task}")
        cells = []
        for stage in ("global", "group", "local"):
            cells.append(" / ".join(fmt(by_stage[stage][key]) for key in keys))
        lines.append(f"| {task} | {' / '.join(keys)} | " + " | ".join(cells) + " |")
    lines += ["", "## Run provenance", "",
              "These rows combine within-task seed 0/1/2 runs, not five-seed results. "
              "Task-specific Stage-3 settings may differ across tasks. "
              "The Movie Tagging runs use the legacy Long cohort; LaMP-5 uses "
              "the later fast-configuration Long cohort. The group-mapping "
              "files are byte-identical across these three seeds within "
              "every task, so this is training-seed variability conditional "
              "on fixed clusters, not a full pipeline seed study with "
              "K-means reinitialized for each seed. The dataset shuffle "
              "calls also use a fixed seed of 42 in the current training "
              "scripts.", "",
              "| Task | Seed | Summary |", "|---|---:|---|"]
    for task, seed, path in provenance:
        lines.append(f"| {task} | {seed} | `{path.relative_to(ROOT)}` |")
    output = args.output
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
