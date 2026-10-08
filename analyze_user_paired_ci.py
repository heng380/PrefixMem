"""Paired user-cluster bootstrap for the completed Long-history three-seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from aggregate_main_table_multiseed import ROOT, TASKS, path_for
from eval.evaluation import normalize_news_category_prediction


STAGES = ("global", "group", "local")
COMPARISONS = (("group", "global"), ("local", "group"), ("local", "global"))
CLASSIFICATION = {"citation", "movie_tagging", "news_categorize"}


def primary_metric(task: str) -> str:
    if task in CLASSIFICATION:
        return "accuracy"
    if task == "product_rating":
        return "MAE"
    return "rouge-L"


def score_query(task: str, prediction: str, gold: str, scorer) -> float:
    if task == "news_categorize":
        prediction = normalize_news_category_prediction(prediction)
    if task in CLASSIFICATION:
        return float(prediction.strip() == gold.strip())
    if task == "product_rating":
        try:
            rating = float(prediction.strip())
        except ValueError:
            match = re.search(r"score\s*:\s*([1-5])", prediction.strip(), re.I)
            actual = float(gold.strip())
            rating = float(match.group(1)) if match else (
                1.0 if abs(1-actual) > abs(5-actual) else 5.0)
        return -abs(rating - float(gold.strip()))
    return scorer.score(gold.strip(), prediction.strip())["rougeL"].fmeasure


def query_users(data_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    records = json.loads(data_path.read_text())
    ownership = {}
    history = {}
    for record in records:
        user = str(record["user_id"])
        if user in history:
            raise ValueError(f"Duplicate focal user {user}: {data_path}")
        history[user] = len(record["profile"])
        for query in record["query"]:
            question = str(query["id"])
            if question in ownership:
                raise ValueError(f"Duplicate query {question}: {data_path}")
            ownership[question] = user
    return ownership, history


def predictions(path: Path, expected_ids: set[str]) -> dict[str, str]:
    records = json.loads(path.read_text())["golds"]
    result = {str(item["id"]): str(item["output"]) for item in records}
    if len(result) != len(records) or set(result) != expected_ids:
        raise ValueError(f"Predictions do not match focal query IDs: {path}")
    return result


def load_task(task: str) -> tuple[dict, dict[str, int], list[dict]]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    seeds = []
    reference_ownership = None
    reference_history = None
    for seed in range(3):
        run = path_for(task, seed, ROOT / "runs/main-table-3seed", True)
        summary = json.loads((run / "summary.json").read_text())
        data_path = run / "data" / task / "user_top_100_history.json"
        gold_path = run / "data" / task / "user_top_100_history_label.json"
        ownership, history = query_users(data_path)
        golds = {str(row["id"]): str(row["output"])
                 for row in json.loads(gold_path.read_text())["golds"]}
        if set(golds) != set(ownership):
            raise ValueError(f"Golds and focal queries do not match: {run}")
        if reference_ownership is None:
            reference_ownership, reference_history = ownership, history
        elif ownership != reference_ownership or history != reference_history:
            raise ValueError(f"Seed cohorts differ for {task}: {run}")
        stage_scores = {}
        for stage in STAGES:
            path = Path(summary["predictions"][stage]["path"])
            pred = predictions(path, set(ownership))
            score_by_user = {user: [0.0, 0] for user in history}
            for question, user in ownership.items():
                score = score_query(task, pred[question], golds[question], scorer)
                score_by_user[user][0] += score
                score_by_user[user][1] += 1
            summary_metric = primary_metric(task)
            observed = sum(value[0] for value in score_by_user.values()) / len(ownership)
            recorded = summary["predictions"][stage]["metrics"][summary_metric]
            expected = -recorded if task == "product_rating" else recorded
            if abs(observed - expected) > 0.01:
                raise ValueError(f"Per-query scoring differs from summary in {run}/{stage}: "
                                 f"{observed:.6f} vs {expected:.6f}")
            stage_scores[stage] = score_by_user
        seeds.append(stage_scores)
    return reference_ownership, reference_history, seeds


def stage_mean(users: list[str], seeds: list[dict], stage: str) -> float:
    return sum(
        sum(seeds[seed][stage][user][0] for user in users)
        / sum(seeds[seed][stage][user][1] for user in users)
        for seed in range(len(seeds))
    ) / len(seeds)


def paired_ci(users: list[str], seeds: list[dict], newer: str, older: str,
              draws: int, rng: random.Random) -> dict:
    point = stage_mean(users, seeds, newer) - stage_mean(users, seeds, older)
    deltas = []
    for _ in range(draws):
        sample = rng.choices(users, k=len(users))
        deltas.append(stage_mean(sample, seeds, newer) - stage_mean(sample, seeds, older))
    deltas.sort()
    return {"delta": point, "ci_low": deltas[int(0.025 * draws)],
            "ci_high": deltas[min(draws - 1, int(0.975 * draws))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/user-paired-ci")
    parser.add_argument("--report", type=Path, default=ROOT / "paired_bootstrap_long_3seed.md")
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("At least 100 bootstrap draws are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.bootstrap_seed)
    payload = {"draws": args.draws, "bootstrap_seed": args.bootstrap_seed, "tasks": {}}
    per_user = []
    per_user_seed = []
    lines = ["# User-level paired bootstrap on Long-history runs", "",
             "Seven tasks and three training seeds (0, 1, 2) are evaluated on the same focal "
             "users and query IDs. Each bootstrap draw resamples users with "
             "replacement, keeps all their queries, and uses the same sampled "
             "users for both stages and all seeds. The three seed-level "
             "query-weighted metric differences are averaged. Percentile "
             "95% intervals reflect user sampling conditional on these "
             "three seeds, fixed group assignments, and fixed-seed data "
             "shuffling; they are not "
             "five-seed or re-clustered pipeline uncertainty intervals.", "",
             "| Task | Users | Queries | Primary metric | Comparison | Delta | 95% CI |",
             "|---|---:|---:|---|---|---:|---:|"]
    for task in TASKS:
        ownership, history, seeds = load_task(task)
        users = sorted(history)
        metric = ("MAE reduction" if task == "product_rating" else
                  "accuracy" if task in CLASSIFICATION else "ROUGE-L F1")
        comparisons = {}
        for newer, older in COMPARISONS:
            name = f"{newer}-{older}"
            result = paired_ci(users, seeds, newer, older, args.draws, rng)
            comparisons[name] = result
            lines.append(f"| {task} | {len(users)} | {len(ownership)} | {metric} | "
                         f"{name} | {result['delta']:+.4f} | "
                         f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] |")
        payload["tasks"][task] = {"users": len(users), "queries": len(ownership),
                                  "metric": metric, "comparisons": comparisons}
        for user in users:
            row = {"task": task, "user_id": user, "history_length": history[user],
                   "query_count": seeds[0]["global"][user][1]}
            for stage in STAGES:
                row[stage] = sum(seed[stage][user][0] / seed[stage][user][1]
                                 for seed in seeds) / len(seeds)
            row["segment_gain"] = row["group"] - row["global"]
            row["individual_gain"] = row["local"] - row["group"]
            per_user.append(row)
        for seed_index, stage_scores in enumerate(seeds):
            run = path_for(task, seed_index, ROOT / "runs/main-table-3seed", True)
            mapping_path = run / "data" / task / "group_mapping_100.json"
            mapping = {str(item["id"]): item["group"] for item in
                       json.loads(mapping_path.read_text())["mapping"]}
            if set(mapping) != set(users):
                raise ValueError(f"Incomplete segment assignments: {mapping_path}")
            for user in users:
                row = {"task": task, "seed": seed_index, "user_id": user,
                       "history_length": history[user],
                       "segment_assignment": mapping[user],
                       "query_count": stage_scores["global"][user][1]}
                for stage in STAGES:
                    row[stage] = (stage_scores[stage][user][0]
                                  / stage_scores[stage][user][1])
                row["segment_gain"] = row["group"] - row["global"]
                row["individual_gain"] = row["local"] - row["group"]
                per_user_seed.append(row)
    lines += ["", "Scores are recomputed from matched predictions: exact-match "
              "accuracy for classification, negative absolute error for "
              "product ratings (so positive deltas mean lower MAE), and "
              "per-query ROUGE-L F1 for generation. Reported aggregate "
              "ROUGE uses an internal bootstrap median, so small differences "
              "from the main table are expected.",
              "Long-cohort OPPU predictions are not available with matching "
              "query IDs; this table makes paired claims only among PrefixMem stages.",
              "", "Machine-readable results: `runs/user-paired-ci/`.", ""]
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_user[0]))
        writer.writeheader()
        writer.writerows(per_user)
    with (args.output_dir / "per_user_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_user_seed[0]))
        writer.writeheader()
        writer.writerows(per_user_seed)
    (args.output_dir / "paired_ci.json").write_text(json.dumps(payload, indent=2) + "\n")
    args.report.write_text("\n".join(lines))
    print(f"Wrote {args.report} and {args.output_dir / 'paired_ci.json'}")


if __name__ == "__main__":
    main()
