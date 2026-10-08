"""User-paired Local PrefixMem versus cleaned OPPU on Short/Medium cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from analyze_user_paired_ci import paired_ci, predictions, query_users
from run_history_budget import ROOT


COHORTS = {
    "movie_tagging": ROOT / "runs/movie-history-cohorts",
    "scholarly_title": ROOT / "runs/lamp5-history-cohorts",
}


def score_by_user(task, ownership, golds, pred, scorer):
    scores = {user: [0.0, 0] for user in set(ownership.values())}
    for question, user in ownership.items():
        if task == "movie_tagging":
            value = float(pred[question].strip() == golds[question].strip())
        else:
            value = scorer.score(golds[question].strip(),
                                 pred[question].strip())["rougeL"].fmeasure
        scores[user][0] += value
        scores[user][1] += 1
    return scores


def main():
    from rouge_score import rouge_scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/cohort-oppu-ci")
    parser.add_argument("--report", type=Path, default=ROOT / "paired_bootstrap_cohort_oppu.md")
    args = parser.parse_args()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rng = random.Random(2026)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"draws": args.draws, "tasks": {}}
    per_user = []
    lines = ["# User-Paired PrefixMem versus OPPU in Sparse-History Cohorts", "",
             "Each Short/Medium cohort compares the same focal users and "
             "query IDs. OPPU predictions use the existing post-generation "
             "cleaning rule; raw files remain unchanged. The 95% percentile "
             "CIs resample users (5,000 draws, seed 2026) and retain all "
             "queries per sampled user. These are single-training-seed "
             "cohort comparisons, not three-seed baseline CIs; training "
             "seeds were not synchronized across methods. Movie Tagging "
             "Medium's cleaned file was regenerated from its retained "
             "raw predictions using `clean_oppu_predictions.py`.", "",
             "| Task | Cohort | Users | Queries | Metric | PrefixMem Local | OPPU | Local - OPPU | 95% CI |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for task, root in COHORTS.items():
        payload["tasks"][task] = {}
        for cohort in ("short", "medium"):
            run = root / cohort
            summary = json.loads((run / "summary.json").read_text())
            data = run / "data" / task
            ownership, history = query_users(data / "user_top_100_history.json")
            golds = {str(item["id"]): str(item["output"])
                     for item in json.loads((data / "user_top_100_history_label.json").read_text())["golds"]}
            if set(golds) != set(ownership):
                raise ValueError(f"Unmatched gold/query IDs in {run}")
            local_path = Path(summary["predictions"]["local"]["path"])
            oppu_path = run / "oppu_eval/preds-base-clean.json"
            local = predictions(local_path, set(ownership))
            oppu = predictions(oppu_path, set(ownership))
            local_scores = score_by_user(task, ownership, golds, local, scorer)
            oppu_scores = score_by_user(task, ownership, golds, oppu, scorer)
            users = sorted(history)
            stage_scores = [{"local": local_scores, "oppu": oppu_scores}]
            ci = paired_ci(users, stage_scores, "local", "oppu", args.draws, rng)
            local_mean = sum(value[0] for value in local_scores.values()) / len(ownership)
            oppu_mean = sum(value[0] for value in oppu_scores.values()) / len(ownership)
            metric = "accuracy" if task == "movie_tagging" else "ROUGE-L F1"
            lines.append(f"| {task} | {cohort} | {len(users)} | {len(ownership)} | "
                         f"{metric} | {local_mean:.4f} | {oppu_mean:.4f} | "
                         f"{ci['delta']:+.4f} | [{ci['ci_low']:+.4f}, "
                         f"{ci['ci_high']:+.4f}] |")
            payload["tasks"][task][cohort] = {"users": len(users),
                                                 "queries": len(ownership),
                                                 "local": local_mean, "oppu": oppu_mean,
                                                 "delta_ci": ci,
                                                 "local_path": str(local_path),
                                                 "oppu_path": str(oppu_path)}
            for user in users:
                local_user = local_scores[user][0] / local_scores[user][1]
                oppu_user = oppu_scores[user][0] / oppu_scores[user][1]
                per_user.append({"task": task, "cohort": cohort,
                                 "user_id": user, "history_length": history[user],
                                 "query_count": local_scores[user][1],
                                 "local": local_user, "oppu": oppu_user,
                                 "local_minus_oppu": local_user-oppu_user})
    lines += ["", "LaMP-5 uses per-query ROUGE-L F1 here; its published "
              "aggregate evaluator takes a bootstrap median, so the point "
              "value can differ slightly. OPPU was trained separately for "
              "these cohorts. This comparison does not establish a paired "
              "Long-history OPPU result.", "",
              "Machine-readable output: `runs/cohort-oppu-ci/`.", ""]
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_user[0]))
        writer.writeheader()
        writer.writerows(per_user)
    (args.output_dir / "paired_ci.json").write_text(json.dumps(payload, indent=2) + "\n")
    args.report.write_text("\n".join(lines))
    print(f"Wrote {args.report} and {args.output_dir / 'paired_ci.json'}")


if __name__ == "__main__":
    main()
