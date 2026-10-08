"""User-level paired stage-gain intervals for natural history cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median

from analyze_user_paired_ci import paired_ci
from run_history_budget import ROOT
from summarize_history_budget import load_scores


COHORTS = {
    "movie_tagging": ROOT / "runs/movie-history-cohorts",
    "scholarly_title": ROOT / "runs/lamp5-history-cohorts",
}


def main():
    from rouge_score import rouge_scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/cohort-stage-ci")
    parser.add_argument("--report", type=Path, default=ROOT / "history_cohort_stage_ci.md")
    args = parser.parse_args()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rng = random.Random(2026)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"draws": args.draws, "tasks": {}}
    per_user = []
    lines = ["# Natural History-Cohort User-Level Stage CIs", "",
             "Short, Medium, and Long use different focal users. Within each "
             "cohort, Stage 1/2/3 predictions have matching query IDs; "
             "bootstrap draws resample users and retain all their queries "
             "(5,000 draws, seed 2026). These are one-seed, within-cohort "
             "stage comparisons. Differences between cohorts are descriptive "
             "and may reflect user composition, not only history length.", "",
             "| Task | Cohort | Users | Median history | Queries | Group - Global (95% CI) | Local - Group (95% CI) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, root in COHORTS.items():
        payload["tasks"][task] = {}
        for cohort in ("short", "medium", "long"):
            run = root / cohort
            ownership, history, result = load_scores(task, run, scorer)
            users = sorted(history)
            scores = result["scores"]
            segment = paired_ci(users, [scores], "group", "global", args.draws, rng)
            individual = paired_ci(users, [scores], "local", "group", args.draws, rng)
            lines.append(f"| {task} | {cohort} | {len(users)} | "
                         f"{median(history.values()):g} | {len(ownership)} | "
                         f"{segment['delta']:+.4f} [{segment['ci_low']:+.4f}, "
                         f"{segment['ci_high']:+.4f}] | "
                         f"{individual['delta']:+.4f} [{individual['ci_low']:+.4f}, "
                         f"{individual['ci_high']:+.4f}] |")
            payload["tasks"][task][cohort] = {
                "users": len(users), "queries": len(ownership),
                "median_history": median(history.values()),
                "segment_gain": segment, "individual_gain": individual,
            }
            mapping_path = run / "data" / task / "group_mapping_100.json"
            mapping = {str(row["id"]): row["group"] for row in
                       json.loads(mapping_path.read_text())["mapping"]}
            if set(mapping) != set(users):
                raise ValueError(f"Incomplete segment mapping: {mapping_path}")
            for user in users:
                row = {"task": task, "cohort": cohort, "seed": 0,
                       "user_id": user, "history_length": history[user],
                       "segment_assignment": mapping[user],
                       "query_count": scores["global"][user][1]}
                for stage in ("global", "group", "local"):
                    row[stage] = scores[stage][user][0] / scores[stage][user][1]
                row["segment_gain"] = row["group"] - row["global"]
                row["individual_gain"] = row["local"] - row["group"]
                per_user.append(row)
    lines += ["", "Movie Tagging reports accuracy; LaMP-5 reports per-query "
              "ROUGE-L F1. LaMP's aggregate evaluator uses an internal "
              "bootstrap median, so small differences from its published "
              "scores are expected. This study does not establish a "
              "causal history-length effect; the controlled budget study "
              "tests that separately.", "",
              "Machine-readable outputs: `runs/cohort-stage-ci/`.", ""]
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_user[0]))
        writer.writeheader()
        writer.writerows(per_user)
    (args.output_dir / "paired_ci.json").write_text(json.dumps(payload, indent=2) + "\n")
    args.report.write_text("\n".join(lines))
    print(f"Wrote {args.report} and {args.output_dir / 'paired_ci.json'}")


if __name__ == "__main__":
    main()
