"""Summarize controlled focal-history budgets and user-level uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET

from analyze_user_paired_ci import predictions, query_users, stage_mean, paired_ci
from run_history_budget import BASE_RUNS, ROOT


FRACTIONS = (25, 50, 75, 100)
STAGES = ("global", "group", "local")


def load_scores(task: str, run: Path, scorer) -> tuple[dict[str, str], dict, dict]:
    summary = json.loads((run / "summary.json").read_text())
    data = run / "data" / task
    ownership, history = query_users(data / "user_top_100_history.json")
    golds = {str(row["id"]): str(row["output"])
             for row in json.loads((data / "user_top_100_history_label.json").read_text())["golds"]}
    if set(golds) != set(ownership):
        raise ValueError(f"Unmatched query and gold IDs: {run}")
    scores = {}
    for stage in STAGES:
        pred = predictions(Path(summary["predictions"][stage]["path"]), set(ownership))
        by_user = {user: [0.0, 0] for user in history}
        for question, user in ownership.items():
            if task == "movie_tagging":
                value = float(pred[question].strip() == golds[question].strip())
            else:
                value = scorer.score(golds[question].strip(),
                                     pred[question].strip())["rougeL"].fmeasure
            by_user[user][0] += value
            by_user[user][1] += 1
        scores[stage] = by_user
    return ownership, history, {"summary": summary, "scores": scores}


def slope(values: dict[str, list[tuple[int, float]]], users: list[str]) -> float:
    numerator = denominator = 0.0
    for user in users:
        pairs = values[user]
        xs = [math.log1p(history) for history, _ in pairs]
        ys = [gain for _, gain in pairs]
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        numerator += sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        denominator += sum((x - xbar) ** 2 for x in xs)
    return numerator / denominator


def percentile_ci(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values)-1, int(0.975 * len(values)))]


def draw_figure(payload: dict, path: Path) -> None:
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                             "width": "960", "height": "410", "viewBox": "0 0 960 410"})

    def add(tag, attrs, label=None):
        element = ET.SubElement(svg, tag, {key: str(value) for key, value in attrs.items()})
        if label is not None:
            element.text = label
        return element

    add("rect", {"x": 0, "y": 0, "width": 960, "height": 410, "fill": "white"})
    series = (("segment_gain", "Segment - Global", "#0f766e"),
              ("individual_gain", "Local - Segment", "#b45309"))
    for panel, (task, data) in enumerate(payload["tasks"].items()):
        left = 72 + panel * 472
        right = left + 338
        top, bottom = 56, 322
        add("text", {"x": left, "y": 28, "font-size": 16,
                     "font-family": "sans-serif", "fill": "#202124"},
            "Movie Tagging" if task == "movie_tagging" else "LaMP-5")
        add("text", {"x": left, "y": 45, "font-size": 11,
                     "font-family": "sans-serif", "fill": "#4b5563"},
            "Incremental gain")
        all_bounds = [0.0]
        for row in data["budgets"].values():
            for key, _, _ in series:
                all_bounds += [row[key]["ci_low"], row[key]["ci_high"]]
        lo, hi = min(all_bounds), max(all_bounds)
        padding = max((hi - lo) * 0.14, 0.01)
        lo, hi = lo - padding, hi + padding

        def xcoord(percent):
            return left + (percent - 25) / 75 * (right - left)

        def ycoord(value):
            return bottom - (value - lo) / (hi - lo) * (bottom - top)

        add("line", {"x1": left, "y1": top, "x2": left, "y2": bottom,
                     "stroke": "#6b7280", "stroke-width": 1})
        add("line", {"x1": left, "y1": bottom, "x2": right, "y2": bottom,
                     "stroke": "#6b7280", "stroke-width": 1})
        zero = ycoord(0)
        add("line", {"x1": left, "y1": zero, "x2": right, "y2": zero,
                     "stroke": "#c9ced3", "stroke-width": 1,
                     "stroke-dasharray": "4 3"})
        for fraction in FRACTIONS:
            x = xcoord(fraction)
            add("line", {"x1": x, "y1": bottom, "x2": x, "y2": bottom+5,
                         "stroke": "#6b7280"})
            add("text", {"x": x, "y": bottom+22, "text-anchor": "middle",
                         "font-size": 12, "font-family": "sans-serif"}, f"{fraction}%")
        for tick in (lo, 0.0, hi):
            y = ycoord(tick)
            add("text", {"x": left-9, "y": y+4, "text-anchor": "end",
                         "font-size": 11, "font-family": "sans-serif",
                         "fill": "#4b5563"}, f"{tick:+.2f}")
        for key, _, color in series:
            points = [(xcoord(fraction), ycoord(data["budgets"][str(fraction)][key]["delta"]))
                      for fraction in FRACTIONS]
            add("polyline", {"points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
                             "fill": "none", "stroke": color, "stroke-width": 2})
            for fraction, (x, y) in zip(FRACTIONS, points):
                result = data["budgets"][str(fraction)][key]
                ylow = ycoord(result["ci_low"])
                yhigh = ycoord(result["ci_high"])
                add("line", {"x1": x, "y1": ylow, "x2": x, "y2": yhigh,
                             "stroke": color, "stroke-width": 1.3})
                for cap in (ylow, yhigh):
                    add("line", {"x1": x-4, "y1": cap, "x2": x+4, "y2": cap,
                                 "stroke": color, "stroke-width": 1.3})
                add("circle", {"cx": x, "cy": y, "r": 4, "fill": color})
        add("text", {"x": (left+right)/2, "y": 374, "text-anchor": "middle",
                     "font-size": 12, "font-family": "sans-serif"}, "History retained")
    for index, (_, label, color) in enumerate(series):
        x = 220 + index * 165
        add("line", {"x1": x, "y1": 22, "x2": x+20, "y2": 22,
                     "stroke": color, "stroke-width": 2})
        add("text", {"x": x+26, "y": 26, "font-size": 11,
                     "font-family": "sans-serif"}, label)
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    from rouge_score import rouge_scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/history-budget-nested-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/history-budget-analysis")
    parser.add_argument("--report", type=Path, default=ROOT / "history_budget_controlled_results.md")
    parser.add_argument("--tasks", nargs="+", choices=tuple(BASE_RUNS), default=list(BASE_RUNS))
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("At least 100 bootstrap draws are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rng = random.Random(2026)
    per_user_rows = []
    payload = {"draws": args.draws, "tasks": {}}
    lines = [
        "# Controlled Focal-History Budget Study", "",
        "Each task uses the same Long focal users at every "
        "budget. Each user's history is shuffled once with seed 0, then "
        "nested 25/50/75/100% subsets are taken and restored to source "
        "order. Retention uses ceil(fraction x history length), at least "
        "one record. K=5 assignments, Stage-1/2 checkpoints and their "
        "predictions are fixed. Only Stage-3 Local memory/mediator is "
        "retrained for 25/50/75%; 100% is the completed K=5 seed-0 "
        "run. These controls differ from the earlier natural Short/Medium/Long "
        "cohorts and should not be pooled with them.", "",
        "Each score below uses the task's primary metric (Movie Tagging "
        "accuracy; LaMP-5 ROUGE-L). Delta and percentile 95% CI are "
        "computed from per-query scores with paired user-cluster resampling "
        "(5,000 draws, seed 2026). One training seed is used; CIs capture "
        "user sampling, not training-seed variation.", "",
        "| Task | History retained | Median records/user | Global | Group | Local | Group - Global (95% CI) | Local - Group (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in args.tasks:
        baseline = BASE_RUNS[task]
        runs = {100: baseline}
        runs.update({percent: args.run_root / task / f"p{percent}"
                     for percent in FRACTIONS if percent != 100})
        loaded = {percent: load_scores(task, run, scorer)
                  for percent, run in runs.items()}
        base_ownership, base_history, base_data = loaded[100]
        users = sorted(base_history)
        mapping_path = baseline / "data" / task / "group_mapping_100.json"
        mapping = {str(item["id"]): item["group"] for item in
                   json.loads(mapping_path.read_text())["mapping"]}
        if set(mapping) != set(users):
            raise ValueError(f"Incomplete segment assignments: {mapping_path}")
        task_rows = {}
        slope_data = {user: [] for user in users}
        segment_ci = paired_ci(users, [base_data["scores"]], "group", "global", args.draws, rng)
        for percent in FRACTIONS:
            ownership, history, data = loaded[percent]
            if ownership != base_ownership:
                raise ValueError(f"Focal queries changed for {task} at {percent}%")
            if percent != 100:
                manifest = json.loads((runs[percent] / "data/history_budget.json").read_text())
                if manifest.get("sampling") != "nested_random" or manifest["fraction"] != percent / 100:
                    raise ValueError(f"Unexpected budget manifest: {runs[percent]}")
            scores = data["scores"]
            for stage in ("global", "group"):
                if scores[stage] != base_data["scores"][stage]:
                    raise ValueError(f"Shared {stage} predictions changed for {task} {percent}%")
            ci = paired_ci(users, [scores], "local", "group", args.draws, rng)
            metric_key = "accuracy" if task == "movie_tagging" else "rouge-L"
            metrics = {}
            for stage in STAGES:
                source_data = data if stage == "local" else base_data
                metrics[stage] = source_data["summary"]["predictions"][stage]["metrics"][metric_key]
            lines.append(
                f"| {task} | {percent}% | {median(history.values()):g} | "
                f"{metrics['global']:.4f} | {metrics['group']:.4f} | "
                f"{metrics['local']:.4f} | {segment_ci['delta']:+.4f} "
                f"[{segment_ci['ci_low']:+.4f}, {segment_ci['ci_high']:+.4f}] | "
                f"{ci['delta']:+.4f} "
                f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] |"
            )
            task_rows[str(percent)] = {"users": len(users), "queries": len(ownership),
                                       "median_retained_history": median(history.values()),
                                       "metrics": metrics, "segment_gain": segment_ci,
                                       "individual_gain": ci,
                                       "summary": str(runs[percent] / "summary.json")}
            for user in users:
                user_score = {stage: scores[stage][user][0] / scores[stage][user][1]
                              for stage in STAGES}
                gain = user_score["local"] - user_score["group"]
                slope_data[user].append((history[user], gain))
                per_user_rows.append({"task": task, "seed": 0, "user_id": user,
                                      "segment_assignment": mapping[user],
                                      "budget_percent": percent,
                                      "original_history": base_history[user],
                                      "retained_history": history[user],
                                      "query_count": scores["local"][user][1],
                                      **user_score, "segment_gain": user_score["group"]-user_score["global"],
                                      "individual_gain": gain})
        point_slope = slope(slope_data, users)
        boot_slopes = [slope(slope_data, rng.choices(users, k=len(users)))
                       for _ in range(args.draws)]
        low, high = percentile_ci(boot_slopes)
        payload["tasks"][task] = {"budgets": task_rows,
                                  "within_user_slope_per_log_record": point_slope,
                                  "slope_ci": [low, high]}
    lines += ["", "## Within-user history slope", "",
              "For each user, we regress the Local-minus-Group per-user "
              "query mean on log(1 + retained history count) across the "
              "four nested budgets, then pool the within-user centered "
              "cross-products. Bootstrap CIs resample users. This is an "
              "association within the controlled budgets, not a guarantee "
              "of monotonic gain for every user.", "",
              "| Task | Slope per log(1 + history) | 95% CI |",
              "|---|---:|---:|"]
    for task, data in payload["tasks"].items():
        low, high = data["slope_ci"]
        lines.append(f"| {task} | {data['within_user_slope_per_log_record']:+.4f} | "
                     f"[{low:+.4f}, {high:+.4f}] |")
    figure_path = args.output_dir / "history_budget_gains.svg"
    draw_figure(payload, figure_path)
    figure_link = os.path.relpath(figure_path.resolve(), args.report.resolve().parent)
    lines += ["", f"[Incremental gain curves with user-level 95% CI]({figure_link})",
              "", "LaMP-5 aggregate ROUGE uses its evaluator's internal "
              "bootstrap median, whereas paired CIs use per-query ROUGE-L "
              "F1; small point-estimate differences are expected. "
              "The fixed Global/Group columns reuse the 100% aggregate "
              "scores, because the evaluator's bootstrap aggregation can "
              "vary slightly even for identical predictions. The retained "
              "subsets do not re-cluster users. The 100% "
              "baseline is the newer K-sensitivity run with the current "
              "Stage-3 learning rate, not the earlier legacy Long result.",
              "", f"Machine-readable outputs: `{args.output_dir}`.", ""]
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_user_rows[0]))
        writer.writeheader()
        writer.writerows(per_user_rows)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    args.report.write_text("\n".join(lines))
    print(f"Wrote {args.report} and {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
