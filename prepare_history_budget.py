"""Create an isolated focal-user history-budget variant of a PrefixMem task."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics


FOCAL_FILE = "user_top_100_history.json"


def describe(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "total": sum(values),
    }


def select_history(profile, user_id, retained, sampling, seed):
    if sampling == "prefix":
        return profile[:retained]
    digest = hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    indices = list(range(len(profile)))
    rng.shuffle(indices)
    selected = set(indices[:retained])
    return [item for index, item in enumerate(profile) if index in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Root containing task data directories")
    parser.add_argument("--output", type=Path, required=True, help="Destination data root")
    parser.add_argument("--task_name", default="movie_tagging")
    parser.add_argument("--fraction", type=float, required=True, help="Per-user history fraction in (0, 1]")
    parser.add_argument("--sampling", choices=("prefix", "nested_random"), default="prefix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")

    source_task = args.source.resolve() / args.task_name
    output_task = args.output.resolve() / args.task_name
    if not source_task.is_dir():
        parser.error(f"task data not found: {source_task}")
    if output_task.exists():
        parser.error(f"destination already exists: {output_task}")

    output_task.mkdir(parents=True)
    for source_file in source_task.iterdir():
        if source_file.is_file() and source_file.name != FOCAL_FILE:
            shutil.copy2(source_file, output_task / source_file.name)

    with (source_task / FOCAL_FILE).open(encoding="utf-8") as handle:
        users = json.load(handle)

    original_lengths = []
    retained_lengths = []
    per_user = []
    for user in users:
        original = len(user["profile"])
        retained = max(1, math.ceil(original * args.fraction))
        original_lengths.append(original)
        retained_lengths.append(retained)
        user["profile"] = select_history(user["profile"], user["user_id"],
                                         retained, args.sampling, args.seed)
        per_user.append({
            "user_id": user["user_id"],
            "original_history": original,
            "retained_history": retained,
        })

    with (output_task / FOCAL_FILE).open("x", encoding="utf-8") as handle:
        json.dump(users, handle, ensure_ascii=False)

    manifest = {
        "task": args.task_name,
        "source": str(source_task),
        "fraction": args.fraction,
        "history_order": ("source-order prefix" if args.sampling == "prefix" else
                          "nested random subset, restored to source order"),
        "sampling": args.sampling,
        "sampling_seed": args.seed,
        "segment_assignment": "fixed from source data",
        "users": len(users),
        "original_history": describe(original_lengths),
        "retained_history": describe(retained_lengths),
        "per_user": per_user,
    }
    manifest_path = args.output.resolve() / "history_budget.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps({key: value for key, value in manifest.items() if key != "per_user"}, indent=2))


if __name__ == "__main__":
    main()
