"""Build natural short/medium/long focal-user cohorts for history sparsity tests.

The published processed split uses ``user_top_100_history.json`` as the focal
population and ``user_others.json`` for shared training.  This script selects
short and medium focal cohorts from the latter, removes all selected users
from shared training, and writes three independent run directories with the
same file layout expected by ``run_pipeline.py``.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from prepare_data import FILES, read, validate
from utils import name2taskid


def describe(users):
    lengths = [len(user["profile"]) for user in users]
    queries = sum(len(user["query"]) for user in users)
    return {
        "users": len(users),
        "history": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "queries": queries,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def link_json(path, target):
    """Expose a shared, immutable training artifact in a cohort directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target.resolve())


def subset_mapping(mapping_data, ids):
    selected = [row for row in mapping_data["mapping"] if str(row["id"]) in ids]
    counts = Counter(int(row["group"]) for row in selected)
    return {
        **mapping_data,
        "num_records": len(selected),
        "cluster_counts": {
            str(group): int(counts.get(group, 0))
            for group in range(int(mapping_data["n_clusters"]))
        },
        "mapping": selected,
    }


def subset_profile(rows, ids):
    return [row for row in rows if str(row["id"]) in ids]


def build_golds(users, task):
    return {
        "task": name2taskid[task],
        "golds": [
            {"id": query["id"], "output": query["gold"]}
            for user in users
            for query in user["query"]
        ],
    }


def build_cohort(source, output, task, cohort, focal_users, train_users,
                 source_train_mapping, source_focal_mapping, source_train_profiles,
                 source_focal_profiles, prompt_templates, selection,
                 shared_train_paths=None):
    task_output = output / "data" / task
    task_output.mkdir(parents=True, exist_ok=False)

    focal_ids = {str(user["user_id"]) for user in focal_users}
    train_ids = {str(user["user_id"]) for user in train_users}
    data = {
        "user_others.json": train_users,
        "user_top_100_history.json": focal_users,
        "user_top_100_history_label.json": build_golds(focal_users, task),
        "profile_user_others.json": subset_profile(source_train_profiles, train_ids),
        "profile_user_100.json": subset_profile(source_focal_profiles, focal_ids),
        "group_mapping_others.json": subset_mapping(source_train_mapping, train_ids),
        "group_mapping_100.json": subset_mapping(source_focal_mapping, focal_ids),
    }
    train_names = {
        "user_others.json", "profile_user_others.json", "group_mapping_others.json"
    }
    for name, value in data.items():
        if shared_train_paths is not None and name in train_names:
            link_json(task_output / name, shared_train_paths[name])
        else:
            write_json(task_output / name, value)

    stats = validate(task, data, prompt_templates[task])
    cohort_manifest = {
        "task": task,
        "cohort": cohort,
        "selection": selection,
        "source": str(source.resolve()),
        "train": describe(train_users),
        "focal": describe(focal_users),
        "validation": stats,
    }
    write_json(output / "data" / "history_cohort.json", cohort_manifest)
    return cohort_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Parent directory containing short/, medium/, and long/ run directories")
    parser.add_argument("--task_name", choices=sorted(name2taskid), default="movie_tagging")
    parser.add_argument("--short_size", type=int, default=100)
    parser.add_argument("--medium_size", type=int, default=100)
    args = parser.parse_args()
    if args.short_size < 1 or args.medium_size < 1:
        parser.error("cohort sizes must be positive")

    source = args.source.resolve() / args.task_name
    if not source.is_dir():
        parser.error(f"task data not found: {source}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_data = {name: read(source / name) for name in FILES}
    prompt_templates = read(Path(__file__).resolve().parent / "prompt" / "prompt.json")
    # Validate the untouched source before selecting users.
    validate(args.task_name, source_data, prompt_templates[args.task_name])

    all_train_users = source_data["user_others.json"]
    existing_long_users = source_data["user_top_100_history.json"]
    if not existing_long_users:
        raise ValueError("the source focal split is empty")

    ordered = sorted(all_train_users, key=lambda user: (len(user["profile"]), int(user["user_id"])))
    if args.short_size + args.medium_size > len(ordered):
        raise ValueError("requested cohorts exceed available non-focal users")
    short_users = ordered[:args.short_size]
    short_ids = {str(user["user_id"]) for user in short_users}
    medium_pool = ordered
    medium_start = (len(medium_pool) - args.medium_size) // 2
    medium_users = medium_pool[medium_start:medium_start + args.medium_size]
    medium_selection_rule = (
        "central users after sorting the non-focal pool by history length; ties by user_id"
    )
    if short_ids & {str(user["user_id"]) for user in medium_users}:
        medium_pool = [user for user in ordered if str(user["user_id"]) not in short_ids]
        medium_start = (len(medium_pool) - args.medium_size) // 2
        medium_users = medium_pool[medium_start:medium_start + args.medium_size]
        medium_selection_rule = (
            "central users after removing the short cohort and sorting by history length; "
            "ties by user_id"
        )
    medium_ids = {str(user["user_id"]) for user in medium_users}
    if short_ids & medium_ids:
        raise AssertionError("short and medium cohort overlap")

    # Keep the shared training pool identical across the three cohort runs.
    held_out_ids = short_ids | medium_ids
    train_users = [user for user in all_train_users if str(user["user_id"]) not in held_out_ids]
    source_train_profiles = source_data["profile_user_others.json"]
    source_focal_profiles = source_data["profile_user_100.json"]
    source_train_mapping = source_data["group_mapping_others.json"]
    source_focal_mapping = source_data["group_mapping_100.json"]

    # The shared Stage 1/2 input can be over a gigabyte for scholarly_title.
    # Materialize it once and link it from each cohort; the values are still
    # validated independently below, and no cohort writes to these files.
    shared_task_output = output / "_shared" / "data" / args.task_name
    shared_task_output.mkdir(parents=True, exist_ok=True)
    shared_data = {
        "user_others.json": train_users,
        "profile_user_others.json": subset_profile(source_train_profiles, {str(user["user_id"]) for user in train_users}),
        "group_mapping_others.json": subset_mapping(source_train_mapping, {str(user["user_id"]) for user in train_users}),
    }
    shared_train_paths = {}
    for name, value in shared_data.items():
        path = shared_task_output / name
        write_json(path, value)
        shared_train_paths[name] = path

    cohorts = {
        "short": (short_users, {
            "rule": "100 shortest user_others histories; ties by user_id",
            "source_pool": "user_others",
            "rank_start_1based": 1,
            "rank_end_1based": args.short_size,
        }, source_train_mapping, source_train_profiles),
        "medium": (medium_users, {
            "rule": medium_selection_rule,
            "source_pool": "user_others",
            "rank_start_1based": medium_start + 1,
            "rank_end_1based": medium_start + args.medium_size,
        }, source_train_mapping, source_train_profiles),
        "long": (existing_long_users, {
            "rule": "published user_top_100_history cohort",
            "source_pool": "user_top_100_history",
            "rank_start_1based": None,
            "rank_end_1based": None,
        }, source_focal_mapping, source_focal_profiles),
    }

    manifests = {}
    for cohort, (focal_users, selection, focal_mapping, focal_profiles) in cohorts.items():
        run_dir = output / cohort
        manifests[cohort] = build_cohort(
            source=source,
            output=run_dir,
            task=args.task_name,
            cohort=cohort,
            focal_users=focal_users,
            train_users=train_users,
            source_train_mapping=source_train_mapping,
            source_focal_mapping=focal_mapping,
            source_train_profiles=source_train_profiles,
            source_focal_profiles=focal_profiles,
            prompt_templates=prompt_templates,
            selection=selection,
            shared_train_paths=shared_train_paths,
        )

    selection_manifest = {
        "task": args.task_name,
        "source": str(source),
        "shared_training": describe(train_users),
        "held_out_for_focal_cohorts": {
            "users": len(held_out_ids),
            "short_users": sorted(map(int, short_ids)),
            "medium_users": sorted(map(int, medium_ids)),
        },
        "cohorts": {
            cohort: {
                "focal": manifest["focal"],
                "selection": manifest["selection"],
                "path": str((output / cohort).resolve()),
            }
            for cohort, manifest in manifests.items()
        },
    }
    write_json(output / "selection_manifest.json", selection_manifest)
    print(json.dumps(selection_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
