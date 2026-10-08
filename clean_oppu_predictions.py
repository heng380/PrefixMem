"""Clean already-generated OPPU predictions and recompute benchmark metrics."""
import argparse
import json
from pathlib import Path

from eval.evaluation import LaMPEvaluation
from eval_oppu_adapters import clean_prediction
from utils import name2taskid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--golds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text())
    golds = json.loads(Path(args.golds).read_text())
    cleaned = []
    for item in predictions["golds"]:
        raw = item.get("raw_output", item.get("output", ""))
        cleaned.append({"id": item["id"], "output": clean_prediction(args.task_name, raw)})
    output = {"task": predictions.get("task", name2taskid[args.task_name]),
              "golds": cleaned, "model": predictions.get("model", "")}
    Path(args.output).write_text(json.dumps(output, indent=2))
    metrics = LaMPEvaluation(single_gold_json_file_addr=args.golds).evaluate_task(
        args.output, name2taskid[args.task_name])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
