"""Evaluate saved OPPU adapters on a cohort.

The adapters are trained once with OPPU base.  This evaluator reuses those
adapters and changes only the inference-time context, which implements the
OPPU+RAG and OPPU+PAG columns in the main table.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.evaluation import LaMPEvaluation
from utils import (
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
    get_first_k_tokens,
    name2taskid,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="/cfs/models/llama/llama3.1-8B")
    parser.add_argument("--task_name", default="movie_tagging")
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--golds_json", required=True)
    parser.add_argument("--profile_json", default="")
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--adapter_glob", default="*OPPU_LoRA-{index}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["base", "rag", "pag"], default="base")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--cut_off", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_users", type=int, default=-1)
    return parser.parse_args()


def get_extract(task_name):
    return {
        "citation": extract_citation_title,
        "movie_tagging": extract_movie,
        "news_categorize": extract_news_cat,
        "news_headline": extract_news_headline,
        "product_rating": extract_product_review,
        "scholarly_title": extract_scholarly_title,
        "tweet_paraphrase": extract_tweet_paraphrasing,
    }[task_name]


MOVIE_LABELS = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia",
                "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking",
                "social commentary", "violence", "true story"]


def clean_prediction(task_name, text):
    """Convert free-form causal-LM continuation to the benchmark target format."""
    text = str(text).replace("\r", "").strip()
    # The model sometimes starts a second prompt/solution after answering.
    text = re.split(r"\n\s*##(?:SOLUTION|ANSWER|EXPLANATION|INSTRUCTION)\s*:", text,
                    maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if task_name == "movie_tagging":
        lower = text.lower()
        # Prefer a complete label and preserve the benchmark's canonical spelling.
        matches = [(lower.find(label), label) for label in MOVIE_LABELS if lower.find(label) >= 0]
        if matches:
            return min(matches, key=lambda item: item[0])[1]
        return text.splitlines()[0].strip(" []\"'") if text else ""
    # LaMP-5 is evaluated as text generation. The first non-empty line is the
    # generated title; explanations after it are not part of the prediction.
    for line in text.splitlines():
        line = line.strip().strip('"')
        if line:
            return line
    return ""


def find_adapter(root, pattern, index):
    candidates = sorted(glob.glob(os.path.join(root, pattern.format(index=index))))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one adapter for user index {index}, found: {candidates}")
    return candidates[0]


def build_context(task_name, item, prompt_template, history, mode, k, profile_text):
    values = {key: get_first_k_tokens(value, 768) if isinstance(value, str) else value for key, value in item.items()}
    if task_name == "citation":
        title = extract_citation_title(item["input"])
        prompt = prompt_template[task_name]["prompt"].format(title, extract_option(item["input"], 1), extract_option(item["input"], 2))
    else:
        article = get_extract(task_name)(item["input"])
        prompt = prompt_template[task_name]["prompt"].format(article)
    prompt = "##INSTRUCTION:\n" + prompt

    if mode == "rag" and history:
        history_values = [{key: get_first_k_tokens(value, 368) if isinstance(value, str) else value for key, value in h.items()} for h in history]
        history_text = [prompt_template[task_name]["retrieval_history"].format(**h) for h in history_values]
        bm25 = BM25Okapi([doc.split(" ") for doc in history_text])
        query_text = prompt_template[task_name]["retrieval_query_wokey"].format(get_extract(task_name)(item["input"])).split(" ")
        selected = bm25.get_top_n(query_text, history_text, n=max(1, k))
        prompt = "##USER HISTORY:\n" + "".join(selected) + "\n" + prompt
    if mode == "pag" and profile_text:
        prompt = "##USER PROFILE:\n" + profile_text + "\n" + prompt
    return prompt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", local_files_only=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    base.config.use_cache = True
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id

    with open(args.test_json) as handle:
        users = json.load(handle)
    if args.max_users > 0:
        users = users[:args.max_users]
    with open(args.golds_json) as handle:
        golds = json.load(handle)
    gold_by_id = {item["id"]: item["output"] for item in golds.get("golds", [])}
    with open("prompt/prompt.json") as handle:
        prompt_template = json.load(handle)

    profiles = {}
    if args.mode == "pag":
        if not args.profile_json:
            raise ValueError("--profile_json is required for --mode pag")
        with open(args.profile_json) as handle:
            profiles = {int(item["id"]): item.get("output", "") for item in json.load(handle)}

    predictions = []
    for index, user in enumerate(tqdm(users)):
        adapter_path = find_adapter(args.adapter_root, args.adapter_glob, index)
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        model.eval()
        context = []
        prompts = []
        query_ids = []
        for query in user["query"]:
            prompts.append(build_context(args.task_name, query, prompt_template, user.get("profile", []), args.mode, args.k, profiles.get(int(user["user_id"]), "")))
            query_ids.append(query["id"])
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.cut_off, return_token_type_ids=False).to(model.device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(**inputs, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=args.max_new_tokens)
            decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.size(1):], skip_special_tokens=True)
            for qid, output in zip(query_ids[start:start + args.batch_size], decoded):
                predictions.append({"id": qid, "output": clean_prediction(args.task_name, output), "raw_output": output})
        del model
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"preds-{args.mode}.json"
    path.write_text(json.dumps({"task": name2taskid[args.task_name], "golds": predictions, "model": args.model_name}, indent=2))
    # For smoke tests with --max_users, evaluate only the corresponding gold IDs.
    eval_gold_path = args.golds_json
    if args.max_users > 0:
        pred_ids = {str(item["id"]) for item in predictions}
        filtered = {"task": golds.get("task", name2taskid[args.task_name]), "golds": [item for item in golds.get("golds", []) if str(item["id"]) in pred_ids]}
        eval_gold_path = str(output_dir / "golds_subset.json")
        Path(eval_gold_path).write_text(json.dumps(filtered))
    metrics = LaMPEvaluation(single_gold_json_file_addr=eval_gold_path).evaluate_task(str(path), name2taskid[args.task_name])
    (output_dir / f"summary-{args.mode}.json").write_text(json.dumps({"task": args.task_name, "mode": args.mode, "metrics": metrics, "count": len(predictions), "prediction_path": str(path)}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
