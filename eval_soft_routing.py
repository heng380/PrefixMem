"""Evaluate hard and soft segment routing with latency measurements.

The soft variant composes the segment task adapter and the segment-specific
mediator with weights derived from the saved K-means centroids.  A user's
individual prefix remains the prefix learned for that user's assigned group;
the soft routing experiment isolates the segment-level routing decision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.evaluation import LaMPEvaluation
from utils import (
    extract_citation_title,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
    extract_option,
    merge_lora_and_unload,
    name2taskid,
    split_batch,
)


TASK_EXTRACTORS = {
    "movie_tagging": extract_movie,
    "news_categorize": extract_news_cat,
    "news_headline": extract_news_headline,
    "product_rating": extract_product_review,
    "scholarly_title": extract_scholarly_title,
    "tweet_paraphrase": extract_tweet_paraphrasing,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="/cfs/models/llama/llama3.1-8B")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--cut_off", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature over centroid distances")
    parser.add_argument("--num_groups", type=int, default=5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--output_dir", default="soft_routing")
    parser.add_argument("--mapping_file", default="")
    parser.add_argument("--golds_json", default="")
    parser.add_argument("--limit_users", type=int, default=0,
                        help="Optional user limit for latency smoke checks")
    return parser.parse_args()


class RouteState:
    def __init__(self):
        self.weights = None


class SoftRouteLinear(nn.Module):
    """A frozen linear layer with per-example weighted LoRA deltas."""

    def __init__(self, base: nn.Module, adapters, route_state: RouteState):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleList()
        self.scales = []
        self.route_indices = []
        self.route_state = route_state
        for adapter in adapters:
            holder = nn.Module()
            holder.register_buffer("a", adapter[0], persistent=False)
            holder.register_buffer("b", adapter[1], persistent=False)
            self.adapters.append(holder)
            self.scales.append(float(adapter[2]))
            self.route_indices.append(int(adapter[3]))

    def forward(self, x):
        out = self.base(x)
        weights = self.route_state.weights
        if weights is None:
            return out
        # x is [batch, sequence, hidden] for attention projections.  The same
        # code also handles the 2-D shape used by a few model implementations.
        for index, adapter in enumerate(self.adapters):
            weight = weights[:, self.route_indices[index]]
            if not torch.any(weight != 0):
                continue
            delta = F.linear(F.linear(x, adapter.a), adapter.b)
            view_shape = [weight.shape[0]] + [1] * (delta.ndim - 1)
            out = out + delta * weight.to(delta.dtype).view(*view_shape) * self.scales[index]
        return out


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _adapter_state(path: Path):
    state_path = path / "adapter_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    return torch.load(state_path, map_location="cpu", weights_only=True)


def _adapter_scale(path: Path) -> float:
    config = _load_json(path / "adapter_config.json")
    return float(config["lora_alpha"]) / float(config["r"])


def _module_key(key: str):
    prefix = "base_model.model."
    if not key.startswith(prefix):
        return None
    key = key[len(prefix):]
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[:-len(suffix)], suffix[1:-len(".weight")]
    return None


def _adapter_by_module(path: Path, device):
    state = _adapter_state(path)
    grouped = {}
    for key, value in state.items():
        parsed = _module_key(key)
        if parsed is None:
            continue
        module_name, kind = parsed
        grouped.setdefault(module_name, {})[kind] = value.to(device=device, dtype=torch.bfloat16)
    scale = _adapter_scale(path)
    return grouped, scale


def _replace_module(root: nn.Module, name: str, value: nn.Module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _load_mapping(args):
    mapping_path = Path(args.mapping_file or f"data/{args.task_name}/group_mapping_100.json")
    mapping = _load_json(mapping_path)
    if args.mode == "soft" and ("centroids" not in mapping or "embeddings" not in mapping):
        raise ValueError(
            f"Soft routing requires centroid and embedding metadata in {mapping_path}; "
            "rerun cluster_profiles.py for this run."
        )
    user_to_group = {int(row["id"]): int(row["group"]) for row in mapping.get("mapping", [])}
    if args.mode == "hard":
        return mapping, user_to_group, None
    centroids = np.asarray(mapping["centroids"], dtype=np.float32)
    embeddings = {
        int(row["id"]): np.asarray(row["embedding"], dtype=np.float32)
        for row in mapping["embeddings"]
    }
    if centroids.shape[0] != args.num_groups:
        raise ValueError(f"Expected {args.num_groups} centroids, found {centroids.shape[0]}")
    return mapping, user_to_group, (centroids, embeddings)


def _routing_weights(user_id, user_to_group, geometry, mode, temperature, num_groups):
    if mode == "hard":
        weights = np.zeros(num_groups, dtype=np.float32)
        weights[user_to_group[user_id]] = 1.0
        return weights
    centroids, embeddings = geometry
    vector = embeddings[user_id]
    distances = np.linalg.norm(centroids - vector[None, :], axis=1)
    logits = -distances / max(float(temperature), 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    return (weights / weights.sum()).astype(np.float32)


def _load_model_and_route(args, ckpt_dir: Path):
    device = torch.device(f"cuda:{args.cuda_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=False,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model_short = args.model_name.split("/")[-1]
    task_path = ckpt_dir / f"k0-{args.task_name}-{model_short}-task_LoRA_ckpt"
    if not task_path.exists():
        raise FileNotFoundError(task_path)
    task_model = PeftModel.from_pretrained(base, str(task_path), is_trainable=False)
    model = merge_lora_and_unload(task_model)
    model.eval()
    model.config.use_cache = True

    adapters_by_group = []
    for group_id in range(args.num_groups):
        group_path = ckpt_dir / f"k0-{args.task_name}-{model_short}-group{group_id}-task_LoRA_ckpt"
        local_path = ckpt_dir / f"k0-{args.task_name}-{model_short}-group{group_id}-local-memory-lora"
        if not group_path.exists() or not local_path.exists():
            raise FileNotFoundError(f"Missing K={args.num_groups} adapters for group {group_id}")
        group_state, group_scale = _adapter_by_module(group_path, device)
        local_state, local_scale = _adapter_by_module(local_path, device)
        combined = {}
        for module_name in set(group_state) | set(local_state):
            entries = []
            if module_name in group_state:
                item = group_state[module_name]
                entries.append((item["lora_A"], item["lora_B"], group_scale))
            if module_name in local_state:
                item = local_state[module_name]
                entries.append((item["lora_A"], item["lora_B"], local_scale))
            combined[module_name] = entries
        adapters_by_group.append(combined)

    route_state = RouteState()
    module_names = sorted({name for group in adapters_by_group for name in group})
    named_modules = dict(model.named_modules())
    for module_name in module_names:
        if module_name not in named_modules:
            raise KeyError(f"Adapter module is not present in base model: {module_name}")
        entries = []
        # Store one pair per group and per adapter type.  The route weight for
        # a group is applied to both its task and mediator deltas.
        for group_id, group in enumerate(adapters_by_group):
            entries.extend((a, b, scale, group_id)
                           for a, b, scale in group.get(module_name, []))
        _replace_module(model, module_name, SoftRouteLinear(named_modules[module_name], entries, route_state))
    return model, tokenizer, route_state


def _build_prompt(task_name, query, prompt_template):
    if task_name == "citation":
        title = extract_citation_title(query["input"])
        return prompt_template[task_name]["prompt"].format(
            title, extract_option(query["input"], 1), extract_option(query["input"], 2)
        )
    article = TASK_EXTRACTORS[task_name](query["input"])
    return prompt_template[task_name]["prompt"].format(article)


def _generate_for_user(model, tokenizer, route_state, prompts, weights, prefix, args):
    route_state.weights = torch.tensor(weights, device=model.device, dtype=torch.float32).view(1, -1)
    timings = []
    outputs = []
    for batch in split_batch(prompts, args.batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=args.cut_off, return_token_type_ids=False).to(model.device)
        prefix_batch = prefix.unsqueeze(0).expand(inputs["input_ids"].size(0), -1, -1)
        token_embeds = model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds = torch.cat([prefix_batch.to(token_embeds.dtype), token_embeds], dim=1)
        prefix_mask = torch.ones(
            (inputs["attention_mask"].size(0), prefix.shape[0]),
            dtype=inputs["attention_mask"].dtype, device=model.device,
        )
        attention_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(model.device)
        timings.append(time.perf_counter() - start)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs, timings


def main():
    args = parse_args()
    if args.mode == "soft" and args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    data_dir = Path("data") / args.task_name
    ckpt_dir = Path("ckpt") / args.task_name
    test_data = _load_json(data_dir / "user_top_100_history.json")
    golds_path = Path(args.golds_json or data_dir / "user_top_100_history_label.json")
    golds = _load_json(golds_path)
    gold_by_id = {row["id"]: row["output"] for row in golds["golds"]}
    prompt_template = _load_json(Path("prompt") / "prompt.json")
    mapping, user_to_group, geometry = _load_mapping(args)
    users = test_data[:args.limit_users] if args.limit_users else test_data

    model, tokenizer, route_state = _load_model_and_route(args, ckpt_dir)
    model_short = args.model_name.split("/")[-1]
    payloads = {}
    for group_id in range(args.num_groups):
        path = ckpt_dir / f"k0-{args.task_name}-{model_short}-group{group_id}-local-memory.pt"
        payloads[group_id] = torch.load(path, map_location="cpu", weights_only=True)

    all_predictions = []
    records = []
    total_generation_seconds = 0.0
    total_queries = 0
    for user in tqdm(users, desc=f"{args.mode} routing"):
        user_id = int(user["user_id"])
        group_id = user_to_group[user_id]
        payload = payloads[group_id]
        user_index = payload["user_id_to_index"].get(str(user_id), payload["user_id_to_index"].get(user_id))
        if user_index is None:
            raise KeyError(f"Missing prefix for user {user_id} in group {group_id}")
        prefix = payload["user_embedding"]["weight"][int(user_index)].view(
            int(payload["prefix_len"]), int(payload["hidden_size"])
        ).to(model.device, dtype=torch.bfloat16)
        weights = _routing_weights(user_id, user_to_group, geometry, args.mode,
                                   args.temperature, args.num_groups)
        prompts = [_build_prompt(args.task_name, query, prompt_template) for query in user["query"]]
        prompts = ["##INSTRUCTION:\n" + prompt for prompt in prompts]
        outputs, timings = _generate_for_user(model, tokenizer, route_state, prompts, weights, prefix, args)
        total_generation_seconds += sum(timings)
        total_queries += len(outputs)
        for query, output in zip(user["query"], outputs):
            if args.task_name == "citation":
                stripped = output.strip()
                if stripped == "1":
                    output = "[1]"
                elif stripped == "2":
                    output = "[2]"
            all_predictions.append({"id": query["id"], "output": output})
            records.append({
                "id": query["id"], "user_id": user_id, "hard_group": group_id,
                "routing_weights": weights.tolist(), "output": output,
                "gold": gold_by_id.get(query["id"], ""),
            })

    output_root = Path(args.output_dir) / args.task_name / args.mode
    output_root.mkdir(parents=True, exist_ok=True)
    pred_path = output_root / f"predictions-{args.task_name}-{args.mode}.json"
    pred_path.write_text(json.dumps({"task": name2taskid[args.task_name], "golds": all_predictions,
                                     "model": args.model_name}, ensure_ascii=False, indent=2) + "\n")
    (output_root / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    eval_golds_path = golds_path
    if len(all_predictions) != len(golds["golds"]):
        # Keep latency smoke checks useful without pretending that a partial
        # user slice is a full benchmark evaluation.
        predicted_ids = {row["id"] for row in all_predictions}
        subset = {"task": golds.get("task", name2taskid[args.task_name]),
                  "golds": [row for row in golds["golds"] if row["id"] in predicted_ids]}
        eval_golds_path = output_root / "golds_subset.json"
        eval_golds_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n")
    evaluator = LaMPEvaluation(single_gold_json_file_addr=str(eval_golds_path))
    metrics = evaluator.evaluate_task(str(pred_path), name2taskid[args.task_name])
    summary = {
        "task": args.task_name, "mode": args.mode, "num_groups": args.num_groups,
        "temperature": args.temperature if args.mode == "soft" else None,
        "queries": total_queries, "users": len(users),
        "generation_seconds": total_generation_seconds,
        "generation_ms_per_query": 1000.0 * total_generation_seconds / max(total_queries, 1),
        "active_segment_modules": args.num_groups if args.mode == "soft" else 1,
        "metrics": metrics, "prediction_file": str(pred_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
