"""Generation, answer extraction, and scoring for GSM8K evaluations.

All real generation goes through vLLM: evaluation is the dominant compute cost
in this project, because 8-shot prompts are long and we run them many times.

Predictions are written to `results/<run>.jsonl` incrementally, one line per
example, so a Colab session that dies halfway through resumes instead of
restarting. The paired bootstrap in `stats.py` reads those files later.

    python -m src.evaluate --config configs/eval_baseline_8shot.yaml
    python -m src.evaluate --config configs/eval_baseline_8shot.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src import data

RESULTS_DIR = Path("results")
CHUNK_SIZE = 64  # Predictions are flushed to disk after every chunk.

DEFAULTS = {
    "name": "unnamed",
    "model": "Qwen/Qwen3-0.6B-Base",
    "lora": None,          # optional path to a LoRA adapter directory
    "max_lora_rank": 32,
    "split": "val",        # val | test
    "shots": data.N_SHOTS,
    "val_size": data.DEFAULT_VAL_SIZE,
    "max_new_tokens": 400,
    "temperature": 0.0,    # greedy: exact-match scoring should not be noisy
    "seed": 0,
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.85,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_dtype() -> str:
    """Pick a dtype at runtime — the assigned Colab GPU varies.

    T4 (Turing) has no bf16 support, so hardcoding bfloat16 crashes there.
    """
    if not torch.cuda.is_available():
        return "auto"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def load_config(path: str, overrides: dict) -> dict:
    config = dict(DEFAULTS)
    if path:
        with open(path) as f:
            config.update(yaml.safe_load(f) or {})
    config.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return config


def standard_error(p: float, n: int) -> float:
    """Binomial standard error. Every accuracy number gets an uncertainty."""
    if n == 0:
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def load_existing(path: Path) -> dict[str, dict]:
    """Read predictions already on disk, dropping any truncated final line."""
    if not path.exists():
        return {}
    records = {}
    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # crashed mid-write; the example is simply redone
            records[record["id"]] = record
    # Rewrite cleanly so appending cannot land after a partial line.
    with path.open("w") as f:
        for record in records.values():
            f.write(json.dumps(record) + "\n")
    return records


def score_one(example: data.Example, generation: str) -> dict:
    predicted = data.extract_answer(generation)
    return {
        "id": example.id,
        "generation": generation,
        "predicted": predicted,
        "gold": example.answer,
        "correct": predicted is not None and predicted == example.answer,
        "well_formed": data.is_well_formed(generation),
    }


def summarize(records: list[dict], config: dict, extra: dict) -> dict:
    n = len(records)
    accuracy = sum(r["correct"] for r in records) / n if n else float("nan")
    adherence = sum(r["well_formed"] for r in records) / n if n else float("nan")
    return {
        "run": config["name"],
        "n": n,
        "exact_match": accuracy,
        "exact_match_stderr": standard_error(accuracy, n),
        "format_adherence": adherence,
        "format_adherence_stderr": standard_error(adherence, n),
        "config": config,
        **extra,
    }


def build_engine(config: dict):
    """Construct the vLLM engine once. Imported lazily so --dry-run works on CPU.

    Building this per chunk would pay ~40s of startup twelve times over a
    750-example run, and would tear the engine down before memory is measured.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=config["model"],
        dtype=select_dtype(),
        max_model_len=config["max_model_len"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        seed=config["seed"],
    )
    if config["lora"]:
        llm_kwargs.update(enable_lora=True, max_lora_rank=config["max_lora_rank"])
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["max_new_tokens"],
        stop=data.STOP_SEQUENCES,
    )
    lora_request = (
        LoRARequest("adapter", 1, config["lora"]) if config["lora"] else None
    )
    return llm, sampling, lora_request


def generate(engine, prompts: list[str]) -> list[str]:
    llm, sampling, lora_request = engine
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    return [output.outputs[0].text for output in outputs]


def gpu_report() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        # vLLM may allocate outside this process, so report both views.
        "peak_torch_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu_mem_used_gb": (total - free) / 1e9,
    }


def run(config: dict, limit: int | None, out: Path, dry_run: bool) -> dict:
    set_seed(config["seed"])

    splits = data.build_splits(val_size=config["val_size"])
    examples = (
        data.load_gsm8k("test") if config["split"] == "test" else splits[config["split"]]
    )
    if limit:
        examples = examples[:limit]
    # `shots` selects how many demonstrations to show, never the split.
    prefix = data.build_fewshot_prefix(splits["fewshot"][: config["shots"]])

    done = load_existing(out)
    todo = [ex for ex in examples if ex.id not in done]
    print(f"{len(examples)} examples, {len(done)} already done, {len(todo)} to generate")

    started = time.time()
    # Nothing to do on a full resume, so do not pay engine startup for it.
    engine = build_engine(config) if todo and not dry_run else None

    with out.open("a") as f:
        for i in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[i : i + CHUNK_SIZE]
            prompts = [data.build_eval_prompt(ex.question, prefix) for ex in chunk]
            if dry_run:
                # Echo the gold completion: exercises prompt building, the
                # writer, resume, and scoring without touching a GPU. Accuracy
                # is 1.0 by construction and means nothing.
                generations = [data.format_completion(ex) for ex in chunk]
            else:
                generations = generate(engine, prompts)
            for example, generation in zip(chunk, generations):
                record = score_one(example, generation)
                done[example.id] = record
                f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  {min(i + CHUNK_SIZE, len(todo))}/{len(todo)}")

    # Read memory before the engine is released.
    gpu = gpu_report()
    del engine

    records = [done[ex.id] for ex in examples]
    metrics = summarize(
        records,
        config,
        {
            "split": config["split"],
            "dry_run": dry_run,
            "wall_clock_s": time.time() - started,
            **gpu,
        },
    )
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--shots", type=int)
    parser.add_argument("--model")
    parser.add_argument("--lora")
    parser.add_argument("--name", help="Override the run name, and so the results filename.")
    parser.add_argument("--limit", type=int, help="Evaluate the first N examples only.")
    parser.add_argument("--out", help="Override the results path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip vLLM and echo gold completions, to smoke-test the plumbing on CPU.",
    )
    args = parser.parse_args()

    config = load_config(
        args.config,
        {"split": args.split, "shots": args.shots, "model": args.model, "lora": args.lora},
    )

    if args.name:
        config["name"] = args.name
    name = config["name"] + ("-dryrun" if args.dry_run else "")
    out = Path(args.out) if args.out else RESULTS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("!! DRY RUN: generations are gold completions, metrics are meaningless\n")
    if config["split"] == "test" and not args.dry_run:
        print("!! TEST SPLIT: this is the one-shot final evaluation\n")

    metrics = run(config, args.limit, out, args.dry_run)

    print(f"\nrun               {metrics['run']}  ({metrics['split']}, n={metrics['n']})")
    print(f"exact match       {metrics['exact_match']:.4f} +/- {metrics['exact_match_stderr']:.4f}")
    print(f"format adherence  {metrics['format_adherence']:.4f} +/- {metrics['format_adherence_stderr']:.4f}")
    print(f"wall clock        {metrics['wall_clock_s']:.1f}s")
    if metrics.get("gpu"):
        print(f"gpu               {metrics['gpu']}  peak {metrics['gpu_mem_used_gb']:.2f} GB")
    print(f"\npredictions  {out}\nmetrics      {out.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    _main()
