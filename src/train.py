"""LoRA supervised fine-tuning on GSM8K, driven by one YAML per run.

The training set is the GSM8K train split minus the few-shot exemplars and the
validation carve-out (see `data.build_splits`). Each example is fed to TRL as a
prompt/completion pair, so the loss is computed on the chain of thought and the
final answer only, not on the question — the model already knows how to read
questions, and we are teaching it how to answer.

    python -m src.train --config configs/lora_r16_seed0.yaml
    python -m src.train --config configs/lora_r16_seed0.yaml --dry-run --limit 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src import data
from src.evaluate import RESULTS_DIR, set_seed

DEFAULTS = {
    "name": "unnamed",
    "model": "Qwen/Qwen3-0.6B-Base",
    "output_dir": "outputs",
    # LoRA
    "rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # Optimization
    "learning_rate": 2.0e-4,  # LoRA tolerates a much higher LR than full SFT
    "num_train_epochs": 2,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "max_length": 640,
    "logging_steps": 20,
    # Data
    "val_size": data.DEFAULT_VAL_SIZE,
    "shots": data.N_SHOTS,
    "seed": 0,
}


def load_config(path: str) -> dict:
    config = dict(DEFAULTS)
    with open(path) as f:
        config.update(yaml.safe_load(f) or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return config


def torch_dtype() -> torch.dtype:
    """bf16 where supported, fp16 on Turing (T4). The Colab GPU varies."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_dataset(config: dict, limit: int | None):
    """Prompt/completion pairs. TRL masks the prompt out of the loss."""
    from datasets import Dataset

    splits = data.build_splits(val_size=config["val_size"], n_shots=config["shots"])
    examples = splits["train"]
    if limit:
        examples = examples[:limit]
    return Dataset.from_list(
        [
            {
                "prompt": data.format_prompt(ex.question),
                "completion": data.format_completion(ex),
            }
            for ex in examples
        ]
    )


def train(config: dict, dataset, limit: int | None) -> dict:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dtype = torch_dtype()
    output_dir = Path(config["output_dir"]) / config["name"]

    peft_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        max_length=config["max_length"],
        logging_steps=config["logging_steps"],
        seed=config["seed"],
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        # A 0.6B model with a rank-32 adapter fits comfortably; checkpointing
        # would only trade speed for memory we are not short of.
        gradient_checkpointing=False,
        # Loss on the completion only. This is the default for prompt/completion
        # datasets, but the choice matters enough to state it.
        completion_only_loss=True,
        save_strategy="no",  # the adapter is saved once, at the end
        report_to="none",
        model_init_kwargs={"dtype": dtype},
    )

    trainer = SFTTrainer(
        model=config["model"],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    wall_clock = time.time() - started

    trainer.save_model(str(output_dir))
    print(f"adapter saved to {output_dir}")

    metrics = {
        "run": config["name"],
        "n_train": len(dataset),
        "limit": limit,
        "train_loss": result.training_loss,
        "global_steps": result.global_step,
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_s": wall_clock,
        "dtype": str(dtype),
        "adapter_dir": str(output_dir),
        "config": config,
    }
    if torch.cuda.is_available():
        metrics["gpu"] = torch.cuda.get_device_name(0)
        metrics["peak_torch_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    # The trainer's own log history carries the loss curve for the README plot.
    metrics["log_history"] = trainer.state.log_history
    return metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on GSM8K.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Train on the first N examples only.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset and stop, to check formatting on CPU.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    dataset = build_dataset(config, args.limit)
    print(f"{len(dataset)} training examples")

    if args.dry_run:
        example = dataset[0]
        print("\n--- prompt (loss masked) ---")
        print(example["prompt"])
        print("\n--- completion (loss computed) ---")
        print(example["completion"])
        return

    metrics = train(config, dataset, args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{config['name']}.train.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nrun          {metrics['run']}")
    print(f"train loss   {metrics['train_loss']:.4f}  over {metrics['global_steps']} steps")
    print(f"wall clock   {metrics['wall_clock_s']:.1f}s")
    if "gpu" in metrics:
        print(f"gpu          {metrics['gpu']}  peak {metrics['peak_torch_mem_gb']:.2f} GB")
    print(f"\nadapter      {metrics['adapter_dir']}\nmetrics      {out}")


if __name__ == "__main__":
    _main()
