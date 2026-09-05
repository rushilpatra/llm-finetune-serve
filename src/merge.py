"""Merge a LoRA adapter into the base weights, producing a standalone model.

Serving a merged model avoids per-request adapter overhead, and lets the
benchmark load it like any other HuggingFace checkpoint.

Merging is arithmetic on weights, and it is easy to get subtly wrong — a dtype
mismatch or a misapplied alpha/r scaling produces a model that loads fine and
answers differently. `--verify` compares logits from the adapter-wrapped model
against the merged one before we trust it.

    python -m src.merge --adapter outputs/lora_r32_seed0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src import data
from src.train import torch_dtype

DEFAULT_OUT = Path("merged")
VERIFY_PROMPTS = 4
# bf16 round-off across a merged matmul is ~1e-2 on logits; anything larger
# means the merge changed the model, not just the numerics.
LOGIT_TOLERANCE = 0.05


def load_causal_lm(path: str, dtype):
    """Load a causal LM across the transformers 4/5 dtype rename.

    Transformers 5 takes `dtype`; 4.x takes `torch_dtype`. Colab runs 5.x, but
    the CPU smoke path on an older local install has to work too.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)


def base_model_of(adapter_dir: Path) -> str:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    return config["base_model_name_or_path"]


def verify(adapter_dir: Path, merged_dir: Path, base_model: str, dtype) -> dict:
    """Check the merged weights reproduce the adapter's outputs."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    splits = data.build_splits()
    prompts = [data.format_prompt(ex.question) for ex in splits["val"][:VERIFY_PROMPTS]]
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}

    base = load_causal_lm(base_model, dtype)
    adapter_model = PeftModel.from_pretrained(base, str(adapter_dir))
    if torch.cuda.is_available():
        adapter_model = adapter_model.cuda()
    adapter_model.eval()
    with torch.no_grad():
        reference = adapter_model(**batch).logits.float()
    del adapter_model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged = load_causal_lm(str(merged_dir), dtype)
    if torch.cuda.is_available():
        merged = merged.cuda()
    merged.eval()
    with torch.no_grad():
        actual = merged(**batch).logits.float()
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    max_diff = (reference - actual).abs().max().item()
    agree = (reference.argmax(-1) == actual.argmax(-1)).float().mean().item()
    return {
        "max_logit_diff": max_diff,
        "argmax_agreement": agree,
        "within_tolerance": max_diff < LOGIT_TOLERANCE,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into base weights.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", help="Defaults to merged/<adapter name>.")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT / adapter_dir.name
    base_model = base_model_of(adapter_dir)
    dtype = torch_dtype()
    print(f"merging {adapter_dir} into {base_model} as {dtype}")

    started = time.time()
    base = load_causal_lm(base_model, dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    # The tokenizer travels with the model: vLLM and the API load one directory.
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))
    elapsed = time.time() - started
    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"merged in {elapsed:.1f}s -> {out_dir}")

    if not args.skip_verify:
        result = verify(adapter_dir, out_dir, base_model, dtype)
        print(f"\nverification against the adapter-wrapped model:")
        print(f"  max logit difference  {result['max_logit_diff']:.5f}")
        print(f"  argmax agreement      {result['argmax_agreement']:.4f}")
        if not result["within_tolerance"]:
            raise SystemExit(
                f"merge changed the model: max logit diff {result['max_logit_diff']:.5f} "
                f"exceeds tolerance {LOGIT_TOLERANCE}"
            )
        print("  merge verified")


if __name__ == "__main__":
    _main()
