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
from src.evaluate import RESULTS_DIR
from src.train import torch_dtype

DEFAULT_OUT = Path("merged")
VERIFY_PROMPTS = 4
VERIFY_NEW_TOKENS = 32


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


def _logits(model_loader, batch) -> "torch.Tensor":
    model = model_loader()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        logits = model(**batch).logits.float().cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _greedy(model_loader, batch, tokenizer, max_new_tokens: int) -> list[str]:
    model = model_loader()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.batch_decode(
        out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def verify(adapter_dir: Path, merged_dir: Path, base_model: str, dtype) -> dict:
    """Check the merged weights reproduce the adapter's behaviour.

    Comparing logits alone cannot distinguish the two failures that matter:
    arithmetic drift from merging in low precision, and a merge that silently
    did not apply and left the plain base model. So the merged model is compared
    against *both* the adapter-wrapped model and the bare base model. If it sits
    closer to the base, the adapter never made it in.

    The pass criterion is behavioural — identical greedy continuations — rather
    than a logit tolerance. Merging in bf16 changes individual logits by
    non-trivial amounts while leaving generated text untouched, so a logit
    threshold would fail correct merges and tell us nothing about the ones that
    matter.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = data.build_splits()
    prompts = [data.format_prompt(ex.question) for ex in splits["val"][:VERIFY_PROMPTS]]
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}

    load_base = lambda: load_causal_lm(base_model, dtype)
    load_adapter = lambda: PeftModel.from_pretrained(load_causal_lm(base_model, dtype), str(adapter_dir))
    load_merged = lambda: load_causal_lm(str(merged_dir), dtype)

    base_logits = _logits(load_base, batch)
    adapter_logits = _logits(load_adapter, batch)
    merged_logits = _logits(load_merged, batch)

    # Left padding means the leading positions predict from pad tokens, where
    # outputs are meaningless and differ arbitrarily between any two models.
    # Including them made every pair look equally different — the bare base and
    # the adapter appeared to disagree on 42% of predictions, which cannot be
    # true of models that agree on 76% of final answers.
    real = batch["attention_mask"].bool().cpu()

    def compare(a, b) -> dict:
        diff = (a - b).abs()[real]
        agree = (a.argmax(-1) == b.argmax(-1))[real].float()
        return {
            "max_logit_diff": diff.max().item(),
            "mean_logit_diff": diff.mean().item(),
            "argmax_agreement": agree.mean().item(),
        }

    vs_adapter = compare(merged_logits, adapter_logits)
    vs_base = compare(merged_logits, base_logits)
    adapter_vs_base = compare(adapter_logits, base_logits)

    # The adapter has to actually do something, or "merged matches adapter" is
    # vacuous — it would also match the base model.
    adapter_is_active = adapter_vs_base["argmax_agreement"] < 0.999
    merge_applied = vs_adapter["mean_logit_diff"] < vs_base["mean_logit_diff"]

    adapter_text = _greedy(load_adapter, batch, tokenizer, VERIFY_NEW_TOKENS)
    merged_text = _greedy(load_merged, batch, tokenizer, VERIFY_NEW_TOKENS)
    identical = sum(a == m for a, m in zip(adapter_text, merged_text))

    return {
        "merged_vs_adapter": vs_adapter,
        "merged_vs_base": vs_base,
        "adapter_vs_base": adapter_vs_base,
        "adapter_is_active": adapter_is_active,
        "merge_applied": merge_applied,
        "greedy_identical": identical,
        "greedy_total": len(prompts),
        "sample_adapter": adapter_text[0],
        "sample_merged": merged_text[0],
        "passed": bool(adapter_is_active and merge_applied and identical == len(prompts)),
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
    # Merge in fp32 and cast afterwards. PEFT computes W + (alpha/r)BA in the
    # model's dtype, and bf16 has ~8 mantissa bits, so merging directly in bf16
    # rounds every merged weight. The model is small enough that fp32 is free.
    base = load_causal_lm(base_model, torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model = model.to(dtype)
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
        print("\nverification")
        for key in ("merged_vs_adapter", "merged_vs_base", "adapter_vs_base"):
            c = result[key]
            print(f"  {key:18s} max {c['max_logit_diff']:8.4f}  "
                  f"mean {c['mean_logit_diff']:7.5f}  "
                  f"argmax agreement {c['argmax_agreement']:.4f}")
        print(f"  adapter is active   {result['adapter_is_active']}")
        print(f"  merge applied       {result['merge_applied']}")
        print(f"  greedy identical    {result['greedy_identical']}/{result['greedy_total']}")
        if not result["passed"]:
            print(f"\n  adapter says: {result['sample_adapter'][:160]!r}")
            print(f"  merged  says: {result['sample_merged'][:160]!r}")
            raise SystemExit("merged model does not reproduce the adapter's behaviour")
        print("  merge verified")

        (RESULTS_DIR / f"merge_{adapter_dir.name}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    _main()
