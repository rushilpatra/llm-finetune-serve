"""Concurrent load testing: HF `generate()` static batching vs vLLM.

Accuracy came out a tie, so the interesting question is cost. The 8-shot
baseline carries ~900 extra prompt tokens on every request; the fine-tuned model
does not. Five configurations separate the effects:

    HF   | 8-shot base          | static batching
    HF   | fine-tuned zero-shot | static batching
    vLLM | 8-shot base          | prefix caching OFF
    vLLM | 8-shot base          | prefix caching ON
    vLLM | fine-tuned zero-shot | prefix caching ON

The APC off/on pair is what makes any prefix-caching claim attributable; without
it, a difference between stacks is confounded with continuous batching and
PagedAttention. Predictions were pre-registered in
`results/benchmark_predictions.md` before this file was written.

Every request generates exactly `max_new_tokens` — EOS is ignored and stop
strings are disabled on both stacks. Otherwise the fine-tuned model, which is
trained to stop early, would post better latency for a reason that has nothing
to do with the serving stack, and the prompt-length effect we are trying to
isolate would be confounded with an output-length effect. How much sooner it
stops naturally is a model property, already measured during evaluation.

    python -m src.benchmark --verify-adapter outputs/lora_r32_seed0
    python -m src.benchmark --configs hf_8shot --limit 2 \
        --model-override sshleifer/tiny-gpt2 --concurrency 1     # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from src import data
from src.evaluate import RESULTS_DIR, set_seed
from src.train import torch_dtype

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_MERGED = "merged/lora_r32_seed0"
DEFAULT_CONCURRENCY = [1, 8, 32, 64]
DEFAULT_REQUESTS = 64
DEFAULT_MAX_NEW_TOKENS = 256
PROMPT_SEED = 0

CONFIGS = [
    {"name": "hf_8shot", "stack": "hf", "model": "base", "shots": 8},
    {"name": "hf_finetuned", "stack": "hf", "model": "merged", "shots": 0},
    {"name": "vllm_8shot_apc_off", "stack": "vllm", "model": "base", "shots": 8,
     "prefix_caching": False},
    {"name": "vllm_8shot_apc_on", "stack": "vllm", "model": "base", "shots": 8,
     "prefix_caching": True},
    {"name": "vllm_finetuned", "stack": "vllm", "model": "merged", "shots": 0,
     "prefix_caching": True},
]


def build_prompts(n: int, shots: int) -> list[str]:
    """The same questions for every configuration, so arms are comparable."""
    splits = data.build_splits()
    exemplars = data.sample_fewshot(splits, None)
    prefix = data.build_fewshot_prefix(exemplars[:shots])
    test = data.load_gsm8k("test")
    idx = random.Random(PROMPT_SEED).sample(range(len(test)), min(n, len(test)))
    return [data.build_eval_prompt(test[i].question, prefix) for i in idx]


def gpu_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


class HFRunner:
    """HuggingFace `generate()`, static batching, closed-loop.

    All N prompts are present at t=0, padded to the longest, and returned
    together — so every request waits for the slowest generation in the batch.
    That stall is the architectural difference against continuous batching, not
    an artifact of handicapping the baseline.
    """

    label = "HF generate(), static batching"

    def __init__(self, model_path: str, max_new_tokens: int):
        from transformers import AutoTokenizer

        from src.merge import load_causal_lm

        dtype = torch_dtype()
        # Decoder-only batched generation requires left padding, or short
        # prompts continue from pad tokens and the outputs are garbage.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = load_causal_lm(model_path, dtype)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _encode(self, prompts: list[str]):
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        return batch

    def _generate(self, prompts: list[str], max_new_tokens: int) -> int:  # noqa: D401
        batch = self._encode(prompts)
        with torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                # Forces the full length: without it, sequences finishing early
                # would make the token count depend on the model rather than on
                # the work the stack was asked to do.
                min_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = out.shape[1] - batch["input_ids"].shape[1]
        return generated * out.shape[0]

    def prefill(self, prompts: list[str]) -> float:
        """One decode step, as a TTFT proxy — batched generate() cannot report
        per-sequence first-token times."""
        started = time.perf_counter()
        self._generate(prompts, 1)
        return time.perf_counter() - started

    def run(self, prompts: list[str]) -> tuple[float, int, list[float] | None]:
        started = time.perf_counter()
        tokens = self._generate(prompts, self.max_new_tokens)
        elapsed = time.perf_counter() - started
        # Static batching returns the whole batch at once, so every request in
        # the round genuinely completed at the same instant. Nothing is lost.
        return elapsed, tokens, None

    def cache_stats(self) -> dict | None:
        return None

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMRunner:
    """vLLM offline engine: all N prompts submitted at once, continuously batched."""

    def __init__(self, model_path: str, max_new_tokens: int, prefix_caching: bool,
                 max_model_len: int, gpu_memory_utilization: float):
        from vllm import LLM, SamplingParams

        self.label = f"vLLM (prefix caching {'on' if prefix_caching else 'off'})"
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=prefix_caching,
            seed=0,
        )
        self._sampling = SamplingParams
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompts: list[str], max_tokens: int):
        # ignore_eos and no stop strings: identical work per request on both
        # stacks, so latency differences are the stack, not the model.
        params = self._sampling(temperature=0.0, max_tokens=max_tokens,
                                ignore_eos=True)
        return self.llm.generate(prompts, params, use_tqdm=False)

    @staticmethod
    def _per_request_latencies(outputs) -> list[float] | None:
        """Real completion times, when the engine records them.

        Continuous batching finishes requests at different moments; falling back
        to the round time would understate vLLM. If any request is missing
        timing, return None rather than mixing two definitions.
        """
        latencies = []
        for output in outputs:
            metrics = getattr(output, "metrics", None)
            arrival = getattr(metrics, "arrival_time", None)
            finished = getattr(metrics, "finished_time", None)
            if arrival is None or finished is None:
                return None
            latencies.append(finished - arrival)
        return latencies or None

    def prefill(self, prompts: list[str]) -> float:
        started = time.perf_counter()
        self._generate(prompts, 1)
        return time.perf_counter() - started

    def run(self, prompts: list[str]) -> tuple[float, int, list[float] | None]:
        started = time.perf_counter()
        outputs = self._generate(prompts, self.max_new_tokens)
        elapsed = time.perf_counter() - started
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return elapsed, tokens, self._per_request_latencies(outputs)

    def cache_stats(self) -> dict | None:
        """Prefix cache hit rate, if the engine exposes it.

        Without this, a prefix-caching claim is untestable: an APC-on arm that
        never actually hit the cache would look identical to APC-off for reasons
        having nothing to do with the hypothesis.
        """
        try:
            metrics = self.llm.get_metrics()
        except Exception:
            return None
        found = {}
        for metric in metrics:
            name = getattr(metric, "name", "")
            if "prefix_cache" not in name:
                continue
            value = getattr(metric, "value", None)
            if value is None:
                value = getattr(metric, "sum", None)
            if value is not None:
                found[name] = value
        if not found:
            return None
        hits = next((v for k, v in found.items() if "hit" in k), None)
        queries = next((v for k, v in found.items() if "quer" in k), None)
        stats = dict(found)
        if hits is not None and queries:
            stats["hit_rate"] = hits / queries
        return stats

    def close(self) -> None:
        del self.llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def measure(
    runner, prompts: list[str], concurrency: int, requests: int
) -> tuple[dict, list[dict]]:
    """One (config, concurrency) cell: warm up, then time closed-loop rounds.

    Returns the summary and the per-request rows behind it. Summary statistics
    hide the shape of the distribution, and the shape is the point: static
    batching should show every request in a round completing at the same
    instant, continuous batching should not.
    """
    rounds = max(1, math.ceil(requests / concurrency))
    batches = [prompts[i * concurrency : (i + 1) * concurrency] for i in range(rounds)]
    batches = [b for b in batches if b]

    runner.run(batches[0])  # discarded warmup

    prefill_s = runner.prefill(batches[0])

    latencies, per_request, total_tokens = [], [], 0
    started = time.perf_counter()
    for round_index, batch in enumerate(batches):
        elapsed, tokens, reported = runner.run(batch)
        total_tokens += tokens
        # Prefer the engine's own per-request times. Falling back to the round
        # time is exact for static batching and conservative for vLLM, which
        # finishes some requests before the round ends.
        if reported is not None and len(reported) == len(batch):
            round_latencies, source = reported, "per_request"
        else:
            round_latencies, source = [elapsed] * len(batch), "round"
        latencies.extend(round_latencies)
        for i, latency in enumerate(round_latencies):
            per_request.append(
                {
                    "concurrency": concurrency,
                    "round": round_index,
                    "index": i,
                    "latency_s": latency,
                    "round_wall_s": elapsed,
                    "source": source,
                }
            )
    wall = time.perf_counter() - started

    ordered = sorted(latencies)
    summary = {
        "concurrency": concurrency,
        "rounds": len(batches),
        "requests": len(latencies),
        "prefill_s": prefill_s,
        "latency_p50_s": statistics.median(ordered),
        # At n=64 the p95 is the third-worst observation, not a stable
        # percentile. Reported, but the max is the honest tail statistic here.
        "latency_p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "latency_max_s": ordered[-1],
        "latency_source": per_request[0]["source"] if per_request else None,
        "wall_clock_s": wall,
        "output_tokens": total_tokens,
        "output_tokens_per_s": total_tokens / wall if wall else float("nan"),
        "requests_per_s": len(latencies) / wall if wall else float("nan"),
        "gpu_mem_used_gb": gpu_memory_gb(),
        "prefix_cache": runner.cache_stats(),
    }
    return summary, per_request


def verify_weights(adapter_dir: Path, merged_dir: Path) -> dict:
    """Confirm we are about to time the weights we think we are."""
    from src.merge import base_model_of, verify

    result = verify(adapter_dir, merged_dir, base_model_of(adapter_dir), torch_dtype())
    print(f"weight check: max logit diff {result['max_logit_diff']:.5f}, "
          f"argmax agreement {result['argmax_agreement']:.4f}")
    if not result["within_tolerance"]:
        raise SystemExit(
            "merged model does not match base+adapter; benchmarking it would "
            "time the wrong weights"
        )
    print("weight check passed\n")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Serving benchmark.")
    parser.add_argument("--configs", nargs="+", default=[c["name"] for c in CONFIGS])
    parser.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--model-override", help="Use this model for every arm (CPU smoke tests).")
    parser.add_argument("--verify-adapter", help="Check merged weights against this adapter first.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark.json"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke test: this many requests and at most 8 new tokens.",
    )
    args = parser.parse_args()

    set_seed(0)
    requests = args.limit if args.limit else args.requests
    max_new_tokens = min(args.max_new_tokens, 8) if args.limit else args.max_new_tokens

    if args.verify_adapter:
        verify_weights(Path(args.verify_adapter), Path(args.merged))

    selected = [c for c in CONFIGS if c["name"] in args.configs]
    if not selected:
        raise SystemExit(f"no configs matched {args.configs}")

    rows, request_rows = [], []
    for config in selected:
        model_path = args.model_override or (
            BASE_MODEL if config["model"] == "base" else args.merged
        )
        prompts = build_prompts(requests, config["shots"])
        print(f"\n{'=' * 70}\n{config['name']}  ({model_path}, {config['shots']}-shot)\n{'=' * 70}")

        if config["stack"] == "hf":
            runner = HFRunner(model_path, max_new_tokens)
        else:
            runner = VLLMRunner(
                model_path,
                max_new_tokens,
                config["prefix_caching"],
                args.max_model_len,
                args.gpu_memory_utilization,
            )

        try:
            for concurrency in args.concurrency:
                row, per_request = measure(runner, prompts, concurrency, requests)
                for entry in per_request:
                    entry.update(config=config["name"], stack=config["stack"])
                request_rows.extend(per_request)
                row.update(
                    config=config["name"],
                    stack=config["stack"],
                    label=runner.label,
                    model=model_path,
                    shots=config["shots"],
                    max_new_tokens=max_new_tokens,
                )
                rows.append(row)
                cache = row["prefix_cache"] or {}
                rate = cache.get("hit_rate")
                print(
                    f"  c={concurrency:<3d} prefill {row['prefill_s']:6.3f}s  "
                    f"p50 {row['latency_p50_s']:7.3f}s  "
                    f"max {row['latency_max_s']:7.3f}s  "
                    f"(p95 {row['latency_p95_s']:7.3f}s)  "
                    f"{row['output_tokens_per_s']:8.1f} tok/s"
                    + (f"  cache {rate:.3f}" if rate is not None else "")
                )
        finally:
            runner.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")

    raw = out.with_name(out.stem + "_latencies.jsonl")
    with raw.open("w") as f:
        for entry in request_rows:
            f.write(json.dumps(entry) + "\n")
    print(f"\nwritten to {out}\nper-request latencies to {raw}")


if __name__ == "__main__":
    _main()
