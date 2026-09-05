"""FastAPI wrapper around the served model.

The benchmark decided what to deploy: the merged fine-tuned model, served
zero-shot on vLLM. It matches 8-shot prompting on accuracy, produces a
parseable answer 99% of the time against 95%, and sustains ~2x the throughput
because it carries no ~900-token demonstration prefix.

The service returns the *extracted numeric answer*, not just raw text. Answer
extraction is the same code path the evaluation used, so what a caller receives
is exactly what was scored — a service whose parsing disagreed with the
benchmark's would be reporting accuracy it does not have.

    MODEL_PATH=merged/lora_r32_seed0 uvicorn src.serve:app --port 8000
    BACKEND=hf MODEL_PATH=sshleifer/tiny-gpt2 uvicorn src.serve:app  # CPU
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import data

MODEL_PATH = os.environ.get("MODEL_PATH", "merged/lora_r32_seed0")
BACKEND = os.environ.get("BACKEND", "vllm")
# 0 for the fine-tuned model, which no longer needs demonstrations. Set to 8 to
# serve the base model with the 8-shot prompt instead.
SHOTS = int(os.environ.get("SHOTS", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))

state: dict = {}


class Question(BaseModel):
    question: str = Field(..., min_length=1, description="A grade-school math word problem.")
    max_tokens: int | None = Field(None, ge=1, le=1024)


class Answer(BaseModel):
    answer: str | None = Field(..., description="The extracted numeric answer, null if none was produced.")
    reasoning: str = Field(..., description="The model's full chain of thought.")
    well_formed: bool = Field(..., description="Whether the model emitted a parseable '#### <answer>'.")
    latency_ms: float
    prompt_tokens: int | None = None


class VLLMBackend:
    name = "vllm"

    def __init__(self) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            # The 8-shot prefix, if used, is identical on every request.
            enable_prefix_caching=True,
        )
        self.sampling = SamplingParams

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        params = self.sampling(
            temperature=0.0, max_tokens=max_tokens, stop=data.STOP_SEQUENCES
        )
        output = self.llm.generate([prompt], params, use_tqdm=False)[0]
        return output.outputs[0].text, len(output.prompt_token_ids)


class HFBackend:
    """Reference backend, and the only one testable without a GPU."""

    name = "hf"

    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from src.merge import load_causal_lm
        from src.train import torch_dtype

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = load_causal_lm(MODEL_PATH, torch_dtype())
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        batch = self.tokenizer(prompt, return_tensors="pt")
        if self.torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        with self.torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = batch["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True), prompt_len


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once at startup: the benchmark showed engine construction costs
    # tens of seconds, which is not something to pay per request.
    state["backend"] = VLLMBackend() if BACKEND == "vllm" else HFBackend()
    # The deployed configuration is zero-shot, so there is nothing to build and
    # no reason for the container to download GSM8K at startup. Only a
    # deliberately 8-shot deployment pays that cost.
    if SHOTS:
        splits = data.build_splits()
        exemplars = data.sample_fewshot(splits, None)
        state["prefix"] = data.build_fewshot_prefix(exemplars[:SHOTS])
    else:
        state["prefix"] = ""
    yield
    state.clear()


app = FastAPI(
    title="GSM8K solver",
    description="Qwen3-0.6B, LoRA fine-tuned on GSM8K, served zero-shot.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    ready = "backend" in state
    return {
        "status": "ok" if ready else "starting",
        "model": MODEL_PATH,
        "backend": state["backend"].name if ready else BACKEND,
        "shots": SHOTS,
    }


@app.post("/solve", response_model=Answer)
def solve(request: Question) -> Answer:
    if "backend" not in state:
        raise HTTPException(status_code=503, detail="model still loading")

    prompt = data.build_eval_prompt(request.question.strip(), state["prefix"])
    started = time.perf_counter()
    try:
        text, prompt_tokens = state["backend"].generate(
            prompt, request.max_tokens or MAX_TOKENS
        )
    except Exception as exc:  # surface the failure rather than a bare 500
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    return Answer(
        answer=data.extract_answer(text),
        reasoning=text.strip(),
        well_formed=data.is_well_formed(text),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
    )
