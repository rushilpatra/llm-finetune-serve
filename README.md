# LoRA fine-tuning vs 8-shot prompting on GSM8K

Does parameter-efficient fine-tuning beat few-shot prompting for grade-school
math on a small base model — and what does serving the result cost?

**Model:** `Qwen/Qwen3-0.6B-Base` · **Data:** GSM8K · **Adaptation:** LoRA SFT
(ranks 8/16/32 × 2 seeds) · **Serving:** HF `generate()` vs vLLM under concurrency

> **Status.** Training and evaluation are complete. The serving benchmark is in
> progress; that section is marked accordingly and contains no numbers yet.

---

## Headline result

**Fine-tuning did not beat 8-shot prompting on accuracy.** On the held-out test
split it slightly underperformed. It did reliably improve output formatting, and
it removes ~900 prompt tokens from every request.

### Test split (n = 1319, evaluated once)

| System | Exact match | vs baseline | 95% CI | p |
|---|---|---|---|---|
| **8-shot prompting (base model)** | **0.5519** ± 0.0137 | — | — | — |
| LoRA r32, seed 0 (zero-shot) | 0.5383 ± 0.0137 | −0.0136 | [−0.039, +0.014] | 0.345 |
| LoRA r32, seed 1 (zero-shot) | 0.5148 ± 0.0138 | −0.0371 | [−0.064, −0.010] | 0.009 |
| LoRA r32, seed-averaged | 0.5265 | −0.0254 | [−0.049, −0.001] | 0.044 |

The seed-averaged interval clears zero by 0.0008, and the two seeds differ from
each other by as much as the effect. The defensible claim is **"fine-tuning did
not beat prompting, and if anything underperformed it"** — not "fine-tuning is
2.5 points worse."

**Format adherence** (produced a parseable `#### <answer>`): 0.9545 → **0.9913**,
difference +0.0368, CI [+0.025, +0.049]. This is the one large, unambiguous
effect, and it holds on both splits.

**Is 55% a sane number?** The Qwen3 technical report lists **59.59** for
Qwen3-0.6B-Base on GSM8K. Our 8-shot baseline reaching 55.2% under a different
harness, prompt format and answer extractor puts us in the right neighbourhood;
several people have opened issues reporting difficulty reproducing the
published figure exactly, so a few points of gap is unremarkable. It is not a
competitive score against frontier models, which sit in the 90s — but the
absolute number is not what this project measures.

### Validation split (n = 750, used for all config selection)

| System | Exact match | vs baseline | 95% CI | p |
|---|---|---|---|---|
| 8-shot prompting | 0.6240 ± 0.0177 | — | — | — |
| LoRA rank 8 (seed mean) | 0.6180 | −0.0060 | [−0.039, +0.028] | 0.749 |
| LoRA rank 16 (seed mean) | 0.6180 | −0.0060 | [−0.039, +0.028] | 0.746 |
| LoRA rank 32 (seed mean) | 0.6273 | +0.0033 | [−0.029, +0.036] | 0.850 |

No rank differs significantly from the baseline, and rank has no detectable
effect on accuracy.

---

## Three things worth more than the headline

### 1. Rank bought a better fit and no better answers

| Rank | Trainable params | Final train loss | Val accuracy |
|---|---|---|---|
| 8 | 5.0M (0.84%) | 0.513 | 0.6180 |
| 16 | 10.1M (1.67%) | 0.496 | 0.6180 |
| 32 | 20.2M (3.28%) | 0.477 | 0.6273 |

Training loss falls monotonically with capacity while accuracy does not move.
SFT maximises the likelihood of GSM8K's reference solutions; nothing in that
objective rewards being correct. The model got better at sounding like the
dataset.

### 2. The aggregate tie hides substantial churn

On the test split, the baseline and LoRA r32 seed 0 disagree on **322 of 1319
questions (24%)** and trade wins almost evenly:

| | count |
|---|---|
| both correct | 558 |
| both wrong | 439 |
| baseline only | 170 |
| fine-tuned only | 152 |

These are not the same model reaching the same answers. Two summary statistics
would have hidden this entirely; per-example prediction logs are what make it
visible.

### 3. The winner's curse, on schedule

Rank 32 / seed 1 was the **best** config on validation (0.6347, highest of all
six runs) and the **worst** on test (0.5148, −12.0 points). Its sibling seed,
identical configuration, scored 0.6200 on validation — so luck alone moved that
number by 1.5 points, and the maximum of six noisy measurements is high partly
*because* it got lucky.

Config selection therefore used **seed-averaged accuracy per rank**, and both
seeds of the selected rank were evaluated on test. Selecting on single-run
validation maximum would have produced a headline number inflated by ~12 points.

---

## Scope and caveats

**The baseline is a control, not a target.** This project measures a *difference*
between two adaptation methods on one model. Both arms land near 53%; on a model
that scored 90%, both arms would land near 90% and the comparison would be the
same comparison. Nothing in "LoRA SFT and 8-shot prompting are equivalent in
accuracy, and fine-tuning wins on serving cost" depends on 53% being impressive.

**Accuracy near 50% is where binomial variance is largest.** That is part of why
the intervals here are as wide as they are, and part of why an exemplar-choice
effect was large enough to overturn the original finding. A system operating at
90% would produce noticeably tighter intervals for the same sample size, and
would need a smaller effect to reach significance.

**The conclusion is scoped to a model already competent at the task.** Fine-tuning
did not help because the base model already had the capability, and GSM8K's terse
reference solutions were not better reasoning than what it already produced —
evidenced by training loss falling with rank while accuracy did not move. On a
model that genuinely cannot do the task, SFT would very likely help a great deal.
The finding is *"parameter-efficient fine-tuning on a benchmark's own training
set adds little to a model that can already do the benchmark"*, not
*"fine-tuning does not work"*.

---

## Method

### One text format everywhere

```
Question: <question>
Answer: <chain of thought>
#### <final numeric answer>
```

The baseline prepends eight fully worked examples in this format. The fine-tuned
model sees the same template with none. Because the format is identical, a single
answer extractor scores both systems, so the comparison cannot be confounded by
one system's output being easier to parse.

Extraction reads the text after `####`, falling back to the last number in the
generation. The fallback matters: an unadhering base model often trails off
without the marker, and without it we would score format failures as wrong
answers and conflate the two metrics. `well_formed` tracks the marker separately.

### Splits

The official train split is shuffled once with a fixed seed, independent of any
training seed, and cut at fixed boundaries:

| Split | Size | Use |
|---|---|---|
| few-shot exemplars | 8 | demonstrations for the baseline prompt |
| validation | 750 | all rank and seed selection |
| train | 6715 | LoRA SFT |
| test (official) | 1319 | evaluated **once**, on the selected config |

Boundaries do not depend on how many demonstrations a prompt uses. Every system
is scored on an identical validation set, and no run trains on a question another
run is evaluated on.

### Statistics

Two systems answering the same questions are not independent samples: most
questions are easy for both or hard for both, and only disagreements carry
information. Comparing standard errors discards the pairing and overstates
uncertainty.

`src/stats.py` runs a **paired bootstrap** — resample *questions* 10 000 times,
recompute the accuracy difference on each resample, report the distribution. An
interval containing zero means the systems are indistinguishable on this data.

The interval covers uncertainty over which questions are on the exam. It does
**not** cover seed-to-seed training variation, which here is comparable in size —
hence the per-seed rows alongside the seed-averaged ones.

### Training

LoRA SFT via TRL, one YAML per run.

- **Loss on the completion only.** Question tokens are masked out; we are teaching
  the model to answer, not to write GSM8K questions.
- **`alpha = 2r` across all ranks.** LoRA's update scales as `alpha/r`; holding
  the ratio fixed means the rank ablation varies capacity rather than effective
  learning rate.
- **LR 2e-4, 2 epochs, batch 16, cosine schedule.** LoRA tolerates roughly 10× the
  learning rate of full fine-tuning since only the adapter moves.
- Dtype is selected at runtime (`bf16` where supported, `fp16` on Turing).

---

## Two bugs found, and what they cost

### The split-offset bug

`build_splits` originally used the shot count as the offset for the validation
and train slices:

```python
"val": shuffled[n_shots : n_shots + val_size]
```

The 8-shot baseline (`shots: 8`) was scored on `shuffled[8:758]`. The zero-shot
fine-tuned evaluations (`shots: 0`) were scored on `shuffled[0:750]` — **two
validation sets differing by eight questions**, which the paired bootstrap
silently intersected away. Worse, the fine-tuned training set began at index 750
and so contained eight questions from the baseline's validation set.

Eight of 750 is small, but the effect being measured was about one point, so the
contamination was the same size as the finding — and "both systems saw the same
questions" is the assumption the entire comparison rests on.

All six fine-tuned runs were **retrained**, not merely re-scored, since the
adapters had seen validation data. The baseline was unaffected: under corrected
splits its validation set is identical to the one it had already run on.

The real failure was not the off-by-eight. It was that two configurations derived
their splits through the same function with no assertion tying the results
together. `stats.align()` now **refuses** mismatched question sets rather than
intersecting them.

### Two more of the same species

An audit prompted by the first bug found:

- **Resume matched on question id alone.** Re-running an evaluation with a
  different adapter or shot count would have silently kept the previous run's
  generations for ids already on disk. Predictions now carry a fingerprint of
  every field that determines output, and a mismatched re-run stops.
- **`peak_torch_mem_gb` was always 0.0**, because vLLM runs its engine in a
  separate process. It was on track to become a wrong number in this README.

All three are the same failure mode: a silent merge where there should have been
a refusal.

---

## Why fine-tuning did not win

The obvious explanation — that GSM8K's terse reference solutions taught the model
to reason more briefly — **does not hold up**. Measured on test:

- 7% fewer generated tokens (61 vs 65 median)
- identical median reasoning steps (3 vs 3)
- accuracy differences by problem length of +0.000, −0.054, +0.007, +0.000

If reasoning had been damaged, hard problems would suffer most. They did not.

What is left is less dramatic and better supported:

**The base model already knew this.** Qwen3-0.6B-Base was pretrained on far more
mathematics, better written, than GSM8K's 7 000 terse solutions contain. This was
not a blank slate being taught a skill; it was a competent model being nudged with
narrower data than it was built on.

**The objective is not accuracy.** SFT maximises next-token likelihood on
reference text. The rank ablation above is direct evidence: loss down, accuracy
flat.

**8-shot prompting is a strong intervention.** Nine hundred tokens of worked
arithmetic in context before every question is not a formality.

### A note on the validation-to-test drop

Every system fell ~7 points from validation to test. Difficulty mix explains
almost none of it: the splits have near-identical solution lengths (mean 3.59 vs
3.65 steps) and step distributions, and reweighting test accuracy to the
validation difficulty mix moves the baseline only from 0.5519 to 0.5587 — 0.7 of
the 7.2-point gap. The remainder is a genuine distribution shift between GSM8K's
train and test splits that solution length does not capture. Validation numbers
carved from the training pool were optimistic for every system; because the shift
hit all systems similarly, the comparison survives.

---

## Serving benchmark

*In progress — no numbers yet.*

Because accuracy is a tie, the interesting question becomes cost. The baseline
carries ~900 extra prompt tokens on every request; the fine-tuned model does not.
Whether 2.5 points of accuracy is worth that is a serving question, so the
benchmark measures five configurations rather than one:

| Stack | Model | Prefix caching |
|---|---|---|
| HF `generate()` | 8-shot base | — |
| HF `generate()` | fine-tuned, zero-shot | — |
| vLLM | 8-shot base | off |
| vLLM | 8-shot base | on |
| vLLM | fine-tuned, zero-shot | on |

The off/on pair isolates prefix caching specifically: the 8-shot prefix is
byte-identical across requests, so vLLM can compute it once. Without that arm,
any difference would be confounded with continuous batching and PagedAttention.

Metrics: prefill latency, p50 / p95 / max latency, tokens/sec, peak GPU memory,
at concurrency 1 / 8 / 32 / 64. Per-request latencies are written to JSONL
alongside the summary, because summary statistics hide the shape of the
distribution and the shape is the point — static batching should show every
request in a round completing at the same instant, continuous batching should
not.

Whichever system the numbers justify is what gets deployed, with the reasoning
stated — not whichever one the original plan assumed would win.

### Known limitations of this benchmark

Recorded before the runs, alongside the pre-registered predictions.

**Generation length is unrealistic, deliberately.** Every request generates
exactly 256 tokens with EOS ignored, while real generations here run ~63 tokens
at the median. That is roughly 4× more decoding than production would do, and it
is not a neutral choice: prefill is a one-time cost amortised over the decode
steps, while the 8-shot arm pays its ~900-key attention penalty on *every*
decode step. So this setting systematically overweights decode and underweights
prefill — the exact axis the prefix-caching prediction sits on. It is the right
choice for isolating the mechanism and the wrong one for describing production
cost, so a shorter-generation check is run separately where time allows.

Fixed lengths are nonetheless necessary: the fine-tuned model is trained to stop
early, so letting each arm stop naturally would credit it with latency wins
belonging to the model rather than the serving stack, confounding the
prompt-length effect with an output-length effect.

**p95 at n = 64 is the third-worst observation, not a stable percentile.** It is
reported, but tables lead with p50 and max; the max is the more honest statistic
about tail behaviour at this sample size.

**Latency measurement is conservative toward vLLM.** Where the engine reports
per-request completion times they are used; otherwise a request is charged the
completion time of its whole round. That is exact for static batching, where
every request genuinely waits for the slowest, and understates vLLM, which
finishes some requests earlier. Since vLLM is the arm expected to win, whatever
bias remains works against the conclusion rather than for it. Fixed output
lengths also mean sequences within a round are the same length by construction,
so the effect is small.

**TTFT is a prefill-latency proxy**, measured as a separate `max_new_tokens=1`
call rather than a streamed first token, because batched HF `generate()` cannot
report per-sequence first-token times. One definition across both stacks was
worth more than a more faithful measurement on only one of them.

---

## Reproducing

Development is local; GPU execution is Google Colab (browser, or the Colab
extension for VS Code — the kernel is remote either way, so code reaches the GPU
through this repo).

```bash
pip install -r requirements.txt

python -m src.data --split val --limit 2                                    # splits, format
python -m src.evaluate --config configs/eval_baseline_8shot.yaml            # the baseline
python -m src.train    --config configs/lora_r32_seed0.yaml                 # one LoRA run
python -m src.evaluate --config configs/eval_lora.yaml \
    --lora outputs/lora_r32_seed0 --name eval_lora_r32_seed0                # score it
python -m src.stats --report                                                # validation stats
python -m src.stats --report-test                                           # the test comparison
python -m src.merge --adapter outputs/lora_r32_seed0                        # merge for serving
```

Every script takes `--limit N` to run on a tiny subset, and evaluations take
`--dry-run` to exercise prompt construction, the incremental writer, resume and
scoring on CPU without a GPU. `notebooks/run.ipynb` is a thin Colab driver that
clones, installs, and calls these scripts — no logic lives in it.

Evaluations write one JSON line per question (id, generation, extracted answer,
gold answer, correct, well-formed) and flush every 64 examples, so a dead session
resumes rather than restarting. Those files are what the paired bootstrap reads.

## Layout

```
src/data.py        GSM8K loading, prompt format, splits, answer extraction
src/train.py       LoRA SFT, config-driven
src/evaluate.py    vLLM generation, scoring, resumable per-example predictions
src/stats.py       paired bootstrap, confidence intervals
src/merge.py       merge adapter into base weights, with verification
src/benchmark.py   concurrent load testing            (in progress)
src/serve.py       FastAPI app                        (in progress)
configs/           one YAML per run, logged with results
results/           per-example JSONL, metrics, statistics
notebooks/run.ipynb  thin Colab driver
```

## Environment

Requires `transformers>=4.51.0` (older versions raise `KeyError: 'qwen3'`). In
Colab, vLLM must own the torch stack — uninstall the preinstalled
`torch`/`torchvision`/`torchaudio` *and* vLLM itself before installing, or pip
treats vLLM as satisfied and never restores torch. Colab's `torchao` 0.10 must
also be removed; PEFT under transformers 5.x rejects anything below 0.16.

Runs reported here used an A100-40GB: ~5.3 min per training run, ~90 s per
750-question evaluation.
