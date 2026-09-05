# Serving benchmark — predictions, recorded before the code was written

Written before `src/benchmark.py` existed, so the git history shows these were
not adjusted to fit the numbers. Each prediction states what would falsify it.

**Configurations.** {HF `generate()` static batching, vLLM} × {8-shot base,
zero-shot fine-tuned}, plus vLLM 8-shot with automatic prefix caching disabled,
at concurrency 1 / 8 / 32 / 64.

The APC-off arm exists so that any prefix-caching effect is attributable to
prefix caching rather than to the other things that differ between the two
stacks — continuous batching and PagedAttention.

---

### 1. TTFT: the 8-shot penalty mostly disappears with prefix caching on

The 8-shot prefix is byte-identical across every request, so vLLM should compute
it once and reuse it. Expect vLLM-8shot-APC-on prefill latency to approach
vLLM-fine-tuned, with vLLM-8shot-APC-off well above both.

*Falsified if* APC-on prefill stays close to APC-off, or if the gap to the
zero-shot arm remains a large fraction of the APC-off gap.

*Untestable if* the measured prefix cache hit rate on the APC-on arm is low; in
that case caching never engaged and the comparison says nothing. Hit rate is
logged for exactly this reason.

### 2. Decode latency: the 8-shot penalty persists regardless of caching

Prefix caching removes recomputation of the prefix, not attention over it. Every
decode step still attends across ~900 additional keys, so per-output-token time
should stay higher for the 8-shot arms whether APC is on or off.

*Falsified if* per-token decode time for vLLM-8shot-APC-on matches
vLLM-fine-tuned within noise.

### 3. Throughput: direction unknown

Two effects oppose each other. Prefix caching saves prefill work, raising
throughput. But the 8-shot arm's longer sequences occupy more KV cache per
request, so fewer run concurrently at high load, lowering it. I do not know
which dominates at concurrency 64.

*This prediction cannot be falsified* — it is recorded as a genuine unknown so
that whichever way it lands is not retrofitted into a story.

### 4. HF vs vLLM: the gap widens with concurrency

At concurrency 1 the two stacks should be close; both run one sequence. As
concurrency rises, static batching pays the slowest-generation-in-batch stall
while continuous batching does not, so the gap should grow monotonically.

*Falsified if* the ratio is flat across concurrency levels, or if HF is
competitive at 32 and 64.

---

### Measurement definitions

- **HF arm:** static batching, closed-loop — N prompts padded to the longest,
  one `generate()` call, all returning together, every request present at t=0.
  This is the strongest honest HF baseline; serialized requests would hand vLLM
  a factor-of-N win for the trivial reason that we declined to batch.
- **TTFT** is measured as a separate `max_new_tokens=1` call and is a **prefill
  latency proxy**, not a streamed first-token time. Batched HF `generate()`
  cannot report per-sequence first-token times, so this keeps one definition
  across both stacks.
- **Latency** is round completion time. For static batching that is exactly
  right, since every request waits for the slowest. For vLLM it is conservative:
  requests that finish early are not credited. Per-request finish times are
  recorded instead wherever the engine exposes them.
- Greedy decoding, identical prompt set drawn from the test split, one discarded
  warmup round, GPU memory read device-wide because vLLM's engine runs
  out-of-process.
