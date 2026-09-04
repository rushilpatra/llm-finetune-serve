# llm-finetune-serve

Does LoRA fine-tuning beat 8-shot prompting on GSM8K for a 0.6B base model,
and what does serving the winner cost under concurrency?

- **Model:** `Qwen/Qwen3-0.6B-Base`
- **Dataset:** GSM8K (`openai/gsm8k`, `main`)
- **Baseline:** base model, 8-shot chain-of-thought prompting
- **Treatment:** LoRA SFT (ranks 8 / 16 / 32, multiple seeds)
- **Serving:** merged adapter, HF `generate()` vs vLLM at concurrency 1/8/32/64

Results tables and plots go here once the runs are done. Every number in this
README is produced by the scripts in `src/`.

## Layout

```
src/data.py        GSM8K loading, prompt format, 8-shot construction
src/train.py       LoRA SFT (config-driven)
src/evaluate.py    generation, answer extraction, scoring
src/stats.py       paired bootstrap, confidence intervals
src/merge.py       merge LoRA adapter into base weights
src/serve.py       FastAPI app
src/benchmark.py   concurrent load testing
configs/           one YAML per training run
notebooks/run.ipynb  thin Colab driver (clone, install, call scripts)
results/           per-example JSONL predictions, metrics, plots
```

## Prompt format

One format is used for the baseline, for training, and for evaluation:

```
Question: <question>
Answer: <chain of thought>
#### <final numeric answer>
```

The 8-shot baseline prepends eight fully worked examples. The fine-tuned model
sees the same template with no demonstrations. Answer extraction reads the text
after `####`, falling back to the last number in the generation.

## Splits

The official train split is shuffled once with a fixed seed (`SPLIT_SEED = 0`,
independent of the training seed) and cut into:

| Split | Size | Use |
|---|---|---|
| few-shot exemplars | 8 | demonstrations for the baseline prompt |
| validation | 750 | rank and seed selection |
| train | remainder (~6.7k) | LoRA SFT |
| test (official) | 1319 | evaluated **once**, on the selected config |

Exemplars are held out from both training and validation.

## Running

Development is local; GPU execution is Colab. Open `notebooks/run.ipynb` in
Colab, point it at this repo, and run the cells — all logic lives in `src/`.

Local smoke test (CPU, no GPU needed):

```bash
pip install datasets
python -m src.data --split val --limit 2
```
