"""GSM8K loading, prompt formatting, and 8-shot prompt construction.

The same text format is used everywhere in this project:

    Question: <question>
    Answer: <chain of thought>
    #### <final numeric answer>

The 8-shot baseline stacks eight fully-worked examples in front of the target
question. The fine-tuned model is evaluated zero-shot on the same template, so
answer extraction is shared between both settings (see `extract_answer`).
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass

from datasets import load_dataset

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"

# Fixed forever, and deliberately independent of the training seed: the
# few-shot exemplars and the validation split must be identical across every
# run, otherwise config selection is comparing different validation sets.
SPLIT_SEED = 0

N_SHOTS = 8
DEFAULT_VAL_SIZE = 750

ANSWER_MARKER = "####"

# Generation stops when the model starts inventing the next question, which is
# how a base model behaves after a few-shot block.
STOP_SEQUENCES = ["\nQuestion:", "\n\nQuestion:"]

_CALCULATOR_RE = re.compile(r"<<[^>]*>>")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Example:
    """One GSM8K item, already split into reasoning and final answer."""

    id: str
    question: str
    reasoning: str
    answer: str


def normalize_number(text: str) -> str:
    """Canonicalize a numeric string so string equality is a fair comparison."""
    text = text.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        value = float(text)
    except ValueError:
        return text
    if value == int(value):
        return str(int(value))
    return str(value)


def parse_solution(solution: str) -> tuple[str, str]:
    """Split a raw GSM8K solution into (reasoning, final answer)."""
    reasoning, _, answer = solution.partition(ANSWER_MARKER)
    reasoning = _CALCULATOR_RE.sub("", reasoning).strip()
    return reasoning, normalize_number(answer)


def extract_answer(generation: str) -> str | None:
    """Pull the predicted final answer out of a model generation.

    Prefers the text after the `####` marker. Falls back to the last number in
    the generation, which is what an unadhering base model usually leaves us.
    Returns None if there is no number at all, which is the format-adherence
    failure case we report as a secondary metric.
    """
    text = generation.split("Question:")[0]
    if ANSWER_MARKER in text:
        tail = text.rsplit(ANSWER_MARKER, 1)[1]
        match = _NUMBER_RE.search(tail)
        if match:
            return normalize_number(match.group())
    matches = _NUMBER_RE.findall(text)
    if matches:
        return normalize_number(matches[-1])
    return None


def is_well_formed(generation: str) -> bool:
    """Secondary metric: did the model produce the `#### <number>` format."""
    text = generation.split("Question:")[0]
    if ANSWER_MARKER not in text:
        return False
    tail = text.rsplit(ANSWER_MARKER, 1)[1]
    return _NUMBER_RE.search(tail) is not None


def load_gsm8k(split: str) -> list[Example]:
    """Load an official GSM8K split as `Example` objects with stable ids."""
    rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    examples = []
    for i, row in enumerate(rows):
        reasoning, answer = parse_solution(row["answer"])
        examples.append(
            Example(
                id=f"{split}-{i}",
                question=row["question"].strip(),
                reasoning=reasoning,
                answer=answer,
            )
        )
    return examples


def build_splits(val_size: int = DEFAULT_VAL_SIZE) -> dict[str, list[Example]]:
    """Carve the official train split into few-shot / validation / train.

    The boundaries deliberately do NOT depend on how many exemplars a given
    prompt actually uses. A zero-shot evaluation still reserves the same eight
    exemplars, so every run — 8-shot baseline and zero-shot fine-tuned model
    alike — is scored on an identical validation set, and no run trains on a
    question another run is evaluated on.

    Callers that want fewer than N_SHOTS demonstrations slice the `fewshot`
    list; they must not change the split.
    """
    train = load_gsm8k("train")
    order = list(range(len(train)))
    random.Random(SPLIT_SEED).shuffle(order)
    shuffled = [train[i] for i in order]
    return {
        "fewshot": shuffled[:N_SHOTS],
        "val": shuffled[N_SHOTS : N_SHOTS + val_size],
        "train": shuffled[N_SHOTS + val_size :],
    }


def format_prompt(question: str) -> str:
    """The part of an example the model is asked to continue."""
    return f"Question: {question}\nAnswer:"


def format_completion(example: Example) -> str:
    """The target continuation, including the `####` final-answer line."""
    return f" {example.reasoning}\n{ANSWER_MARKER} {example.answer}"


def format_example(example: Example) -> str:
    """A fully worked example, used as a few-shot demonstration."""
    return format_prompt(example.question) + format_completion(example)


def build_fewshot_prefix(exemplars: list[Example]) -> str:
    """The static block of demonstrations prepended to every baseline prompt."""
    if not exemplars:
        return ""
    return "\n\n".join(format_example(ex) for ex in exemplars) + "\n\n"


def build_eval_prompt(question: str, fewshot_prefix: str = "") -> str:
    return fewshot_prefix + format_prompt(question)


def build_training_text(example: Example) -> str:
    """Zero-shot text the SFT trainer learns to produce."""
    return format_example(example)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the GSM8K splits.")
    parser.add_argument("--split", default="val", choices=["train", "val", "fewshot", "test"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--shots", type=int, default=N_SHOTS)
    parser.add_argument("--limit", type=int, default=2, help="How many examples to print.")
    args = parser.parse_args()

    splits = build_splits(val_size=args.val_size)
    splits["test"] = load_gsm8k("test")

    print("split sizes:")
    for name, items in splits.items():
        print(f"  {name:8s} {len(items)}")

    prefix = build_fewshot_prefix(splits["fewshot"][: args.shots])
    print(f"\n8-shot prefix: {len(prefix)} chars\n")

    for example in splits[args.split][: args.limit]:
        prompt = build_eval_prompt(example.question, prefix if args.split != "fewshot" else "")
        print("=" * 70)
        print(prompt[-600:] if len(prompt) > 600 else prompt)
        print(f"\n[gold] {example.answer}")
        # Round-trip check: the gold completion must score as correct.
        gold_generation = format_completion(example)
        print(f"[extracted from gold completion] {extract_answer(gold_generation)}")
        print(f"[well formed] {is_well_formed(gold_generation)}")


if __name__ == "__main__":
    _main()
