"""Paired bootstrap comparisons over per-example correctness.

Two systems evaluated on the same 750 questions are not two independent
samples. Comparing them by their standard errors throws away the pairing and
badly overstates the uncertainty: most questions are either easy for both
systems or hard for both, and only the disagreements carry information.

The paired bootstrap resamples *questions* (not runs), recomputes the
difference in accuracy on each resample, and reports the distribution of that
difference. A confidence interval containing zero means we cannot distinguish
the systems on this validation set.

    python -m src.stats --report
    python -m src.stats --baseline results/baseline_8shot.jsonl \
                        --system results/eval_lora_r32_seed1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0


def load_correctness(path: Path) -> dict[str, float]:
    """Map question id -> 1.0 if the run got it right."""
    scores = {}
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            scores[record["id"]] = float(record["correct"])
    return scores


def average_over_seeds(paths: list[Path]) -> dict[str, float]:
    """Per-example score averaged across seeds of the same config.

    A question both seeds get right scores 1.0, one seed 0.5, neither 0.0.
    Averaging first means the comparison is against the config's expected
    behaviour rather than against whichever seed happened to run best.
    """
    runs = [load_correctness(p) for p in paths]
    ids = set(runs[0])
    for run in runs[1:]:
        ids &= set(run)
    return {i: float(np.mean([run[i] for run in runs])) for i in ids}


def align(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Require identical question sets. A pairwise test on a silently
    intersected subset is not the test we think we are running — this caught a
    real bug where two runs used validation sets differing by eight examples.
    """
    if set(a) != set(b):
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        raise ValueError(
            f"runs were evaluated on different question sets: "
            f"{only_a} only in baseline, {only_b} only in system"
        )
    ids = sorted(a)
    return (
        np.array([a[i] for i in ids]),
        np.array([b[i] for i in ids]),
    )


def paired_bootstrap(
    baseline: np.ndarray,
    system: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap the difference (system - baseline) by resampling questions."""
    n = len(baseline)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    diffs = system[idx].mean(axis=1) - baseline[idx].mean(axis=1)

    observed = float(system.mean() - baseline.mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled difference lands on
    # the other side of zero from the observed one.
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "baseline_acc": float(baseline.mean()),
        "system_acc": float(system.mean()),
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, 2 * tail)),
        "significant": not (low <= 0.0 <= high),
        "n": n,
    }


def disagreements(baseline: np.ndarray, system: np.ndarray) -> dict:
    """Where the two systems actually differ — the only informative questions."""
    both = int(np.sum((baseline == 1) & (system == 1)))
    neither = int(np.sum((baseline == 0) & (system == 0)))
    baseline_only = int(np.sum((baseline == 1) & (system == 0)))
    system_only = int(np.sum((baseline == 0) & (system == 1)))
    return {
        "both_correct": both,
        "both_wrong": neither,
        "baseline_only": baseline_only,
        "system_only": system_only,
    }


def compare(baseline_path: Path, system_paths: list[Path], label: str) -> dict:
    baseline = load_correctness(baseline_path)
    system = (
        average_over_seeds(system_paths)
        if len(system_paths) > 1
        else load_correctness(system_paths[0])
    )
    a, b = align(baseline, system)
    result = {"label": label, **paired_bootstrap(a, b)}
    if len(system_paths) == 1:
        result["disagreements"] = disagreements(a, b)
    return result


def _format(result: dict) -> str:
    verdict = "significant" if result["significant"] else "not significant"
    return (
        f"{result['label']:22s} {result['system_acc']:.4f}  "
        f"diff {result['difference']:+.4f}  "
        f"95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
        f"p={result['p_value']:.3f}  {verdict}"
    )


def _report() -> None:
    baseline_path = RESULTS_DIR / "baseline_8shot.jsonl"
    ranks = [8, 16, 32]
    seeds = [0, 1]

    print(f"Baseline: 8-shot prompting, {load_correctness(baseline_path).__len__()} examples")
    print(f"Paired bootstrap, {N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}\n")

    results = []

    print("Individual runs vs baseline")
    for rank in ranks:
        for seed in seeds:
            path = RESULTS_DIR / f"eval_lora_r{rank}_seed{seed}.jsonl"
            if not path.exists():
                continue
            result = compare(baseline_path, [path], f"r{rank} seed{seed}")
            results.append(result)
            print("  " + _format(result))

    print("\nSeed-averaged per rank vs baseline")
    rank_results = []
    for rank in ranks:
        paths = [RESULTS_DIR / f"eval_lora_r{rank}_seed{s}.jsonl" for s in seeds]
        paths = [p for p in paths if p.exists()]
        if len(paths) < 2:
            continue
        result = compare(baseline_path, paths, f"rank {rank} (mean)")
        rank_results.append(result)
        results.append(result)
        print("  " + _format(result))

    if rank_results:
        best = max(rank_results, key=lambda r: r["system_acc"])
        print(
            f"\nBest rank by seed-averaged validation accuracy: "
            f"{best['label']} at {best['system_acc']:.4f}"
        )

    out = RESULTS_DIR / "stats.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwritten to {out}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparisons.")
    parser.add_argument("--report", action="store_true", help="Full comparison table.")
    parser.add_argument("--baseline", help="Baseline predictions JSONL.")
    parser.add_argument("--system", nargs="+", help="System predictions JSONL(s).")
    args = parser.parse_args()

    if args.report:
        _report()
        return
    if not (args.baseline and args.system):
        parser.error("pass --report, or both --baseline and --system")

    paths = [Path(p) for p in args.system]
    result = compare(Path(args.baseline), paths, Path(paths[0]).stem)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
