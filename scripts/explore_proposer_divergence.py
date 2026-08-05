"""Proposer-divergence metrics for one run dir (neutral vs adversarial analysis).

  M7a raw    sequence-JSD between the round's two slot theories on each
             proposer's FIRST proposed design (experiment_attempt_00). Pre-gate:
             isolates the proposer from the shared acceptance machinery.
  M7a excess raw minus a baseline: mean sequence-JSD of k size-matched random
             designs (src.ablations.random_design) for the SAME theory pair.
             Controls for the pair's intrinsic separability. NOTE: the baseline
             samples random_design's validity distribution (uniform 0.55-0.95,
             anchored 0.95/0.55), which is a methodological choice — proposers
             may explore validities outside this range.
  M9         number of experiment_attempt_NN files per proposer per round
             (attempts before the acceptance gate passed).

Correctness properties (pinned by tests/test_explore_proposer_divergence.py):
  * proposers are enumerated from each round's theories.json slot labels —
    never from directory-name patterns;
  * a malformed LLM design is skipped and counted in n_parse_failures, it
    never crashes the analysis;
  * every stochastic quantity is seeded per (round, label, --seed), so results
    are exactly reproducible and each proposal's baseline is independent of
    every other proposal;
  * all JSDs are in nats (bounded by ln 2).

Usage: python scripts/explore_proposer_divergence.py <run_dir> \
           [--n_runs 120] [--k_baseline 5] [--seed 0]
Prints one JSON dict.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ablations import random_design  # noqa: E402
from src.decision_making_binary_features.experiment import (  # noqa: E402
    DecisionMakingBinaryExperiment,
)
from src.jsd import sequence_jsd  # noqa: E402
from src.theory import Theory  # noqa: E402

RESPONSE_JSON_RE = re.compile(r"## Response\s+```json\s+(\{.*?\})\s+```", re.DOTALL)
DESIGN_KEYS = ("validities", "trial_a_ratings", "trial_b_ratings")


def _round_dirs(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rounds").glob("round_*"))


def _slot_entries(round_dir: Path) -> list[dict]:
    """The round's starting_theories entries, in slot order (the authoritative
    list of who proposed this round)."""
    data = json.loads((round_dir / "theories.json").read_text())
    return sorted(data.get("starting_theories", []), key=lambda s: s.get("slot", 0))


def _parse_design(attempt_md: Path) -> DecisionMakingBinaryExperiment | None:
    """Extract the proposed design from a prompt log. Returns None on any
    malformed response (bad JSON, missing fields, inconsistent lengths) —
    LLM output is untrusted input and must never crash the analysis."""
    m = RESPONSE_JSON_RE.search(attempt_md.read_text())
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
        spec = {k: raw[k] for k in DESIGN_KEYS if k in raw}
        if set(DESIGN_KEYS) - set(spec):
            return None
        return DecisionMakingBinaryExperiment(**spec)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
        return None


def _proposal_seed(base_seed: int, round_name: str, label: str) -> int:
    """Stable per-proposal seed: independent of every other proposal, of
    iteration order, and of Python hash randomization."""
    return (zlib.crc32(f"{round_name}/{label}".encode()) ^ base_seed) % 2**31


def _reshuffle_trials(
    design: DecisionMakingBinaryExperiment, *, seed: int
) -> None:
    """Re-apply the constructor's trial shuffle with a seeded Generator.

    DecisionMakingBinaryExperiment shuffles its trial sequence at construction
    with an unseeded `np.random.default_rng()` (experiment.py:236), which no
    global seeding can reach. Trial order is part of the quantity we measure
    (sequence-JSD is order-sensitive for history-dependent theories), so we
    rebuild the sequence exactly as the constructor does — unique pairs ×
    n_repeats, shuffled — but from `seed`. The class itself cannot grow a seed
    field because it doubles as the LLM response schema in the live pipeline."""
    levels = [
        (tuple(a), tuple(b))
        for a, b in zip(design.trial_a_ratings, design.trial_b_ratings)
    ]
    sequence = list(levels) * design._n_repeats
    np.random.default_rng(seed).shuffle(sequence)
    design._trials = sequence
    design._reset_history()


def _seeded_sequence_jsd(
    t1: Theory, t2: Theory, design: DecisionMakingBinaryExperiment,
    *, n_runs: int, seed: int,
) -> float:
    # Three randomness sources must be pinned for exact reproducibility:
    # trial order (constructor shuffle — re-applied seeded here), parameter
    # draws (stdlib `random` in src/sample_parameters.py), and action
    # sampling (numpy legacy global RNG via the theory policy's
    # np.random.choice).
    _reshuffle_trials(design, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    return sequence_jsd(t1, t2, design, n_runs=n_runs)


def _baseline_jsd(
    t1: Theory, t2: Theory, design: DecisionMakingBinaryExperiment,
    *, n_runs: int, k: int, seed: int,
) -> float:
    """Mean theory-separation JSD over k random designs matched to `design`'s
    size (same n_features, n_pairs), for the same theory pair. Seeded from
    `seed` only — independent of all other proposals."""
    rng = np.random.default_rng(seed)
    nf = len(design.validities)
    npairs = len(design.trial_a_ratings)
    vals = []
    for i in range(k):
        base_design = random_design(rng, n_features=nf, n_pairs=npairs)
        vals.append(_seeded_sequence_jsd(
            t1, t2, base_design, n_runs=n_runs, seed=seed + i + 1,
        ))
    return float(np.mean(vals))


def analyse(
    run_dir: Path,
    *,
    n_runs: int,
    k_baseline: int = 5,
    base_seed: int = 0,
    max_round: int | None = None,
) -> dict:
    """Compute per-proposal divergence records for the first `max_round`
    rounds (None = all). One record per proposal — parsed or not — so the
    output cannot misalign. Rounds whose theories.json is missing or does not
    hold exactly 2 slot theories (crashed/partial writes) are skipped and
    reported in `skipped_rounds`."""
    proposals: list[dict] = []
    skipped_rounds: list[str] = []
    for rd in _round_dirs(run_dir)[:max_round]:
        try:
            entries = _slot_entries(rd)
        except FileNotFoundError:
            skipped_rounds.append(rd.name)
            continue
        if len(entries) != 2:
            skipped_rounds.append(rd.name)
            continue
        t1 = Theory.model_validate(entries[0]["theory"])
        t2 = Theory.model_validate(entries[1]["theory"])
        for entry in entries:
            label = entry["label"]
            attempts = sorted((rd / label / "prompts").glob("experiment_attempt_*.md"))
            if not attempts:
                continue  # this slot made no proposal this round
            record: dict = {
                "label": f"{rd.name}/{label}",
                "n_attempts": len(attempts),
                "parsed": False,
                "raw_jsd": None,
                "baseline_jsd": None,
                "excess_jsd": None,
            }
            proposals.append(record)
            design = _parse_design(attempts[0])  # FIRST proposal (pre-gate)
            if design is None:
                continue
            seed = _proposal_seed(base_seed, rd.name, label)
            raw = _seeded_sequence_jsd(t1, t2, design, n_runs=n_runs, seed=seed)
            base = _baseline_jsd(
                t1, t2, design, n_runs=n_runs, k=k_baseline,
                seed=seed + 100_003,  # disjoint from the raw-JSD seed
            )
            record.update(
                parsed=True,
                raw_jsd=round(raw, 6),
                baseline_jsd=round(base, 6),
                excess_jsd=round(raw - base, 6),
            )
    parsed = [p for p in proposals if p["parsed"]]
    attempts_all = [p["n_attempts"] for p in proposals]
    return {
        "run_dir": run_dir.name,
        "n_proposals": len(proposals),
        "n_designs_parsed": len(parsed),
        "n_parse_failures": len(proposals) - len(parsed),
        "skipped_rounds": skipped_rounds,
        "proposals": proposals,
        "M7a_raw_design_jsd_mean": (
            float(np.mean([p["raw_jsd"] for p in parsed])) if parsed else None
        ),
        "M7a_excess_jsd_mean": (
            float(np.mean([p["excess_jsd"] for p in parsed])) if parsed else None
        ),
        "M9_attempts_mean": (
            float(np.mean(attempts_all)) if attempts_all else None
        ),
        "n_runs": n_runs,
        "k_baseline": k_baseline,
        "seed": base_seed,
        "max_round": max_round,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--n_runs", type=int, default=120)
    p.add_argument("--k_baseline", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_round", type=int, default=None,
                   help="Analyse only the first N rounds (matched-budget "
                        "comparison against longer runs).")
    args = p.parse_args()
    print(json.dumps(
        analyse(args.run_dir, n_runs=args.n_runs,
                k_baseline=args.k_baseline, base_seed=args.seed,
                max_round=args.max_round),
        indent=2,
    ))


if __name__ == "__main__":
    main()
