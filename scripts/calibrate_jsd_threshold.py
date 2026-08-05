"""Calibrate the jsd_metric acceptance threshold.

Computes sequence-aware JSD for the three canonical pairs on N random
designs and writes the distribution + percentile thresholds to
configs/jsd_threshold.json. Choose the percentile that roughly matches the
control condition's experiment-acceptance strictness (default: 50th —
i.e. an LLM-proposed design must beat the median random design)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ablations import random_design
from src.jsd import sequence_jsd
from src.theory import Theory

THEORIES_DIR = Path("theories/heuristic_decision_making")
CANONICAL = ("ttb_sampling", "wadd_sampling", "tallying_sampling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_designs", type=int, default=50)
    parser.add_argument("--n_runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="configs/jsd_threshold.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    theories = {n: Theory.from_yaml(THEORIES_DIR / f"{n}.yaml") for n in CANONICAL}
    jsds: list[float] = []
    for k in range(args.n_designs):
        exp = random_design(rng)
        for n1, n2 in itertools.combinations(CANONICAL, 2):
            jsds.append(sequence_jsd(theories[n1], theories[n2], exp, n_runs=args.n_runs))
        print(f"[{k + 1}/{args.n_designs}] running mean={np.mean(jsds):.4f}")

    percentiles = {f"p{p}": float(np.percentile(jsds, p)) for p in (25, 50, 75, 90)}
    out = {
        "n_designs": args.n_designs,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "pairs": list(itertools.combinations(CANONICAL, 2)),
        "percentiles": percentiles,
        "threshold": percentiles["p50"],
        "all_jsds": jsds,
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"threshold={out['threshold']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
