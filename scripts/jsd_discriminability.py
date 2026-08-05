"""Compute static and sequence-aware JSD between two theories on one design.

Usage:
  python scripts/jsd_discriminability.py \
    --theory_1 theories/heuristic_decision_making/ttb_sampling.yaml \
    --theory_2 theories/heuristic_decision_making/wadd_sampling.yaml \
    --experiment design.json --n_runs 300

Prints a JSON dict to stdout: static_jsd, sequence_jsd, and per-trial
profiles for both marginals.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.decision_making_binary_features.experiment import (  # noqa: E402
    DecisionMakingBinaryExperiment,
)
from src.jsd import _per_trial_bernoulli, _per_trial_lag1, choice_matrix, jsd  # noqa: E402
from src.theory import Theory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory_1", required=True)
    parser.add_argument("--theory_2", required=True)
    parser.add_argument("--experiment", required=True,
                        help="JSON file with validities + trial ratings")
    parser.add_argument("--n_runs", type=int, default=300)
    args = parser.parse_args()

    spec = json.loads(Path(args.experiment).read_text())
    t1 = Theory.from_yaml(args.theory_1)
    t2 = Theory.from_yaml(args.theory_2)

    # src.logger prints [INFO] lines to stdout; route them to stderr so
    # stdout stays machine-parseable JSON.
    with contextlib.redirect_stdout(sys.stderr):
        experiment = DecisionMakingBinaryExperiment(**spec)
        m1 = choice_matrix(t1, experiment, n_runs=args.n_runs)
        m2 = choice_matrix(t2, experiment, n_runs=args.n_runs)
    static = [jsd(a, b) for a, b in zip(_per_trial_bernoulli(m1), _per_trial_bernoulli(m2))]
    seq = [jsd(a, b) for a, b in zip(_per_trial_lag1(m1), _per_trial_lag1(m2))]
    print(json.dumps({
        "static_jsd": float(np.mean(static)),
        "sequence_jsd": float(np.mean(seq)),
        "per_trial_static_jsd": static,
        "per_trial_sequence_jsd": seq,
        "n_runs": args.n_runs,
    }, indent=2))


if __name__ == "__main__":
    main()
