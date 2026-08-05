"""Jensen-Shannon discriminability between two theories on one experiment.

Marginalises each theory's parameters by pooling simulated subjects (one
parameter draw per simulated run, via `Experiment.simulate`), then compares
the two predictive distributions:

* static:   per-trial Bernoulli over the binary response, JSD per trial,
            mean over trials. Ignores sequential structure.
* sequence: per-trial joint distribution over (response_{t-1}, response_t)
            (4 cells; first trial uses the marginal), JSD per trial, mean.
            First-order sequence-awareness — enough to separate
            perseveration/alternating from static heuristics.

All JSDs are in nats, bounded by ln 2. Plug-in estimates are upward-biased
at finite n_runs; comparisons (e.g. against a calibrated threshold) must
use the same n_runs on both sides so the bias cancels.
"""
from __future__ import annotations

import numpy as np

from src.experiment import Experiment
from src.theory import Theory


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD (nats) between two discrete distributions on the same support."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def choice_matrix(theory: Theory, experiment: Experiment, *, n_runs: int) -> np.ndarray:
    """Simulate `n_runs` subjects (one parameter draw each) and return the
    (n_runs, n_trials) binary response matrix in trial order."""
    data = experiment.simulate(theory, n_runs=n_runs)
    rows = [
        subj_df["response"].to_numpy(dtype=int)
        for _, subj_df in data.groupby("subject_id", sort=False)
    ]
    return np.vstack(rows)


def _per_trial_bernoulli(m: np.ndarray) -> np.ndarray:
    """(n_trials, 2) array of [P(0), P(1)] per trial, pooled over runs."""
    p1 = m.mean(axis=0)
    return np.stack([1.0 - p1, p1], axis=1)


def _per_trial_lag1(m: np.ndarray) -> np.ndarray:
    """(n_trials, 4) array; trial 0 is the marginal padded into cells
    [P(0), P(1), 0, 0]; trial t>0 is the joint over
    (response_{t-1}, response_t) in cell order (00, 01, 10, 11)."""
    n_runs, n_trials = m.shape
    out = np.zeros((n_trials, 4))
    out[0, 0] = float((m[:, 0] == 0).mean())
    out[0, 1] = float((m[:, 0] == 1).mean())
    for t in range(1, n_trials):
        idx = 2 * m[:, t - 1] + m[:, t]
        out[t] = np.bincount(idx, minlength=4) / n_runs
    return out


def static_jsd(
    t1: Theory, t2: Theory, experiment: Experiment, *, n_runs: int = 300
) -> float:
    m1 = choice_matrix(t1, experiment, n_runs=n_runs)
    m2 = choice_matrix(t2, experiment, n_runs=n_runs)
    p1, p2 = _per_trial_bernoulli(m1), _per_trial_bernoulli(m2)
    return float(np.mean([jsd(a, b) for a, b in zip(p1, p2)]))


def sequence_jsd(
    t1: Theory, t2: Theory, experiment: Experiment, *, n_runs: int = 300
) -> float:
    m1 = choice_matrix(t1, experiment, n_runs=n_runs)
    m2 = choice_matrix(t2, experiment, n_runs=n_runs)
    p1, p2 = _per_trial_lag1(m1), _per_trial_lag1(m2)
    return float(np.mean([jsd(a, b) for a, b in zip(p1, p2)]))
