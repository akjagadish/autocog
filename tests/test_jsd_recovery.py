"""Tests for sequence-aware JSD mechanism-recovery scoring.

The premise the whole analysis rests on: the Hilbig evaluation design, scored
with the sequence-aware (lag-1) JSD, can tell a history-dependent ground truth
apart from a marginal-matched static surrogate — which the static JSD (and
static choice-proportion correlation) cannot. These are analytic corners:
self-JSD is ~0, and the perseveration/alternating pair has matched marginals
but opposite lag-1 structure.
"""
import random

import numpy as np

from scripts.jsd_recovery import build_eval_experiment, jsd_to_gt
from src.theory import Theory

THEORIES_DIR = "theories/heuristic_decision_making"


def _load(name: str) -> Theory:
    return Theory.from_yaml(f"{THEORIES_DIR}/{name}.yaml")


def test_eval_experiment_uses_human_validities_and_is_nondegenerate():
    exp = build_eval_experiment()
    assert list(exp.validities) == [0.9, 0.8, 0.7, 0.6]
    assert len(exp.trial_a_ratings) == len(exp.trial_b_ratings) > 0
    for a, b in zip(exp.trial_a_ratings, exp.trial_b_ratings):
        assert len(a) == len(b) == 4


def test_self_jsd_is_near_zero_floor():
    # A theory vs itself: both JSDs sit at the plug-in bias floor (~0).
    exp = build_eval_experiment()
    pers = _load("perseveration")
    d = jsd_to_gt(pers, pers, exp, n_runs=300)
    assert d["sequence_jsd"] < 0.03
    assert d["static_jsd"] < 0.03


def test_sequence_jsd_sees_history_that_static_misses():
    # Perseveration vs alternating: both ignore the stimulus, so their per-trial
    # MARGINALS match (~0.5) and static JSD is small; but their lag-1 joints are
    # opposite (repeat vs switch), so the SEQUENCE JSD is much larger. This is
    # exactly the history signal static choice-proportion scoring is blind to.
    exp = build_eval_experiment()
    pers, alt = _load("perseveration"), _load("alternating")
    d = jsd_to_gt(alt, pers, exp, n_runs=300)
    assert d["sequence_jsd"] > d["static_jsd"]
    assert d["sequence_jsd"] > 5 * max(d["static_jsd"], 1e-4)


def test_seeding_both_rngs_makes_jsd_reproducible():
    # Reproducibility requires seeding BOTH numpy (policy draws) AND stdlib
    # random (Theory parameter draws). With both seeded, identical results.
    exp = build_eval_experiment()
    ttb = _load("ttb_sampling")

    def run():
        random.seed(0)
        np.random.seed(0)
        return jsd_to_gt(ttb, ttb, exp, n_runs=50)

    assert run() == run()
