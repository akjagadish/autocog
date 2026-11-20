"""Tests for `scripts/explore_revise.py`.

Closed-form checks:

1. Lineage: each round's two alive theories, the one killed, and its
   replacement, truncated to `max_rounds`.
2. Seed ids resolve to the kept (de-duplicated) copy.
3. RMS distance over experiments, each experiment weighted equally.
4. A revision step: size, progress toward the ground truth, whether the
   worse theory was killed, whether the replacement beats both.
5. Experiment targeting ratios against the family's typical experiment.
6. Novelty: distance to the nearest experiment from an earlier round.
7. Run summary: gain of the best theory and path efficiency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _round(alive, killed, repl, idx):
    return {"round_idx": idx, "verdict": "new_theory",
            "starting_theories": [{"slot": i, "label": lab, "killed": lab == killed,
                                   "theory": {}} for i, lab in enumerate(alive)],
            "replacement": {"label": repl, "slot": alive.index(killed), "theory": {}}}


def test_load_lineage(tmp_path):
    from scripts.explore_revise import load_lineage

    run = tmp_path / "run"
    rounds = [(["pi_1", "pi_2"], "pi_1", "pi_3"), (["pi_3", "pi_2"], "pi_2", "pi_4"),
              (["pi_3", "pi_4"], "pi_3", "pi_5")]
    for r, (alive, killed, repl) in enumerate(rounds):
        d = run / "rounds" / f"round_{r:03d}"
        d.mkdir(parents=True)
        (d / "theories.json").write_text(json.dumps(_round(alive, killed, repl, r)))
    lin = load_lineage(run, max_rounds=2)
    assert len(lin) == 2
    assert lin[1].round == 1
    assert lin[1].alive == ("pi_3", "pi_2")
    assert (lin[1].killed, lin[1].survivor, lin[1].replacement) == ("pi_2", "pi_3", "pi_4")
    assert lin[0].seeds == frozenset({"pi_1", "pi_2"})


def test_model_id_resolves_seed_alias():
    from scripts.explore_revise import model_id

    info = {"llm": "x", "family": "f", "replicate": "2"}
    known = {"x/f/run2/pi_3", "x/f/run1/pi_1"}
    kept_seed = {("f", "pi_1"): "x/f/run1/pi_1"}
    assert model_id(info, "pi_3", seed=False, known=known, kept_seed=kept_seed) == "x/f/run2/pi_3"
    assert model_id(info, "pi_1", seed=True, known=known, kept_seed=kept_seed) == "x/f/run1/pi_1"
    assert model_id(info, "pi_9", seed=False, known=known, kept_seed=kept_seed) is None


def test_rms_weights_experiments_equally():
    from scripts.explore_revise import rms

    # Two experiments (2 pairs and 1 pair); theories 0 and 1.
    P = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0], [0.0]])]
    # exp0 mean sq = 1, exp1 = 0 -> sqrt(0.5)
    assert rms(P, 0, 1, [0, 1]) == pytest.approx(np.sqrt(0.5))
    assert rms(P, 0, 1, [0]) == pytest.approx(1.0)


def test_revision_step_closed_form():
    from scripts.explore_revise import revision_step

    # One experiment, one pair. Rows: killed, survivor, new, ground truth.
    P = [np.array([[0.9], [0.5], [0.3], [0.2]])]
    row = {"k": 0, "s": 1, "n": 2, "gt": 3}
    s = revision_step(P, row, [0], killed="k", survivor="s", new="n", gt="gt")
    assert s["step_behaviour"] == pytest.approx(0.6)
    assert s["T_killed"] == pytest.approx(0.7)
    assert s["T_survivor"] == pytest.approx(0.3)
    assert s["T_new"] == pytest.approx(0.1)
    assert s["killed_the_worse"]
    assert s["progress"] == pytest.approx(0.6)
    assert s["efficiency"] == pytest.approx(1.0)  # moved straight toward truth
    assert s["new_beats_best"]
    assert s["best_before"] == pytest.approx(0.3)
    assert s["best_after"] == pytest.approx(0.1)


def test_targeting_ratios():
    from scripts.explore_revise import targeting

    # Theories a, b (alive; b killed) and gt on three experiments of one pair.
    P = [np.array([[0.0], [1.0], [0.0]]),   # a, b differ by 1; b off gt by 1
         np.array([[0.0], [0.0], [0.0]]),   # no difference
         np.array([[0.0], [0.5], [0.5]])]   # differ by .5; b on gt
    row = {"a": 0, "b": 1, "gt": 2}
    t = targeting(P, row, 0, [0, 1, 2], alive=("a", "b"), killed="b", gt="gt")
    # Mean rival separation over the family = 0.5; this experiment 1.
    assert t["rival_sep"] == pytest.approx(2.0)
    # Killed theory's mean error over the family = 1/3; this experiment 1.
    assert t["killed_exposed"] == pytest.approx(3.0)


def test_novelty_is_distance_to_nearest_earlier_round():
    from scripts.explore_revise import novelty

    D = np.array([[0, 1, 4, 2],
                  [1, 0, 3, 5],
                  [4, 3, 0, 6],
                  [2, 5, 6, 0]], dtype=float)
    out = novelty(D, np.array([0, 0, 1, 2]))
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(3.0)
    assert out[3] == pytest.approx(2.0)


def test_run_summary_gain_and_path_efficiency():
    from scripts.explore_revise import run_summary

    steps = pd.DataFrame({"run": ["r"] * 2, "llm": ["x"] * 2, "family": ["f"] * 2,
                          "round": [0, 1], "best_before": [0.4, 0.3],
                          "best_after": [0.3, 0.2], "step_behaviour": [0.1, 0.3],
                          "progress": [0.1, 0.0]})
    s = run_summary(steps).iloc[0]
    assert s["best_start"] == pytest.approx(0.4)
    assert s["best_end"] == pytest.approx(0.2)
    assert s["gain"] == pytest.approx(0.2)
    assert s["path"] == pytest.approx(0.4)
    assert s["path_efficiency"] == pytest.approx(0.5)
