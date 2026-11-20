"""Tests for `scripts/compare_experiments.py`.

Closed-form checks:

1. Loading: one row per proposed experiment, truncated to the first
   `max_rounds` rounds, linked to its proposer theory's model id.
2. Structural features on a hand-built design (first discriminating cue,
   tally gap, TTB-vs-tally conflict, A/B balance).
3. Discrimination profile = mean |P_a - P_b| over the experiment's pairs;
   the panel's tallying model is exactly 0.5 on an equal-tally pair.
4. Classical MDS recovers a planar configuration exactly.
5. TF-IDF cosine distance: identical texts 0, disjoint texts 1.
6. Joint theory x experiment distances on hand-computed predictions.
7. Mantel: identical matrices correlate at 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.theory import Theory

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = REPO_ROOT / "theories" / "heuristic_decision_making"


def _obs(proposer: str, validities, a_rows, b_rows, rationale="r"):
    return {"experiment": {"rationale": rationale, "validities": validities,
                           "trial_a_ratings": a_rows, "trial_b_ratings": b_rows},
            "metric": {"metric_source": "def metric(data):\n    return 0.0\n"},
            "proposer_label": proposer}


def _write_run(root: Path, n_rounds: int) -> Path:
    run = root / "dmb_ground_truth_single_cue_noise=0.0_llm-x_run2"
    (run / "observations").mkdir(parents=True)
    rounds = []
    for r in range(n_rounds):
        rounds.append({"observations": [
            _obs(f"pi_{r + 1}", [0.9, 0.6], [[1, 0]], [[0, 1]]),
            _obs(f"pi_{r + 2}", [0.8, 0.7, 0.6], [[1, 0, 0], [0, 1, 1]],
                 [[0, 1, 1], [1, 0, 0]]),
        ]})
    (run / "observations" / "state.json").write_text(json.dumps({"rounds": rounds}))
    return run


# ---------------------------------------------------------------------------
# 1. Loading.
# ---------------------------------------------------------------------------


def test_load_experiments_truncates_and_links_proposer(tmp_path):
    from scripts.compare_experiments import load_experiments

    run = _write_run(tmp_path, n_rounds=4)
    exps = load_experiments([run], max_rounds=3)
    assert len(exps) == 6
    assert [e.round for e in exps] == [0, 0, 1, 1, 2, 2]
    e = exps[3]
    assert (e.llm, e.family, e.replicate) == ("llm-x", "single_cue", "2")
    assert e.proposer_model == "llm-x/single_cue/run2/pi_3"
    assert e.validities == (0.8, 0.7, 0.6)
    assert e.pairs == (((1, 0, 0), (0, 1, 1)), ((0, 1, 1), (1, 0, 0)))
    assert e.exp_id == "llm-x/single_cue/run2/r1e1"


# ---------------------------------------------------------------------------
# 2. Structural features.
# ---------------------------------------------------------------------------


def test_structural_features_closed_form():
    from scripts.compare_experiments import structural_features

    # Cue 0 is the most valid. Pair 1: A wins on cue 0 only, B wins on
    # cues 1 and 2 -> TTB says A, tally says B (conflict), gap 1.
    # Pair 2: A has cue 1, B nothing -> both say A, first discriminating
    # cue is rank 1 of 3. Pair 3: identical -> nothing discriminates.
    v = [0.9, 0.7, 0.6]
    pairs = [((1, 0, 0), (0, 1, 1)), ((0, 1, 0), (0, 0, 0)), ((1, 1, 0), (1, 1, 0))]
    f = structural_features(v, pairs)
    assert f["n_cues"] == 3 and f["n_pairs"] == 3
    assert f["validity_mean"] == pytest.approx(np.mean(v))
    assert f["frac_no_discrimination"] == pytest.approx(1 / 3)
    # Among the 2 discriminating pairs:
    assert f["mean_discriminating_cues"] == pytest.approx((3 + 1) / 2)
    assert f["frac_ttb_tally_conflict"] == pytest.approx(1 / 2)
    assert f["frac_tally_tie"] == pytest.approx(0.0)
    # First discriminating cue's validity rank, scaled to [0, 1]: 0 and 1/2.
    assert f["first_disc_rank"] == pytest.approx((0 + 0.5) / 2)
    # |tally gap| / n_cues: 1/3 and 1/3.
    assert f["tally_gap"] == pytest.approx(1 / 3)
    # TTB favours A on both discriminating pairs -> fully unbalanced.
    assert f["ab_imbalance"] == pytest.approx(0.5)
    # Ones: pair 1 has 1 + 2, pair 2 has 1 + 0, pair 3 has 2 + 2.
    assert f["cue_density"] == pytest.approx(8 / 18)


# ---------------------------------------------------------------------------
# 3. Discrimination profile.
# ---------------------------------------------------------------------------


def test_discrimination_profile_is_mean_abs_difference():
    from scripts.compare_experiments import discrimination_profile

    preds = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 1.0]),
             "c": np.array([0.5, 0.5])}
    prof = discrimination_profile(preds)
    assert prof == pytest.approx({"a|b": 0.5, "a|c": 0.5, "b|c": 0.5})
    assert list(prof) == ["a|b", "a|c", "b|c"]


def test_panel_tallying_is_indifferent_on_equal_tallies():
    from scripts.compare_experiments import load_panel, panel_predictions

    panel = load_panel(("tallying", "ttb"))
    pairs = [((1, 0, 0, 0), (0, 1, 0, 0)), ((1, 1, 0, 0), (0, 0, 0, 0))]
    preds = panel_predictions(panel, [0.9, 0.8, 0.7, 0.6], pairs, n_draws=20, seed=0)
    assert preds["tallying"][0] == pytest.approx(0.5)
    # TTB follows the most valid discriminating cue (A) on the tie pair.
    assert preds["ttb"][0] < 0.5
    assert preds["tallying"][1] < 0.5


# ---------------------------------------------------------------------------
# 4. Classical MDS and distances.
# ---------------------------------------------------------------------------


def _pairwise(X):
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def test_classical_mds_recovers_planar_configuration():
    from scripts.compare_experiments import classical_mds, kruskal_stress

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    D = _pairwise(X)
    Y, explained = classical_mds(D)
    assert Y.shape == (12, 2)
    np.testing.assert_allclose(_pairwise(Y), D, atol=1e-8)
    assert explained == pytest.approx(1.0)
    assert kruskal_stress(D, Y) == pytest.approx(0.0, abs=1e-8)


def test_tfidf_cosine_distance_extremes():
    from scripts.compare_experiments import tfidf_cosine_distances

    D = tfidf_cosine_distances(["cue validity test", "cue validity test",
                                "banana orange"])
    assert D[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert D[0, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(D, D.T)


def test_mantel_identical_matrices():
    from scripts.compare_experiments import mantel

    rng = np.random.default_rng(1)
    D = _pairwise(rng.normal(size=(10, 2)))
    r, p = mantel(D, D, n_perm=99, seed=0)
    assert r == pytest.approx(1.0)
    assert p <= 0.02


# ---------------------------------------------------------------------------
# 5. Joint theory x experiment space.
# ---------------------------------------------------------------------------


def test_joint_distances_closed_form():
    from scripts.compare_experiments import joint_distances

    # Two experiments (2 and 1 pairs), three theories.
    P = [np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
         np.array([[0.0], [0.0], [1.0]])]
    D_th, D_ex = joint_distances(P)
    # Theory distance: RMS over experiments weighted equally:
    # t0 vs t1: exp0 mean sq = 1, exp1 = 0 -> sqrt(0.5).
    assert D_th[0, 1] == pytest.approx(np.sqrt(0.5))
    # t0 vs t2: exp0 0.25, exp1 1 -> sqrt(0.625).
    assert D_th[0, 2] == pytest.approx(np.sqrt(0.625))
    # Experiment profile over theory pairs (01, 02, 12):
    # exp0: (1, .5, .5), exp1: (0, 1, 1) -> distance sqrt(1 + .25 + .25).
    assert D_ex[0, 1] == pytest.approx(np.sqrt(1.5))


def test_drop_incomplete_theories():
    from scripts.compare_experiments import complete_theories

    P = [np.array([[0.1], [np.nan], [0.3]]), np.array([[0.2, 0.4], [0.5, 0.5], [np.nan, 0.1]])]
    keep = complete_theories(P)
    assert keep.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# 6. Theory predictions on a new design reuse best-fit free parameters.
# ---------------------------------------------------------------------------


def test_theory_on_experiment_uses_experiment_validities():
    from scripts.compare_experiments import predict_on_experiment

    th = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    # Best-fit params were found on 4 cues; validities must come from the
    # new experiment, the free parameters from the fit.
    best = {"validities": [0.9, 0.8, 0.7, 0.6], "beta": 20.0, "epsilon": 0.0}
    pairs = [((0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1))]
    p = predict_on_experiment(th, best, [0.5, 0.5, 0.5, 0.5, 0.6, 0.9], pairs)
    assert p[0] > 0.99  # cue 5 (validity 0.9) favours B
