"""Tests for `scripts/hilibig_battery.py`.

These tests pin the public surface of the Hilbig / heuristic-decision-making
battery analog of `recovery_battery.py`. They are intentionally minimal —
the script is research-code plotting infrastructure, and the high-value
invariants are:

1. We read the right things out of `observations/state.json` (validities,
   rating_max, the unique set of trial pairs across rounds).
2. Trajectories have shape `(T, 2, n_features)` and cover the requested
   pairs without dropping any.
3. Ground-truth-vs-self distance is identically zero (sanity floor).
4. Per-model aggregate accuracy vs. ground truth is in [0, 1].
5. When a surfaced theory declares a `validities` parameter, the
   parameter sampler pins it to the run's actual validities rather than
   sampling from the YAML prior (which is `[(0.0, 1.0)] * n_features`
   and would be nonsense as a ground-truth comparison).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = REPO_ROOT / "theories" / "heuristic_decision_making"


# ---------------------------------------------------------------------------
# Run-metadata loading.
# ---------------------------------------------------------------------------


def _first_available_run_dir() -> Path:
    """Find any hilbig run-dir under results/ for integration tests."""
    # hilibig_battery requires a CARDINAL run: load_run_metadata reads
    # experiment["rating_max"], which binary runs (results/recovery/) omit
    # entirely. The cardinal run that ships is the human study.
    for base in (REPO_ROOT / "results" / "human_decision_making_cardinal",):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "observations" / "state.json").is_file():
                return d
    pytest.skip("No cardinal run-dir with observations/state.json under results/.")


def test_load_run_metadata_returns_validities_and_trial_pairs():
    from scripts.hilibig_battery import load_run_metadata

    run_dir = _first_available_run_dir()
    meta = load_run_metadata(run_dir)  # default round_idx="last"

    assert set(meta) == {"validities", "rating_max", "trial_pairs", "round_idx"}
    val = meta["validities"]
    assert isinstance(val, list) and all(0.0 <= v <= 1.0 for v in val)
    assert len(val) >= 2, "Hilbig uses at least 2 features."
    assert isinstance(meta["rating_max"], int) and meta["rating_max"] >= 1
    assert isinstance(meta["round_idx"], int) and meta["round_idx"] >= 0

    pairs = meta["trial_pairs"]
    assert len(pairs) > 0
    # Every pair is ((a0..an), (b0..bn)) with length == n_features.
    n = len(val)
    for a, b in pairs:
        assert len(a) == n and len(b) == n
        assert all(0 <= int(x) <= meta["rating_max"] for x in (*a, *b))


def test_load_run_metadata_respects_round_idx():
    """'first' and 'last' can differ — the LLM arbiter may swap designs."""
    from scripts.hilibig_battery import load_run_metadata

    run_dir = _first_available_run_dir()
    first = load_run_metadata(run_dir, round_idx="first")
    last = load_run_metadata(run_dir, round_idx="last")
    assert first["round_idx"] == 0
    assert last["round_idx"] >= first["round_idx"]
    # Explicit integer works and must match.
    again = load_run_metadata(run_dir, round_idx=first["round_idx"])
    assert again["trial_pairs"] == first["trial_pairs"]
    assert again["validities"] == first["validities"]


# ---------------------------------------------------------------------------
# Trajectory generation.
# ---------------------------------------------------------------------------


def test_sample_trajectory_shape_and_completeness():
    from scripts.hilibig_battery import sample_trajectory

    pairs = [((1, 0, 0, 0), (0, 1, 1, 1)),
             ((0, 0, 1, 1), (1, 1, 0, 0)),
             ((1, 1, 1, 0), (0, 0, 0, 1))]
    rng = np.random.default_rng(0)
    traj = sample_trajectory(rng, trial_pairs=pairs, n_blocks=2)

    assert traj.stimuli.shape == (6, 2, 4)
    assert traj.stimuli.dtype.kind in {"i", "f"}

    # The concatenation should contain each unique pair exactly n_blocks times.
    seen = [tuple(map(tuple, stim.tolist())) for stim in traj.stimuli]
    expected = [tuple(map(tuple, p)) for p in pairs]
    for p in expected:
        assert seen.count(p) == 2


# ---------------------------------------------------------------------------
# Parameter sampling: pin validities when the theory uses them.
# ---------------------------------------------------------------------------


def test_sample_hilibig_params_pins_validities_when_present():
    from src.theory import Theory
    from scripts.hilibig_battery import sample_hilibig_params

    ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    pinned = [0.9, 0.8, 0.7, 0.6]
    p = sample_hilibig_params(
        ttb, validities=pinned, rating_max=5, n_features=4, seed_override=0,
    )
    assert p["validities"] == pinned
    assert "beta" in p and "epsilon" in p


def test_sample_hilibig_params_no_ops_for_theories_without_validities():
    """Tallying has no `validities` parameter — must not crash, and the
    returned dict simply should not contain that key."""
    from src.theory import Theory
    from scripts.hilibig_battery import sample_hilibig_params

    tal = Theory.from_yaml(YAML_DIR / "tallying.yaml")
    p = sample_hilibig_params(
        tal, validities=[0.9, 0.8, 0.7, 0.6], rating_max=5, n_features=4,
        seed_override=0,
    )
    assert "validities" not in p
    assert "beta" in p and "epsilon" in p


def test_sample_hilibig_params_resolves_passthrough_rating_max():
    """A theory that declares `rating_max: 'rating_max'` (the
    context-passthrough convention from src/sample_parameters.py) must
    receive the run's actual rating_max as an int, not the literal
    string 'rating_max'."""
    from src.theory import Theory
    from scripts.hilibig_battery import sample_hilibig_params

    theory = Theory.model_validate({
        "description": "dummy",
        "predict_source": (
            "def predict(parameters, stimulus, history):\n"
            "    import numpy as np\n"
            "    rm = float(parameters['rating_max'])\n"
            "    return np.array([0.5, 0.5])\n"
        ),
        "policy_source": "def policy(p):\n    return 0\n",
        "parameters": {"rating_max": "rating_max"},
    })
    p = sample_hilibig_params(
        theory, validities=[0.9, 0.8, 0.7, 0.6], rating_max=5, n_features=4,
        seed_override=0,
    )
    assert p["rating_max"] == 5


# ---------------------------------------------------------------------------
# Simulation + distance.
# ---------------------------------------------------------------------------


def test_predict_trajectory_stateless_shape():
    """Ground-truth TTB on a small trajectory returns shape (T, 2) with
    rows that sum to 1."""
    from src.theory import Theory
    from scripts.hilibig_battery import predict_trajectory, sample_trajectory

    ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    pairs = [((1, 0, 0, 0), (0, 1, 1, 1)),
             ((1, 1, 0, 0), (0, 0, 1, 1))]
    rng = np.random.default_rng(0)
    traj = sample_trajectory(rng, trial_pairs=pairs, n_blocks=1)
    params = {
        "beta": 10.0, "epsilon": 0.0,
        "validities": [0.9, 0.8, 0.7, 0.6],
    }
    probs = predict_trajectory(ttb.predict, params, traj)
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-9)


def test_self_distance_is_zero():
    """A model's distance to itself (same parameters, same trajectory)
    must be identically 0 — this guards against off-by-one trajectory
    bugs in the distance computation."""
    from src.theory import Theory
    from scripts.hilibig_battery import (
        predict_trajectory, sample_trajectory, per_trial_mae_mse,
    )

    ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    pairs = [((1, 0, 0, 0), (0, 1, 1, 1)),
             ((1, 1, 0, 0), (0, 0, 1, 1))]
    rng = np.random.default_rng(0)
    traj = sample_trajectory(rng, trial_pairs=pairs, n_blocks=2)
    params = {"beta": 5.0, "epsilon": 0.1,
              "validities": [0.9, 0.8, 0.7, 0.6]}

    P = predict_trajectory(ttb.predict, params, traj)
    Q = predict_trajectory(ttb.predict, params, traj)

    mae, mse = per_trial_mae_mse(P[None, :, :], Q[None, :, :])  # (1 subj,)
    assert float(mae.max()) == 0.0
    assert float(mse.max()) == 0.0


# ---------------------------------------------------------------------------
# Theory resolution from a real run-dir (integration).
# ---------------------------------------------------------------------------


def test_resolve_surfaced_theories_returns_nonempty_dict():
    from scripts.hilibig_battery import resolve_surfaced_theories

    run_dir = _first_available_run_dir()
    surfaced = resolve_surfaced_theories(run_dir)
    assert len(surfaced) >= 1
    # Every value must be a usable Theory with a bound predict.
    for label, theory in surfaced.items():
        assert hasattr(theory, "predict") and callable(theory.predict)
        assert isinstance(label, str) and label.startswith("pi_")
