"""Tests for `scripts/eval_hilbig.py`.

The high-value invariants the script must enforce:

1. Stimulus columns from `hilbig/exp1.txt` (string-formatted numpy arrays
   like `"[0 1 1 1]"`) parse to integer tuples.
2. Per-stimulus human choice proportions are computed correctly, and the
   pooled-vs-within-participant aggregation actually disagrees when
   participants are unbalanced — the user explicitly asked for both.
3. Replaying a YAML theory on a list of stimulus pairs returns valid
   probabilities (in [0, 1]) and respects the theory's known qualitative
   behavior (TTB picks the option that wins on the highest-validity
   discriminating cue).
4. The model-vs-human distance shrinks when the model IS the data-
   generating process — a sanity floor that catches sign / column-mixup
   bugs in the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = REPO_ROOT / "theories" / "heuristic_decision_making"
HUMAN_VALIDITIES = [0.9, 0.8, 0.7, 0.6]


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


def test_parse_array_string_handles_brackets_and_whitespace():
    from scripts.eval_hilbig import _parse_array_string

    assert _parse_array_string("[0 1 1 1]") == (0, 1, 1, 1)
    assert _parse_array_string("[1   0  0   1]") == (1, 0, 0, 1)
    assert _parse_array_string("  [0 0 0 0]  ") == (0, 0, 0, 0)


def test_load_human_choices_parses_real_exp1_file():
    """Smoke-test the real Hilbig exp1.txt — the script's primary input."""
    from scripts.eval_hilbig import load_human_choices

    path = REPO_ROOT / "hilbig" / "exp1.txt"
    if not path.is_file():
        pytest.skip(f"{path} not present.")
    df = load_human_choices(path)

    assert {"participant", "trial", "choice", "option_a", "option_b"} <= set(
        df.columns
    )
    assert df["choice"].isin((0, 1)).all()
    # All option vectors must have a consistent feature count.
    lens = df["option_a"].map(len).unique().tolist()
    assert lens == [len(HUMAN_VALIDITIES)], (
        f"Expected exactly {len(HUMAN_VALIDITIES)} features; got {lens}."
    )
    # Binary ratings.
    flat = np.concatenate([np.asarray(t) for t in df["option_a"].tolist()])
    assert set(flat.tolist()) <= {0, 1}


# ---------------------------------------------------------------------------
# Human stimulus-choice proportions.
# ---------------------------------------------------------------------------


def test_compute_human_proportions_pooled_matches_hand_count():
    from scripts.eval_hilbig import compute_human_proportions

    df = pd.DataFrame({
        "participant": [0, 0, 1, 1],
        "trial": [0, 1, 0, 1],
        "choice": [0, 1, 1, 1],
        "option_a": [(0, 1)] * 4,
        "option_b": [(1, 0)] * 4,
    })
    out = compute_human_proportions(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_trials"] == 4
    assert row["n_choose_b"] == 3
    assert row["p_b_pooled"] == pytest.approx(0.75)
    assert row["n_participants"] == 2


def test_pooled_and_within_subject_disagree_when_unbalanced():
    """A heavy-trial participant choosing A drowns out a light-trial
    participant who chose B — pooled tilts toward A while the within-
    subject mean (simple participant-average) is 0.5. This is exactly
    why the user asked for both metrics; the test exists to lock the
    distinction in.
    """
    from scripts.eval_hilbig import compute_human_proportions

    df = pd.DataFrame({
        "participant": [0] * 10 + [1],
        "trial": list(range(10)) + [0],
        "choice": [0] * 10 + [1],
        "option_a": [(0, 1)] * 11,
        "option_b": [(1, 0)] * 11,
    })
    out = compute_human_proportions(df)
    row = out.iloc[0]
    assert row["p_b_pooled"] == pytest.approx(1 / 11)
    assert row["p_b_within_subj_mean"] == pytest.approx(0.5)


def test_compute_human_proportions_separates_distinct_pairs():
    from scripts.eval_hilbig import compute_human_proportions

    df = pd.DataFrame({
        "participant": [0, 0, 0, 0],
        "trial": [0, 1, 2, 3],
        "choice": [0, 1, 1, 0],
        "option_a": [(0, 1), (0, 1), (1, 1), (1, 1)],
        "option_b": [(1, 0), (1, 0), (0, 0), (0, 0)],
    })
    out = compute_human_proportions(df)
    by_pair = {
        (a, b): row for (a, b), row in zip(
            zip(out["option_a"], out["option_b"]),
            out.to_dict(orient="records"),
        )
    }
    assert by_pair[((0, 1), (1, 0))]["n_trials"] == 2
    assert by_pair[((0, 1), (1, 0))]["p_b_pooled"] == pytest.approx(0.5)
    assert by_pair[((1, 1), (0, 0))]["n_trials"] == 2
    assert by_pair[((1, 1), (0, 0))]["p_b_pooled"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Replaying a YAML theory on stimulus pairs.
# ---------------------------------------------------------------------------


def _ttb() -> "Theory":  # type: ignore[name-defined]
    from src.theory import Theory

    if not (YAML_DIR / "ttb.yaml").is_file():
        pytest.skip("No ttb.yaml in theories/heuristic_decision_making/.")
    return Theory.from_yaml(YAML_DIR / "ttb.yaml")


def test_model_proportions_returns_probs_in_unit_interval():
    from scripts.eval_hilbig import model_proportions_for_theory

    pairs = [
        ((1, 0, 0, 0), (0, 1, 1, 1)),
        ((0, 1, 1, 1), (1, 0, 0, 0)),
        ((0, 0, 0, 0), (1, 1, 1, 1)),
    ]
    preds = model_proportions_for_theory(
        _ttb(),
        stimulus_pairs=pairs,
        validities=HUMAN_VALIDITIES,
        rating_max=1,
        n_subjects=8,
        base_seed=42,
    )
    assert set(preds) == set(pairs)
    for v in preds.values():
        assert 0.0 <= v <= 1.0


def test_ttb_prefers_winner_on_highest_validity_cue():
    """Behavioral spec for TTB: the option with the higher value on the
    top-validity discriminating cue should be picked far more often than
    50/50 — averaged across noise draws. If the predict path mixes up A
    and B, this test fails."""
    from scripts.eval_hilbig import model_proportions_for_theory

    a_wins = ((1, 0, 0, 0), (0, 1, 1, 1))   # A wins on validity[0]=0.9
    b_wins = ((0, 1, 1, 1), (1, 0, 0, 0))   # B wins on validity[0]=0.9

    preds = model_proportions_for_theory(
        _ttb(),
        stimulus_pairs=[a_wins, b_wins],
        validities=HUMAN_VALIDITIES,
        rating_max=1,
        n_subjects=64,
        base_seed=0,
    )
    assert preds[a_wins] < 0.5 < preds[b_wins]


# ---------------------------------------------------------------------------
# Canonical baselines: --include-canonical loads every YAML under
# theories/heuristic_decision_making/ and reports them with role='canonical'.
# ---------------------------------------------------------------------------


def test_load_canonical_theories_returns_all_yamls():
    from scripts.eval_hilbig import load_canonical_theories

    if not YAML_DIR.is_dir():
        pytest.skip("No canonical YAML dir.")
    out = load_canonical_theories(YAML_DIR)
    expected = {p.stem for p in YAML_DIR.glob("*.yaml")}
    assert set(out) == expected
    # Sanity: every loaded entry is a real Theory we can predict with.
    from src.theory import Theory
    for v in out.values():
        assert isinstance(v, Theory)


def test_evaluate_models_includes_canonical_role():
    from scripts.eval_hilbig import (
        compute_human_proportions, evaluate_models, load_canonical_theories,
    )

    if not YAML_DIR.is_dir():
        pytest.skip("No canonical YAML dir.")
    canonical = load_canonical_theories(YAML_DIR)
    df = pd.DataFrame({
        "participant": [0, 0],
        "trial": [0, 1],
        "choice": [1, 0],
        "option_a": [(1, 0, 0, 0), (0, 1, 1, 1)],
        "option_b": [(0, 1, 1, 1), (1, 0, 0, 0)],
    })
    human_props = compute_human_proportions(df)
    _, summary = evaluate_models(
        base={}, surfaced={}, canonical=canonical,
        human_props=human_props,
        validities=HUMAN_VALIDITIES, rating_max=1,
        n_subjects=4, base_seed=0,
    )
    assert (summary["role"] == "canonical").all()
    assert set(summary["model"]) == set(canonical)


# ---------------------------------------------------------------------------
# Shared-seed invariant: when the same theory shows up under two roles
# (e.g. a seed that started the run AND survived to the last round),
# the two rows must produce identical per-stimulus P(B) and identical
# MAE. Different rows for the same theory should never differ from
# Monte-Carlo seed noise alone.
# ---------------------------------------------------------------------------


def test_same_theory_in_base_and_surfaced_yields_identical_rows():
    from scripts.eval_hilbig import compute_human_proportions, evaluate_models

    ttb = _ttb()
    df = pd.DataFrame({
        "participant": [0, 0, 1, 1],
        "trial": [0, 1, 0, 1],
        "choice": [1, 0, 1, 0],
        "option_a": [(1, 0, 0, 0), (0, 1, 1, 1)] * 2,
        "option_b": [(0, 1, 1, 1), (1, 0, 0, 0)] * 2,
    })
    human_props = compute_human_proportions(df)

    per_stim, summary = evaluate_models(
        base={"pi_1": ttb},
        surfaced={"pi_1": ttb},   # same theory, dual role
        human_props=human_props,
        validities=HUMAN_VALIDITIES, rating_max=1,
        n_subjects=8, base_seed=7,
    )
    base_row = summary[summary["role"] == "base"].iloc[0]
    surf_row = summary[summary["role"] == "surfaced"].iloc[0]
    assert base_row["mae"] == pytest.approx(surf_row["mae"])
    assert base_row["mse"] == pytest.approx(surf_row["mse"])

    base_p = per_stim[per_stim["role"] == "base"]["p_b_model"].to_numpy()
    surf_p = per_stim[per_stim["role"] == "surfaced"]["p_b_model"].to_numpy()
    assert np.allclose(base_p, surf_p)


# ---------------------------------------------------------------------------
# Sanity floor: when the model IS the data-generating process, the
# model-vs-human MAE should be small. This catches column / sign / index
# mixups that any plausible stimulus-proportion comparison would miss
# silently.
# ---------------------------------------------------------------------------


def test_self_mae_is_small_when_model_generates_the_data():
    """Synthesize human-like data from TTB itself, then compare TTB's
    predicted P(B) to that synthetic 'human' P(B). MAE should shrink
    toward zero as we add subjects."""
    import random as _random

    from scripts.eval_hilbig import (
        compute_human_proportions,
        evaluate_models,
    )

    ttb = _ttb()
    pairs = [
        ((1, 0, 0, 0), (0, 1, 1, 1)),
        ((0, 1, 1, 1), (1, 0, 0, 0)),
        ((1, 1, 0, 0), (0, 0, 1, 1)),
        ((0, 0, 1, 1), (1, 1, 0, 0)),
    ]

    # Synthesize 30 'subjects', each making 6 trials per pair, choosing
    # by sampling from TTB's predicted P(B) under freshly-drawn params.
    rng = np.random.default_rng(0)
    rows = []
    n_subjects, trials_per_pair = 30, 6
    for s in range(n_subjects):
        _random.seed(s)
        params = ttb.sample_parameters({
            "n_features": 4, "n_labels": 2, "validities": HUMAN_VALIDITIES,
        })
        params["validities"] = HUMAN_VALIDITIES
        for a, b in pairs:
            stim = np.asarray([list(a), list(b)], dtype=float)
            p = np.asarray(
                ttb.predict(params, stim, {
                    "stimulus": [], "label": [],
                    "previous_stimuli": [], "previous_labels": [],
                }),
                dtype=float,
            )
            choices = rng.binomial(1, p[1], size=trials_per_pair)
            for t, c in enumerate(choices):
                rows.append({
                    "participant": s, "trial": len(rows),
                    "choice": int(c), "option_a": a, "option_b": b,
                })
    syn_df = pd.DataFrame(rows)
    human_props = compute_human_proportions(syn_df)

    _, summary = evaluate_models(
        base={"ttb": ttb}, surfaced={},
        human_props=human_props,
        validities=HUMAN_VALIDITIES, rating_max=1,
        n_subjects=64, base_seed=123,
        target_col="p_b_pooled",
    )
    mae = float(summary.loc[summary["model"] == "ttb", "mae"].iloc[0])
    # Loose bound — sampling noise floor for ~6 trials/pair × 30 subjects.
    assert mae < 0.10, f"Self-comparison MAE too large: {mae:.3f}"
