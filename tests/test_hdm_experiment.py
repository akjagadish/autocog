"""Tests for LLM-proposable n_features + validities on HeuristicDecisionMakingExperiment.

Goal: the experiment class must accept `validities` and `trial_pairs` of arbitrary
feature count from the LLM (no longer hardcoded to 4 features with fixed validities
0.9/0.8/0.7/0.6). This lets an experiment-proposer vary both the feature size and
the per-feature validities, on top of the trial-pair structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.heuristic_decision_making.experiment import (
    HeuristicDecisionMakingExperiment,
)
from src.theory import Theory


def _ttb_like_theory() -> Theory:
    """Minimal TTB-flavoured theory. Reads `parameters['validities']` dynamically
    (no hardcoded length) and operates on a (2, n_features) stimulus — so it
    works at any n_features the experiment decides to use."""
    predict_src = (
        "def predict(parameters, state, history):\n"
        "    import numpy as np\n"
        "    stim = np.asarray(state, dtype=float)\n"
        "    val = np.asarray(parameters['validities'], dtype=float)\n"
        "    order = np.argsort(-val)  # highest-validity cue first\n"
        "    for k in order:\n"
        "        diff = stim[0, k] - stim[1, k]\n"
        "        if diff > 0:\n"
        "            return np.array([1.0, 0.0])\n"
        "        if diff < 0:\n"
        "            return np.array([0.0, 1.0])\n"
        "    return np.array([0.5, 0.5])\n"
    )
    policy_src = (
        "def policy(probs):\n"
        "    import numpy as np\n"
        "    return int(np.argmax(probs))\n"
    )
    return Theory(
        description="TTB-like",
        predict_source=predict_src,
        policy_source=policy_src,
        # Symbolic reference: `sample_parameters` pulls the experiment's
        # instance-level `validities` from `parameter_context()` into this
        # theory's runtime `parameters` dict.
        parameters={"validities": "validities"},
    )


def test_llm_proposable_n_features_and_validities():
    """The experiment accepts (n_features=3) + LLM-chosen validities,
    exposes them via parameter_context, and simulates cleanly."""
    # Asymmetric validities (not descending from 0.9 — the LLM may pick any
    # order within [0.5, 1.0]) to stress that nothing depends on the
    # historical H&M sort order.
    validities = [0.9, 0.7, 0.5]
    # Parallel lists (analogous to CL's stimuli/labels split). Gemini's
    # structured-output schema rejects both tuple/prefixItems AND $ref/$defs
    # nested models, so keeping everything as parallel primitive-list fields
    # is the only Gemini-friendly shape.
    trial_a_ratings = [[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]]
    trial_b_ratings = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]

    exp = HeuristicDecisionMakingExperiment(
        validities=validities,
        rating_max=1,
        trial_a_ratings=trial_a_ratings,
        trial_b_ratings=trial_b_ratings,
    )

    # parameter_context reports instance-level values, not stale classvars.
    ctx = exp.parameter_context()
    assert ctx["n_features"] == 3, (
        f"expected n_features=3 (derived from validities), got {ctx['n_features']!r}"
    )
    assert list(ctx["validities"]) == validities, (
        f"expected validities={validities}, got {ctx['validities']!r}"
    )
    assert ctx["rating_max"] == 1, (
        f"expected rating_max=1, got {ctx['rating_max']!r}"
    )

    # simulate() round-trips the new shape without a length-4 assertion error.
    theory = _ttb_like_theory()
    df = exp.simulate(theory, n_runs=1)
    assert len(df) > 0
    assert len(df["option_a_ratings"].iloc[0]) == 3
    assert len(df["option_b_ratings"].iloc[0]) == 3


def test_validities_length_must_match_ratings():
    """Rating list length and validities length must be consistent."""
    with pytest.raises(ValueError):
        HeuristicDecisionMakingExperiment(
            validities=[0.9, 0.7],  # 2 validities...
            rating_max=1,
            trial_a_ratings=[[1, 0, 1]],  # ...but 3-feature ratings
            trial_b_ratings=[[0, 1, 0]],
        )


def test_validities_bounded_to_half_to_one():
    """Each validity must be in [0.5, 1.0]; the LLM can't propose a cue that's
    worse than chance."""
    with pytest.raises(ValueError):
        HeuristicDecisionMakingExperiment(
            validities=[0.9, 0.3, 0.5],  # 0.3 < 0.5
            rating_max=1,
            trial_a_ratings=[[1, 0, 1]],
            trial_b_ratings=[[0, 1, 0]],
        )


def test_validities_bounds_exposed_in_json_schema():
    # Without item-level ge/le, Gemini's structured-output doesn't know the
    # bound and can emit validities<0.5, which then fails model_post_init
    # and crashes the run with an opaque "parse failed" error. The bound
    # must be on the list items, not the list itself.
    schema = HeuristicDecisionMakingExperiment.model_json_schema()
    items = schema["properties"]["validities"]["items"]
    assert items.get("minimum") == 0.5, items
    assert items.get("maximum") == 1.0, items


def test_cardinal_ratings_dissociate_ew_from_tallying():
    """With rating_max=5, Equal-Weight and Tallying can produce different
    decisions on the same trial — the whole point of adding cardinal
    features. This is a conceptual test: a single crafted trial where a
    large A-advantage on one feature is outweighed (in Tallying) by small
    B-wins on two other features, but EW picks A by sum."""
    import numpy as np

    validities = [0.9, 0.7, 0.6]
    exp = HeuristicDecisionMakingExperiment(
        validities=validities,
        rating_max=5,
        # A=[5, 0, 0] vs B=[0, 2, 2]:  EW sum A=5 > B=4 -> pick A
        #                              Tallying  A-wins=1, B-wins=2 -> pick B
        trial_a_ratings=[[5, 0, 0]],
        trial_b_ratings=[[0, 2, 2]],
    )

    # EW theory: sum each option, pick higher.
    ew = Theory(
        description="EW",
        predict_source=(
            "def predict(parameters, state, history):\n"
            "    import numpy as np\n"
            "    s = np.asarray(state, dtype=float)\n"
            "    a, b = s[0].sum(), s[1].sum()\n"
            "    return np.array([1.0, 0.0]) if a >= b else np.array([0.0, 1.0])\n"
        ),
        policy_source=(
            "def policy(probs):\n"
            "    import numpy as np\n"
            "    return int(np.argmax(probs))\n"
        ),
        parameters={},
    )
    # Tallying: sign-based feature-wise wins.
    tallying = Theory(
        description="Tallying",
        predict_source=(
            "def predict(parameters, state, history):\n"
            "    import numpy as np\n"
            "    s = np.asarray(state, dtype=float)\n"
            "    a_wins = int(np.sum(s[0] > s[1]))\n"
            "    b_wins = int(np.sum(s[1] > s[0]))\n"
            "    if a_wins > b_wins:\n"
            "        return np.array([1.0, 0.0])\n"
            "    if b_wins > a_wins:\n"
            "        return np.array([0.0, 1.0])\n"
            "    return np.array([0.5, 0.5])\n"
        ),
        policy_source=(
            "def policy(probs):\n"
            "    import numpy as np\n"
            "    return int(np.argmax(probs))\n"
        ),
        parameters={},
    )

    ew_df = exp.simulate(ew, n_runs=1)
    tallying_df = exp.simulate(tallying, n_runs=1)
    # All EW trials pick A (response=0); all Tallying trials pick B (response=1).
    assert (ew_df["response"] == 0).all(), (
        f"EW should pick A on A=[5,0,0]/B=[0,2,2]; got {ew_df['response'].tolist()}"
    )
    assert (tallying_df["response"] == 1).all(), (
        f"Tallying should pick B on A=[5,0,0]/B=[0,2,2]; got {tallying_df['response'].tolist()}"
    )


def test_rating_value_out_of_range_rejected():
    """Rating values above rating_max must be rejected."""
    with pytest.raises(ValueError):
        HeuristicDecisionMakingExperiment(
            validities=[0.9, 0.7, 0.5],
            rating_max=3,
            trial_a_ratings=[[5, 0, 0]],  # 5 > rating_max=3
            trial_b_ratings=[[0, 2, 2]],
        )
