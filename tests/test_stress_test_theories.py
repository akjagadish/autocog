"""Behavioral tests for the four non-canonical stress-test baseline theories:

  - theories/heuristic_decision_making/perseveration.yaml
  - theories/heuristic_decision_making/take_the_worst.yaml
  - theories/heuristic_decision_making/single_cue.yaml
  - theories/heuristic_decision_making/cue_parity.yaml

These mirror tests/test_anti_majority_alternating.py: each theory is loaded and
executed exactly as the pipeline runs it (`Theory.from_yaml` ->
`theory.predict(parameters, state, history)`), so the tests exercise the real
YAML code, not a re-implementation.

`state` is the per-trial stimulus the experiment yields: `(option_a_ratings,
option_b_ratings)`, i.e. a (2, n_features) array once `np.asarray`-d. `history`
is the running dict the experiment maintains; only its "response" key (the list
of realized 0/1 choices so far) matters for the sequential theory.

Each theory stresses a different recovery axis (see SKILL.md):
  * perseveration  - sequential; mirror of `alternating` (repeats instead of
                     switches -> lag-1 autocorr +1 vs -1).
  * take_the_worst - feature-based, but consults cues by ASCENDING validity
                     (the cue we do NOT bias for).
  * single_cue     - extreme frugality: only the single least-valid cue is read.
  * cue_parity     - non-monotone in evidence (parity of A-favoring cues), a
                     hard case like `anti_majority`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.decision_making_binary_features.experiment import (
    DecisionMakingBinaryExperiment,
)
from src.theory import Theory

REPO_ROOT = Path(__file__).resolve().parents[1]
THEORIES_DIR = REPO_ROOT / "theories/heuristic_decision_making"

PERSEVERATION_YAML = THEORIES_DIR / "perseveration.yaml"
TAKE_THE_WORST_YAML = THEORIES_DIR / "take_the_worst.yaml"
SINGLE_CUE_YAML = THEORIES_DIR / "single_cue.yaml"
CUE_PARITY_YAML = THEORIES_DIR / "cue_parity.yaml"

# Near-deterministic, lapse-free knobs so `predict` reads off the pure winner.
_SHARP = {"beta": 20.0, "epsilon": 0.0}


def _empty_history() -> dict:
    return {"option_a_ratings": [], "option_b_ratings": [], "response": []}


@pytest.fixture
def perseveration() -> Theory:
    return Theory.from_yaml(str(PERSEVERATION_YAML))


@pytest.fixture
def take_the_worst() -> Theory:
    return Theory.from_yaml(str(TAKE_THE_WORST_YAML))


@pytest.fixture
def single_cue() -> Theory:
    return Theory.from_yaml(str(SINGLE_CUE_YAML))


@pytest.fixture
def cue_parity() -> Theory:
    return Theory.from_yaml(str(CUE_PARITY_YAML))


# ---------------------------------------------------------------------------
# Perseveration (pure repeat baseline)
#
# Contract: ignore the stimulus entirely. First trial = coin flip; every later
# trial = a REPEAT of the previous realized response. The exact mirror image of
# `alternating`, so a discovery procedure that nails one must distinguish it
# from the other purely by the SIGN of the sequential dependency.
# ---------------------------------------------------------------------------


def test_perseveration_first_trial_is_coin_flip(perseveration: Theory):
    """No prior response -> uniform over the two options."""
    probs = perseveration.predict({}, ([5, 2], [1, 9]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


def test_perseveration_repeats_last_choice(perseveration: Theory):
    """Last response A (0) -> deterministically repeat A (0); same for B."""
    after_a = perseveration.predict({}, ([5, 2], [1, 9]), {"response": [0]})
    assert np.allclose(after_a, [1.0, 0.0])

    after_b = perseveration.predict({}, ([5, 2], [1, 9]), {"response": [1]})
    assert np.allclose(after_b, [0.0, 1.0])


def test_perseveration_uses_only_most_recent_response(perseveration: Theory):
    """Only history[-1] matters; earlier choices are irrelevant. Last = 1 here,
    so the next prediction must repeat B (1)."""
    probs = perseveration.predict({}, ([3, 3], [7, 1]), {"response": [0, 1, 0, 1]})
    assert np.allclose(probs, [0.0, 1.0])


def test_perseveration_ignores_the_stimulus(perseveration: Theory):
    """Same history, wildly different stimuli -> identical prediction."""
    hist = {"response": [0]}
    p1 = perseveration.predict({}, ([0, 0], [9, 9]), hist)
    p2 = perseveration.predict({}, ([9, 9], [0, 0]), hist)
    assert np.allclose(p1, p2)
    assert np.allclose(p1, [1.0, 0.0])


def test_perseveration_produces_constant_runs_end_to_end(perseveration: Theory):
    """End-to-end through the real binary experiment: every subject's realized
    response sequence must be CONSTANT (r[i] == r[i-1] for all i>=1) — the
    opposite of alternating. The first response may be either option (coin
    flip); only the repeat is enforced."""
    exp = DecisionMakingBinaryExperiment(
        validities=[0.9, 0.6],
        trial_a_ratings=[[1, 0], [0, 1], [1, 1]],
        trial_b_ratings=[[0, 1], [1, 0], [0, 0]],
    )
    df = exp.simulate(perseveration, n_runs=5)

    for subject_id, group in df.groupby("subject_id"):
        responses = group["response"].tolist()
        assert len(responses) > 1, "need multiple trials to test perseveration"
        consecutive_equal = [
            responses[i] == responses[i - 1] for i in range(1, len(responses))
        ]
        assert all(consecutive_equal), (
            f"subject {subject_id} failed to perseverate: {responses}"
        )


# ---------------------------------------------------------------------------
# Take The Worst (least-valid one-reason)
#
# Contract: TTB's cue cascade, but cues are consulted in ASCENDING validity.
# The first discriminating (strict-inequality) cue decides; guess if none.
# ---------------------------------------------------------------------------


def test_take_the_worst_decides_on_least_valid_cue(take_the_worst: Theory):
    """validities [0.9, 0.6], A=[1,0], B=[0,1]. The least-valid cue (feature 1)
    discriminates first: B>A there, so B (1) wins — the OPPOSITE of TTB, which
    would stop on the high-validity feature 0 and pick A."""
    params = {**_SHARP, "validities": [0.9, 0.6]}
    probs = take_the_worst.predict(params, ([1, 0], [0, 1]), _empty_history())
    assert int(np.argmax(probs)) == 1
    assert probs[1] > 0.99


def test_take_the_worst_mirror_picks_A(take_the_worst: Theory):
    """Flip the stimulus: A=[0,1], B=[1,0]. Least-valid cue (feature 1) has
    A>B, so A (0) wins."""
    params = {**_SHARP, "validities": [0.9, 0.6]}
    probs = take_the_worst.predict(params, ([0, 1], [1, 0]), _empty_history())
    assert int(np.argmax(probs)) == 0
    assert probs[0] > 0.99


def test_take_the_worst_cascades_when_worst_cue_ties(take_the_worst: Theory):
    """If the least-valid cue ties, move UP to the next-least-valid cue.
    validities [0.9, 0.6], A=[1,1], B=[0,1]: feature 1 ties (1==1), so feature 0
    decides: A>B -> A (0). (single_cue, by contrast, would guess here.)"""
    params = {**_SHARP, "validities": [0.9, 0.6]}
    probs = take_the_worst.predict(params, ([1, 1], [0, 1]), _empty_history())
    assert int(np.argmax(probs)) == 0
    assert probs[0] > 0.99


def test_take_the_worst_guesses_when_no_cue_discriminates(take_the_worst: Theory):
    params = {**_SHARP, "validities": [0.9, 0.6]}
    probs = take_the_worst.predict(params, ([1, 1], [1, 1]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Single-Cue (least-valid cue only)
#
# Contract: read ONLY the single lowest-validity cue; the option ahead on that
# one cue wins; guess if it ties. No other cue is ever consulted.
# ---------------------------------------------------------------------------


def test_single_cue_uses_only_the_least_valid_cue(single_cue: Theory):
    """validities [0.9, 0.6, 0.55] -> least valid is feature 2. A=[1,1,0],
    B=[0,0,1]: feature 2 has B>A, so B (1) wins even though A leads on the two
    higher-validity cues."""
    params = {**_SHARP, "validities": [0.9, 0.6, 0.55]}
    probs = single_cue.predict(params, ([1, 1, 0], [0, 0, 1]), _empty_history())
    assert int(np.argmax(probs)) == 1
    assert probs[1] > 0.99


def test_single_cue_mirror_picks_A(single_cue: Theory):
    params = {**_SHARP, "validities": [0.9, 0.6, 0.55]}
    probs = single_cue.predict(params, ([0, 0, 1], [1, 1, 0]), _empty_history())
    assert int(np.argmax(probs)) == 0
    assert probs[0] > 0.99


def test_single_cue_guesses_when_least_valid_cue_ties(single_cue: Theory):
    """The defining contrast with take_the_worst: when the least-valid cue ties,
    single_cue GUESSES (it never consults another cue), even though a higher cue
    discriminates. validities [0.9, 0.6], A=[1,1], B=[0,1]: feature 1 ties ->
    guess, despite feature 0 favoring A."""
    params = {**_SHARP, "validities": [0.9, 0.6]}
    probs = single_cue.predict(params, ([1, 1], [0, 1]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Cue-Parity (XOR of A-favoring cues)
#
# Contract: count the cues where A strictly beats B; prefer A iff that count is
# ODD, else prefer B. Non-monotone: adding one more A-winning cue flips the
# preference, so dominance does NOT imply choice.
# ---------------------------------------------------------------------------


def test_cue_parity_odd_a_wins_picks_A(cue_parity: Theory):
    """One A-winning cue (odd) -> A. A=[1,0], B=[0,0]: A beats B on feature 0
    only -> a_wins=1 (odd) -> A (0)."""
    params = {**_SHARP}
    probs = cue_parity.predict(params, ([1, 0], [0, 0]), _empty_history())
    assert int(np.argmax(probs)) == 0
    assert probs[0] > 0.99


def test_cue_parity_even_a_wins_picks_B_despite_dominance(cue_parity: Theory):
    """The signature non-monotone case: A DOMINATES B (a_wins=2, even) yet the
    model picks B. A=[1,1], B=[0,0] -> a_wins=2 -> B (1)."""
    params = {**_SHARP}
    probs = cue_parity.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert int(np.argmax(probs)) == 1
    assert probs[1] > 0.99


def test_cue_parity_zero_a_wins_picks_B(cue_parity: Theory):
    """Zero A-winning cues is even -> B. A=[0,0], B=[1,1] -> a_wins=0 -> B (1)."""
    params = {**_SHARP}
    probs = cue_parity.predict(params, ([0, 0], [1, 1]), _empty_history())
    assert int(np.argmax(probs)) == 1


def test_cue_parity_three_a_wins_picks_A(cue_parity: Theory):
    """Three A-winning cues (odd) -> A. A=[1,1,1], B=[0,0,0] -> a_wins=3 -> A."""
    params = {**_SHARP}
    probs = cue_parity.predict(params, ([1, 1, 1], [0, 0, 0]), _empty_history())
    assert int(np.argmax(probs)) == 0


# ---------------------------------------------------------------------------
# Shared noise-channel checks (the beta/epsilon softmax+lapse block, reused
# from ttb.yaml). Exercised on the feature-based theories.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name, stimulus, params",
    [
        ("take_the_worst", ([1, 0], [0, 1]), {"validities": [0.9, 0.6]}),
        ("single_cue", ([1, 1, 0], [0, 0, 1]), {"validities": [0.9, 0.6, 0.55]}),
        ("cue_parity", ([1, 0], [0, 0]), {}),
    ],
)
def test_feature_theories_return_valid_distribution(
    request, fixture_name, stimulus, params
):
    theory = request.getfixturevalue(fixture_name)
    probs = theory.predict({**_SHARP, **params}, stimulus, _empty_history())
    assert probs.shape == (2,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


@pytest.mark.parametrize(
    "fixture_name, stimulus, params",
    [
        ("take_the_worst", ([1, 0], [0, 1]), {"validities": [0.9, 0.6]}),
        ("cue_parity", ([1, 0], [0, 0]), {}),
    ],
)
def test_feature_theories_low_beta_is_near_random(
    request, fixture_name, stimulus, params
):
    """beta -> 0 collapses the softmax over the binary winner score toward
    50/50: the chosen direction survives but confidence vanishes."""
    theory = request.getfixturevalue(fixture_name)
    probs = theory.predict(
        {"beta": 0.0, "epsilon": 0.0, **params}, stimulus, _empty_history()
    )
    assert np.allclose(probs, [0.5, 0.5])


@pytest.mark.parametrize(
    "fixture_name, stimulus, params",
    [
        ("take_the_worst", ([1, 0], [0, 1]), {"validities": [0.9, 0.6]}),
        ("cue_parity", ([1, 0], [0, 0]), {}),
    ],
)
def test_feature_theories_epsilon_lapse_mixes_toward_uniform(
    request, fixture_name, stimulus, params
):
    """Lapse is a convex mixture (1-eps)*p_core + eps*uniform. With a sharp
    winner (p_core ~ one-hot) and eps=0.5 the output is pulled exactly halfway
    to uniform — one coordinate 0.75, the other 0.25."""
    theory = request.getfixturevalue(fixture_name)
    probs = theory.predict(
        {"beta": 20.0, "epsilon": 0.5, **params}, stimulus, _empty_history()
    )
    hi, lo = max(probs), min(probs)
    assert np.isclose(hi, 0.75, atol=1e-6)
    assert np.isclose(lo, 0.25, atol=1e-6)
