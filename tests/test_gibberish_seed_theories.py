"""Behavioral tests for the two 'gibberish' seed theories used to stress-test
recovery from non-canonical seeds:

  - theories/heuristic_decision_making/coin_flip.yaml
  - theories/heuristic_decision_making/stimulus_parity.yaml

Both are deliberately nonsense rules, maximally *far* from ttb / wadd /
tallying (all of which rank the two options by cue evidence): coin_flip
ignores everything and guesses; stimulus_parity keys the choice on the parity
of the total feature-bit sum, a chaotic hash that does not track which option
has better/more-valid cues. They exist only to seed the adversarial loop with
garbage so the discovery of the ground truth (e.g. wadd) is unassisted.

The theories are loaded and executed exactly as the pipeline runs them
(`Theory.from_yaml` -> `theory.predict(parameters, stimulus, history)`), so
the tests exercise the real YAML code, not a re-implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.theory import Theory

REPO_ROOT = Path(__file__).resolve().parents[1]
COIN_FLIP_YAML = REPO_ROOT / "theories/heuristic_decision_making/coin_flip.yaml"
PARITY_YAML = REPO_ROOT / "theories/heuristic_decision_making/stimulus_parity.yaml"

# Near-deterministic, lapse-free knobs so `predict` reads off the pure winner.
_SHARP = {"beta": 20.0, "epsilon": 0.0}


def _empty_history() -> dict:
    return {"option_a_ratings": [], "option_b_ratings": [], "response": []}


@pytest.fixture
def coin_flip() -> Theory:
    return Theory.from_yaml(str(COIN_FLIP_YAML))


@pytest.fixture
def parity() -> Theory:
    return Theory.from_yaml(str(PARITY_YAML))


# ---------------------------------------------------------------------------
# coin_flip — pure uniform guess, ignores stimulus, history, and parameters.
# ---------------------------------------------------------------------------


def test_coin_flip_is_exactly_uniform(coin_flip: Theory):
    probs = coin_flip.predict({}, ([1, 0], [0, 1]), _empty_history())
    assert probs.shape == (2,)
    assert np.allclose(probs, [0.5, 0.5])


def test_coin_flip_ignores_stimulus_and_history(coin_flip: Theory):
    """Wildly different stimuli / histories -> identical uniform output."""
    p1 = coin_flip.predict({}, ([1, 1], [0, 0]), {"response": []})
    p2 = coin_flip.predict({}, ([0, 0], [1, 1]), {"response": [0, 1, 0]})
    assert np.allclose(p1, p2)
    assert np.allclose(p1, [0.5, 0.5])


def test_coin_flip_probabilities_are_a_valid_distribution(coin_flip: Theory):
    probs = coin_flip.predict({}, ([1, 0], [0, 1]), _empty_history())
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# stimulus_parity — winner = A if the total feature-bit sum is even, else B;
# then the standard softmax(beta) + lapse(epsilon) noise channel.
# ---------------------------------------------------------------------------


def test_parity_even_total_picks_A(parity: Theory):
    """A=[1,1], B=[0,0]: total = 2 (even) -> winner A (0)."""
    probs = parity.predict(_SHARP, ([1, 1], [0, 0]), _empty_history())
    assert probs.shape == (2,)
    assert int(np.argmax(probs)) == 0
    assert probs[0] > 0.99


def test_parity_odd_total_picks_B(parity: Theory):
    """A=[1,0], B=[0,0]: total = 1 (odd) -> winner B (1)."""
    probs = parity.predict(_SHARP, ([1, 0], [0, 0]), _empty_history())
    assert int(np.argmax(probs)) == 1
    assert probs[1] > 0.99


def test_parity_is_not_an_alias_of_wadd(parity: Theory):
    """On a stimulus where WADD strictly prefers A, parity must pick B, proving
    the parity hash does NOT track cue evidence.

      A=[1,1], B=[1,0], validities=[0.9, 0.6], weights=[1,1]:
        WADD:   score_A = 0.9 + 0.6 = 1.5  >  score_B = 0.9   -> prefers A (0)
        parity: total = 1+1+1+0 = 3 (odd)                     -> picks   B (1)
    """
    probs = parity.predict(_SHARP, ([1, 1], [1, 0]), _empty_history())
    assert int(np.argmax(probs)) == 1, "parity must diverge from WADD's A-preference"


def test_parity_low_beta_is_near_random(parity: Theory):
    """beta -> 0 collapses the softmax over the binary winner score to 50/50."""
    params = {"beta": 0.0, "epsilon": 0.0}
    probs = parity.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


def test_parity_epsilon_lapse_mixes_toward_uniform(parity: Theory):
    """(1-eps)*p_core + eps*uniform. Sharp winner A (p_core ~= [1, 0]) with
    eps=0.5 -> exactly [0.75, 0.25]."""
    params = {"beta": 20.0, "epsilon": 0.5}
    probs = parity.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert np.allclose(probs, [0.75, 0.25], atol=1e-6)


def test_parity_probabilities_are_a_valid_distribution(parity: Theory):
    probs = parity.predict(_SHARP, ([1, 0], [1, 0]), _empty_history())
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)
