"""Behavioral tests for the two sequential/contrarian baseline theories:

  - theories/heuristic_decision_making/anti_majority.yaml
  - theories/heuristic_decision_making/alternating.yaml

These theories are loaded and executed exactly as the pipeline runs them
(`Theory.from_yaml` -> `theory.predict(parameters, state, history)`), so the
tests exercise the real YAML code, not a re-implementation.

`state` is the per-trial stimulus the experiment yields: `(option_a_ratings,
option_b_ratings)`, i.e. a (2, n_features) array once `np.asarray`-d.
`history` is the running dict the experiment maintains; only its "response"
key (the list of realized 0/1 choices so far) matters here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.heuristic_decision_making.experiment import (
    HeuristicDecisionMakingExperiment,
)
from src.theory import Theory

REPO_ROOT = Path(__file__).resolve().parents[1]
ANTI_MAJORITY_YAML = REPO_ROOT / "theories/heuristic_decision_making/anti_majority.yaml"
ALTERNATING_YAML = REPO_ROOT / "theories/heuristic_decision_making/alternating.yaml"


@pytest.fixture
def anti_majority() -> Theory:
    return Theory.from_yaml(str(ANTI_MAJORITY_YAML))


@pytest.fixture
def alternating() -> Theory:
    return Theory.from_yaml(str(ALTERNATING_YAML))


# ---------------------------------------------------------------------------
# Anti-Majority Ensemble
#
# Contract: run Tallying / TTB / WADD as hard (argmax) votes, take the
# majority of the three, and target its COMPLEMENT. Choice noise (softmax
# over beta + lapse over epsilon) is applied ONCE to that flipped decision.
# ---------------------------------------------------------------------------

# Near-deterministic, lapse-free knobs so `predict` reads off the pure flip.
_SHARP = {"beta": 20.0, "epsilon": 0.0}


def _empty_history() -> dict:
    return {"option_a_ratings": [], "option_b_ratings": [], "response": []}


def test_anti_majority_flips_unanimous_A_to_B(anti_majority: Theory):
    """A dominates B on every cue, so Tallying, TTB and WADD all vote A (0).
    The majority is A; anti-majority must therefore target B (1).

    This is the strongest "anti-majority" check: the chosen option opposes
    ALL THREE heuristics at once, which no 'just follow heuristic X' rule
    could ever produce."""
    params = {**_SHARP, "validities": [0.9, 0.6], "weights": [1.0, 1.0]}
    probs = anti_majority.predict(params, ([1, 1], [0, 0]), _empty_history())

    assert probs.shape == (2,)
    assert int(np.argmax(probs)) == 1, "must target B, the complement of the A-majority"
    assert probs[1] > 0.99


def test_anti_majority_flips_unanimous_B_to_A(anti_majority: Theory):
    """Mirror image: B dominates, all three vote B (1), so the model targets A (0)."""
    params = {**_SHARP, "validities": [0.9, 0.6], "weights": [1.0, 1.0]}
    probs = anti_majority.predict(params, ([0, 0], [1, 1]), _empty_history())

    assert int(np.argmax(probs)) == 0, "must target A, the complement of the B-majority"
    assert probs[0] > 0.99


def test_anti_majority_uses_majority_not_unanimity(anti_majority: Theory):
    """A genuine 2-1 split. With validities [0.9, 0.6, 0.55] and
    A=[0,1,1], B=[1,0,0]:

      - Tallying: A wins features 1 & 2, B wins feature 0  -> votes A (0)
      - TTB:      top-validity cue (feature 0) has B>A      -> votes B (1)
      - WADD:     weighted sum A=0.6+0.55=1.15 > B=0.9      -> votes A (0)

    Majority = A (2 of 3). The model must target B even though the vote is
    not unanimous — proving the >=2 majority is computed and complemented,
    not that it merely inverts a unanimous block."""
    params = {**_SHARP, "validities": [0.9, 0.6, 0.55], "weights": [1.0, 1.0, 1.0]}
    probs = anti_majority.predict(params, ([0, 1, 1], [1, 0, 0]), _empty_history())

    assert int(np.argmax(probs)) == 1, "majority is A (2-1); complement B must be targeted"
    assert probs[1] > 0.99


def test_anti_majority_ignores_self_consistent_minority_heuristic(anti_majority: Theory):
    """Reverse the split so the majority lands on B and the model targets A,
    confirming the direction tracks the majority side rather than any fixed
    heuristic. A=[1,0,0], B=[0,1,1], validities [0.9, 0.6, 0.55]:

      - Tallying: B wins features 1 & 2          -> votes B (1)
      - TTB:      top cue (feature 0) has A>B     -> votes A (0)
      - WADD:     B=1.15 > A=0.9                  -> votes B (1)

    Majority = B; complement = A (0)."""
    params = {**_SHARP, "validities": [0.9, 0.6, 0.55], "weights": [1.0, 1.0, 1.0]}
    probs = anti_majority.predict(params, ([1, 0, 0], [0, 1, 1]), _empty_history())

    assert int(np.argmax(probs)) == 0, "majority is B (2-1); complement A must be targeted"
    assert probs[0] > 0.99


def test_anti_majority_probabilities_are_a_valid_distribution(anti_majority: Theory):
    params = {**_SHARP, "validities": [0.9, 0.6], "weights": [1.0, 1.0]}
    probs = anti_majority.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert probs.shape == (2,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_anti_majority_low_beta_is_near_random(anti_majority: Theory):
    """beta -> 0 makes the softmax over the binary flip score collapse toward
    50/50: the contrarian *direction* survives but the confidence vanishes."""
    params = {"beta": 0.0, "epsilon": 0.0, "validities": [0.9, 0.6], "weights": [1.0, 1.0]}
    probs = anti_majority.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


def test_anti_majority_epsilon_lapse_mixes_toward_uniform(anti_majority: Theory):
    """The lapse is a convex mixture: (1-eps)*p_core + eps*uniform. With a
    sharp flip toward B (p_core ~= [0, 1]) and eps=0.5 the output must be
    exactly [0.25, 0.75], i.e. pulled halfway to uniform but still leaning B."""
    params = {"beta": 20.0, "epsilon": 0.5, "validities": [0.9, 0.6], "weights": [1.0, 1.0]}
    probs = anti_majority.predict(params, ([1, 1], [0, 0]), _empty_history())
    assert np.allclose(probs, [0.25, 0.75], atol=1e-6)


# ---------------------------------------------------------------------------
# Alternation (pure switch baseline)
#
# Contract: ignore the stimulus entirely. First trial = coin flip; every
# later trial = the opposite of the previous realized response.
# ---------------------------------------------------------------------------


def test_alternating_first_trial_is_coin_flip(alternating: Theory):
    """No prior response -> uniform over the two options."""
    probs = alternating.predict({}, ([5, 2], [1, 9]), _empty_history())
    assert np.allclose(probs, [0.5, 0.5])


def test_alternating_switches_away_from_last_choice(alternating: Theory):
    """Last response A (0) -> deterministically pick B (1), and vice versa."""
    after_a = alternating.predict({}, ([5, 2], [1, 9]), {"response": [0]})
    assert np.allclose(after_a, [0.0, 1.0])

    after_b = alternating.predict({}, ([5, 2], [1, 9]), {"response": [1]})
    assert np.allclose(after_b, [1.0, 0.0])


def test_alternating_uses_only_the_most_recent_response(alternating: Theory):
    """Only history[-1] matters; earlier choices are irrelevant. Last = 0 here
    despite a longer history, so the next prediction must be B (1)."""
    probs = alternating.predict({}, ([3, 3], [7, 1]), {"response": [1, 0, 1, 0]})
    assert np.allclose(probs, [0.0, 1.0])


def test_alternating_ignores_the_stimulus(alternating: Theory):
    """Same history, wildly different stimuli -> identical prediction."""
    hist = {"response": [1]}
    p1 = alternating.predict({}, ([0, 0], [9, 9]), hist)
    p2 = alternating.predict({}, ([9, 9], [0, 0]), hist)
    assert np.allclose(p1, p2)
    assert np.allclose(p1, [1.0, 0.0])


def test_alternating_produces_strictly_alternating_runs_end_to_end(alternating: Theory):
    """End-to-end through the real experiment: every subject's realized
    response sequence must strictly alternate (r[i] != r[i-1] for all i>=1).
    The first response may be either option (coin flip); only the switching
    is enforced."""
    exp = HeuristicDecisionMakingExperiment(
        validities=[0.9, 0.6],
        rating_max=1,
        trial_a_ratings=[[1, 0], [0, 1], [1, 1]],
        trial_b_ratings=[[0, 1], [1, 0], [0, 0]],
    )
    df = exp.simulate(alternating, n_runs=5)

    for subject_id, group in df.groupby("subject_id"):
        responses = group["response"].tolist()
        assert len(responses) > 1, "need multiple trials to test alternation"
        consecutive_equal = [
            responses[i] == responses[i - 1] for i in range(1, len(responses))
        ]
        assert not any(consecutive_equal), (
            f"subject {subject_id} failed to strictly alternate: {responses}"
        )
