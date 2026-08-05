import math

import numpy as np
import pytest

from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.jsd import jsd, choice_matrix, static_jsd, sequence_jsd
from src.theory import Theory

LN2 = math.log(2.0)

THEORIES_DIR = "theories/heuristic_decision_making"


def _load(name: str) -> Theory:
    return Theory.from_yaml(f"{THEORIES_DIR}/{name}.yaml")


def _experiment(trial_a, trial_b, validities) -> DecisionMakingBinaryExperiment:
    return DecisionMakingBinaryExperiment(
        validities=validities,
        trial_a_ratings=trial_a,
        trial_b_ratings=trial_b,
    )


# A design where TTB and WADD disagree: the single most valid cue favours A,
# but the three lower-validity cues all favour B (so weighted/added evidence
# favours B for any validity spread where 3 moderate cues outweigh 1 strong).
CONFLICT_A = [[1, 0, 0, 0]] * 4
CONFLICT_B = [[0, 1, 1, 1]] * 4
# A design where every compensatory and non-compensatory heuristic agrees:
# option A dominates on every cue.
AGREE_A = [[1, 1, 1, 1]] * 4
AGREE_B = [[0, 0, 0, 0]] * 4
VALIDITIES = [0.95, 0.7, 0.65, 0.6]


def test_jsd_basic_properties():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    u = np.array([0.5, 0.5])
    assert jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert jsd(p, q) == pytest.approx(jsd(q, p), abs=1e-12)
    assert 0.0 <= jsd(p, q) <= LN2 + 1e-12
    # maximally separated point masses hit the ln2 bound
    assert jsd(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(LN2)
    assert jsd(p, u) < jsd(p, q)


def test_choice_matrix_shape():
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    ttb = _load("ttb_sampling")
    m = choice_matrix(ttb, exp, n_runs=10)
    assert m.ndim == 2 and m.shape[0] == 10
    assert set(np.unique(m)) <= {0, 1}


def test_self_jsd_near_zero():
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    ttb = _load("ttb_sampling")
    assert static_jsd(ttb, ttb, exp, n_runs=300) < 0.02
    assert sequence_jsd(ttb, ttb, exp, n_runs=300) < 0.02


def test_conflict_design_separates_more_than_agree_design():
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    conflict = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    agree = _experiment(AGREE_A, AGREE_B, VALIDITIES)
    j_conflict = static_jsd(ttb, wadd, conflict, n_runs=300)
    j_agree = static_jsd(ttb, wadd, agree, n_runs=300)
    assert j_conflict > 5 * max(j_agree, 1e-4)


def _coin(p: float) -> Theory:
    """A stimulus-ignoring Bernoulli theory: P(choose B) = p on every trial.

    Gives an ANALYTIC ground truth: the per-trial choice distribution is
    exactly [1-p, p] independent of the design, so the true static JSD
    between two coins is the closed-form `jsd([1-p1, p1], [1-p2, p2])`.
    """
    return Theory(
        theory=f"Always pick B with probability {p}, ignoring the stimulus.",
        predict=f"def predict(parameters, stimulus, history):\n    return np.array([1.0 - {p}, {p}])\n",
        policy=(
            "def policy(probabilities):\n"
            "    probabilities = probabilities / probabilities.sum()\n"
            "    return int(np.random.choice(len(probabilities), p=probabilities))\n"
        ),
        parameters={},
    )


@pytest.mark.parametrize("p1,p2", [(0.5, 0.5), (0.2, 0.8), (0.5, 0.9), (0.3, 0.7)])
def test_static_jsd_matches_closed_form_bernoulli(p1, p2):
    # Analytic oracle: estimated static JSD must converge to the exact
    # Bernoulli JSD, not merely order correctly. n_runs/tol set so the
    # Monte-Carlo + plug-in error stays well under the tolerance.
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    analytic = jsd(np.array([1 - p1, p1]), np.array([1 - p2, p2]))
    estimate = static_jsd(_coin(p1), _coin(p2), exp, n_runs=1500)
    assert estimate == pytest.approx(analytic, abs=0.02)


def test_static_jsd_exact_corners():
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    # Deterministic opposite choices -> exactly ln 2 every run (no sampling).
    assert static_jsd(_coin(0.0), _coin(1.0), exp, n_runs=200) == pytest.approx(LN2, abs=1e-9)
    # Identical coins -> 0 up to plug-in bias.
    assert static_jsd(_coin(0.7), _coin(0.7), exp, n_runs=1500) < 0.01


def test_sequence_jsd_separates_perseveration_from_alternation():
    # ANALYTIC corner: perseveration and alternation have ~identical per-trial
    # marginals (both ~50/50) -> static JSD ~ 0; but opposite deterministic
    # lag-1 joints -> sequence JSD ~ ln 2 (diluted only by the random first
    # trial). This is the one job sequence-awareness exists for.
    pers, alt = _load("perseveration"), _load("alternating")
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    assert static_jsd(pers, alt, exp, n_runs=1000) < 0.02
    assert sequence_jsd(pers, alt, exp, n_runs=1000) == pytest.approx(LN2, abs=0.03)


def test_sequence_jsd_catches_history_dependence():
    # Perseveration repeats its previous choice; its per-trial MARGINAL on a
    # symmetric design is ~uniform, matching alternation's — but their lag-1
    # joints are opposite point masses ((0,0)/(1,1) vs (0,1)/(1,0)).
    pers = _load("perseveration")
    alt = _load("alternating")
    exp = _experiment(CONFLICT_A, CONFLICT_B, VALIDITIES)
    j_static = static_jsd(pers, alt, exp, n_runs=300)
    j_seq = sequence_jsd(pers, alt, exp, n_runs=300)
    assert j_seq > j_static
