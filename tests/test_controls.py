import pytest

from src.controls import NeutralAutoCog, render_neutral_experiment_proposal
from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.theory import Theory

THEORIES_DIR = "theories/heuristic_decision_making"


def _load(name: str) -> Theory:
    return Theory.from_yaml(f"{THEORIES_DIR}/{name}.yaml")


def test_neutral_prompt_is_symmetric_and_complete():
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    system, user = render_neutral_experiment_proposal(
        experiment_class=DecisionMakingBinaryExperiment,
        theory_1=ttb, theory_2=wadd, ledger=[],
    )
    # both theories' code shown; no advocacy framing
    assert ttb.predict_source.strip()[:40] in user
    assert wadd.predict_source.strip()[:40] in user
    for banned in ("advocat", "your theory", "competing theory", "adversary"):
        assert banned not in system.lower()
        assert banned not in user.lower()
    assert "distinguish" in (system + user).lower()


def test_neutral_autocog_overrides_only_experiment_proposal():
    # propose_round (metric + Welch machinery) must be inherited unchanged;
    # only the experiment-proposal LLM call is swapped.
    from src.autocog import AutoCog

    assert issubclass(NeutralAutoCog, AutoCog)
    assert "propose_round" not in NeutralAutoCog.__dict__
    assert "_llm_propose_experiment" in NeutralAutoCog.__dict__


# --- JsdAutoCog (jsd_metric control) ----------------------------------------

from src.controls import JsdAutoCog, make_projection_metric
from src.jsd import choice_matrix, _per_trial_bernoulli
from src.observation import Observations


def _conflict_experiment():
    return DecisionMakingBinaryExperiment(
        validities=[0.95, 0.7, 0.65, 0.6],
        trial_a_ratings=[[1, 0, 0, 0]] * 4,
        trial_b_ratings=[[0, 1, 1, 1]] * 4,
    )


def _mixed_experiment():
    # Heterogeneous pairs so trial ORDER carries information — required to
    # make the order-invariance test non-trivial.
    return DecisionMakingBinaryExperiment(
        validities=[0.95, 0.7, 0.65, 0.6],
        trial_a_ratings=[[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
        trial_b_ratings=[[0, 1, 1, 1], [1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0]],
    )


def test_projection_metric_separates_proposing_theories():
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    exp = _conflict_experiment()
    metric = make_projection_metric(exp, choice_matrix(ttb, exp, n_runs=200),
                                    choice_matrix(wadd, exp, n_runs=200))
    est_ttb = metric(exp.simulate(ttb, n_runs=100))
    est_wadd = metric(exp.simulate(wadd, n_runs=100))
    # projection onto (p_ttb - p_wadd) must score ttb data higher
    assert est_ttb.value > est_wadd.value
    assert est_ttb.variance is not None  # Welch downstream needs variance


def test_projection_metric_is_trial_order_invariant():
    # Experiments reshuffle their trial order on every (re)construction
    # (model_post_init, unseeded). The metric is persisted and re-applied to
    # data simulated from RECONSTRUCTED instances (crash-resume, rescoring),
    # so it must key weights by pair CONTENT, not trial position: permuting
    # rows within a subject must not change the metric value.
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    exp = _mixed_experiment()
    metric = make_projection_metric(exp, choice_matrix(ttb, exp, n_runs=200),
                                    choice_matrix(wadd, exp, n_runs=200))
    data = exp.simulate(ttb, n_runs=50)
    # Global row shuffle: scrambles each subject's trial order (and interleaves
    # subjects, which the metric re-groups by subject_id). Keeps the
    # subject_id column intact — unlike groupby.apply, which drops grouping
    # columns on pandas >= 3.0.
    shuffled = data.sample(frac=1.0, random_state=0).reset_index(drop=True)
    est = metric(data)
    est_shuffled = metric(shuffled)
    assert est.value == pytest.approx(est_shuffled.value, abs=1e-12)


def test_jsd_autocog_accepts_discriminating_experiment(monkeypatch):
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1 = JsdAutoCog(label="pi_1", theory=ttb,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    pi_2 = JsdAutoCog(label="pi_2", theory=wadd,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    # stub the single LLM call: always "propose" the conflict design
    monkeypatch.setattr(
        JsdAutoCog, "_llm_propose_experiment",
        lambda self, adversary, **kw: _conflict_experiment(),
    )
    pool = Observations()
    obs = pi_1.propose_round(pi_2, pool, n_runs=200)
    assert obs.metric is not None  # JSD-to-self metric attached
    p1 = obs.prediction_by_label("pi_1")
    p2 = obs.prediction_by_label("pi_2")
    assert p1 is not None and p2 is not None
    # On JSD-to-self the proposer sits near its own floor (~0) while the
    # adversary is the gate JSD away from it, so they separate.
    assert p1.value != p2.value
    assert p1.value < p2.value


def test_sequence_projection_separates_history_dependent_theories():
    # The STATIC projection metric is blind to pure history dependence: for
    # perseveration vs alternation, every per-trial-content choice prob is
    # ~0.5, so static weights ~0 and the metric cannot separate them. The
    # SEQUENCE projection keys on (content, previous response), so it sees the
    # stay/switch structure and separates them strongly. Analytic expectation:
    # under a (perseveration-vs-alternation) sequence metric, perseveration
    # data scores well above alternation data; the static metric gives ~0 gap.
    from src.controls import make_sequence_projection_metric

    pers, alt = _load("perseveration"), _load("alternating")
    exp = _conflict_experiment()
    m_pers = choice_matrix(pers, exp, n_runs=300)
    m_alt = choice_matrix(alt, exp, n_runs=300)

    static = make_projection_metric(exp, m_pers, m_alt)
    static_gap = static(exp.simulate(pers, n_runs=150)).value - \
        static(exp.simulate(alt, n_runs=150)).value
    assert abs(static_gap) < 0.05  # static is blind to history

    seq = make_sequence_projection_metric(exp, m_pers, m_alt)
    seq_gap = seq(exp.simulate(pers, n_runs=150)).value - \
        seq(exp.simulate(alt, n_runs=150)).value
    assert seq_gap > 0.3  # sequence metric separates them strongly


def test_sequence_projection_still_separates_stimulus_theories():
    # The sequence metric must SUBSUME the static one: for stimulus-driven
    # theories (ttb vs wadd) it must still score self-theory data higher.
    from src.controls import make_sequence_projection_metric

    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    exp = _conflict_experiment()
    seq = make_sequence_projection_metric(
        exp, choice_matrix(ttb, exp, n_runs=300), choice_matrix(wadd, exp, n_runs=300))
    assert seq(exp.simulate(ttb, n_runs=150)).value > \
        seq(exp.simulate(wadd, n_runs=150)).value


def test_jsd_autocog_survives_a_crashing_experiment(monkeypatch):
    # Baseline AutoCog.propose_round catches metric-eval exceptions, rejects
    # that attempt, and continues. JsdAutoCog must do the same: one bad
    # experiment (e.g. choice_matrix throws on a degenerate design) must not
    # crash a multi-round run — it should be rejected and a later good
    # attempt accepted.
    import src.controls as controls

    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1 = JsdAutoCog(label="pi_1", theory=ttb,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    pi_2 = JsdAutoCog(label="pi_2", theory=wadd,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    monkeypatch.setattr(
        JsdAutoCog, "_llm_propose_experiment",
        lambda self, adversary, **kw: _conflict_experiment(),
    )
    real_choice_matrix = controls.choice_matrix
    calls = {"n": 0}

    def _flaky_choice_matrix(theory, experiment, *, n_runs):
        calls["n"] += 1
        if calls["n"] == 1:  # first attempt's first sim throws
            raise RuntimeError("simulated degenerate-design failure")
        return real_choice_matrix(theory, experiment, n_runs=n_runs)

    monkeypatch.setattr(controls, "choice_matrix", _flaky_choice_matrix)
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=200, max_experiments=3)
    assert obs is not None and obs.metric is not None  # did not crash
    assert obs.prediction_by_label("pi_1") is not None


# --- JSD-to-self metric (jsd_metric downstream distance) --------------------
# Matches the baseline metric's structure: defined at PROPOSAL time from the
# proposing theory's simulations alone, absolute scale [0, ln2], frozen
# thereafter. metric(data) = count-weighted mean over (content, prev) states
# of the binary JSD between the data's conditional choice probability and the
# proposing theory's baked one.


def _content_coin(p_first_cue_high: float, p_otherwise: float):
    """Stimulus-dependent Bernoulli theory with ANALYTIC conditionals:
    P(choose B) = p_first_cue_high when option A's first cue is 1, else
    p_otherwise. History-independent, so every (content, prev) cell of its
    conditional profile equals the per-content p exactly."""
    return Theory(
        theory=(
            f"Choose B with probability {p_first_cue_high} when option A's "
            f"first cue is 1, else {p_otherwise}; ignores everything else."
        ),
        predict=(
            "def predict(parameters, stimulus, history):\n"
            f"    p = {p_first_cue_high} if stimulus[0][0] == 1 else {p_otherwise}\n"
            "    return np.array([1.0 - p, p])\n"
        ),
        policy=(
            "def policy(probabilities):\n"
            "    probabilities = probabilities / probabilities.sum()\n"
            "    return int(np.random.choice(len(probabilities), p=probabilities))\n"
        ),
        parameters={},
    )


def _two_content_experiment():
    # Two distinct trial contents, distinguishable by option A's first cue:
    # X = ([1,0,0,0] vs [0,1,1,1]) and Y = ([0,1,1,0] vs [1,0,0,1]),
    # in equal proportion (each duplicated, so 48/48 of the 96 trials).
    return DecisionMakingBinaryExperiment(
        validities=[0.95, 0.7, 0.65, 0.6],
        trial_a_ratings=[[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
        trial_b_ratings=[[0, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
    )


def test_jsd_to_self_metric_matches_closed_form():
    # ANALYTIC ORACLE. Baked theory: P(B)=0.9 on content X, 0.7 on content Y.
    # Data theory: P(B)=0.2 everywhere. X and Y are equally frequent, so
    #   metric = ½·JSD([.8,.2],[.1,.9]) + ½·JSD([.8,.2],[.3,.7]) ≈ 0.2040 nats
    # (`jsd` itself is validated against closed forms in test_jsd.py).
    import numpy as np

    from src.controls import make_jsd_to_self_metric
    from src.jsd import jsd

    exp = _two_content_experiment()
    baked = _content_coin(0.9, 0.7)
    data_theory = _content_coin(0.2, 0.2)
    metric = make_jsd_to_self_metric(exp, choice_matrix(baked, exp, n_runs=1000))
    est = metric(exp.simulate(data_theory, n_runs=400))
    analytic = 0.5 * (
        jsd(np.array([0.8, 0.2]), np.array([0.1, 0.9]))
        + jsd(np.array([0.8, 0.2]), np.array([0.3, 0.7]))
    )
    assert est.value == pytest.approx(analytic, abs=0.02)
    assert est.variance is not None  # single-subject re-application must work


def test_jsd_to_self_metric_exact_corners():
    import math

    from src.controls import make_jsd_to_self_metric

    exp = _two_content_experiment()
    # Identical theory on both sides -> plug-in floor near 0.
    coin7 = _content_coin(0.7, 0.7)
    floor_metric = make_jsd_to_self_metric(exp, choice_matrix(coin7, exp, n_runs=1000))
    assert floor_metric(exp.simulate(coin7, n_runs=400)).value < 0.01
    # Deterministic opposite choice per content -> every visited
    # (content, prev) cell is a 0-vs-1 point mass -> exactly ln 2 (no
    # sampling noise; relies only on the 96-trial shuffle containing all
    # four content transitions, which fails with negligible probability).
    det = _content_coin(1.0, 0.0)
    anti = _content_coin(0.0, 1.0)
    metric = make_jsd_to_self_metric(exp, choice_matrix(det, exp, n_runs=50))
    assert metric(exp.simulate(anti, n_runs=20)).value == pytest.approx(
        math.log(2.0), abs=1e-9
    )


def test_jsd_to_self_fires_on_pool_shared_error_where_projection_is_silent():
    # Miniature of the anti_majority run failure. The design mixes
    # "conflict" pairs (TTB vs WADD disagree -> that is the projection
    # axis) with "majority-trap" pairs (TTB, Tallying and WADD all pick A,
    # so anti_majority picks B -> a direction the pool SHARES, orthogonal
    # to the projection axis). anti_majority mimics TTB on conflict pairs
    # (majority says B, so anti says A) and flips on trap pairs. The
    # projection metric therefore scores anti_majority data as fitting the
    # pool; JSD-to-self reads the absolute divergence and fires.
    from src.controls import make_jsd_to_self_metric, make_sequence_projection_metric

    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    anti = _load("anti_majority")
    exp = DecisionMakingBinaryExperiment(
        validities=[0.95, 0.7, 0.65, 0.6],
        trial_a_ratings=[[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
        trial_b_ratings=[[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
    )
    m_ttb = choice_matrix(ttb, exp, n_runs=300)
    m_wadd = choice_matrix(wadd, exp, n_runs=300)
    data = exp.simulate(anti, n_runs=150)
    ttb_sims = exp.simulate(ttb, n_runs=150)
    wadd_sims = exp.simulate(wadd, n_runs=150)

    proj = make_sequence_projection_metric(exp, m_ttb, m_wadd)
    p_t1, p_t2, p_data = (
        proj(ttb_sims).value, proj(wadd_sims).value, proj(data).value,
    )
    # The data's projection lands INSIDE the interval the pool spans: it
    # reads as "intermediate between Theory 1 and Theory 2" — the arbiter's
    # verbatim failure mode in the anti_majority run — not as an outlier.
    assert min(p_t1, p_t2) < p_data < max(p_t1, p_t2)

    jsd_self = make_jsd_to_self_metric(exp, m_ttb)
    self_floor = jsd_self(ttb_sims).value
    adv_dist = jsd_self(wadd_sims).value
    data_dist = jsd_self(data).value
    # On the absolute ruler the data is farther from the proposer than even
    # the adversary is — outside the pool's spread: the alarm fires.
    assert self_floor < 0.01
    assert data_dist > adv_dist > self_floor
    assert data_dist - self_floor > 0.10


def test_jsd_autocog_rejects_nondiscriminating_experiment(monkeypatch):
    # All-agree design: JSD ~ 0, so with any reasonable threshold the design
    # is rejected on every attempt and the last attempt is returned anyway
    # (mirrors the baseline's "no attempt accepted" fall-through).
    agree = DecisionMakingBinaryExperiment(
        validities=[0.95, 0.7, 0.65, 0.6],
        trial_a_ratings=[[1, 1, 1, 1]] * 4,
        trial_b_ratings=[[0, 0, 0, 0]] * 4,
    )
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1 = JsdAutoCog(label="pi_1", theory=ttb,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    pi_2 = JsdAutoCog(label="pi_2", theory=wadd,
                     experiment_class=DecisionMakingBinaryExperiment,
                     llm_client=None, jsd_threshold=0.05)
    calls = {"n": 0}

    def _propose(self, adversary, **kw):
        calls["n"] += 1
        return agree

    monkeypatch.setattr(JsdAutoCog, "_llm_propose_experiment", _propose)
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=200, max_experiments=2)
    assert calls["n"] == 2  # retried, never accepted
    assert obs.metric is not None  # last attempt still returned with metric
