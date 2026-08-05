import numpy as np
import pytest

import src.ablations as ablations
from src.ablations import BlindAutoCog, BlindJsdAutoCog, _derive_rng, random_design
from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.metric import Metric
from src.observation import Observations
from src.theory import Theory

THEORIES_DIR = "theories/heuristic_decision_making"

MEAN_RESPONSE_METRIC = (
    "def metric(data):\n"
    "    return float(data['response'].mean())\n"
)
CONSTANT_METRIC = (
    "def metric(data):\n"
    "    return 0.0\n"
)
RAISING_METRIC = (
    "def metric(data):\n"
    "    raise ValueError('bad metric')\n"
)
NONSCALAR_METRIC = (
    "def metric(data):\n"
    "    return 'not a number'\n"  # non-None but non-scalar -> must be rejected, not crash
)


def _load(name: str) -> Theory:
    return Theory.from_yaml(f"{THEORIES_DIR}/{name}.yaml")


def _blind(label: str, theory: Theory, seed: int = 0) -> BlindAutoCog:
    return BlindAutoCog(
        label=label, theory=theory,
        experiment_class=DecisionMakingBinaryExperiment,
        llm_client=None, seed=seed,
    )


def test_random_design_is_valid_and_nondegenerate():
    rng = np.random.default_rng(0)
    exp = random_design(rng, n_features=4, n_pairs=8)
    assert len(exp.validities) == 4
    assert all(0.5 <= v <= 1.0 for v in exp.validities)
    assert len(exp.trial_a_ratings) == len(exp.trial_b_ratings) == 8
    for a, b in zip(exp.trial_a_ratings, exp.trial_b_ratings):
        assert len(a) == len(b) == 4
        assert set(a) | set(b) <= {0, 1}
        assert a != b  # no null trials


# --- Task 1: _derive_rng seeding helper ---------------------------------------


def test_derive_rng_is_deterministic():
    a = _derive_rng(7, "pi_1", 0).integers(0, 2**32)
    b = _derive_rng(7, "pi_1", 0).integers(0, 2**32)
    assert a == b


def test_derive_rng_varies_by_seed_label_and_round():
    base = _derive_rng(7, "pi_1", 0).integers(0, 2**32)
    assert _derive_rng(8, "pi_1", 0).integers(0, 2**32) != base   # seed
    assert _derive_rng(7, "pi_2", 0).integers(0, 2**32) != base   # slot label
    assert _derive_rng(7, "pi_1", 1).integers(0, 2**32) != base   # round


# --- Task 2: BlindAutoCog.propose_round ----------------------------------------


def test_blind_returns_wellformed_observation(monkeypatch):
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind("pi_1", ttb), _blind("pi_2", wadd)
    monkeypatch.setattr(
        BlindAutoCog, "_llm_propose_metric",
        lambda self, adv, exp, **kw: Metric(metric_source=MEAN_RESPONSE_METRIC),
    )
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=50)
    assert isinstance(obs.experiment, DecisionMakingBinaryExperiment)
    assert obs.metric is not None
    assert obs.prediction_by_label("pi_1") is not None
    assert obs.prediction_by_label("pi_2") is not None


def test_blind_accepts_nonseparating_design_no_welch(monkeypatch):
    # Constant metric: both theories score 0.0 -> the adversarial baseline's
    # Welch gate would REJECT (no separation); blind_design has no gate, so it
    # must accept and return. This is the defining behavioral difference.
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind("pi_1", ttb), _blind("pi_2", wadd)
    monkeypatch.setattr(
        BlindAutoCog, "_llm_propose_metric",
        lambda self, adv, exp, **kw: Metric(metric_source=CONSTANT_METRIC),
    )
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=50)
    assert obs is not None
    assert obs.prediction_by_label("pi_1").value == 0.0
    assert obs.prediction_by_label("pi_2").value == 0.0


def test_blind_simulate_is_unguarded(monkeypatch):
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind("pi_1", ttb), _blind("pi_2", wadd)

    def boom(self, *a, **k):
        raise RuntimeError("sim failed")

    monkeypatch.setattr(DecisionMakingBinaryExperiment, "simulate", boom)
    with pytest.raises(RuntimeError, match="sim failed"):
        pi_1.propose_round(pi_2, Observations(), n_runs=50)


def test_blind_redraws_a_fresh_design_when_all_metrics_error(monkeypatch):
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind("pi_1", ttb), _blind("pi_2", wadd)

    seen = []
    real_random_design = ablations.random_design

    def spy(rng, **kw):
        exp = real_random_design(rng, **kw)
        seen.append(exp)
        return exp

    monkeypatch.setattr(ablations, "random_design", spy)
    monkeypatch.setattr(
        BlindAutoCog, "_llm_propose_metric",
        lambda self, adv, exp, **kw: Metric(metric_source=RAISING_METRIC),
    )
    pi_1.propose_round(pi_2, Observations(), n_runs=50, max_experiments=3, max_metrics=2)
    assert len(seen) == 3  # one fresh design per outer iteration (no metric ever valid)


def test_blind_handles_nonscalar_metric_value_without_crashing(monkeypatch):
    # A malformed LLM metric returning a non-None, non-scalar value must be
    # rejected gracefully (coerced to float inside the try, like the baseline),
    # NOT crash propose_round with an uncaught pydantic ValidationError.
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind("pi_1", ttb), _blind("pi_2", wadd)
    monkeypatch.setattr(
        BlindAutoCog, "_llm_propose_metric",
        lambda self, adv, exp, **kw: Metric(metric_source=NONSCALAR_METRIC),
    )
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=50, max_experiments=1, max_metrics=1)
    assert obs is not None
    assert obs.prediction_by_label("pi_1").value is None  # coercion failed -> recorded None


# --- Task 3: BlindJsdAutoCog (blind_design_jsd: random design + lag-1 JSD metric) ---


def _blind_jsd(label: str, theory: Theory, seed: int = 0, jsd_n_runs: int = 200) -> BlindJsdAutoCog:
    return BlindJsdAutoCog(
        label=label, theory=theory,
        experiment_class=DecisionMakingBinaryExperiment,
        llm_client=None, seed=seed, jsd_n_runs=jsd_n_runs,
    )


def _forbid_llm(*_a, **_k):
    raise AssertionError("blind_design_jsd must not call the LLM")


def test_blind_jsd_uses_jsd_to_self_metric_and_never_calls_llm(monkeypatch):
    # The defining contract: the design is drawn by random_design (no LLM
    # experiment proposal) and the metric is the fixed lag-1 JSD-to-self
    # metric (no LLM metric proposal). Both LLM entry points are wired to
    # raise, so a returned observation proves neither was touched.
    monkeypatch.setattr(BlindJsdAutoCog, "_llm_propose_experiment", _forbid_llm, raising=False)
    monkeypatch.setattr(BlindJsdAutoCog, "_llm_propose_metric", _forbid_llm, raising=False)

    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    pi_1, pi_2 = _blind_jsd("pi_1", ttb), _blind_jsd("pi_2", wadd)
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=100)

    assert isinstance(obs.experiment, DecisionMakingBinaryExperiment)
    # Metric is the auto-generated lag-1 JSD-to-self metric, not an LLM one.
    assert "JSD-to-self" in (obs.metric.rationale or "")
    # The metric is anchored to the PROPOSING theory's conditional profile, so
    # the proposer scores ~the self-JSD floor and the differing adversary
    # scores strictly above it.
    p_self = obs.prediction_by_label("pi_1").value
    p_adv = obs.prediction_by_label("pi_2").value
    assert p_self is not None and p_adv is not None
    assert p_adv > p_self


def test_blind_jsd_accepts_nonseparating_design_and_predicts_self_floor():
    # Two pis wrapping the SAME theory: the design cannot separate them
    # (analytic corner — a theory vs itself has lag-1 JSD == 0). A gated
    # condition would reject for non-separation; blind_design_jsd has NO gate,
    # so it must still return a well-formed observation, and BOTH predictions
    # sit at the self-JSD plug-in floor (~0, well under ln2 == 0.693).
    ttb = _load("ttb_sampling")
    pi_1 = _blind_jsd("pi_1", ttb, jsd_n_runs=300)
    pi_2 = _blind_jsd("pi_2", ttb, jsd_n_runs=300)
    obs = pi_1.propose_round(pi_2, Observations(), n_runs=300)
    assert obs is not None
    p_self = obs.prediction_by_label("pi_1").value
    p_adv = obs.prediction_by_label("pi_2").value
    assert 0.0 <= p_self < 0.05  # self-JSD floor (theory vs its own profile)
    assert 0.0 <= p_adv < 0.05   # identical theory -> also at the floor


def test_blind_jsd_design_is_seed_deterministic():
    # propose_round must derive its random design from (seed, label, round) via
    # _derive_rng, so two pis with the same seed/label draw the SAME design.
    ttb, wadd = _load("ttb_sampling"), _load("wadd_sampling")
    obs_a = _blind_jsd("pi_1", ttb, seed=5, jsd_n_runs=60).propose_round(
        _blind_jsd("pi_2", wadd, jsd_n_runs=60), Observations(), n_runs=40
    )
    obs_b = _blind_jsd("pi_1", ttb, seed=5, jsd_n_runs=60).propose_round(
        _blind_jsd("pi_2", wadd, jsd_n_runs=60), Observations(), n_runs=40
    )
    # Validities come straight off the design rng and are not reshuffled at
    # construction, so they pin the drawn design deterministically.
    assert list(obs_a.experiment.validities) == list(obs_b.experiment.validities)
