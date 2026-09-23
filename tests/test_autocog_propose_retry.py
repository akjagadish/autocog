"""An experiment proposal that is schema-valid JSON but violates the
experiment's own invariants (pydantic model validator) used to crash the
whole run. z-ai/glm-5.3 wrote 6-element rating lists for 7- and 24-feature
designs twice in one run (2026-09-19); a fresh sample usually passes, so
the proposal loop should spend another of its `max_experiments` attempts
instead of aborting, and fail clearly only when every attempt is invalid."""
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.autocog import AutoCog
from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.llm import MockClient
from src.observation import Observations
from src.theory import Theory


def _agent(label):
    return AutoCog(
        label=label,
        theory=Theory.from_yaml("theories/heuristic_decision_making/ttb_sampling.yaml"),
        experiment_class=DecisionMakingBinaryExperiment,
        llm_client=MockClient(),
    )


def _validation_error():
    try:
        DecisionMakingBinaryExperiment(**{})
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


def test_invalid_proposal_is_retried_not_fatal():
    agent, adversary = _agent("pi_1"), _agent("pi_2")
    with patch.object(
        agent, "_llm_propose_experiment",
        side_effect=[_validation_error(), RuntimeError("sentinel")],
    ) as m:
        with pytest.raises(RuntimeError, match="sentinel"):
            agent.propose_round(adversary, Observations(), max_experiments=3)
    assert m.call_count == 2, "the ValidationError must consume one attempt, not abort"


def test_all_invalid_proposals_raise_clearly():
    agent, adversary = _agent("pi_1"), _agent("pi_2")
    with patch.object(
        agent, "_llm_propose_experiment", side_effect=[_validation_error()] * 3
    ) as m:
        with pytest.raises(RuntimeError, match="3 experiment proposals failed validation"):
            agent.propose_round(adversary, Observations(), max_experiments=3)
    assert m.call_count == 3
