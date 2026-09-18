"""Tests for the plain-OpenAI provider path in src/llm.py and the per-run LLM
override on the agent `from_config` constructors.

Motivation: the recovery battery was run with Gemini; re-running it with
OpenAI models needs (a) a working `provider="openai"` branch, (b) the
chat-completions kwargs that the current OpenAI reasoning models accept
(`gpt-6-astra` rejects `max_tokens`, like gpt-5/o-series), and (c) the CLI
`--llm_provider/--llm_model` reaching the Arbiter/Improver/TheoryGenerator,
which otherwise silently read Gemini from configs/default.yaml.
"""
import os
from unittest.mock import MagicMock, patch

from src.arbiter import Arbiter
from src.config import LLMConfig
from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.improver import Improver
from src.llm import (
    MockClient,
    OpenAIClient,
    _completion_token_kwarg,
    _temperature_kwarg,
    make_client,
)
from src.theory_generator import TheoryGenerator


class TestReasoningModelKwargs:
    def test_gpt6_uses_max_completion_tokens(self):
        assert _completion_token_kwarg("gpt-6-astra", 4096) == {
            "max_completion_tokens": 4096
        }

    def test_gpt6_omits_temperature(self):
        assert _temperature_kwarg("gpt-6-astra", 0.7) == {}

    def test_gpt5_mini_is_reasoning(self):
        assert _completion_token_kwarg("gpt-5.4-mini", 100) == {
            "max_completion_tokens": 100
        }
        assert _temperature_kwarg("gpt-5.4-mini", 0.7) == {}

    def test_legacy_gpt4o_keeps_max_tokens_and_temperature(self):
        assert _completion_token_kwarg("gpt-4o-mini", 4096) == {"max_tokens": 4096}
        assert _temperature_kwarg("gpt-4o-mini", 0.7) == {"temperature": 0.7}


class TestMakeClientOpenAI:
    def test_openai_provider_builds_openai_client(self):
        cfg = LLMConfig(provider="openai", model="gpt-6-astra")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), patch(
            "openai.OpenAI"
        ) as mock_ctor:
            mock_ctor.return_value = MagicMock()
            client = make_client(cfg)

        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-6-astra"
        assert client.temperature == cfg.temperature
        assert client.max_tokens == cfg.max_tokens
        # Plain OpenAI endpoint: no Portkey base_url, key from the environment.
        mock_ctor.assert_called_once_with()


class TestFromConfigLLMOverride:
    """`llm=` overrides the YAML's llm section for every agent."""

    def test_arbiter_uses_override(self):
        arbiter = Arbiter.from_config(
            experiment_class=DecisionMakingBinaryExperiment,
            llm=LLMConfig(provider="mock"),
        )
        assert isinstance(arbiter.llm_client, MockClient)

    def test_improver_and_feedback_use_override(self):
        improver = Improver.from_config(
            experiment_class=DecisionMakingBinaryExperiment,
            llm=LLMConfig(provider="mock"),
        )
        assert isinstance(improver.llm_client, MockClient)
        assert improver.feedback.llm_client is improver.llm_client

    def test_theory_generator_and_feedback_use_override(self):
        gen = TheoryGenerator.from_config(
            experiment_class=DecisionMakingBinaryExperiment,
            llm=LLMConfig(provider="mock"),
        )
        assert isinstance(gen.llm_client, MockClient)
        assert gen.feedback.llm_client is gen.llm_client


class TestAutoCogFromYamlLLMOverride:
    def test_from_yaml_uses_override(self):
        from src.autocog import AutoCog

        agent = AutoCog.from_yaml(
            "theories/heuristic_decision_making/ttb_sampling.yaml",
            label="pi_1",
            experiment_class=DecisionMakingBinaryExperiment,
            llm=LLMConfig(provider="mock"),
        )
        assert isinstance(agent.llm_client, MockClient)


class TestFromConfigYamlPath:
    """Without `llm=`, the YAML llm section is still what gets used."""

    def test_arbiter_reads_yaml_when_no_override(self):
        arbiter = Arbiter.from_config(
            experiment_class=DecisionMakingBinaryExperiment,
            config_path="configs/mock.yaml",
        )
        assert isinstance(arbiter.llm_client, MockClient)
