"""Run configuration loaded from YAML at startup.

All per-run knobs live here. Domain-level framing (what a stimulus is, fit
protocol, canonical stimuli) lives in domains/<name>/domain.yaml instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


_PRINCETON_MODEL_PREFIXES: tuple[str, ...] = ("gpt-", "o1", "o3", "o4")


class LLMConfig(BaseModel):
    provider: Literal["gemini", "openai", "anthropic", "princeton", "mock"] = "gemini"
    model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.7
    max_tokens: int = 32768
    thinking_budget: int = 8096

    @model_validator(mode="after")
    def _check_princeton_model(self) -> "LLMConfig":
        if self.provider == "princeton" and not self.model.startswith(
            _PRINCETON_MODEL_PREFIXES
        ):
            raise ValueError(
                f"model {self.model!r} doesn't look like an OpenAI-family model. "
                f"The Princeton AI Sandbox (Portkey gateway) exposes OpenAI models "
                f"only; expected a name starting with one of "
                f"{_PRINCETON_MODEL_PREFIXES}."
            )
        return self


class RunConfig(BaseModel):
    domain: str
    seed_theories: list[str] = Field(..., min_length=2, max_length=2)
    llm: LLMConfig = LLMConfig()
    n_rounds: int = 5
    seed: int = 42
    max_verify_retries: int = 5
    max_gecco_retries: int = 3
    disabled_stages: list[str] = Field(default_factory=list)
    run_tag: str = "default"
    diversity_requirement: str = ""


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text())
    return RunConfig.model_validate(data)
