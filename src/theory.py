from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import AliasChoices, Field, PrivateAttr

from src.sample_parameters import sample_parameters as _sample
from src.promptable import Promptable

def _format_parameters(parameters: dict[str, str]) -> str:
    if not parameters:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in parameters.items())

class Theory(Promptable):
    """
    A scientific theory operationalized as code.

    Each Theory carries a prose `description` of its claim plus two Python
    source strings that implement the claim:

        predict(parameters, state, history) -> np.ndarray   # choice probabilities
        policy(probs)                       -> int           # action choice

    Parameters are declared as ranges or choices (e.g. "[1.0, 10.0]" or "{1, 2}")
    and sampled per simulated subject using the experiment's parameter context.
    """

    name: ClassVar[str] = "Unknown Theory"

    description: str = Field(
        ...,
        validation_alias=AliasChoices("theory", "description"),
        description="The theoretical claim this theory makes about cognition.",
    )
    predict_source: str = Field(
        ...,
        validation_alias=AliasChoices("predict", "predict_source"),
        description="Python source defining def predict(parameters, state, history) -> np.ndarray.",
    )
    policy_source: str = Field(
        ...,
        validation_alias=AliasChoices("policy", "policy_source"),
        description="Python source defining def policy(probs) -> int.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter ranges (e.g. \"[1.0, 10.0]\") or choices (e.g. \"{1, 2}\"), sampled per subject.",
    )

    rationale: str | None = Field(
        default=None,
        description="Reasoning behind this theory design; used to explain why this theory is better then the previous ones."
    )

    _predict_impl: Any = PrivateAttr(default=None)
    _policy_impl: Any = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        predict_ns: dict[str, Any] = {"np": np}
        exec(self.predict_source, predict_ns)
        object.__setattr__(self, "_predict_impl", predict_ns["predict"])

        policy_ns: dict[str, Any] = {"np": np}
        exec(self.policy_source, policy_ns)
        object.__setattr__(self, "_policy_impl", policy_ns["policy"])

    def predict(self, parameters: dict[str, Any], state: Any, history: Any) -> np.ndarray:
        return self._predict_impl(parameters, state, history)

    def policy(self, probs: np.ndarray) -> int:
        return self._policy_impl(probs)

    def sample_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        return _sample(self.parameters, context=context)
    
    def pretty_print(self) -> str:
        return (
        f"**Description:** {self.description}\n\n"
        f"**Parameters:**\n{_format_parameters(self.parameters)}\n\n"
        f"**`predict source code`:**\n"
        f"```python\n{self.predict_source.rstrip()}\n```\n\n"
        f"**`policy source code`:**\n"
        f"```python\n{self.policy_source.rstrip()}\n```\n"
    )


class Model(Promptable):
    """
    A scientific theory operationalized as code.

    Each Theory carries a prose `description` of its claim plus two Python
    source strings that implement the claim:

        predict(parameters, state, history) -> np.ndarray   # choice probabilities
        policy(probs)                       -> int           # action choice

    Parameters are declared as ranges or choices (e.g. "[1.0, 10.0]" or "{1, 2}")
    and sampled per simulated subject using the experiment's parameter context.
    """

    predict_source: str = Field(
        ...,
        validation_alias=AliasChoices("predict", "predict_source"),
        description="Python source defining def predict(parameters, state, history) -> np.ndarray.",
    )
    policy_source: str = Field(
        ...,
        validation_alias=AliasChoices("policy", "policy_source"),
        description="Python source defining def policy(probs) -> int.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter ranges (e.g. \"[1.0, 10.0]\") or choices (e.g. \"{1, 2}\"), sampled per subject.",
    )

    rationale: str | None = Field(
        default=None,
        description="Reasoning behind this model design; used to explain why this model is better then the previous ones."
    )

    _predict_impl: Any = PrivateAttr(default=None)
    _policy_impl: Any = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        predict_ns: dict[str, Any] = {"np": np}
        exec(self.predict_source, predict_ns)
        object.__setattr__(self, "_predict_impl", predict_ns["predict"])

        policy_ns: dict[str, Any] = {"np": np}
        exec(self.policy_source, policy_ns)
        object.__setattr__(self, "_policy_impl", policy_ns["policy"])

    def predict(self, parameters: dict[str, Any], state: Any, history: Any) -> np.ndarray:
        return self._predict_impl(parameters, state, history)

    def policy(self, probs: np.ndarray) -> int:
        return self._policy_impl(probs)

    def sample_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        return _sample(self.parameters, context=context)
