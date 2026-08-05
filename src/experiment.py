from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from src.logger import info
from src.promptable import Promptable

if TYPE_CHECKING:
    from src.theory import Theory


class Experiment(Promptable, ABC):
    """
    An experimental design that can be iterated trial-by-trial, updated with
    subject responses, and converted to a tidy DataFrame.

    Each domain subclass declares its contract via the ClassVars below
    (`name`, `description`, `output_columns`, `parameter_variables`) and
    implements the iteration / update / to_df / parameter_context methods.
    """

    # --- class-level contract data (not part of the LLM response schema) -----

    name: ClassVar[str] = "Unknown Experiment"
    description: ClassVar[str] = ""
    """Free-form description of the experimental domain."""

    output_columns: ClassVar[dict[str, str]] = {}
    """Columns of the data collected from this experiment, with descriptions of what each column contains."""

    parameter_variables: ClassVar[dict[str, str]] = {}
    """Variable names available when sampling model parameters (e.g. n_features), with descriptions."""

    introduction_text: ClassVar[str] = ""
    """Subject-facing instructions; also shown to the LLM as task framing."""

    @classmethod
    def pretty_print_header(cls) -> str:
        """One-paragraph design summary aimed at an experiment PROPOSER.

        This is the audience that will pick free design knobs (stimuli,
        n_unique_stimuli, etc.), so subclasses may write it in the
        imperative ("Choose ... based on ..."). Embedded in
        `experiment_proposal` and `metric_proposal` prompts.
        """
        return cls.description

    @classmethod
    def pretty_print_protocol(cls) -> str:
        """One-paragraph protocol summary aimed at a THEORIST / MODELER.

        This is the audience that will write `predict` / `policy` against
        an already-fixed protocol; they don't choose anything about it.
        Subclasses should describe the protocol in the indicative
        ("Each subject sees N blocks ..."), with no imperative "Choose ..."
        guidance. Embedded in `theory_generation` and `model_improvement`
        prompts.

        Default falls back to `pretty_print_header()` so subclasses that
        don't override still produce something — but a tailored override is
        strongly recommended whenever `pretty_print_header()` carries
        proposer-only instructions.
        """
        return cls.pretty_print_header()

    def pretty_print_design(self) -> str:
        """
        Human-readable rendering of this concrete design. Used in proposal
        prompts so the LLM sees a compact, paired view instead of raw JSON.
        Subclasses should override; the default falls back to the JSON dump.
        """
        return self.pretty_print()

    def pretty_print(self) -> str:
        """
        Human-readable rendering of this concrete design. Used in proposal
        prompts so the LLM sees a compact, paired view instead of raw JSON.
        Subclasses should override; the default falls back to the JSON dump.
        """
        return self.model_dump_json(indent=2)

    # --- instance fields (part of the LLM response schema) -------------------

    rationale: str | None = Field(
        default=None,
        description="Reasoning behind this experimental design; used when interpreting results.",
    )

    # --- runtime state -------------------------------------------------------

    _history: Any = PrivateAttr(default_factory=dict)

    # --- abstract iteration interface ----------------------------------------

    @abstractmethod
    def __iter__(self) -> "Experiment": ...

    @abstractmethod
    def __next__(self) -> tuple[Any, Any]: ...

    @abstractmethod
    def update(self, response: Any) -> None: ...

    @abstractmethod
    def to_df(self) -> pd.DataFrame: ...

    # --- parameter materialization for Model.simulate ------------------------

    def parameter_context(self) -> dict[str, Any]:
        """Materialize concrete values for `parameter_variables` from this instance."""
        return {}

    # --- canonical data shape ------------------------------------------------

    @classmethod
    def to_canonical_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Force `df` into the canonical column order declared by `output_columns`.

        Single source of truth for "what does a row of data from this
        experiment look like" — both `simulate` (model-generated data) and
        `run_online` (human-generated data) funnel through this so the LLM's
        metric sees the exact same shape no matter where the data came from.
        """
        if not cls.output_columns:
            return df
        return df.reindex(columns=list(cls.output_columns.keys()))

    # --- execution: humans (run_online, on subclasses) or models (simulate) --

    def simulate(
        self,
        theory: "Theory",
        n_runs: int = 1,
        *,
        action_noise: float = 0.0,
        rng: "np.random.Generator | None" = None,
    ) -> pd.DataFrame:
        """
        Run this experiment with `theory` standing in for human subjects, n_runs times.
        Returns a DataFrame matching `output_columns`, with `subject_id` per run.

        `action_noise` (in [0, 1]) injects epsilon-greedy policy noise on top
        of `theory.policy(probs)`: with probability `action_noise` the chosen
        action is replaced by a uniform draw over `len(probs)` options;
        otherwise the theory's policy output is used unchanged. This is a
        synthetic-test-case knob used to simulate noisy "ground-truth"
        subjects without modifying the theory YAMLs themselves, so the
        theory's own prediction logic is untouched. `rng` seeds the noise
        draw (default: an unseeded `default_rng()`). The default
        `action_noise=0.0` preserves byte-for-byte parity with the prior
        noiseless behaviour.
        """
        if not (0.0 <= action_noise <= 1.0):
            raise ValueError(
                f"action_noise must be in [0, 1]; got {action_noise!r}."
            )
        info(
            f"{type(self).__name__}: simulating data — n_runs={n_runs}"
            + (f" action_noise={action_noise}" if action_noise > 0.0 else "")
        )
        noise_rng = rng if rng is not None else np.random.default_rng()
        frames: list[pd.DataFrame] = []
        for run in range(n_runs):
            parameters = theory.sample_parameters(self.parameter_context())
            for state, history in self:
                probs = theory.predict(parameters, state, history)
                if action_noise > 0.0 and noise_rng.random() < action_noise:
                    action = int(noise_rng.integers(0, len(probs)))
                else:
                    action = theory.policy(probs)
                self.update(action)
            df = self.to_df()
            df["subject_id"] = run
            frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.to_canonical_data(df)
