"""Arbiter — the judging agent.

Sibling to `AutoCog` and `Gecco`, but stateless: one call → one verdict. Its
sole job is, given a `Round` of two `Observation`s (one per pi), to render
the arbitration prompt, call the LLM with `response_schema=ArbiterVerdict`,
and return the parsed structured verdict.

The verdict is categorical (`"new_model"` vs `"new_theory"`) plus freeform
interpretation and recommendation; the orchestrator decides what to do with
it (e.g. dispatch to gecco when `verdict == "new_theory"`, or to the
model-improver when `verdict == "new_model"`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import load_config
from src.llm import LLMClient, make_client
from src.arbiter_verdict import ArbiterVerdict
from src.experiment import Experiment
from src.observation import Observations, Round
from src.prompts import arbitration


class Arbiter:
    """
    Stateless arbitration agent.

    Constructor takes only the experiment class and an LLM client. Per-call
    `workspace` is supplied at `arbitrate` time so logs can be organised
    round-wise by the orchestrator.
    """

    def __init__(
        self,
        *,
        experiment_class: type[Experiment],
        llm_client: LLMClient,
    ):
        self.experiment_class = experiment_class
        self.llm_client = llm_client

    @classmethod
    def from_config(
        cls,
        *,
        experiment_class: type[Experiment],
        config_path: str | Path = "configs/default.yaml",
    ) -> "Arbiter":
        run_cfg = load_config(Path(config_path))
        llm_client = make_client(run_cfg.llm)
        return cls(
            experiment_class=experiment_class,
            llm_client=llm_client,
        )

    # --- llm helper (mirrors AutoCog / Gecco) --------------------------------

    def _generate_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type | None = None,
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> Any:
        result = self.llm_client.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            response_schema=response_schema,
        )
        if workspace is not None and log_label is not None:
            self._write_prompt_log(workspace, log_label, system_prompt, user_prompt, result)
        return result.parsed if response_schema is not None else result.text

    @staticmethod
    def _write_prompt_log(
        workspace: Path,
        label: str,
        system_prompt: str,
        user_prompt: str,
        result: Any,
    ) -> None:
        if hasattr(result.parsed, "model_dump"):
            response_block = json.dumps(result.parsed.model_dump(), indent=2)
        else:
            response_block = result.text or "(no text)"
        body = (
            f"# {label}\n\n"
            "## System Prompt\n\n"
            f"{system_prompt}\n\n"
            "## User Prompt\n\n"
            f"{user_prompt}\n\n"
            "## Response\n\n"
            f"```json\n{response_block}\n```\n\n"
            "## Usage\n\n"
            f"```json\n{json.dumps(result.usage, indent=2)}\n```\n"
        )
        prompts_dir = workspace / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"{label}.md").write_text(body)

    # --- arbitration --------------------------------------------------------

    def arbitrate(
        self,
        round: Round,
        *,
        pool: Observations,
        workspace: Path | None = None,
        log_label: str | None = "arbitration",
    ) -> ArbiterVerdict:
        """Render the arbitration prompt for `round` and return the verdict.

        `round` must contain exactly two `Observation`s, each carrying its
        own pi-label-tagged predictions (recorded by `AutoCog.propose_round`).
        `pool` supplies the other rounds' observations for the PERFORMANCE
        ON OTHER EXPERIMENTS section; the round being arbitrated is
        excluded. `workspace` is the per-call log directory (typically a
        round-scoped folder).
        """
        other_observations = [
            o
            for r in pool.rounds
            if r is not round
            for o in r.observations
        ]
        system_prompt, user_prompt = arbitration.render(
            experiment_class=self.experiment_class,
            round=round,
            arbiter_verdict=ArbiterVerdict,
            other_observations=other_observations,
        )
        parsed: ArbiterVerdict = self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=ArbiterVerdict,
            workspace=workspace,
            log_label=log_label,
        )
        return ArbiterVerdict(**parsed.model_dump())