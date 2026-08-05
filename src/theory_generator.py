"""TheoryGenerator — the brand-new-theory agent.

Sibling to `Improver`, but for a different verdict. Where `Improver` keeps
the theory description verbatim and only regenerates the runnable bits when
the arbiter says `"new_model"`, `TheoryGenerator` proposes a brand-new
`Theory` (description + model) from scratch when the arbiter says
`"new_theory"`.

Statelessness:
- No `history`, no `save()` / `load()` — each `propose_theory(...)` call
  stands alone.
- A per-call `workspace` is used only for writing per-attempt prompt logs to
  `<workspace>/prompts/`, mirroring `AutoPi` / `Improver` / `Arbiter`.

Retry machinery:
- The LLM's structured output is parsed into a `Theory` (from
  `src/theory.py`). Constructing `Theory` execs the LLM-emitted `predict`
  and `policy` source; an exception there triggers a retry, up to
  `max_attempts` attempts. The accepted attempt is returned; if all fail,
  the last exception is raised.

The output is a runnable `Theory`. Persist it via
`dump_theory_yaml(theory, path)` so `AutoPi.from_yaml(...)` can pick it up
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config
from src.llm import LLMClient, make_client
from src.experiment import Experiment
from src.feedback import (
    Feedback,
    PriorIteration,
    aggregate_loss,
    save_fit_results,
    simulate_candidate,
)
from src.metric import Estimate
from src.observation import Observation
from src.prompts import theory_generation
from src.run_config import (
    FEEDBACK_N_RUNS,
    MAX_CRITIQUE_ITERS,
    THEORY_GENERATOR_MAX_ATTEMPTS,
)
from src.theory import Theory


class TheoryGenerator:
    """
    Stateless theory-generation agent.

    Constructor takes the experiment class (so prompts can pull domain name +
    parameter variables), an LLM client, and an optional `Feedback` agent
    used by the inner critique loop. Per-call `workspace` is supplied at
    `propose_theory` time so logs can be organised round-wise.
    """

    def __init__(
        self,
        *,
        experiment_class: type[Experiment],
        llm_client: LLMClient,
        feedback: Feedback | None = None,
    ):
        self.experiment_class = experiment_class
        self.llm_client = llm_client
        self.feedback = feedback or Feedback(
            experiment_class=experiment_class,
            llm_client=llm_client,
        )

    @classmethod
    def from_config(
        cls,
        *,
        experiment_class: type[Experiment],
        config_path: str | Path = "configs/default.yaml",
    ) -> "TheoryGenerator":
        run_cfg = load_config(Path(config_path))
        llm_client = make_client(run_cfg.llm)
        return cls(
            experiment_class=experiment_class,
            llm_client=llm_client,
            feedback=Feedback(
                experiment_class=experiment_class,
                llm_client=llm_client,
            ),
        )

    # --- llm helper (mirrors AutoPi / Improver / Arbiter) -------------------

    def _generate_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type | None = None,
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> Any:
        """Single point of contact with the LLM. Returns parsed structured
        output when `response_schema` is given, else the raw text string.
        Also writes a prompt log if `workspace` + `log_label` are given."""
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

    # --- proposal -----------------------------------------------------------

    def propose_theory(
        self,
        *,
        arbiter_guide: str,
        arbiter_theory_labels: tuple[str | None, str | None] | None = None,
        arbiter_target_idx: int | None = None,
        observations: list[Observation] | None = None,
        max_attempts: int = THEORY_GENERATOR_MAX_ATTEMPTS,
        max_critique_iters: int = MAX_CRITIQUE_ITERS,
        feedback_n_runs: int = FEEDBACK_N_RUNS,
        workspace: Path | None = None,
        leaderboard: list[tuple[str, Theory, float]] | None = None,
    ) -> Theory:
        """
        Ask the LLM for a brand-new `Theory`, wrapped in an inner critique
        loop with the same programmatic accept gate as `Improver` (see
        `Improver.propose_model` for the canonical write-up; the same
        rules apply here).

        ## Accept gate (programmatic)
          - The proposer always builds on `accepted_theory` (the running
            best), never on a rejected attempt.
          - A new candidate is ACCEPTED iff `loss < accepted_loss`
            (strict, finite). Iter 0 auto-accepts.
          - A `"continue"` verdict on a REJECTED candidate is downgraded
            to "regenerate" (returning a worse-than-running-best
            candidate would defeat the gate).

        Per-iteration prompt logs go to:
          <workspace>/prompts/generation_iter_II_attempt_NN.md
          <workspace>/prompts/feedback_iter_II.md

        Returns the running-best `accepted_theory`. Raises if iter 0
        fails to produce any compiling candidate.
        """
        observations = observations or []
        # `prior_iterations` carries one `PriorIteration(rationale,
        # estimates, loss, accepted)` per critique iteration. `accepted`
        # is the SOURCE OF TRUTH for "did the critic's advice in iter k
        # actually help" (= "did the candidate it elicited in iter k+1
        # beat the running best?").
        prior_iterations: list[PriorIteration] = []
        last_compile_exc: Exception | None = None
        # Running-best state — only changes on ACCEPT.
        accepted_theory: Theory | None = None
        accepted_results: list[tuple[Observation, Estimate]] = []
        accepted_loss: float = float("inf")
        accepted_iter: int = -1

        for crit in range(max_critique_iters):
            # --- 1. propose a candidate (validation retry loop) ---
            system_prompt, user_prompt = theory_generation.render(
                experiment_class=self.experiment_class,
                response_schema=Theory,
                arbiter_guide=arbiter_guide,
                arbiter_theory_labels=arbiter_theory_labels,
                arbiter_target_idx=arbiter_target_idx,
                observations=observations,
                previous_candidate=accepted_theory,
                leaderboard=leaderboard,
                prior_iterations=prior_iterations or None,
            )
            candidate: Theory | None = None
            for attempt in range(max_attempts):
                try:
                    text: str = self._generate_response(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_schema=None,
                        workspace=workspace,
                        log_label=f"generation_iter_{crit:02d}_attempt_{attempt:02d}",
                    )
                    candidate = Theory.from_llm_text(text)
                    print(f"[theory_generator crit {crit} attempt {attempt}] compiled.")
                    break
                except Exception as e:
                    last_compile_exc = e
                    print(
                        f"[theory_generator crit {crit} attempt {attempt}] rejected "
                        f"({type(e).__name__}: {e}); retrying."
                    )
            if candidate is None:
                if accepted_theory is not None:
                    print(
                        f"[theory_generator crit {crit}] no compile after "
                        f"{max_attempts} tries; returning running-best accepted "
                        f"theory (iter {accepted_iter}, loss={accepted_loss:.4f})."
                    )
                    if workspace is not None and accepted_results:
                        save_fit_results(accepted_results, workspace=workspace)
                    return accepted_theory
                assert last_compile_exc is not None
                raise RuntimeError(
                    f"theory_generator: no compiling theory after "
                    f"{max_attempts} attempts on critique iter {crit}."
                ) from last_compile_exc

            # --- 2. simulate candidate on every existing experiment ---
            results = simulate_candidate(
                candidate, observations, n_runs=feedback_n_runs
            )

            # --- 3. PROGRAMMATIC ACCEPT GATE ---
            # Strict, finite improvement only — except iter 0, which
            # auto-accepts unconditionally so `accepted_theory` is never
            # None even when the run can't be scored (e.g. no
            # observations, or every candidate produces +inf). Without
            # this carve-out a degenerate iter-0 run would crash on the
            # post-loop "running-best" assertion.
            loss = aggregate_loss(results)
            if accepted_theory is None:
                was_accepted = True
            else:
                was_accepted = (
                    loss < accepted_loss
                    and loss != float("inf")
                )
            if was_accepted:
                accepted_theory = candidate
                accepted_results = results
                accepted_loss = loss
                accepted_iter = crit
            print(
                f"[theory_generator crit {crit}] aggregate_loss={loss:.4f} "
                f"-> {'ACCEPTED' if was_accepted else 'REJECTED'} "
                f"(running best: iter {accepted_iter} @ {accepted_loss:.4f})."
            )

            # --- 4. ask the feedback agent (always — even on REJECT) ---
            verdict = self.feedback.critique(
                theory=candidate,
                candidate_results=results,
                prior_iterations=prior_iterations or None,
                current_loss=loss,
                current_accepted=was_accepted,
                accepted_loss=accepted_loss,
                arbiter_guide=arbiter_guide,
                arbiter_theory_labels=arbiter_theory_labels,
                arbiter_target_idx=arbiter_target_idx,
                workspace=workspace,
                log_label=f"feedback_iter_{crit:02d}",
            )
            print(
                f"[theory_generator crit {crit}] feedback verdict={verdict.verdict} "
                f"({len(results)} experiments scored)."
            )
            if verdict.verdict == "continue" and was_accepted:
                if workspace is not None:
                    save_fit_results(results, workspace=workspace)
                return candidate
            if verdict.verdict == "continue" and not was_accepted:
                print(
                    f"[theory_generator crit {crit}] critic said 'continue' but "
                    f"candidate was REJECTED by accept gate; downgrading to "
                    f"'regenerate' to keep the running-best base intact."
                )
            prior_iterations.append(
                PriorIteration(
                    rationale=(
                        f"Verdict: regenerate\n"
                        f"Interpretation: {verdict.interpretation}\n"
                        f"Rationale: {verdict.rationale}"
                    ),
                    estimates=[est for _, est in results],
                    loss=loss,
                    accepted=was_accepted,
                )
            )

        # Loop exhausted. Hand back the running-best accepted theory.
        assert accepted_theory is not None
        print(
            f"[theory_generator] {max_critique_iters} critique iterations "
            f"exhausted; returning running-best accepted theory "
            f"(iter {accepted_iter}, aggregate_loss={accepted_loss:.4f})."
        )
        if workspace is not None and accepted_results:
            save_fit_results(accepted_results, workspace=workspace)
        return accepted_theory


# --- module-level helpers --------------------------------------------------


def dump_theory_yaml(
    theory: Theory,
    path: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Write a `Theory` to disk in seed-YAML format.

    The output uses the LLM-facing keys (`name`/`theory`/`predict`/`policy`/
    `parameters`) so `AutoPi.from_yaml(path, ...)` consumes it unchanged.
    """
    out = {
        "name": name or "Generated Theory",
        "theory": theory.description,
        "predict": theory.predict_source,
        "policy": theory.policy_source,
        "parameters": theory.parameters,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(out, default_flow_style=False, sort_keys=False))
    return path
