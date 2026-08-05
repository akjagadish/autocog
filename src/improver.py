"""Improver — the model-regeneration agent.

Sibling to `Gecco`, but for a different verdict. Where `Gecco` proposes a
brand-new `Theory` (description + model) when the arbiter says
`"new_theory"`, `Improver` keeps the theory description verbatim and
regenerates only the runnable bits — `predict`, `policy`, `parameters` — when
the arbiter says `"new_model"`.

Statelessness:
- No `history`, no `save()` / `load()` — each `propose_model(...)` call
  stands alone.
- A per-call `workspace` is used only for writing per-attempt prompt logs to
  `<workspace>/prompts/`, mirroring `AutoPi` / `Gecco` / `Arbiter`.

Retry machinery:
- The LLM's structured output is parsed into a `Model` (from
  `src/theory.py`). Constructing `Model` execs the LLM-emitted `predict` and
  `policy` source; an exception there triggers a retry, up to `max_attempts`
  attempts. The accepted attempt is returned; if all fail, the last
  exception is raised.

The output is a runnable `Model`. To attach it back to the unchanged theory
description, use `make_theory(theory, model)` (or write a YAML via
`dump_theory_yaml(theory, model, path)` so `AutoPi.from_yaml(...)` can pick
it up unchanged).
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
from src.prompts import model_improvement
from src.run_config import (
    FEEDBACK_N_RUNS,
    IMPROVER_MAX_ATTEMPTS,
    MAX_CRITIQUE_ITERS,
)
from src.theory import Model, Theory


class Improver:
    """
    Stateless model-regeneration agent.

    Constructor takes the experiment class (so prompts can pull domain name +
    parameter variables), an LLM client, and an optional `Feedback` agent
    used by the inner critique loop. Per-call `workspace` is supplied at
    `propose_model` time so logs can be organised round-wise.
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
    ) -> "Improver":
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

    # --- llm helper (mirrors AutoPi / Gecco / Arbiter) ----------------------

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

    def propose_model(
        self,
        *,
        theory: Theory,
        arbiter_guide: str,
        arbiter_theory_labels: tuple[str | None, str | None] | None = None,
        arbiter_target_idx: int | None = None,
        other_theory: Theory | None = None,
        observations: list[Observation] | None = None,
        max_attempts: int = IMPROVER_MAX_ATTEMPTS,
        max_critique_iters: int = MAX_CRITIQUE_ITERS,
        feedback_n_runs: int = FEEDBACK_N_RUNS,
        workspace: Path | None = None,
        leaderboard: list[tuple[str, Theory, float]] | None = None,
    ) -> Model:
        """
        Ask the LLM for a new `Model` under the given Theory's description,
        wrapped in an inner critique loop.

        `observations` is the empirical evidence the new model must explain
        (typically every Observation in the pool). The prompt formats each
        as (experimental design, metric, real value).

        ## Accept gate (programmatic, no LLM in the loop)
        The loop tracks an `accepted_model` (the running-best candidate so
        far) and `accepted_loss = aggregate_loss(accepted_model)`. On every
        iteration:
          - The proposer renders against `previous_candidate=accepted_model`
            (NEVER the most recent attempt if it was rejected) plus the
            full `prior_iterations` log so it sees what was already tried
            and which past critic feedback was ACCEPTED vs REJECTED.
          - The new candidate's loss is computed and compared:
              * `loss < accepted_loss` (strict, finite) → ACCEPT, becomes
                the new `accepted_model` / `accepted_loss`.
              * else → REJECT, base unchanged. The next iter will build
                on the same `accepted_model` again, but with the new
                critic feedback.
          - Iter 0 auto-accepts (any finite loss beats the initial `+inf`).
        This guarantees the running-best monotonically improves and the
        proposer never has to "remember to revert" a regression — the loop
        does it programmatically.

        ## Critic role
        The critic is consulted every iter (so it can give a fresh
        direction even when the proposal was rejected). It receives:
          - the current candidate's results,
          - `current_accepted` (was this candidate kept?),
          - `accepted_loss` (the bar that had to be beaten),
          - the full `prior_iterations` log with per-iter ACCEPTED /
            REJECTED markers.
        A `"continue"` verdict on a REJECTED candidate is silently
        downgraded to "regenerate" — returning a candidate worse than the
        running best would defeat the accept gate. A `"continue"` verdict
        on an ACCEPTED candidate exits the loop early and returns the
        candidate.

        Per-iteration prompt logs go to:
          <workspace>/prompts/improvement_iter_II_attempt_NN.md
          <workspace>/prompts/feedback_iter_II.md

        Returns the running-best `accepted_model`. Raises if iter 0 fails
        to produce any compiling candidate.
        """
        observations = observations or []
        # `prior_iterations` carries one `PriorIteration(rationale,
        # estimates, loss, accepted)` per critique iteration. `accepted`
        # is the SOURCE OF TRUTH for "did the critic's advice in iter k
        # actually help" (= "did the candidate it elicited in iter k+1
        # beat the running best?") — both prompts use this to grade
        # critic feedback and avoid stacking on regressions.
        prior_iterations: list[PriorIteration] = []
        last_compile_exc: Exception | None = None
        # Running-best state — only changes on ACCEPT.
        accepted_model: Model | None = None
        accepted_results: list[tuple[Observation, Estimate]] = []
        accepted_loss: float = float("inf")
        accepted_iter: int = -1

        for crit in range(max_critique_iters):
            # --- 1. propose a candidate (validation retry loop) ---
            # The proposer always builds on the running-best base, never
            # on a rejected attempt — this is the accept gate's whole
            # point.
            previous_candidate = (
                make_theory(theory, accepted_model)
                if accepted_model is not None
                else None
            )
            system_prompt, user_prompt = model_improvement.render(
                experiment_class=self.experiment_class,
                theory=theory,
                response_schema=Model,
                arbiter_guide=arbiter_guide,
                arbiter_theory_labels=arbiter_theory_labels,
                arbiter_target_idx=arbiter_target_idx,
                other_theory=other_theory,
                observations=observations,
                previous_candidate=previous_candidate,
                leaderboard=leaderboard,
                prior_iterations=prior_iterations or None,
            )
            candidate: Model | None = None
            for attempt in range(max_attempts):
                try:
                    # Freeform text — Gemini rejects dict-bearing schemas
                    # (`additionalProperties` not supported); `Model.from_llm_text`
                    # strips any markdown fence and validates via
                    # `model_post_init`, which compiles `predict`/`policy`.
                    text: str = self._generate_response(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_schema=None,
                        workspace=workspace,
                        log_label=f"improvement_iter_{crit:02d}_attempt_{attempt:02d}",
                    )
                    candidate = Model.from_llm_text(text)
                    print(f"[improver crit {crit} attempt {attempt}] compiled.")
                    break
                except Exception as e:
                    last_compile_exc = e
                    print(
                        f"[improver crit {crit} attempt {attempt}] rejected "
                        f"({type(e).__name__}: {e}); retrying."
                    )
            if candidate is None:
                # Every attempt this iteration failed to compile. If we
                # already have an accepted base, hand it back; otherwise
                # iter 0 with no compiling candidate is a hard failure.
                if accepted_model is not None:
                    print(
                        f"[improver crit {crit}] no compile after {max_attempts} "
                        f"tries; returning running-best accepted model "
                        f"(iter {accepted_iter}, loss={accepted_loss:.4f})."
                    )
                    if workspace is not None and accepted_results:
                        save_fit_results(accepted_results, workspace=workspace)
                    return accepted_model
                assert last_compile_exc is not None
                raise RuntimeError(
                    f"improver: no compiling model after "
                    f"{max_attempts} attempts on critique iter {crit}."
                ) from last_compile_exc

            # --- 2. simulate candidate on every existing experiment ---
            candidate_theory = make_theory(theory, candidate)
            results = simulate_candidate(
                candidate_theory, observations, n_runs=feedback_n_runs
            )

            # --- 3. PROGRAMMATIC ACCEPT GATE ---
            # Strict, finite improvement only — except iter 0, which
            # auto-accepts unconditionally so `accepted_model` is never
            # None even when the run can't be scored (e.g. no
            # observations, or every candidate produces +inf). Without
            # this carve-out a degenerate iter-0 run would crash on the
            # post-loop "running-best" assertion.
            loss = aggregate_loss(results)
            if accepted_model is None:
                was_accepted = True
            else:
                was_accepted = (
                    loss < accepted_loss
                    and loss != float("inf")
                )
            if was_accepted:
                accepted_model = candidate
                accepted_results = results
                accepted_loss = loss
                accepted_iter = crit
            print(
                f"[improver crit {crit}] aggregate_loss={loss:.4f} "
                f"-> {'ACCEPTED' if was_accepted else 'REJECTED'} "
                f"(running best: iter {accepted_iter} @ {accepted_loss:.4f})."
            )

            # --- 4. ask the feedback agent (always — even on REJECT, we
            # need fresh direction for the next iter) ---
            verdict = self.feedback.critique(
                theory=candidate_theory,
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
                f"[improver crit {crit}] feedback verdict={verdict.verdict} "
                f"({len(results)} experiments scored)."
            )
            # `continue` only honored when the candidate was accepted —
            # otherwise we'd be returning a candidate strictly worse than
            # the running best, defeating the accept gate.
            if verdict.verdict == "continue" and was_accepted:
                if workspace is not None:
                    save_fit_results(results, workspace=workspace)
                return candidate
            if verdict.verdict == "continue" and not was_accepted:
                print(
                    f"[improver crit {crit}] critic said 'continue' but "
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

        # Loop exhausted. Hand back the running-best accepted model.
        assert accepted_model is not None
        print(
            f"[improver] {max_critique_iters} critique iterations exhausted; "
            f"returning running-best accepted model (iter {accepted_iter}, "
            f"aggregate_loss={accepted_loss:.4f})."
        )
        if workspace is not None and accepted_results:
            save_fit_results(accepted_results, workspace=workspace)
        return accepted_model


# --- module-level helpers --------------------------------------------------


def make_theory(theory: Theory, model: Model) -> Theory:
    """Combine an unchanged Theory description with a new Model.

    Used after `Improver.propose_model(...)` to assemble the next-round
    Theory (same prose claim, regenerated runnable bits).
    """
    return Theory(
        description=theory.description,
        predict_source=model.predict_source,
        policy_source=model.policy_source,
        parameters=model.parameters,
        rationale=model.rationale,
    )


def dump_theory_yaml(
    theory: Theory,
    model: Model,
    path: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Write `(theory.description, model.*)` to disk in seed-YAML format.

    The output uses the LLM-facing keys (`name`/`theory`/`predict`/`policy`/
    `parameters`) so `AutoPi.from_yaml(path, ...)` consumes it unchanged.
    """
    out = {
        "name": name or "Improved Model",
        "theory": theory.description,
        "predict": model.predict_source,
        "policy": model.policy_source,
        "parameters": model.parameters,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(out, default_flow_style=False, sort_keys=False))
    return path
