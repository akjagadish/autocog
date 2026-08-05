"""Feedback — the candidate-critic agent.

Sibling to `Arbiter`, but invoked from inside the `Improver` /
`TheoryGenerator` inner loop rather than from `main_.py`. Its job is
narrow: given a freshly proposed candidate (Theory or Theory-built-from-
Model) and that candidate's simulated performance on every existing
experiment, return a `FeedbackVerdict`:

  * `"continue"`   — accept the candidate as-is.
  * `"regenerate"` — reject; the proposing agent should produce a new
                      candidate, taking the rationale into account.

Statelessness:
- No persistence of its own. Per-call `workspace` controls where prompt
  logs go (`<workspace>/prompts/<log_label>.md`), mirroring AutoPi /
  Arbiter / Improver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from src.config import load_config
from src.llm import LLMClient, make_client
from src.arbiter_verdict import FeedbackVerdict
from src.experiment import Experiment
from src.metric import Estimate
from src.observation import Observation
from src.prompts import feedback as feedback_prompt
from src.theory import Theory


class PriorIteration(NamedTuple):
    """One critique iteration's record, threaded through the propose-loop.

    Carries everything the proposer and critic need to grade *the loop's
    own trajectory*:
      - `rationale`: the critic's verdict + interpretation + rationale
        text from this iteration (rendered into both PRIOR FEEDBACK and
        YOUR PRIOR CRITIQUES sections).
      - `estimates`: per-observation candidate estimates for THIS
        iteration's candidate, in the same order as the loop's
        `observations`. Used by the per-experiment trajectory blocks.
      - `loss`: this iteration's `aggregate_loss` (lower = better, +inf =
        unscorable). Surfaced to both agents.
      - `accepted`: whether the propose-loop's accept gate took THIS
        iteration's candidate as the new base (`True`) or rejected it and
        kept the previous base (`False`). The accept gate is purely
        programmatic — `loss < accepted_loss` (strict, finite) — so this
        flag is the source of truth for "did the critic's advice that
        produced this iteration's candidate actually help" and replaces
        the old HELPED/HURT loss-delta heuristic.
    """

    rationale: str
    estimates: list[Estimate]
    loss: float
    accepted: bool


class Feedback:
    """
    Stateless candidate-critic agent.

    Used inside `Improver` / `TheoryGenerator` to grade a proposed
    Theory/Model after it's been simulated on every existing experiment.
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
    ) -> "Feedback":
        run_cfg = load_config(Path(config_path))
        llm_client = make_client(run_cfg.llm)
        return cls(
            experiment_class=experiment_class,
            llm_client=llm_client,
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

    # --- critique -----------------------------------------------------------

    def critique(
        self,
        *,
        theory: Theory,
        candidate_results: list[tuple[Observation, Estimate]],
        prior_iterations: list[PriorIteration] | None = None,
        current_loss: float | None = None,
        current_accepted: bool | None = None,
        accepted_loss: float | None = None,
        arbiter_guide: str | None = None,
        arbiter_theory_labels: tuple[str | None, str | None] | None = None,
        arbiter_target_idx: int | None = None,
        workspace: Path | None = None,
        log_label: str = "feedback",
    ) -> FeedbackVerdict:
        """Render the feedback prompt for one candidate and return the verdict.

        `candidate_results` is `[(observation, candidate_estimate), ...]` —
        one entry per existing experiment the candidate was simulated on.
        Each `Estimate` is a `(value, variance)` namedtuple: the metric's
        full-data scalar paired with its between-subject variance.

        `prior_iterations` is the history of *this* propose-loop's earlier
        critique iterations as `PriorIteration(rationale, estimates, loss,
        accepted)` entries in chronological order. Each entry's
        `estimates` list is in the SAME order as `candidate_results`;
        `loss` is the `aggregate_loss` of that iteration's candidate;
        `accepted` records whether the loop's programmatic accept gate
        took that candidate as the new base. The critic uses these to
        (a) see the per-experiment trajectory and detect oscillation,
        and (b) grade whether its OWN past advice helped — advice in
        iter k is "good" iff the candidate it elicited in iter k+1 was
        ACCEPTED (i.e. strictly improved the running-best loss).

        `current_loss` is `aggregate_loss` of the candidate currently
        being critiqued. `current_accepted` says whether the loop will
        keep that candidate as the new base (True) or discard it and
        keep `accepted_loss`'s base on the next iter (False); when False,
        a `"continue"` verdict from the critic is silently ignored by
        the loop because returning a worse candidate would defeat the
        gate. `accepted_loss` is the running-best loss the new
        candidate had to beat.

        `arbiter_guide` (+ optional `arbiter_theory_labels` /
        `arbiter_target_idx`) is the recommendation that drove this
        propose-loop. Forwarding it to the prompt lets the critic grade
        FAITHFULNESS to the prescribed mechanism family and limit its
        own suggestions to minor in-family adjustments — switching to a
        different mechanism family is the arbiter's prerogative, not the
        critic's.
        """
        system_prompt, user_prompt = feedback_prompt.render(
            experiment_class=self.experiment_class,
            theory=theory,
            candidate_results=candidate_results,
            feedback_verdict=FeedbackVerdict,
            prior_iterations=prior_iterations,
            current_loss=current_loss,
            current_accepted=current_accepted,
            accepted_loss=accepted_loss,
            arbiter_guide=arbiter_guide,
            arbiter_theory_labels=arbiter_theory_labels,
            arbiter_target_idx=arbiter_target_idx,
        )
        parsed: FeedbackVerdict = self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=FeedbackVerdict,
            workspace=workspace,
            log_label=log_label,
        )
        return FeedbackVerdict(**parsed.model_dump())


# --- shared helper ----------------------------------------------------------


def simulate_candidate(
    theory: Theory,
    observations: list[Observation],
    *,
    n_runs: int = 50,
) -> list[tuple[Observation, Estimate]]:
    """Simulate `theory` on each observation's experiment and score its metric.

    Returns `[(observation, Estimate(value, variance)), ...]`. The
    `Estimate` mirrors `Metric.__call__`'s return: `value` is the canonical
    full-data scalar (what acceptance / loss read), `variance` is the
    between-subject variance reported alongside it for downstream prompts.
    A failure to simulate or evaluate the metric produces an
    `Estimate(None, None)` (rather than aborting the loop) so the feedback
    agent still sees the experiment and can react accordingly.
    """
    out: list[tuple[Observation, Estimate]] = []
    for obs in observations:
        try:
            sim = obs.experiment.simulate(theory, n_runs=n_runs)
            # `.copy()` so an in-place mutation in the LLM metric doesn't
            # corrupt `obs.data` aliases or sibling experiments downstream.
            est = obs.metric(sim.copy())
            value: float | None = (
                float(est.value) if est.value is not None else None
            )
            variance: float | None = (
                float(est.variance) if est.variance is not None else None
            )
        except Exception as e:
            print(
                f"[simulate_candidate] failed on experiment "
                f"{obs.experiment.__class__.__name__} "
                f"({type(e).__name__}: {e}); recording None."
            )
            value = None
            variance = None
        out.append((obs, Estimate(value=value, variance=variance)))
    return out

def aggregate_loss(
    results: list[tuple[Observation, Estimate]],
) -> float:
    """Per-iteration scalar loss for a candidate; lower is better.

    For each observation we compute `|real - candidate|` and divide by the
    spread of all numeric values known for that observation's metric
    (real, current candidate, every prior theory's prediction). This makes
    per-experiment losses comparable when the metrics live on very
    different scales (e.g. an accuracy in `[0, 1]` vs. a difference score
    in `[-1, 1]`); the final loss is the mean across observations.

    Edge cases:
      - `obs.real_value is None`: observation skipped (no ground truth to
        compare against).
      - `value is None`: candidate failed to simulate / score → assigned
        the maximum normalized distance (`1.0`) so a broken candidate can
        never beat a working one.
      - `spread <= 0`: every known value is identical → fall back to raw
        absolute distance (likely 0 anyway).
      - No usable observations → returns `+inf`.

    Used by `Improver` / `TheoryGenerator` to pick the best critique
    iteration when no iteration says `"continue"`, instead of returning
    whatever happened to be last.
    """
    losses: list[float] = []
    for obs, est in results:
        value = est.value
        if obs.real_value is None:
            continue
        if value is None:
            losses.append(1.0)
            continue
        known = [obs.real_value, value]
        known.extend(
            p.value for p in obs.predicted_values if p.value is not None
        )
        spread = max(known) - min(known)
        d = abs(value - obs.real_value)
        losses.append(d / spread if spread > 0 else d)
    return sum(losses) / len(losses) if losses else float("inf")


def save_fit_results(
    candidate_results: list[tuple[Observation, Estimate]],
    *,
    workspace: Path,
) -> Path:
    """Save scalar metric values (human vs. model) for each observation.

    Writes `<workspace>/fit_results.json` with one entry per observation
    containing the human real_value/variance and the candidate's simulated
    value/variance.
    """
    records = []
    for idx, (obs, est) in enumerate(candidate_results):
        records.append({
            "observation_index": idx,
            "human_value": obs.real_value,
            "human_variance": obs.real_variance,
            "model_value": est.value,
            "model_variance": est.variance,
            "metric_rationale": obs.metric.rationale,
        })
    workspace.mkdir(parents=True, exist_ok=True)
    out_path = workspace / "fit_results.json"
    out_path.write_text(json.dumps(records, indent=2))
    return out_path
