from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scipy import stats

from src.config import load_config
from src.llm import LLMClient, make_client
from src.experiment import Experiment
from src.metric import Estimate, Metric
from src.observation import Observation, Observations
from src.prompts import experiment_proposal, interpret_results, metric_proposal
from src.run_config import REAL_N_SUBJECTS
from src.theory import Theory
from src.logger import log


class AutoCog:
    """
    One side of the adversarial debate.

    Stateless w.r.t. both evidence and persistence: AutoCog never owns
    observations and has no constructor workspace. The orchestrator holds
    the shared `Observations` pool and passes a per-call `workspace` to
    each public method (where prompt logs are written). AutoCog carries:

      * `label` — stable identifier used to tag Predictions and Observations
        across the loop (e.g. `"pi_1"`, `"pi_1_1"` after improver regenerates
        the model, `"pi_3"` after theory-generator drops in a brand-new
        theory). The orchestrator owns the naming scheme; AutoCog just uses
        whatever it's given.
      * `theory` — the theory it currently advocates,
      * `experiment_class` — the domain it operates in,
      * `llm_client` — how to talk to the model.

    Theories can change across rounds (via improver / theory_generator). Each
    `Observation` written to the pool snapshots the proposer theory + label
    at Round level, so Predictions inside it (tagged by pi label) remain
    attributable to a concrete (label, theory) pair even after the live pi
    has moved on.
    """

    def __init__(
        self,
        *,
        label: str,
        theory: Theory,
        experiment_class: type[Experiment],
        llm_client: LLMClient,
    ):
        self.label = label
        self._theory = theory
        self.experiment_class = experiment_class
        self.llm_client = llm_client

    @classmethod
    def from_yaml(
        cls,
        theory_path: str | Path,
        *,
        label: str,
        experiment_class: type[Experiment],
        config_path: str | Path = "configs/default.yaml",
    ) -> "AutoCog":
        theory = Theory.from_yaml(theory_path)
        run_cfg = load_config(Path(config_path))
        llm_client = make_client(run_cfg.llm)
        return cls(
            label=label,
            theory=theory,
            experiment_class=experiment_class,
            llm_client=llm_client,
        )

    @property
    def theory(self) -> Theory:
        """The current (next-to-be-used) theory."""
        return self._theory

    # --- llm helper ---------------------------------------------------------

    def _generate_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type | None = None,
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> Any:
        """
        Single point of contact with the LLM. Returns the parsed structured
        output when `response_schema` is given, otherwise the raw text string.
        Swap this method to switch backends without touching the proposal
        methods.

        If `workspace` and `log_label` are both given, write a markdown log to
        `<workspace>/prompts/<log_label>.md` containing the system prompt,
        user prompt, response, and usage info.
        """
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

    # --- low-level llm proposals (no observation side effects) --------------

    def _llm_propose_experiment(
        self,
        adversary: "AutoCog",
        *,
        ledger: list[Experiment],
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> Experiment:
        """One LLM call: render → call → parse. No append, no save."""
        log(f"Proposing experiment for {self.label} vs {adversary.label}")
        system_prompt, user_prompt = experiment_proposal.render(
            experiment_class=self.experiment_class,
            advocating=self._theory,
            competing=adversary.theory,
            ledger=ledger,
        )
        parsed = self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=self.experiment_class,
            workspace=workspace,
            log_label=log_label,
        )
        return self.experiment_class(**parsed.model_dump())

    def _llm_propose_metric(
        self,
        adversary: "AutoCog",
        experiment: Experiment,
        *,
        ledger: list[tuple[Metric, str]],
        real_n_subjects: int,
        alpha: float,
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> Metric:
        """One LLM call: render → call → parse. No append, no save."""
        log(f"Proposing metric for {self.label} vs {adversary.label}")
        system_prompt, user_prompt = metric_proposal.render(
            experiment_class=self.experiment_class,
            experiment=experiment,
            advocating=self._theory,
            competing=adversary.theory,
            metric_class=Metric,
            ledger=ledger,
            real_n_subjects=real_n_subjects,
            alpha=alpha,
        )
        parsed = self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=Metric,
            workspace=workspace,
            log_label=log_label,
        )
        return Metric(**parsed.model_dump())

    # --- validated round (the real loop) ------------------------------------

    def propose_round(
        self,
        adversary: "AutoCog",
        pool: Observations,
        *,
        workspace: Path | None = None,
        n_runs: int = 50,
        max_experiments: int = 3,
        max_metrics: int = 4,
        real_n_subjects: int = REAL_N_SUBJECTS,
        alpha: float = 0.01,
    ) -> Observation:
        """
        Run the adversarial inner loop and return one Observation.

        `pool` is read-only here: it provides the ledger of past experiments
        and metrics so the LLM doesn't repeat itself. The caller is
        responsible for committing the returned Observation to the pool
        (`pool.add(obs)`) — this method has no persistence side effects.

        `workspace`, if given, is the directory under which prompt logs for
        this call go (`<workspace>/prompts/...`). It's a per-call argument
        because logs are normally organised round-wise by the orchestrator.

        For up to `max_experiments` experiment proposals, and for each up to
        `max_metrics` metric proposals:

          1. propose an experiment (excluding pool + previously rejected)
          2. simulate it once with each theory (`n_runs` simulated subjects
             per theory; data is cached across metric attempts on the same
             experiment)
          3. propose a metric (excluding pool + previously rejected)
          4. evaluate the metric on both datasets and ask: would the two
             theories' predictions be statistically distinguishable when the
             experiment is actually run with `real_n_subjects` HUMAN
             subjects? Concretely, run a Welch's two-sample t-test on
             `(v_self, var_self, real_n_subjects)` vs.
             `(v_adv, var_adv, real_n_subjects)`, where `var_*` is the
             between-subject variance reported alongside each estimate by
             `Metric.__call__`. The metric is ACCEPTED iff the test's
             two-sided p-value is below `alpha`; otherwise it is rejected
             (the gap could be drowned out by within-theory variability at
             that sample size) and the next metric is tried. If all
             metrics fail, propose a new experiment.

        `real_n_subjects` defaults to `REAL_N_SUBJECTS` from
        `src/run_config.py`, the same N `main.py` uses to collect the real
        data — so the discriminability check uses the exact sample size
        humans will be run at.

        Returns the first accepted (experiment, metric) Observation, or — if
        no attempt wins — the last attempted Observation. Predictions are
        tagged by pi label (`self.label`, `adversary.label`); the resulting
        Observation also carries `proposer_label = self.label`.
        """
        log(f"Proposing round for {self.label} vs {adversary.label}")
        round_idx = len(pool)
        rejected_experiments: list[Experiment] = []
        # Per-round ledger of failed metric attempts paired with the exact
        # outcome string we want to surface to the LLM next round.
        failed_metric_attempts: list[tuple[Metric, str]] = []
        last: Observation | None = None

        for k_exp in range(max_experiments):
            experiment = self._llm_propose_experiment(
                adversary,
                ledger=list(pool.experiments) + rejected_experiments,
                workspace=workspace,
                log_label=f"experiment_attempt_{k_exp:02d}",
            )
            data_self = experiment.simulate(self._theory, n_runs=n_runs)
            data_adv = experiment.simulate(adversary.theory, n_runs=n_runs)

            for k_metric in range(max_metrics):
                metric = self._llm_propose_metric(
                    adversary,
                    experiment,
                    ledger=failed_metric_attempts,
                    real_n_subjects=real_n_subjects,
                    alpha=alpha,
                    workspace=workspace,
                    log_label=f"metric_exp{k_exp:02d}_attempt_{k_metric:02d}",
                )
                outcome: str
                est_self: Estimate | None = None
                est_adv: Estimate | None = None
                try:
                    # `.copy()`: LLM-generated metrics often mutate the input
                    # frame in place (e.g. add helper columns); without a
                    # copy those leak across `max_metrics` attempts on the
                    # same `data_self` / `data_adv`. `metric(...)` returns an
                    # `Estimate(value, variance)` namedtuple — value is the
                    # full-data scalar (the acceptance check looks at this);
                    # variance is the between-subject variability we record
                    # alongside it for downstream prompts.
                    est_self = metric(data_self.copy())
                    est_adv = metric(data_adv.copy())
                    v_self: float | None = (
                        float(est_self.value) if est_self.value is not None else None
                    )
                    v_adv: float | None = (
                        float(est_adv.value) if est_adv.value is not None else None
                    )
                    var_self = est_self.variance
                    var_adv = est_adv.variance
                    if v_self is None or v_adv is None:
                        print(f"Metric evaluation failed for {self.label} or {adversary.label}")
                        continue
                    # Acceptance rule: would `real_n_subjects` HUMAN subjects
                    # actually distinguish the two theories on this metric?
                    # We treat each side's simulation as a finite-sample
                    # estimate of (mean, variance) for the metric in that
                    # theory's data-generating process, then run Welch's
                    # two-sample t-test parameterised by the real-run sample
                    # size. Accepted iff the two-sided p-value is below
                    # `alpha` (default 0.05). Welch (equal_var=False) tolerates
                    # heteroscedastic between-theory variance, which is the
                    # common case here (different mechanisms -> different
                    # subject-to-subject spreads).
                    if (
                        var_self is None
                        or var_adv is None
                        or math.isnan(var_self)
                        or math.isnan(var_adv)
                    ):
                        accepted = False
                        outcome = (
                            f"self_sim={v_self:.4f} (var=n/a) "
                            f"adversary_sim={v_adv:.4f} (var=n/a) "
                            f"-> reject (variance unavailable; cannot test "
                            f"discriminability at N={real_n_subjects})"
                        )
                        print(
                            f"[round {round_idx} exp{k_exp} metric{k_metric}] "
                            f"{outcome}"
                        )
                        failed_metric_attempts.append((metric, outcome))
                        continue
                    std_self = math.sqrt(max(var_self, 0.0))
                    std_adv = math.sqrt(max(var_adv, 0.0))
                    t_stat, p_value_raw = stats.ttest_ind_from_stats(
                        mean1=v_self,
                        std1=std_self,
                        nobs1=real_n_subjects,
                        mean2=v_adv,
                        std2=std_adv,
                        nobs2=real_n_subjects,
                        equal_var=False,
                    )
                    # Welch returns NaN when both stds are 0 and means are
                    # identical (0/0 degenerate); treat it as "no separation".
                    p_value = (
                        1.0 if math.isnan(float(p_value_raw)) else float(p_value_raw)
                    )
                    t_stat_f = float(t_stat)
                    accepted: bool = p_value < alpha
                    outcome = (
                        f"self_sim={v_self:.4f} (var={var_self:.4f}) "
                        f"adversary_sim={v_adv:.4f} (var={var_adv:.4f}) "
                        f"welch_t={t_stat_f:+.3f} p={p_value:.4g} "
                        f"(N={real_n_subjects}, alpha={alpha:g}) "
                        f"-> {'ACCEPT' if accepted else 'reject'}"
                    )
                    print(
                        f"[round {round_idx} exp{k_exp} metric{k_metric}] "
                        f"{outcome}"
                    )
                except Exception as e:
                    est_self = est_adv = None
                    v_self = v_adv = None
                    var_self = var_adv = None
                    accepted = False
                    outcome = (
                        f"evaluation failed ({type(e).__name__}: {e})"
                    )
                    print(
                        f"[round {round_idx} exp{k_exp} metric{k_metric}] "
                        f"metric eval failed ({type(e).__name__}: {e}); rejecting."
                    )

                last = Observation(
                    experiment=experiment,
                    metric=metric,
                    proposer_theory=self._theory,
                    proposer_label=self.label,
                )
                last.record_prediction(
                    v_self, variance=var_self, label=self.label
                )
                last.record_prediction(
                    v_adv, variance=var_adv, label=adversary.label
                )
                if accepted:
                    return last
                failed_metric_attempts.append((metric, outcome))

            rejected_experiments.append(experiment)

        assert last is not None  # max_experiments >= 1 and max_metrics >= 1
        print(
            f"[round {round_idx}] no attempt accepted after "
            f"{max_experiments} experiments × {max_metrics} metrics; "
            f"returning last attempt."
        )
        return last

    # --- interpretation -----------------------------------------------------

    def interpret_round(
        self,
        adversary: "AutoCog",
        observation: Observation,
        *,
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> str:
        """
        Ask the LLM for a freeform interpretation of one Observation.

        The Observation is passed in directly (it lives in the orchestrator's
        pool, not on this pi). Returns the interpretation text; the caller
        owns whatever they want to do with it. `workspace` is the per-call
        log directory (typically a round-scoped folder).
        """
        pred_self = observation.prediction_by_label(self.label)
        pred_adv = observation.prediction_by_label(adversary.label)
        system_prompt, user_prompt = interpret_results.render(
            experiment_class=self.experiment_class,
            experiment=observation.experiment,
            metric=observation.metric,
            advocating=observation.proposer_theory,
            competing=adversary.theory,
            predicted_self=(
                pred_self.as_estimate() if pred_self is not None else None
            ),
            predicted_adversary=(
                pred_adv.as_estimate() if pred_adv is not None else None
            ),
            real=observation.real_as_estimate(),
        )
        return self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=None,
            workspace=workspace,
            log_label=log_label,
        )
