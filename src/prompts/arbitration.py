from __future__ import annotations

from typing import TYPE_CHECKING

from src.metric import Estimate, fmt_estimate
from src.prompts.interpret_results import ESTIMATE_NOTE, _fmt  # noqa: F401

if TYPE_CHECKING:
    from src.arbiter_verdict import ArbiterVerdict
    from src.experiment import Experiment
    from src.observation import Round
    from src.observation import Observation

SYSTEM_PROMPT = """\
You are a renowned cognitive scientist arbitrating between two theories across \
multiple experiments in the {domain} domain.

Each experiment was proposed alongside a metric and an expected outcome. For \
each experiment you are shown the design, the metric, both theories' \
predicted metric values (from simulated data), and the observed metric value \
on real data. The two theories are tagged by stable labels (e.g. "{label_1}" \
and "{label_2}") and the same labels are reused on each experiment's \
predictions.

{estimate_note}

The goal of the arbitration is to surface theories that are task-invariant: that is,\
 theories that can explain data across all experiments in the same domain. \
Perform a deep dive: which among the two theories better captures the observed data?\
Do not just look at the newest experiments, but look at all experiments together. \
If a theory is good at explaining all the data keep it. However, if the \
both theories are good at explaining some experiment but not all, then \
it might be a good idea to propose a completly new theory that can potentially \
explain all the data. \
It is often better to propose a new theory than to propose a new model. \
Even if one theory is clearly better than the other, instead of proposing \
a new model, you can propose a new theory that is a stronger competitor to \
the winning theory instead of proposing a new model. \
Only propose a new model if both theories are very good and \
you are confident that the new model will be better than the current one \
clearly distinguish the two theories. \
 Then issue a verdict: either "new_model" (keep the \
current theory description, but regenerate new predict / policy / parameter \
ranges such that they better capture the observed data across all experiments) or \
 "new_theory" (the current theory is degenerate; propose a \
brand-new theory that can better capture the observed data across all experiments). \
`target_theory_idx` is 1 if you are acting on the theory \
labelled "{label_1}" (THEORY 1 below), or 2 if you are acting on the theory \
labelled "{label_2}" (THEORY 2 below). Justify your choice.
"""


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

## THEORY 1 — {label_1}
{theory_1_description}

`predict(parameters, state, history) -> np.ndarray`:
{theory_1_predict}

`policy(probs) -> int`:
{theory_1_policy}

## THEORY 2 — {label_2}
{theory_2_description}

`predict(parameters, state, history) -> np.ndarray`:
{theory_2_predict}

`policy(probs) -> int`:
{theory_2_policy}

## EXPERIMENT 1 (proposed by {label_1})

### DESIGN
{design_block_1}

### METRIC
Rationale:
{metric_rationale_1}

Source:
{metric_source_1}

### RESULTS
- Predicted under {label_1} (simulated): {theory_1_prediction_experiment_1}
- Predicted under {label_2} (simulated): {theory_2_prediction_experiment_1}
- Observed on real data: {real_value_experiment_1}

## EXPERIMENT 2 (proposed by {label_2})

### DESIGN
{design_block_2}

### METRIC
Rationale:
{metric_rationale_2}

Source:
{metric_source_2}

### RESULTS
- Predicted under {label_1} (simulated): {theory_1_prediction_experiment_2}
- Predicted under {label_2} (simulated): {theory_2_prediction_experiment_2}
- Observed on real data: {real_value_experiment_2}

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
{experimental_results}

## RESPONSE FORMAT

Return a JSON object with the following fields:
{instruction_format}
"""


def format_other_experiments(
    others: list["Observation"],
    label_1: str,
    label_2: str,
) -> str:
    """Render a block for each non-current-round observation, showing only
    the two current theories' predictions (by label) alongside the real
    value — no misleading "Candidate (simulated) value" line."""
    if not others:
        return "(no other experiments)"
    blocks: list[str] = []
    for i, obs in enumerate(others, start=3):
        p1 = obs.prediction_by_label(label_1)
        p2 = obs.prediction_by_label(label_2)
        block = (
            f"### Experiment {i}\n"
            f"**Design**\n{obs.experiment.pretty_print_design()}\n\n"
            f"**Metric**\n"
            f"```python\n{obs.metric.metric_source}\n```\n\n"
            f"**Observed (real) value:** {fmt_estimate(obs.real_as_estimate())}\n"
            f"**Predicted under {label_1}:** "
            f"{fmt_estimate(p1.as_estimate() if p1 else None)}\n"
            f"**Predicted under {label_2}:** "
            f"{fmt_estimate(p2.as_estimate() if p2 else None)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def render(
    *,
    experiment_class: type["Experiment"],
    round: "Round",
    arbiter_verdict: type["ArbiterVerdict"],
    other_observations: list["Observation"],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for arbitrating one round.

    The Round is expected to contain exactly two Observations, one per pi.
    Each Observation carries `proposer_label` (the authoring pi) plus a
    Prediction tagged with the proposer's label (own simulation) and a
    Prediction tagged with the adversary's label (cross-simulation), as
    written by `AutoCog.propose_round`.
    """
    if len(round) != 2:
        raise ValueError(
            f"arbitration.render: expected a round of 2 observations, got {len(round)}."
        )
    obs_1, obs_2 = round[0], round[1]
    label_1 = obs_1.proposer_label or "theory_1"
    label_2 = obs_2.proposer_label or "theory_2"
    theory_1, theory_2 = obs_1.proposer_theory, obs_2.proposer_theory

    def _est(prediction) -> Estimate | None:
        # In each obs, the proposer's prediction is tagged with its own label;
        # the cross prediction with the other pi's label. Wrap as Estimate so
        # downstream rendering picks up both value AND variance.
        return prediction.as_estimate() if prediction is not None else None

    t1_pred_e1 = _est(obs_1.prediction_by_label(label_1))
    t2_pred_e1 = _est(obs_1.prediction_by_label(label_2))
    t1_pred_e2 = _est(obs_2.prediction_by_label(label_1))
    t2_pred_e2 = _est(obs_2.prediction_by_label(label_2))

    system_prompt = SYSTEM_PROMPT.format(
        domain=experiment_class.name,
        label_1=label_1,
        label_2=label_2,
        estimate_note=ESTIMATE_NOTE,
    )
    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        label_1=label_1,
        label_2=label_2,
        # THEORY 1
        theory_1_description=theory_1.description,
        theory_1_predict=theory_1.predict_source,
        theory_1_policy=theory_1.policy_source,
        # THEORY 2
        theory_2_description=theory_2.description,
        theory_2_predict=theory_2.predict_source,
        theory_2_policy=theory_2.policy_source,
        # EXPERIMENT 1 — design rationale is already inside pretty_print()
        design_block_1=obs_1.experiment.pretty_print(),
        metric_rationale_1=obs_1.metric.rationale or "(no rationale provided)",
        metric_source_1=obs_1.metric.metric_source,
        theory_1_prediction_experiment_1=fmt_estimate(t1_pred_e1),
        theory_2_prediction_experiment_1=fmt_estimate(t2_pred_e1),
        real_value_experiment_1=fmt_estimate(obs_1.real_as_estimate()),
        # EXPERIMENT 2 — design rationale is already inside pretty_print()
        design_block_2=obs_2.experiment.pretty_print(),
        metric_rationale_2=obs_2.metric.rationale or "(no rationale provided)",
        metric_source_2=obs_2.metric.metric_source,
        theory_1_prediction_experiment_2=fmt_estimate(t1_pred_e2),
        theory_2_prediction_experiment_2=fmt_estimate(t2_pred_e2),
        real_value_experiment_2=fmt_estimate(obs_2.real_as_estimate()),
        # OTHER EXPERIMENTS
        experimental_results=format_other_experiments(
            other_observations, label_1, label_2
        ),
        # FORMAT
        instruction_format=arbiter_verdict.instruction_format(),
    )
    return system_prompt, user_prompt
