from __future__ import annotations

from typing import TYPE_CHECKING

from src.metric import Estimate, fmt_estimate

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.metric import Metric
    from src.theory import Theory


# Single source of truth for the "what do these two numbers mean?" note.
# Imported by every prompt that displays metric values so the explanation
# is consistent across the system.
ESTIMATE_NOTE = (
    "Each metric value below is shown as `point_estimate (var=X)`, where "
    "`point_estimate` is `metric(data)` evaluated on the full pooled "
    "dataset and `var` is the population (between-subject) variance of "
    "the same metric re-applied per `subject_id`. The point estimate is "
    "the canonical scalar; `var` reports how consistent that estimate is "
    "across subjects (lower = more consistent). `var=n/a` means the "
    "metric could not be applied to a single-subject slice."
)


SYSTEM_PROMPT = """\
You are a renowned cognitive scientist interpreting the results of an experiment you \
designed in the {domain} domain.

You pre-registered a metric expected to be HIGHER under your advocated theory \
than under the competing theory. Below you are shown both theories, the \
experimental design, the metric, the metric's predicted value under each \
theory (from simulated data), and its observed value on the real data.

{estimate_note}

Write a freeform interpretation: does the observed value support the \
advocated theory, the competing theory, neither, or both? Flag any \
confounds, alternative explanations, or weaknesses in the design or metric \
that should temper the conclusion. Be honest about ambiguity — do not \
overclaim.
"""


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

{design_header}

## CHOSEN EXPERIMENTAL DESIGN
{experiment_block}

## ADVOCATED THEORY
{advocating_description}

`predict(parameters, state, history) -> np.ndarray`:
{advocating_predict}

`policy(probs) -> int`:
{advocating_policy}

## COMPETING THEORY
{competing_description}

`predict(parameters, state, history) -> np.ndarray`:
{competing_predict}

`policy(probs) -> int`:
{competing_policy}

## METRIC
Rationale:
{metric_rationale}

Source:
{metric_source}

## RESULTS
- Predicted under advocated theory (simulated): {predicted_self}
- Predicted under competing theory (simulated): {predicted_adversary}
- Observed on real data: {real_value}

## INTERPRETATION
Write your interpretation below as plain prose. Cover, at minimum:
1. Whether the observed value is closer to the advocated or competing prediction.
2. Whether the qualitative pattern targeted by the design appears in the data.
3. Confounds, alternative explanations, or weaknesses to flag.
4. What experiment or metric should come next, and why.
"""


def _fmt(value: float | None) -> str:
    """Render a bare scalar (no variance attached). Kept for callers that
    still pass plain floats (e.g. trajectory tables that only carry the
    point estimate). For `Estimate`-shaped values use `fmt_estimate`."""
    return "n/a" if value is None else f"{value:.4f}"


def render(
    *,
    experiment_class: type["Experiment"],
    experiment: "Experiment",
    metric: "Metric",
    advocating: "Theory",
    competing: "Theory",
    predicted_self: "Estimate | None",
    predicted_adversary: "Estimate | None",
    real: "Estimate | None",
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for interpreting a round's results.

    `predicted_self`, `predicted_adversary`, `real` are `Estimate` namedtuples
    (`(value, variance)`) — point estimate plus between-subject variance.
    Both numbers get rendered in the user prompt; the system prompt
    explains what they mean.

    The response is freeform text — no JSON schema is enforced — because the
    interpretation is consumed by a human (or another LLM step) rather than
    executed as code.
    """
    system_prompt = SYSTEM_PROMPT.format(
        domain=experiment_class.name, estimate_note=ESTIMATE_NOTE
    )

    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        design_header=experiment_class.pretty_print_header(),
        experiment_block=experiment.pretty_print(),
        advocating_description=advocating.description,
        advocating_predict=advocating.predict_source,
        advocating_policy=advocating.policy_source,
        competing_description=competing.description,
        competing_predict=competing.predict_source,
        competing_policy=competing.policy_source,
        metric_rationale=metric.rationale or "(no rationale provided)",
        metric_source=metric.metric_source,
        predicted_self=fmt_estimate(predicted_self),
        predicted_adversary=fmt_estimate(predicted_adversary),
        real_value=fmt_estimate(real),
    )
    return system_prompt, user_prompt
