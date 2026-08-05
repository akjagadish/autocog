from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.metric import Metric
    from src.theory import Theory


SYSTEM_PROMPT = """\
You are a psychology researcher proposing a metric in the {domain} domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — \
i.e., its value, computed on data simulated under your advocated theory, should be \
as far as possible from its value computed on data simulated under the competing \
theory. The direction of the gap does not matter; what matters is that the two \
theories produce visibly different numbers on this metric. The metric is computed \
on the data collected from the experimental design provided in the prompt. Produce \
a metric where you're prediction will be much more accurate than the competing \
theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as \
`point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame \
(all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of \
`metric(subj_df)` re-applied per `subject_id`, summarising how stable the \
metric is across subjects. If your metric only makes sense on multi-subject \
data this will fall back to `n/a` and the metric is rejected (the \
acceptance test below cannot run without it). Prefer metrics that work \
both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's \
two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` \
vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the \
number of HUMAN subjects the experiment will actually be run with (a fixed \
small number, currently {real_n_subjects}). Your metric is admitted iff the \
two-sided p-value is below the significance level (currently \
alpha={alpha:g}). Implication: a large between-theory gap is NOT enough — if \
either theory's metric is also highly variable across subjects, N humans \
won't reliably distinguish them and the metric will be rejected. Aim for \
contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.
"""


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

{design_header}

## CHOSEN EXPERIMENTAL DESIGN
{experiment_block}

## ADVOCATED THEORY
{advocating_description}

## COMPETING THEORY
{competing_description}

## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects \
(rows grouped by `subject_id`, in trial order). Columns:
{output_columns}

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / \
np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. \
Those cells are NOT hashable, so operations that hash row values fail with \
`TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder \
for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`, \
    `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns \
    an object array but downstream `set()` / `in dict` will crash), \
    `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list \
    values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer \
    responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be \
    parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did \
NOT discriminate the two theories at the human sample size — either it \
errored, its between-subject variance was unavailable, or Welch's t-test on \
`(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. \
The `outcome` line is the simulation result (means, between-subject \
variances, t-statistic and p-value at the human N) on the same `data_self` \
/ `data_adv` your next metric will be evaluated on. Use the numbers to see \
where your hypothesised contrast collapsed — small mean gap, large \
per-subject variance, or both — and propose something qualitatively \
different. Don't repeat the same idea with cosmetic tweaks.
{ledger}

## RESPONSE FORMAT
Return a JSON object with the following fields:
{instruction_format}
"""


def _format_columns(columns: dict[str, str]) -> str:
    if not columns:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in columns.items())


def _format_ledger(failed: list[tuple["Metric", str]]) -> str:
    """Render past failed metric attempts with their numeric outcomes.

    Each item is `(metric, outcome)`, where `outcome` is the human-readable
    result line printed by the orchestrator for that attempt — e.g.
    `self_sim=0.9600 adversary_sim=2.2675 -> reject` or
    `evaluation failed (TypeError: unhashable type: 'list')`.
    """
    if not failed:
        return "(none yet)"
    return "\n\n".join(
        f"[{i}] rationale: {m.rationale or '(no rationale)'}\n"
        f"metric_source:\n{m.metric_source}\n"
        f"outcome: {outcome}"
        for i, (m, outcome) in enumerate(failed)
    )


def render(
    *,
    experiment_class: type["Experiment"],
    experiment: "Experiment",
    advocating: "Theory",
    competing: "Theory",
    metric_class: type["Metric"],
    ledger: list[tuple["Metric", str]] | None = None,
    real_n_subjects: int,
    alpha: float,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for proposing a metric."""
    system_prompt = SYSTEM_PROMPT.format(
        domain=experiment_class.name,
        real_n_subjects=real_n_subjects,
        alpha=alpha,
    )

    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        design_header=experiment_class.pretty_print_protocol(),
        experiment_block=experiment.pretty_print(),
        advocating_description=advocating.pretty_print(),
        competing_description=competing.pretty_print(),
        output_columns=_format_columns(experiment_class.output_columns),
        ledger=_format_ledger(ledger or []),
        instruction_format=metric_class.instruction_format(),
    )
    return system_prompt, user_prompt
