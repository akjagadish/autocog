from __future__ import annotations

from typing import TYPE_CHECKING

from src.prompts.interpret_results import ESTIMATE_NOTE
from src.prompts.model_improvement import (
    _format_arbiter_theory_key,
    _format_history_keys,
    _format_leaderboard_block,
    _format_loss_trajectory_block,
    _format_observations,
    _format_previous_candidate,
    _format_prior_feedback,
)

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.feedback import PriorIteration
    from src.observation import Observation
    from src.promptable import Promptable
    from src.theory import Theory

SYSTEM_PROMPT = """\
You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the {domain} domain \
based on the feedback provided by an arbiter. \
The feedback contains diagnoses of mechanistic failures of the previous \
theory along with suggestions for a new \
theory family that overcomes those failures. \
The newly proposed theory and model should display human-like behavior when simulated on experiment(s). \

The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,\
theories that explain data across the majority of experiments. \

You will see a list of theories that have been proposed in the past but you should \
only use them as inspiration and not to choose from them. Propose a new theory that is \
different. \

If they fail to do so, you will receive feedback on their performance \
on the same experiment(s) and you will have to propose another new theory and \
model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback \
that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, \
but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the \
candidate's `aggregate_loss` is compared against the running-best loss; \
strict improvement -> ACCEPTED (the candidate becomes the new running-best \
base); otherwise -> REJECTED (the candidate is discarded and the base is \
unchanged). You do NOT need to manually "revert" a regressed edit — the gate \
already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the \
running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it \
was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. \
Use this as ground truth on which past critic advice actually moved the loop \
forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique \
with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight \
critic advice whose previous candidates were REJECTED, and reinforce / extend \
advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit \
should plausibly land at-or-below it, otherwise the gate will reject your \
attempt and the base stays put.

{estimate_note}

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) \
to a *string* value that specifies its domain. Every value MUST be a string — \
never a bare list, number, tuple, or expression. Use exactly one of these \
notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{{v1, v2, ...}}"
   Example: "{{1, 2}}"

3. Vector of intervals whose length is set by the experiment — a bracketed \
tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), \
used when the parameter takes its value from an experiment-defined constant \
rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {{2, 3}}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in \
`parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic \
identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent \
new identifier names. If a parameter's shape doesn't fit any of those \
variables, fall back to a literal interval (notation 1) or discrete set \
(notation 2). Use these names so the model adapts to any experiment in \
this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
{parameter_variables}

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared \
`parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is \
domain-specific — see the experiment description above and the `history` key \
list below, which mirrors the per-trial variables carried in `state`). \
Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a \
list-of-dicts. The per-trial keys are:
{history_keys_doc}
Iterating `for x in history:` iterates the dict KEYS (strings); \
to walk trials index the lists in lock-step, e.g. \
`for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's \
discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen \
action. If you sample with `np.random.choice(..., p=probs)`, normalise \
first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) \
to avoid the "probabilities do not sum to 1" ValueError from float drift.
"""


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

{design_header}

## ARBITER GUIDE
{arbiter_theory_key}{arbiter_guide}
{leaderboard_block}{loss_trajectory_block}
## EXPERIMENTAL RESULTS
{experimental_results}
{previous_candidate_block}{prior_feedback_block}

## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


{proposal_directive}
## RESPONSE FORMAT
Return a JSON object with the following fields:
{instruction_format}
"""
# TODO PREVIOUS MODEL INSTANCES
# {previous_models}
# DIVERSITY REQUIREMENT
# {diversity_requirement}


def _format_parameter_variables(variables: dict[str, str]) -> str:
    if not variables:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in variables.items())


def _format_proposal_directive(
    previous_candidate: "Theory | None",
    *,
    target: str,
) -> str:
    """Render the `## PROPOSAL` block.

    Two regimes:
      - First iteration of the loop (`previous_candidate is None`): tell
        the LLM to propose from scratch (the original behaviour). There
        is nothing to edit yet, so a "minimal-diff" framing would be a
        no-op.
      - Subsequent iterations (`previous_candidate is not None`): tell
        the LLM to apply a MINIMAL-DIFF EDIT on the source rendered
        verbatim under `## PREVIOUS CANDIDATE (this loop)`. Rewriting the
        whole `predict` / `policy` for a small parameter or equation
        change wastes ~2k tokens per iteration and tends to introduce
        regressions. Pinning the diff to the smallest change that resolves
        the critic's diagnosis keeps the loop converging instead of
        oscillating across full rewrites.

    `target` controls the noun used in the directive
    (`"theory"` for theory_generation, `"model"` for model_improvement) so
    the same helper renders well in both prompts.
    """
    if previous_candidate is None:
        return (
            f"## PROPOSAL\n"
            f"Propose a novel {target} from scratch based on all the information "
            f"available, faithfully implementing the mechanism family the arbiter "
            f"prescribed above. Do NOT simply reuse anything generated in past "
            f"rounds.\n"
        )
    return (
        "## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)\n"
        "The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim "
        "above under `## PREVIOUS CANDIDATE (this loop)`. This is the "
        "base the loop's accept gate is currently keeping; the most "
        "recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) "
        "should be applied on top of it. Apply the SMALLEST edit that "
        "addresses the critic's diagnosis while staying inside the "
        "arbiter's prescribed mechanism family:\n"
        "  - Re-emit the previous source verbatim, then change ONLY the lines "
        "needed to address the critic (a parameter range, a normalization, a "
        "softmax temperature, an attention scheme, a gating term, a buggy "
        "indexing line, etc.).\n"
        "  - Keep all unaffected functions, equations, parameter names, and the "
        "overall mechanism intact.\n"
        "  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would "
        "do, and do NOT switch mechanism families — that is the arbiter's "
        "decision, not yours in this loop.\n"
        "  - Briefly explain the minimal edit in `rationale`.\n"
        "If you genuinely believe a larger rewrite is required, you may do one, "
        "but justify in `rationale` why the minimal-diff path was insufficient.\n"
    )


def render(
    *,
    experiment_class: type["Experiment"],
    response_schema: type["Promptable"],
    arbiter_guide: str,
    arbiter_theory_labels: tuple[str | None, str | None] | None = None,
    arbiter_target_idx: int | None = None,
    observations: list["Observation"] | None = None,
    previous_candidate: "Theory | None" = None,
    leaderboard: list[tuple[str, "Theory", float]] | None = None,
    prior_iterations: list["PriorIteration"] | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for proposing a brand-new Theory.

    `arbiter_theory_labels` is the `(theory_1_pi_label, theory_2_pi_label)`
    pair the arbiter saw under its `THEORY 1` / `THEORY 2` headings; when
    provided, a small lookup key is rendered above `arbiter_guide` so the
    LLM can resolve any numeric "Theory 1 / Theory 2" references in the
    arbiter's free-text recommendation back to concrete pi labels.
    `arbiter_target_idx` (1 or 2) flags which of those theories the
    recommendation acts on (i.e. the one being replaced wholesale here).

    `previous_candidate`, when provided, is the most recent in-loop attempt
    rendered verbatim under `## PREVIOUS CANDIDATE (this loop)` so the LLM
    can iterate on its own source rather than reconstructing it from the
    critic's prose feedback. Pass `None` on the first iteration.

    `observations` is the empirical evidence the new theory must explain —
    typically every `Observation` in the latest Round. Each is rendered as
    (experimental design, metric, real value, this loop's prior candidate
    trajectory when `prior_iterations` is given, other theories' values on
    the same metric) via `_format_observations` (shared with
    `model_improvement`).

    `response_schema` is the Pydantic class the LLM's output will be parsed
    into (typically `Theory` itself); its `instruction_format()` produces the
    field-list block at the bottom of the user prompt.

    `prior_iterations` is the chronological history of this propose-loop as
    `PriorIteration(rationale, estimates, loss, accepted)` entries. The
    estimate list is in the SAME order as `observations` so the
    EXPERIMENTAL RESULTS block can render a per-experiment trajectory;
    the `loss` and `accepted` fields together power both the
    `## LOSS TRAJECTORY` block (one row per iteration with the
    ACCEPTED / REJECTED tag from the loop's accept gate) and the
    "Outcome of this advice" line in `## PRIOR FEEDBACK ITERATIONS` so
    the proposer can down-weight past critiques whose candidates were
    REJECTED by the gate.
    """
    system_prompt = SYSTEM_PROMPT.format(
        domain=experiment_class.name,
        parameter_variables=_format_parameter_variables(
            experiment_class.parameter_variables
        ),
        history_keys_doc=_format_history_keys(experiment_class.output_columns),
        estimate_note=ESTIMATE_NOTE,
    )

    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        design_header=experiment_class.pretty_print_protocol(),
        arbiter_guide=arbiter_guide,
        arbiter_theory_key=_format_arbiter_theory_key(
            arbiter_theory_labels, arbiter_target_idx
        ),
        leaderboard_block=_format_leaderboard_block(leaderboard, observations),
        loss_trajectory_block=_format_loss_trajectory_block(prior_iterations),
        experimental_results=_format_observations(
            observations or [], prior_iterations=prior_iterations
        ),
        previous_candidate_block=_format_previous_candidate(
            previous_candidate, include_description=True
        ),
        prior_feedback_block=_format_prior_feedback(prior_iterations),
        proposal_directive=_format_proposal_directive(
            previous_candidate, target="theory"
        ),
        instruction_format=response_schema.instruction_format(),
    )
    return system_prompt, user_prompt
