"""Prompt: refine the *model* (predict / policy / parameter ranges) under a
fixed *theory description*.

Used by the `Improver` agent when the arbiter's verdict is `"new_model"`. The
LLM keeps the prose theory verbatim and only regenerates the runnable bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.metric import Estimate, fmt_estimate
from src.prompts.interpret_results import ESTIMATE_NOTE, _fmt

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.feedback import PriorIteration
    from src.observation import Observation
    from src.promptable import Promptable
    from src.theory import Theory


SYSTEM_PROMPT = """\
You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a NEW model instantiation of an EXISTING theory, given \
arbiter feedback on the previous instantiation. The theory's prose claim is \
fixed — you are NOT redefining the theory. You are regenerating only the \
runnable bits: the `predict` function, the `policy` function, and the \
`parameters` ranges. The newly proposed model should display human-like \
behavior when simulated on experiments in the {domain} domain.

The goal of the model improvement process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,\
theories that explain data across multiple experiments. \

If your model fails to compile or behaves badly, you may receive feedback and \
have to propose another instantiation. Iterate until accepted.

If you think the failure to capture human behavior is due to arbiter feedback \
that is inaccurate or unhelpful, you can propose a new model instance that \
ignores the feedback, but you must provide rationale for why you are ignoring \
it and how your proposal overcomes the identified mechanistic failures.

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
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` \
are reserved for the vector-of-intervals notation in (3).
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

## ROUND THEORIES
The arbiter compared the two theories below this round. Your job is to \
regenerate ONLY the runnable bits (`predict`, `policy`, `parameters`) of the \
one tagged **TO REVISE**, keeping its description verbatim. The other theory \
is shown for context — it is NOT being changed.

{round_theories_block}
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


def _format_one_round_theory(
    theory: "Theory",
    *,
    idx: int,
    label: str | None,
    is_target: bool,
) -> str:
    """Render one theory in the `## ROUND THEORIES` section.

    Mirrors the arbiter's `THEORY 1` / `THEORY 2` headings (anchored to pi
    labels) so any "Theory N" reference in the arbiter's free-text
    recommendation lands on a concrete subsection here. The target (the
    theory being revised) is tagged **TO REVISE**; the other one is tagged
    `(other theory — context only, NOT revised)` and rendered with the same
    depth (description + predict + policy + parameters) so the LLM can
    reason about the contrast.
    """
    label_str = f"`{label}`" if label else "(no label)"
    if is_target:
        suffix = "**TO REVISE**"
        body_header = (
            "**Description (PRESERVE VERBATIM — do not rewrite the prose):**"
        )
        model_header = (
            "**Previous model instance — REGENERATE `predict`, `policy`, "
            "and `parameters` ranges:**"
        )
    else:
        suffix = "(other theory — context only, NOT revised)"
        body_header = "**Description:**"
        model_header = "**Model instance (shown for context):**"
    return (
        f"### THEORY {idx} — {label_str} {suffix}\n\n"
        f"{body_header}\n{theory.description}\n\n"
        f"{model_header}\n\n"
        f"`predict(parameters, state, history) -> np.ndarray`:\n"
        f"{theory.predict_source}\n\n"
        f"`policy(probs) -> int`:\n{theory.policy_source}\n\n"
        f"`parameters`:\n{_format_parameters(theory.parameters)}\n"
    )


def _format_round_theories(
    target_theory: "Theory",
    *,
    other_theory: "Theory | None",
    labels: tuple[str | None, str | None] | None,
    target_idx: int | None,
) -> str:
    """Render the `## ROUND THEORIES` block: both round theories side by
    side, with the target tagged **TO REVISE**.

    Falls back to the legacy "single theory" view (just the killed theory,
    no THEORY 1/2 framing) when `other_theory` or `target_idx` is missing —
    keeps direct callers of `render` working without forcing them to plumb
    round metadata through.
    """
    target_idx_eff = target_idx if target_idx in (1, 2) else 1
    target_label = labels[target_idx_eff - 1] if labels else None
    target_block = _format_one_round_theory(
        target_theory,
        idx=target_idx_eff,
        label=target_label,
        is_target=True,
    )
    if other_theory is None:
        return target_block + "\n"
    other_idx = 3 - target_idx_eff
    other_label = labels[other_idx - 1] if labels else None
    other_block = _format_one_round_theory(
        other_theory,
        idx=other_idx,
        label=other_label,
        is_target=False,
    )
    blocks = sorted(
        [(target_idx_eff, target_block), (other_idx, other_block)],
        key=lambda x: x[0],
    )
    return "\n---\n\n".join(b for _, b in blocks) + "\n"


def _format_arbiter_theory_key(
    labels: tuple[str | None, str | None] | None,
    target_idx: int | None,
) -> str:
    """Render the THEORY 1 / THEORY 2 -> pi-label lookup key for the arbiter
    guide.

    The arbitration prompt rendered the round's two theories as `THEORY 1`
    and `THEORY 2` (anchored to `pi_1` / `pi_2` labels in that prompt), and
    the arbiter's free-text recommendation refers back to those numeric
    labels. Once the recommendation is forwarded here as `arbiter_guide`
    that anchoring is gone, so we re-emit the key right above the guide
    text. Returns "" (no extra block) when labels aren't supplied — old
    callers stay unchanged.

    The returned string ends in a blank line so it slots in cleanly above
    the guide text in the TEMPLATE.
    """
    if not labels or labels[0] is None or labels[1] is None:
        return ""
    lines = [
        "The arbiter labelled this round's two theories in its recommendation as follows:",
        f"- THEORY 1 = `{labels[0]}`",
        f"- THEORY 2 = `{labels[1]}`",
    ]
    if target_idx in (1, 2):
        target_label = labels[target_idx - 1]
        lines.append(
            f"- The recommendation below acts on THEORY {target_idx} "
            f"(= `{target_label}`)."
        )
    return "\n".join(lines) + "\n\n"


def _format_parameter_variables(variables: dict[str, str]) -> str:
    if not variables:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in variables.items())


def _format_history_keys(output_columns: dict[str, str]) -> str:
    """Render the per-trial `history` dict keys from the experiment's
    `output_columns`, skipping `subject_id` (which is a DataFrame column
    but not a per-trial history key). Mirrors what `Experiment._reset_history`
    sets up on each subclass so the LLM sees the actual keys available
    inside `predict`.

    Each description is emitted verbatim (no `.lower()`, no "list of past "
    prefix surgery) because `output_columns` descriptions are authored as
    per-trial value captions — turning them into "list of …" phrases would
    mangle full sentences (e.g. `"0 if subject chose A, 1 if subject chose B"`).
    A single header line tells the LLM that each value is a list in trial order.
    """
    keys = [(k, v) for k, v in output_columns.items() if k != "subject_id"]
    if not keys:
        return "  (no per-trial keys declared)"
    lines = [
        "  Each value below is a Python list in trial order; entry `i` is the "
        "value for trial `i`. On the first trial all lists are empty."
    ]
    lines.extend(f"  - `\"{k}\"`: {v}" for k, v in keys)
    return "\n".join(lines)


def _format_parameters(parameters: dict[str, str]) -> str:
    if not parameters:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in parameters.items())


def _format_other_predictions(obs: "Observation") -> str:
    """Render previously-recorded theory predictions on this Observation's
    metric, one per line, as a calibration reference for the proposer.

    Mirrors the same block shown to the feedback critic so the proposer
    sees the same competitor landscape it will be judged against.
    """
    if not obs.predicted_values:
        return "(none)"
    lines = [
        f"- {p.label}: {fmt_estimate(p.as_estimate())}"
        for p in obs.predicted_values
    ]
    return "\n".join(lines)


def _format_prior_trajectory(
    obs_idx: int,
    real_value: float | None,
    prior_iterations: list["PriorIteration"] | None,
) -> str:
    """Render the per-experiment trajectory of THIS loop's prior candidates.

    The proposer hasn't simulated the next candidate yet, so (unlike the
    feedback prompt) there is no "current" row — we only show iterations
    that have already been critiqued. The most recent entry corresponds to
    the source rendered verbatim under `## PREVIOUS CANDIDATE (this loop)`.
    Each row carries both the point estimate and the between-subject
    variance so a candidate that's drifting in noise (rather than mean) is
    visible in the trace.
    """
    if not prior_iterations:
        return ""
    rows: list[str] = []
    last = len(prior_iterations)
    for k, it in enumerate(prior_iterations, start=1):
        est = it.estimates[obs_idx] if obs_idx < len(it.estimates) else None
        delta = (
            f" (Δ vs real {est.value - real_value:+.4f})"
            if est is not None
            and est.value is not None
            and real_value is not None
            else ""
        )
        suffix = " (most recent)" if k == last else ""
        rows.append(f"  - iter {k}{suffix}: {fmt_estimate(est)}{delta}")
    return "**Previous candidate values (this loop):**\n" + "\n".join(rows)


def _format_observations(
    observations: list["Observation"],
    *,
    prior_iterations: list["PriorIteration"] | None = None,
) -> str:
    """Render a list of Observations as the EXPERIMENTAL RESULTS block.

    For each observation we show:
      - the experimental design (delegated to `experiment.pretty_print()`),
      - the metric (rationale + source),
      - the observed real value,
      - (when `prior_iterations` is non-empty) the per-experiment trajectory
        of this loop's prior candidate values, so the proposer can see how
        its own iterates have moved on this metric and avoid oscillating,
      - the values produced by every other theory previously evaluated
        against this metric, as a calibration reference (same block the
        feedback critic sees).
    """
    if not observations:
        return "(no observations yet)"
    blocks: list[str] = []
    for i, obs in enumerate(observations, start=1):
        trajectory_block = _format_prior_trajectory(
            i - 1, obs.real_value, prior_iterations
        )
        trajectory_section = f"{trajectory_block}\n" if trajectory_block else ""
        block = (
            f"### Experiment {i}\n"
            f"**Design**\n{obs.experiment.pretty_print_design()}\n\n"
            f"**Metric**\n"
            f"```python\n{obs.metric.metric_source}\n```\n\n"
            f"**Observed (real) value:** {fmt_estimate(obs.real_as_estimate())}\n"
            f"{trajectory_section}"
            f"**Other theories' values on this metric (for reference):**\n"
            f"{_format_other_predictions(obs)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_leaderboard_entry(
    label: str,
    theory: "Theory",
    score: float,
    observations: list["Observation"] | None,
) -> str:
    """Render one leaderboard entry as a sub-section.

    Includes the theory's prose description, predict / policy source,
    parameter ranges, overall score, and per-observation `real= … vs this= …`
    fit table — i.e. enough for the LLM to compare its candidate against
    this competitor on the *same* evidence.
    """
    rows: list[str] = []
    for i, obs in enumerate(observations or [], start=1):
        pred = obs.prediction_by_label(label)
        rows.append(
            f"- Experiment {i}: "
            f"real={fmt_estimate(obs.real_as_estimate())} "
            f"vs this={fmt_estimate(pred.as_estimate() if pred is not None else None)}"
        )
    fits = "\n".join(rows) if rows else "(no observations to compare on yet)"
    return (
        f"### `{label}` (overall score: {score:.3f})\n\n"
        f"**Description**\n{theory.description}\n\n"
        f"`predict(parameters, state, history) -> np.ndarray`:\n"
        f"{theory.predict_source}\n\n"
        f"`policy(probs) -> int`:\n{theory.policy_source}\n\n"
        f"`parameters`:\n{_format_parameters(theory.parameters)}\n\n"
        f"**Per-experiment fit (real vs this theory's metric value):**\n"
        f"{fits}\n"
    )


def _format_leaderboard_block(
    leaderboard: list[tuple[str, "Theory", float]] | None,
    observations: list["Observation"] | None,
) -> str:
    """Render the `## THEORY LEADERBOARD` section.

    `leaderboard` is a list of `(label, theory, overall_score)` triples
    chosen by `main._select_leaderboard` per the run-level `LEADERBOARD`
    config (`("none" | "best" | "sample", n)`). Each entry expands into a
    full sub-section (description + predict + policy + parameters +
    per-experiment fits + overall score) so the LLM can compare its
    candidate against concrete prior competitors. Returns `""` when the
    leaderboard is empty.

    The arbiter-killed theory is intentionally absent — it is shown above
    as `## PREVIOUS MODEL INSTANCE` and `main._select_leaderboard` filters
    it out of the leaderboard to avoid duplicating its body.
    """
    if not leaderboard:
        return ""
    entries = "\n\n---\n\n".join(
        _format_leaderboard_entry(label, theory, score, observations)
        for label, theory, score in leaderboard
    )
    return (
        "\n## THEORY LEADERBOARD\n"
        "A small set of prior picked theories shown for reference. "
        "Overall score is in `[0, 1]`, higher = better, computed as "
        "`1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. "
        "1.0 means closest to the real value on every experiment+metric "
        "pair; 0.0 means farthest. Each entry below carries the same "
        "depth of detail as the PREVIOUS MODEL INSTANCE above so you can "
        "borrow concrete mechanisms when useful.\n\n"
        f"{entries}\n"
    )


def _format_previous_candidate(
    candidate: "Theory | None",
    *,
    include_description: bool,
    section_title: str = "PREVIOUS CANDIDATE (this loop)",
) -> str:
    """Render the previous in-loop candidate's full body.

    Used by both `model_improvement` and `theory_generation` prompts so the
    LLM can see — verbatim — the source it just tried, rather than
    reconstructing it from the critic's prose. Returns `""` when no
    candidate is given (i.e. on the first iteration of the inner loop).

    `include_description` controls whether the prose description is also
    rendered. The improver already shows the (unchanged) theory description
    above as `## THEORY (UNCHANGED …)` so we omit it there to avoid a
    duplicate; the theory_generator changes the description each iteration
    so it must be included.
    """
    if candidate is None:
        return ""
    desc_block = (
        f"**Description**\n{candidate.description}\n\n"
        if include_description else ""
    )
    rationale = candidate.rationale or "(none)"
    return (
        f"\n## {section_title}\n"
        "The RUNNING-BEST (last ACCEPTED) candidate in this critique loop "
        "— i.e. the source the loop's accept gate kept as the best base so "
        "far. If your most recent attempt was REJECTED by the gate, this "
        "is NOT that attempt; it is the previously-accepted base the gate "
        "rolled back to. Iterate on this source — the next critic feedback "
        "should be applied on top of it.\n\n"
        f"{desc_block}"
        f"`predict(parameters, state, history) -> np.ndarray`:\n"
        f"{candidate.predict_source}\n\n"
        f"`policy(probs) -> int`:\n{candidate.policy_source}\n\n"
        f"`parameters`:\n{_format_parameters(candidate.parameters)}\n\n"
        f"`rationale`: {rationale}\n"
    )


def _format_proposal_directive(previous_candidate: "Theory | None") -> str:
    """Render the `## PROPOSAL` block for the model-improvement prompt.

    Two regimes (mirrors the helper in `theory_generation`, but scoped to
    "model" since the description is fixed in this prompt):
      - First iteration of the loop (`previous_candidate is None`): tell
        the LLM to propose a fresh model instance for the unchanged
        theory description above. There is nothing to edit yet, so a
        "minimal-diff" framing would be a no-op.
      - Subsequent iterations (`previous_candidate is not None`): tell
        the LLM to apply a MINIMAL-DIFF EDIT on the source rendered
        verbatim under `## PREVIOUS CANDIDATE (this loop)`. Rewriting
        `predict` / `policy` end-to-end for a small change wastes tokens
        and tends to introduce regressions; pinning the diff to the
        smallest change that resolves the critic's diagnosis keeps the
        loop converging instead of oscillating across full rewrites.
    """
    if previous_candidate is None:
        return (
            "## PROPOSAL\n"
            "Propose a new model instance for the theory above from scratch. "
            "Keep the theory's prose claim implicit but unchanged — your only "
            "job is to regenerate `predict`, `policy`, and the `parameters` "
            "ranges so that the implementation actually displays the behavior "
            "the theory claims, while faithfully implementing the mechanism "
            "family the arbiter prescribed.\n"
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
        "  - Re-emit the previous source verbatim, then change ONLY the "
        "lines needed to address the critic (a parameter range, a "
        "normalization, a softmax temperature, an attention scheme, a "
        "gating term, a buggy indexing line, etc.).\n"
        "  - Keep the theory's prose claim implicit but unchanged, and keep "
        "all unaffected functions, equations, parameter names, and the "
        "overall mechanism intact.\n"
        "  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines "
        "would do, and do NOT switch mechanism families — that is the "
        "arbiter's decision, not yours in this loop.\n"
        "  - Briefly explain the minimal edit in `rationale`.\n"
        "If you genuinely believe a larger rewrite is required, you may do "
        "one, but justify in `rationale` why the minimal-diff path was "
        "insufficient.\n"
    )


def _fmt_loss(loss: float | None) -> str:
    """Render an `aggregate_loss` value compactly. `None` is shown as `n/a`,
    `+inf` as `+inf` (unscorable), everything else with 4 decimals.
    Mirrors the helper in `src/prompts/feedback.py` so both sides of the
    propose-loop format losses the same way."""
    if loss is None:
        return "n/a"
    if loss == float("inf"):
        return "+inf"
    if loss == float("-inf"):
        return "-inf"
    return f"{loss:.4f}"


def _accept_marker(accepted: bool | None) -> str:
    """Render the accept-gate verdict for an iteration. Mirrors the helper
    in `src/prompts/feedback.py` (kept local here to avoid a cycle between
    the two prompt modules)."""
    if accepted is None:
        return "?"
    return "ACCEPTED" if accepted else "REJECTED"


def _format_loss_trajectory_block(
    prior_iterations: list["PriorIteration"] | None,
) -> str:
    """Render the propose-loop's `aggregate_loss` trajectory for the proposer.

    Each row carries `loss` plus the loop's programmatic ACCEPTED /
    REJECTED tag (`loss < accepted_loss` -> ACCEPTED, else REJECTED).
    Unlike the critic's view there is no "current candidate" row — the
    proposer hasn't simulated the next candidate yet. The block ends with
    a one-line summary pointing the proposer at the running-best (last
    ACCEPTED) candidate, which is exactly the source rendered verbatim
    below under `## PREVIOUS CANDIDATE (this loop)`.

    Returns "" on iter 0 (nothing to render).
    """
    if not prior_iterations:
        return ""
    rows: list[str] = []
    accepted_iters: list[tuple[int, float]] = []
    for k, it in enumerate(prior_iterations, start=1):
        rows.append(
            f"- iter {k}: loss={_fmt_loss(it.loss)} "
            f"-> {_accept_marker(it.accepted)}"
        )
        if it.accepted and it.loss != float("inf"):
            accepted_iters.append((k, it.loss))
    if accepted_iters:
        best_k, best_l = min(accepted_iters, key=lambda kl: kl[1])
        best_line = (
            f"\nRunning-best (last ACCEPTED) base: iter {best_k} at loss="
            f"{_fmt_loss(best_l)} — this is the source shown verbatim below "
            f"under `## PREVIOUS CANDIDATE (this loop)`. Push the next "
            f"edit's loss strictly below that floor or the gate will "
            f"reject it."
        )
    else:
        best_line = (
            "\nNo iteration has been ACCEPTED yet by the gate. The source "
            "shown below as `## PREVIOUS CANDIDATE (this loop)` is the "
            "best-so-far attempt; any finite-loss improvement on it will be "
            "auto-accepted."
        )
    return (
        "\n## LOSS TRAJECTORY (this propose-loop)\n"
        "Aggregate loss across iterations of THIS propose-loop "
        "(lower = better, 0 = perfect, `+inf` = unscorable). The "
        "ACCEPTED / REJECTED tag is the loop's programmatic accept-gate "
        "decision — only ACCEPTED candidates have ever been used as the "
        "base for a subsequent iteration. Use this together with PRIOR "
        "FEEDBACK ITERATIONS below to grade which past critic advice "
        "actually paid off.\n\n"
        + "\n".join(rows)
        + best_line
        + "\n"
    )


def _format_prior_feedback(
    prior_iterations: list["PriorIteration"] | None,
) -> str:
    """Render the chronological log of prior critic verdicts, each annotated
    with the accept-gate outcome of the candidate it elicited.

    The proposer needs this in two ways:
      1. The literal prose feedback to address (mostly the most recent one).
      2. A per-iteration ACCEPTED / REJECTED marker so it knows which past
         pieces of advice ended up improving the running best and which
         did not — advice whose candidates were REJECTED should be
         discounted, even when the current critic still pushes in the
         same direction.

    For iter k's "outcome" we look at `prior_iterations[k].accepted` (=
    the next candidate's accept status). For the most recent prior
    critique (the one whose advice the proposer is about to address) the
    next candidate hasn't been built yet — we mark that as PENDING.
    """
    if not prior_iterations:
        return ""
    blocks: list[str] = []
    for i, it in enumerate(prior_iterations, start=1):
        is_last = i == len(prior_iterations)
        if is_last:
            outcome_line = (
                f"**Outcome of this advice:** iter {i} candidate loss="
                f"{_fmt_loss(it.loss)} -> next candidate accept-status=PENDING "
                f"(this is the advice you are addressing now)."
            )
            header = f"### Iteration {i} (most recent — address this)"
        else:
            next_it = prior_iterations[i]
            marker = _accept_marker(next_it.accepted)
            outcome_line = (
                f"**Outcome of this advice:** iter {i} candidate loss="
                f"{_fmt_loss(it.loss)} -> iter {i + 1} candidate loss="
                f"{_fmt_loss(next_it.loss)} -> the gate marked it {marker}."
            )
            header = f"### Iteration {i}"
        blocks.append(f"{header}\n{it.rationale}\n\n{outcome_line}")
    rendered = "\n\n".join(blocks)
    return (
        "\n## PRIOR FEEDBACK ITERATIONS\n"
        "The critic's verdicts on each previous in-loop candidate, in "
        "order. Each block ends with an **Outcome of this advice** line "
        "saying whether the candidate the proposer produced AFTER this "
        "advice was ACCEPTED (loss strictly beat the running best) or "
        "REJECTED (the gate discarded it). Address the most recent "
        "iteration's feedback in your next edit, but down-weight past "
        "advice whose candidates were REJECTED.\n\n"
        f"{rendered}\n"
    )


def render(
    *,
    experiment_class: type["Experiment"],
    theory: "Theory",
    response_schema: type["Promptable"],
    arbiter_guide: str,
    arbiter_theory_labels: tuple[str | None, str | None] | None = None,
    arbiter_target_idx: int | None = None,
    other_theory: "Theory | None" = None,
    observations: list["Observation"] | None = None,
    previous_candidate: "Theory | None" = None,
    leaderboard: list[tuple[str, "Theory", float]] | None = None,
    prior_iterations: list["PriorIteration"] | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for proposing a new Model under
    an existing Theory.

    `arbiter_theory_labels` is the `(theory_1_pi_label, theory_2_pi_label)`
    pair the arbiter saw under its `THEORY 1` / `THEORY 2` headings; when
    provided, a small lookup key is rendered above `arbiter_guide` so the
    LLM can resolve any numeric "Theory 1 / Theory 2" references in the
    arbiter's free-text recommendation back to concrete pi labels.
    `arbiter_target_idx` (1 or 2) flags which of those theories the
    recommendation acts on.

    `other_theory` is the round's NON-target theory. When provided
    (alongside `arbiter_theory_labels` + `arbiter_target_idx`), the
    `## ROUND THEORIES` block renders both theories side by side under
    `THEORY 1` / `THEORY 2` headings — the target tagged **TO REVISE** —
    so any numeric references in the arbiter's recommendation point at a
    concrete subsection here. When omitted, the block degrades to a single
    theory view (target only).

    `previous_candidate`, when provided, is the most recent in-loop attempt
    rendered verbatim under `## PREVIOUS CANDIDATE (this loop)` so the LLM
    can iterate on its own source rather than reconstructing it from the
    critic's prose feedback. Pass `None` on the first iteration.

    `theory` carries both the prose description (preserved verbatim in the
    prompt) and the previous model instance (predict/policy/parameters)
    shown as the "previous instantiation" the LLM should improve on.

    `observations` is the per-experiment evidence the new model must
    explain — typically every Observation across every Round in the pool
    (regardless of which pi proposed each). The block is auto-formatted via
    `_format_observations`: experimental design → metric → real value →
    (when `prior_iterations` is given) this loop's prior candidate
    trajectory → other theories' values on the same metric, per
    observation.

    `response_schema` is the Pydantic class the LLM's output is parsed into
    (typically `Model` from `src/theory.py`); its `instruction_format()`
    produces the field-list block at the bottom of the user prompt.

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
        round_theories_block=_format_round_theories(
            theory,
            other_theory=other_theory,
            labels=arbiter_theory_labels,
            target_idx=arbiter_target_idx,
        ),
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
            previous_candidate, include_description=False
        ),
        prior_feedback_block=_format_prior_feedback(prior_iterations),
        proposal_directive=_format_proposal_directive(previous_candidate),
        instruction_format=response_schema.instruction_format(),
    )
    return system_prompt, user_prompt
