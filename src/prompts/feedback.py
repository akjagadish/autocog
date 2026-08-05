"""Prompt: critique a freshly proposed candidate theory/model.

Used by the `Feedback` agent inside `Improver` / `TheoryGenerator`'s inner
loop. The candidate is simulated on every existing experiment (each
`Observation` in the pool); for each one we have a real_value (collected on
human/ground-truth data) and a candidate_value (the candidate's metric on
its own simulated data). The feedback agent reads those side-by-side and
returns a `FeedbackVerdict` (`"continue"` or `"regenerate"`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
   
from src.metric import Estimate, fmt_estimate
from src.prompts.interpret_results import ESTIMATE_NOTE, _fmt

if TYPE_CHECKING:
    from src.arbiter_verdict import FeedbackVerdict
    from src.experiment import Experiment
    from src.feedback import PriorIteration
    from src.observation import Observation
    from src.theory import Theory


SYSTEM_PROMPT = """\
You are a renowned cognitive scientist critiquing a freshly proposed candidate \
theory and model in the {domain} domain.

The candidate has been simulated on every previously run experiment. For each \
experiment you are shown the design, the metric, the value the metric takes \
on real data, and the value it takes on the \
candidate's simulated data.

{estimate_note}

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,\
theories that explain data across multiple experiments. \

Your task is to determine whether the candidate captures the human/real behavior well enough \
across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; \
the proposing agent must produce a new candidate, taking your rationale \
into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in \
what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was \
explicitly instructed to implement the mechanism family the arbiter prescribed. \
Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate \
which family should be used — that is the arbiter's call, made one level above \
this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that \
keep the prescribed mechanism intact: tightening / widening parameter ranges, \
adding a temperature, swapping a normalization scheme, fixing a softmax / \
distance metric, re-balancing attention weights, fixing a learning-rate sign, \
correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a \
switch is the arbiter's prerogative; recommending it here will mislead the \
proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate \
has clearly drifted into a different family than the one prescribed, say so in \
the rationale and ask for a return to the prescribed family — again, with \
minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the \
candidate's `aggregate_loss` is compared against the running-best loss \
(`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new \
running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next \
iteration's proposer will build on the SAME `accepted` candidate again, with \
your new feedback on top. Rejected candidates are discarded — the loop guarantees \
the base never regresses, so you do NOT need to ask the proposer to "revert" \
anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning \
`"continue"` is silently downgraded to `"regenerate"` (returning a worse \
candidate would defeat the gate). Spend your rationale on a NEW direction the \
proposer should try on top of the unchanged accepted base, not on defending the \
rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop \
and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration \
ends with an "Outcome of your advice" line that says whether the next candidate \
the proposer produced was ACCEPTED (your advice helped — its loss strictly \
beat the running best) or REJECTED (your advice didn't help — the proposer \
discarded the result and reset to the previous accepted base). This is the \
loop's ground-truth signal on whether *your own previous critique was good*. \
Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. \
Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same \
recommendation; in your new rationale, briefly acknowledge that the previous \
push in that direction was rejected by the gate and try a different in-family \
knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 \
said "decrease α", iter 3 about to say "increase α" again), STOP and recommend \
a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the \
same information at the loop level — consult it before issuing a new \
regenerate-with-direction recommendation.
"""
# If a "Candidate trajectory (this loop)" block is shown for an experiment, \
# inspect it: a candidate that overshot in the previous iteration and now \
# undershoots (or vice-versa) is oscillating. In that case do NOT recommend \
# "swap direction" again; instead recommend bisection — a smaller step, a \
# parameter value between the two flanking iterations, or a mechanism that \
# trades off the two extremes — so the next candidate lands between the \
# prior two on this metric.


TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}
{arbiter_block}
## CANDIDATE THEORY
{theory_description}

`predict(parameters, state, history) -> np.ndarray`:
{theory_predict}

`policy(probs) -> int`:
{theory_policy}

`parameters`:
{theory_parameters}

`rationale`:
{theory_rationale}
{loss_trajectory_block}
## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
{experimental_results}
{prior_critiques_block}
## RESPONSE FORMAT

Return a JSON object with the following fields:
{instruction_format}
"""


def _format_parameters(parameters: dict[str, str]) -> str:
    if not parameters:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in parameters.items())


def _format_other_predictions(obs: "Observation") -> str:
    """Render previously-recorded theory predictions on this Observation's
    metric, one per line, as a calibration reference for the critic.
    Each entry shows both the point estimate and the between-subject
    variance (the same `value (var=...)` shape as everywhere else)."""
    if not obs.predicted_values:
        return "(none)"
    lines = [
        f"- {p.label}: {fmt_estimate(p.as_estimate())}"
        for p in obs.predicted_values
    ]
    return "\n".join(lines)


def _fmt_loss(loss: float | None) -> str:
    """Render an aggregate loss value compactly. `None` is shown as `n/a`,
    `+inf` as `+inf` (unscorable), everything else with 4 decimals."""
    if loss is None:
        return "n/a"
    if loss == float("inf"):
        return "+inf"
    if loss == float("-inf"):
        return "-inf"
    return f"{loss:.4f}"


def _accept_marker(accepted: bool | None) -> str:
    """Render the accept-gate verdict for an iteration. `True` → ACCEPTED
    (became the new running-best base), `False` → REJECTED (discarded;
    base unchanged), `None` → unknown (used for the iter-0 row when no
    accept decision has been recorded yet)."""
    if accepted is None:
        return "?"
    return "ACCEPTED" if accepted else "REJECTED"


def _format_trajectory(
    obs_idx: int,
    real_value: float | None,
    current: "Estimate",
    prior_iterations: list["PriorIteration"] | None,
) -> str:
    """Render a per-experiment "candidate over iterations" trajectory.

    `obs_idx` is the position of this observation inside the loop's
    `candidate_results` list. We pull the `Estimate` at `obs_idx` from each
    prior iteration's estimate list (same indexing as the current `results`).

    The trajectory makes oscillation visible. The Δ-vs-real number compares
    point estimates only (variance is shown alongside but not subtracted).
    Example:
        iter 1: 0.6704 (var=0.0210)  (Δ vs real -0.1550)
        iter 2: 0.9858 (var=0.0040)  (Δ vs real +0.1610)
        iter 3 (current): 0.6371 (var=0.0160)  (Δ vs real -0.1880)
    so the critic can recommend bisection / smaller steps instead of
    flipping direction every iteration.
    """
    rows: list[str] = []
    if prior_iterations:
        for k, it in enumerate(prior_iterations, start=1):
            est = it.estimates[obs_idx] if obs_idx < len(it.estimates) else None
            delta = (
                f" (Δ vs real {est.value - real_value:+.4f})"
                if est is not None
                and est.value is not None
                and real_value is not None
                else ""
            )
            rows.append(f"  - iter {k}: {fmt_estimate(est)}{delta}")
    delta_now = (
        f" (Δ vs real {current.value - real_value:+.4f})"
        if current.value is not None and real_value is not None
        else ""
    )
    rows.append(
        f"  - iter {len(prior_iterations or []) + 1} (current): "
        f"{fmt_estimate(current)}{delta_now}"
    )
    return "**Candidate trajectory (this loop):**\n" + "\n".join(rows)


def format_candidate_results(
    results: list[tuple["Observation", "Estimate"]],
    *,
    prior_iterations: list["PriorIteration"] | None = None,
) -> str:
    """Render `(observation, candidate_estimate)` pairs as the results block.

    Each entry shows the experimental design, the metric, the observed
    real value (with variance), the candidate's simulated value (with
    variance), and (when `prior_iterations` is non-empty) the per-iteration
    trajectory of candidate estimates for the same observation in this
    propose-loop, plus the estimates produced by every other theory
    previously evaluated against this metric.
    """
    if not results:
        return "(no observations to score against)"
    blocks: list[str] = []
    for i, (obs, est) in enumerate(results, start=1):
        if prior_iterations:
            value_block = _format_trajectory(
                i - 1, obs.real_value, est, prior_iterations
            )
        else:
            value_block = (
                f"**Candidate (simulated) value:** {fmt_estimate(est)}"
            )
        block = (
            f"### Experiment {i}\n"
            f"**Design**\n{obs.experiment.pretty_print_design()}\n\n"
            f"**Metric**\n"
            f"```python\n{obs.metric.metric_source}\n```\n\n"
            f"**Observed (real) value:** {fmt_estimate(obs.real_as_estimate())}\n"
            f"{value_block}\n"
            f"**Other theories' values on this metric (for reference):**\n"
            f"{_format_other_predictions(obs)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_loss_trajectory_block(
    prior_iterations: list["PriorIteration"] | None,
    current_loss: float | None,
    current_accepted: bool | None,
    accepted_loss: float | None,
) -> str:
    """Render the propose-loop's `aggregate_loss` trajectory with the
    accept-gate decision per iteration.

    Each row carries the iteration's `aggregate_loss` plus an
    ACCEPTED/REJECTED tag from the loop's accept gate (programmatic,
    `loss < accepted_loss`). The current candidate's accept status is
    derived from `current_accepted` directly; prior iterations carry it
    on their `PriorIteration.accepted` field.

    Returns "" when there is nothing to render (no prior iterations AND
    no current loss) so iter-0 calls without state stay unchanged.
    """
    if not prior_iterations and current_loss is None:
        return ""
    rows: list[str] = []
    accepted_iters: list[tuple[int, float]] = []
    last_index = len(prior_iterations or []) + 1  # current row's 1-based index
    for k, it in enumerate(prior_iterations or [], start=1):
        rows.append(
            f"- iter {k}: loss={_fmt_loss(it.loss)} "
            f"-> {_accept_marker(it.accepted)}"
        )
        if it.accepted and it.loss != float("inf"):
            accepted_iters.append((k, it.loss))
    # Current candidate row.
    cur_marker = _accept_marker(current_accepted)
    rows.append(
        f"- iter {last_index} (current candidate you are grading): "
        f"loss={_fmt_loss(current_loss)} -> {cur_marker}"
    )
    if current_accepted and current_loss is not None and current_loss != float("inf"):
        accepted_iters.append((last_index, current_loss))
    # Running-best summary line.
    if accepted_iters:
        best_k, best_l = min(accepted_iters, key=lambda kl: kl[1])
        running_best_line = (
            f"\nRunning-best (last accepted) base: iter {best_k} at loss="
            f"{_fmt_loss(best_l)}."
        )
    elif accepted_loss is not None and accepted_loss != float("inf"):
        running_best_line = (
            f"\nRunning-best (last accepted) base: loss={_fmt_loss(accepted_loss)}."
        )
    else:
        running_best_line = (
            "\nNo accepted base yet (this is iter 1; any finite loss will be "
            "auto-accepted)."
        )
    return (
        "\n## LOSS TRAJECTORY (this propose-loop)\n"
        "Aggregate loss across iterations of THIS propose-loop "
        "(lower = better, 0 = perfect, `+inf` = unscorable). The "
        "ACCEPTED / REJECTED tag is the loop's programmatic accept-gate "
        "decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), "
        "else REJECTED (base unchanged). Use this together with the "
        "per-experiment values below to grade fit-quality AND your own "
        "past advice (see `## YOUR PRIOR CRITIQUES` below).\n\n"
        + "\n".join(rows)
        + running_best_line
        + "\n"
    )


def _format_prior_critiques(
    prior_iterations: list["PriorIteration"] | None,
    current_loss: float | None,
    current_accepted: bool | None,
) -> str:
    """Render the critic's own past verdicts + rationales, each annotated
    with whether the candidate the proposer produced AFTER following that
    advice was ACCEPTED (advice helped — strict loss improvement) or
    REJECTED (advice didn't help — candidate discarded by the gate) by
    the propose-loop's accept gate.

    The "Outcome of your advice" line is the headline learning signal:
    advice in iter k caused the candidate in iter k+1, whose acceptance
    status is on `prior_iterations[k]` (or, for the most recent prior
    critique, on `current_accepted`).

    Per-experiment values for those iterations are shown inline in
    `format_candidate_results` so the reader can read the full picture
    next to the experiment, not as a separate disconnected section.
    """
    if not prior_iterations:
        return ""
    blocks: list[str] = []
    for i, it in enumerate(prior_iterations, start=1):
        is_last = i == len(prior_iterations)
        if is_last:
            next_loss = current_loss
            next_accepted = current_accepted
            next_label = "CURRENT candidate"
        else:
            next_loss = prior_iterations[i].loss
            next_accepted = prior_iterations[i].accepted
            next_label = f"iter {i + 1} candidate"
        marker = _accept_marker(next_accepted)
        outcome_line = (
            f"**Outcome of your advice:** iter {i} candidate loss="
            f"{_fmt_loss(it.loss)} -> {next_label} loss="
            f"{_fmt_loss(next_loss)} -> the gate marked it {marker}."
        )
        header = (
            f"### Iteration {i} (most recent)" if is_last else f"### Iteration {i}"
        )
        blocks.append(f"{header}\n{it.rationale}\n\n{outcome_line}")
    rendered = "\n\n".join(blocks)
    return (
        "\n## YOUR PRIOR CRITIQUES (for this propose-loop)\n"
        "Each block is one of YOUR previous critique iterations: the verdict "
        "you returned, the interpretation and rationale you wrote, and an "
        "**Outcome of your advice** line that says whether the candidate the "
        "proposer produced AFTER following your advice was ACCEPTED (your "
        "advice helped — its loss strictly beat the running-best) or REJECTED "
        "(your advice didn't help — the gate discarded the candidate). Use "
        "this self-history to (a) reinforce advice that was ACCEPTED, "
        "(b) avoid repeating advice that was REJECTED, and (c) detect your "
        f"own oscillation across iterations.\n\n{rendered}\n"
    )


def _format_arbiter_block(
    arbiter_guide: str | None,
    arbiter_theory_labels: tuple[str | None, str | None] | None,
    arbiter_target_idx: int | None,
) -> str:
    """Render the arbiter's recommendation as a top-level critic-facing block.

    The arbiter's free-text recommendation refers back to its own
    `THEORY 1` / `THEORY 2` headings (anchored to `pi_N` labels in the
    arbitration prompt). Once that recommendation is forwarded into the
    feedback critic, that anchoring is gone, so we re-emit a small key
    above the guide text. Returns "" when no guide is supplied — legacy
    callers stay unchanged.

    The block is the critic's anchor for "stay inside this mechanism
    family" — it pairs with the SCOPE clause in the system prompt.
    """
    if not arbiter_guide:
        return ""
    lines: list[str] = ["", "## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)"]
    if (
        arbiter_theory_labels
        and arbiter_theory_labels[0] is not None
        and arbiter_theory_labels[1] is not None
    ):
        lines.append(
            "The arbiter labelled this round's two theories in its recommendation as follows:"
        )
        lines.append(f"- THEORY 1 = `{arbiter_theory_labels[0]}`")
        lines.append(f"- THEORY 2 = `{arbiter_theory_labels[1]}`")
        if arbiter_target_idx in (1, 2):
            target_label = arbiter_theory_labels[arbiter_target_idx - 1]
            lines.append(
                f"- The recommendation below acts on THEORY {arbiter_target_idx} "
                f"(= `{target_label}`)."
            )
        lines.append("")
    lines.append(arbiter_guide.rstrip())
    lines.append("")
    return "\n".join(lines) + "\n"


def render(
    *,
    experiment_class: type["Experiment"],
    theory: "Theory",
    candidate_results: list[tuple["Observation", "Estimate"]],
    feedback_verdict: type["FeedbackVerdict"],
    prior_iterations: list["PriorIteration"] | None = None,
    current_loss: float | None = None,
    current_accepted: bool | None = None,
    accepted_loss: float | None = None,
    arbiter_guide: str | None = None,
    arbiter_theory_labels: tuple[str | None, str | None] | None = None,
    arbiter_target_idx: int | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for critiquing one candidate.

    `candidate_results` is `[(Observation, Estimate(value, variance)), ...]`;
    one entry per existing experiment the candidate was simulated on.

    `prior_iterations` is the chronological history of this propose-loop as
    `PriorIteration(rationale, estimates, loss, accepted)` entries. The
    estimate list for each iteration is in the SAME order as
    `candidate_results`; `loss` is that iteration's `aggregate_loss`;
    `accepted` is the loop's programmatic accept-gate decision. The
    prompt uses this to render (a) a per-experiment trajectory of
    candidate estimates so the critic can detect oscillation and
    recommend bisection, and (b) a `## YOUR PRIOR CRITIQUES` block where
    each prior iteration's rationale is annotated with whether the next
    candidate was ACCEPTED (advice helped) or REJECTED (advice didn't
    help) — turning the loop into a closed-loop signal the critic can
    learn from.

    `current_loss` is `aggregate_loss` of the candidate being critiqued
    in THIS call. `current_accepted` is whether the loop's accept gate
    will keep that candidate as the new running-best base (True) or
    discard it (False). `accepted_loss` is the running-best loss the
    candidate had to beat. Together they drive the `## LOSS TRAJECTORY`
    block and the "Outcome of your advice" line on the most-recent
    prior critique.

    `arbiter_guide` (+ optional `arbiter_theory_labels` /
    `arbiter_target_idx`) is the recommendation that drove this propose
    loop in the first place. When supplied, it's rendered verbatim under
    a top-level ARBITER RECOMMENDATION block so the critic knows which
    mechanism family the proposer was told to implement and can grade
    fit-quality WITHIN that family rather than recommending a switch
    (mechanism-family switches are the arbiter's prerogative; this loop
    only tunes within the prescribed family).
    """
    system_prompt = SYSTEM_PROMPT.format(
        domain=experiment_class.name, estimate_note=ESTIMATE_NOTE
    )
    user_prompt = TEMPLATE.format(
        experiment_description=experiment_class.description,
        arbiter_block=_format_arbiter_block(
            arbiter_guide, arbiter_theory_labels, arbiter_target_idx
        ),
        theory_description=theory.description,
        theory_predict=theory.predict_source,
        theory_policy=theory.policy_source,
        theory_parameters=_format_parameters(theory.parameters),
        theory_rationale=theory.rationale or "(none)",
        loss_trajectory_block=_format_loss_trajectory_block(
            prior_iterations, current_loss, current_accepted, accepted_loss
        ),
        experimental_results=format_candidate_results(
            candidate_results, prior_iterations=prior_iterations
        ),
        prior_critiques_block=_format_prior_critiques(
            prior_iterations, current_loss, current_accepted
        ),
        instruction_format=feedback_verdict.instruction_format(),
    )
    return system_prompt, user_prompt
