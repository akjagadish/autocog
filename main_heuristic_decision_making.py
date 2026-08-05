"""
Layout under `run_dir/`:

  observations/        # the shared evidence pool (state.json + data/)
  rounds/round_NNN/    # all artifacts for one round, regardless of which
    pi_1/prompts/...   # pi was active. When a pi gets replaced (gecco), the
    pi_1_1/prompts/... # next round just gets a fresh subdir — no stale
    arbiter/prompts/   # workspace tied to a now-gone theory.

The orchestrator owns the single `Observations` pool that both pis contribute
to (organized by Round). Pis / Arbiter / Improver / TheoryGenerator are all
stateless w.r.t. persistence: the run dir + round index drive every log path.

Labelling scheme
----------------
Each pi carries a stable `label` that follows the regeneration chain:
  * Initial pis:                 pi_1, pi_2
  * `Improver` on pi_N:           pi_N_1   (next improvement -> pi_N_2, ...)
  * `Improver` on pi_N_K:         pi_N_<K+1>
  * `TheoryGenerator` (fresh):    pi_<max_base + 1>

Labels propagate to:
  - Predictions (tagged with the pi label that produced them)
  - Observation.proposer_label
  - Round.next_theory_label / next_theory_idx
  - workspace directory names (rounds/round_NNN/<label>/...)
"""

import json
import sys
import random
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from src.arbiter import Arbiter
from src.heuristic_decision_making.experiment import (
    HeuristicDecisionMakingExperiment,
)
from src.improver import Improver, make_theory


def _make_rating_max_locked_class(
    rating_max: int,
) -> type[HeuristicDecisionMakingExperiment]:
    """Return an HDM subclass with `rating_max` pinned to a single integer.

    Pydantic's `Field` re-declaration on the subclass narrows the allowed
    range to `[rating_max, rating_max]` AND makes the default the same
    value — so the LLM response schema still advertises the field but any
    out-of-range proposal is rejected by validation (triggering autocog's
    existing propose-and-retry loop), and a correctly-pinned proposal
    validates cleanly. The class name encodes the bound so run logs stay
    self-describing.
    """

    # Alias to sidestep the name clash between the class-body annotation
    # `rating_max: int = ...` (which shadows the enclosing scope's lookup
    # on its own RHS) and the outer function parameter.
    _rmax: int = rating_max

    class _Locked(HeuristicDecisionMakingExperiment):
        rating_max: int = Field(
            default=_rmax,
            ge=_rmax,
            le=_rmax,
            description=(
                f"[LOCKED to {_rmax} for this run by --force_rating_max] "
                f"Upper bound (inclusive) of each rating value. Ratings "
                f"must be integers in [0, {_rmax}]."
            ),
        )

    _Locked.__name__ = f"HDMExperiment_RatingMaxEq{rating_max}"
    _Locked.__qualname__ = _Locked.__name__
    return _Locked
from src.logger import info
from src.observation import Observations
from src.online_config import OnlineConfig  # noqa: F401  (used in commented snippet)
from src.autocog import AutoCog
from src.run_config import REAL_N_SUBJECTS
from src.theory import Theory
from src.theory_generator import TheoryGenerator

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--n_rounds", type=int, default=1)
parser.add_argument(
    "--ground_truth",
    type=str,
    default="ttb",
    choices=("ttb", "ew", "tallying", "wadd"),
)
parser.add_argument("--llm_provider", type=str, default="gemini")
parser.add_argument("--llm_model", type=str, default="gemini-3.1-pro-preview")
parser.add_argument(
    "--force_rating_max",
    type=int,
    default=None,
    help=(
        "Pin rating_max to this integer for every LLM-proposed experiment "
        "in this run (e.g. --force_rating_max=1 for binary-only). When "
        "unset, the LLM picks per experiment. Useful for studying what "
        "the proposer can and cannot dissociate under a fixed feature "
        "type (e.g. binary blocks Tallying from EW entirely)."
    ),
)
parser.add_argument('--run_id', type=str, help="Unique tag for this run (e.g. timestamp or short hash)")
parser.add_argument(
    "--gt_epsilon",
    type=float,
    default=0.0,
    help=(
        "Epsilon-greedy action noise applied ONLY to the ground-truth "
        "simulation. With probability "
        "(1 - gt_epsilon) the ground-truth theory's policy output is used; "
        "with probability gt_epsilon the action is drawn uniformly over the "
        "available options. Must be in [0, 1]."
    ),
)
parser.add_argument(
    "--gt_seed",
    type=int,
    default=0,
    help=(
        "Seed for the ground-truth action-noise RNG; only has an effect when "
        "--gt_epsilon > 0. Fixed so noise-level sweeps are reproducible."
    ),
)
parser.add_argument(
    "--out_path",
    type=str,
    default="results",
    help=(
        "Results root the run dir is written under. The "
        "<ground_truth>/noise=<eps>/<run_dir> substructure is preserved "
        "beneath it, so one value serves a whole gt/noise/run sweep "
        "(e.g. --out_path results/recovery). Defaults to 'results', reproducing "
        "the historical path."
    ),
)
args = parser.parse_args()

if not (0.0 <= args.gt_epsilon <= 1.0):
    parser.error(f"--gt_epsilon must be in [0, 1]; got {args.gt_epsilon!r}")

LLM_PROVIDER = args.llm_provider
LLM_MODEL = args.llm_model

# --- run-level config -------------------------------------------------------


# How many adversarial rounds to run per invocation of `main()`. Each round
# does (propose → real data → backfill → arbitrate → regenerate → backfill)
# and persists in-place, so this is "additional rounds on top of whatever is
# already in `run_dir/observations/`". Set to 1 for the historical
# single-round behaviour.
N_ROUNDS: int = args.n_rounds

# Seed YAMLs used only on the very first round of a run (when slots 1 and 2
# have no prior `next_theory` to inherit from). Seeds are the two heuristics
# that are NOT the ground truth — so the adversarial pair has to discover the
# ground-truth heuristic from scratch.
THEORIES_DIR = "theories/heuristic_decision_making"
_ALL_THEORIES = ("ttb", "tallying", "wadd") #"ew", 
_competitors = tuple(t for t in _ALL_THEORIES if t != args.ground_truth)
INITIAL_SEEDS: tuple[str, str] = (
    f"{THEORIES_DIR}/{_competitors[0]}.yaml",
    f"{THEORIES_DIR}/{_competitors[1]}.yaml",
)
GROUND_TRUTH_YAML = f"{THEORIES_DIR}/{args.ground_truth}.yaml"

# Pick the experiment class once: either the base (LLM picks rating_max) or
# a dynamically-generated subclass with rating_max pinned. Every downstream
# `experiment_class=` goes through HDM_EXPERIMENT_CLASS so the choice is
# honoured end-to-end (pool loading, AutoCog seeding, Arbiter/Improver/Generator).
HDM_EXPERIMENT_CLASS: type[HeuristicDecisionMakingExperiment] = (
    _make_rating_max_locked_class(args.force_rating_max)
    if args.force_rating_max is not None
    else HeuristicDecisionMakingExperiment
)

# Where the run lives on disk.
_rmax_tag = (
    f"_rmax{args.force_rating_max}" if args.force_rating_max is not None else ""
)
# Keep noiseless runs (gt_epsilon == 0) at the original path so prior
# results stay discoverable; only tag when noise is actually injected
_gteps_tag = f"noise={args.gt_epsilon}" if args.gt_epsilon >= 0.0 else ""
RUN_DIR = Path(
    f'{args.out_path}/{args.ground_truth}/{_gteps_tag}/hdm_ground_truth_{args.ground_truth}{_rmax_tag}_{_gteps_tag}_{LLM_MODEL}_run{args.run_id}'
)

# Ground-truth theory used as the stand-in for human subjects when collecting
# `real` data on each new observation.


LEADERBOARD: tuple[Literal["none", "best", "sample"], int] = ("best", 3)
"""How to populate the `## THEORY LEADERBOARD` section of the Improver /
TheoryGenerator prompts. Each entry is the full body of a prior theory
(description, predict, policy, parameters, per-experiment fits, overall
score) so the LLM can compare its candidate against concrete competitors.
Format is `(mode, n)`:
    ("none",   _): no leaderboard shown.
    ("best",   n): top-n picked theories by `pool.theory_scores()`.
    ("sample", n): n picked theories sampled WITHOUT replacement, weighted
                   by score (uniform fallback if every score is 0).
For the Improver path the arbiter-killed theory is always shown on top of
the prompt as `## PREVIOUS MODEL INSTANCE` and is excluded from the
leaderboard to avoid duplicating its body. The improver always seeds from
the killed theory now (no separate "best/sample seed" knob).
"""


# --- label helpers ----------------------------------------------------------


def _parse_label(label: str) -> tuple[int, int | None]:
    """`pi_3` -> (3, None), `pi_1_2` -> (1, 2)."""
    parts = label.split("_")
    if parts[0] != "pi" or len(parts) not in (2, 3):
        raise ValueError(f"unrecognised pi label: {label!r}")
    if len(parts) == 2:
        return int(parts[1]), None
    return int(parts[1]), int(parts[2])


def _all_labels(pool: Observations, initial: tuple[str, str]) -> set[str]:
    """Every pi label that has ever existed in this run."""
    seen: set[str] = set(initial)
    for r in pool.rounds:
        for o in r.observations:
            if o.proposer_label is not None:
                seen.add(o.proposer_label)
        if r.next_theory_label is not None:
            seen.add(r.next_theory_label)
    return seen


def replay_slot_labels(
    pool: Observations,
    *,
    initial: tuple[str, str] = ("pi_1", "pi_2"),
) -> tuple[str, str]:
    """Walk each round's `next_theory_label` / `next_theory_idx` to compute
    the label currently sitting at slot 1 and slot 2."""
    label_1, label_2 = initial
    for r in pool.rounds:
        if r.next_theory_label is None or r.next_theory_idx is None:
            continue
        if r.next_theory_idx == 1:
            label_1 = r.next_theory_label
        elif r.next_theory_idx == 2:
            label_2 = r.next_theory_label
    return label_1, label_2


def next_model_label(current: str, all_labels: set[str]) -> str:
    """`pi_1` -> `pi_1_1`, `pi_1_1` -> `pi_1_2`, etc.

    Considers every `pi_<base>_*` label ever seen in this run when picking
    the next version, so we never collide with a previously generated model.
    """
    base, _ = _parse_label(current)
    versions = [
        (_parse_label(L)[1] or 0)
        for L in all_labels
        if _parse_label(L)[0] == base
    ]
    return f"pi_{base}_{max(versions) + 1}"


def next_fresh_label(all_labels: set[str]) -> str:
    """`{pi_1, pi_2, pi_1_1}` -> `pi_3`. Bumps the highest base id ever seen."""
    bases = [_parse_label(L)[0] for L in all_labels]
    return f"pi_{max(bases) + 1}"


# --- orchestrator -----------------------------------------------------------


def round_dir(run_dir: Path, round_idx: int) -> Path:
    """Per-round folder; `<role>/prompts/` lives under it for each agent."""
    return run_dir / "rounds" / f"round_{round_idx:03d}"


def _select_leaderboard(
    pool: Observations,
    *,
    mode: Literal["none", "best", "sample"],
    n: int,
    exclude: set[str] | None = None,
) -> list[tuple[str, Theory, float]]:
    """Pick `(label, theory, score)` triples for the prompt's leaderboard.

    `exclude` is a set of labels to drop from consideration — used by the
    improver path to skip the killed theory (already shown above as
    `## PREVIOUS MODEL INSTANCE`).
    """
    if mode == "none" or n <= 0:
        return []
    excluded = exclude or set()
    registry = pool.theory_registry()
    scores = {
        L: s for L, s in pool.theory_scores().items()
        if L not in excluded and L in registry
    }
    if not scores:
        return []
    labels: list[str]
    if mode == "best":
        labels = sorted(scores.keys(), key=lambda L: scores[L], reverse=True)[:n]
    elif mode == "sample":
        # Weighted sampling without replacement; uniform fallback when the
        # weight-mass collapses (everyone tied at 0).
        remaining_labels = list(scores.keys())
        remaining_weights = [max(0.0, scores[L]) for L in remaining_labels]
        labels = []
        for _ in range(min(n, len(remaining_labels))):
            if sum(remaining_weights) <= 0:
                chosen = random.choice(remaining_labels)
            else:
                chosen = random.choices(
                    remaining_labels, weights=remaining_weights, k=1
                )[0]
            idx = remaining_labels.index(chosen)
            remaining_labels.pop(idx)
            remaining_weights.pop(idx)
            labels.append(chosen)
    else:
        return []
    return [(L, registry[L], scores[L]) for L in labels]


def _format_leaderboard(
    scores: dict[str, float],
    *,
    prev_scores: dict[str, float] | None = None,
    title: str,
    bar_width: int = 24,
) -> str:
    """Render `pool.theory_scores()` as a sorted, multi-line block.

    Columns: `#rank  label  score  |bar|  delta-from-prev`. The bar is
    scaled to the row with the highest score in `scores` (so the leader
    always fills the bar), making relative gaps easy to eyeball. `delta`
    is omitted when there is no `prev_scores` to compare against, marked
    `new` when the label only appeared this round, and rendered as a
    signed `+/-` float otherwise. Returns the title line alone if `scores`
    is empty.
    """
    if not scores:
        return f"[{title}] (no theories scored yet)"
    rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    max_score = max(s for _, s in rows) or 1.0
    label_w = max(len(L) for L, _ in rows)
    lines = [f"[{title}]"]
    for rank, (label, score) in enumerate(rows, start=1):
        bar_len = int(round(bar_width * (score / max_score)))
        bar_len = max(0, min(bar_width, bar_len))
        bar = "█" * bar_len + "·" * (bar_width - bar_len)
        if prev_scores is None:
            delta_str = ""
        elif label not in prev_scores:
            delta_str = "  (new)"
        else:
            delta = score - prev_scores[label]
            delta_str = (
                f"  Δ{delta:+.3f}" if abs(delta) >= 1e-4 else "  Δ ·    "
            )
        lines.append(
            f"  #{rank:>2}  {label:<{label_w}}  {score:.3f}  |{bar}|{delta_str}"
        )
    return "\n".join(lines)


def _log_leaderboard(
    pool: Observations,
    *,
    round_idx: int,
    stage: str,
    prev_scores: dict[str, float] | None = None,
    history_path: Path | None = None,
) -> dict[str, float]:
    """Pretty-print the full leaderboard and return the scores dict.

    `stage` labels the moment in the round (e.g. `"post-data"` after the
    new observations got real values + predictions, `"post-admit"` after a
    new theory was admitted). When `history_path` is given, the rendered
    block is also appended to that file under a markdown sub-heading so
    the run keeps a scrollable record across rounds.
    """
    scores = pool.theory_scores()
    block = _format_leaderboard(
        scores,
        prev_scores=prev_scores,
        title=f"round {round_idx} | {stage}",
    )
    info("\n" + block)
    if history_path is not None and scores:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as f:
            f.write(f"## round {round_idx} — {stage}\n\n```\n{block}\n```\n\n")
    return scores


def _theory_block_md(label: str, theory: Theory) -> str:
    """Render one theory as a markdown block for `theories.md`."""
    if theory.parameters:
        params_md = "\n".join(f"  - `{k}`: `{v}`" for k, v in theory.parameters.items())
    else:
        params_md = "  (none)"
    rationale = theory.rationale or "(none)"
    return (
        f"**Description:** {theory.description}\n\n"
        f"**Rationale:** {rationale}\n\n"
        f"**Parameters:**\n{params_md}\n\n"
        f"**`predict(parameters, stimulus, history)`:**\n"
        f"```python\n{theory.predict_source.rstrip()}\n```\n\n"
        f"**`policy(probs)`:**\n"
        f"```python\n{theory.policy_source.rstrip()}\n```\n"
    )


def _save_round_theories(
    round_dir_path: Path,
    *,
    round_idx: int,
    round_obj,
    pi_1_label: str,
    pi_2_label: str,
    verdict_kind: str,
) -> None:
    """Snapshot the two starting theories + the regenerated theory for a round.

    Writes two files to `round_dir_path/`:
      * `theories.json` — machine-readable: round_idx, verdict, the two
        starting theories (with their slot, label, and a `killed` flag),
        and the replacement (label, slot, verdict_kind, theory body).
      * `theories.md`   — human-readable markdown: starting theories
        marked with KILLED / SURVIVED, followed by the replacement theory
        and a one-line note pointing back at the slot it took over.

    Requires `round_obj.set_next_theory(...)` to have been called already
    so `next_theory*` fields are populated; called from `run_round` right
    after admission.
    """
    if round_obj.next_theory is None or round_obj.next_theory_idx is None:
        return  # nothing to snapshot yet
    starting = [
        {
            "slot": 1,
            "label": pi_1_label,
            "killed": round_obj.next_theory_idx == 1,
            "theory": round_obj.observations[0].proposer_theory.model_dump(),
        },
        {
            "slot": 2,
            "label": pi_2_label,
            "killed": round_obj.next_theory_idx == 2,
            "theory": round_obj.observations[1].proposer_theory.model_dump(),
        },
    ]
    replacement = {
        "label": round_obj.next_theory_label,
        "slot": round_obj.next_theory_idx,
        "verdict": verdict_kind,
        "theory": round_obj.next_theory.model_dump(),
    }
    payload = {
        "round_idx": round_idx,
        "verdict": verdict_kind,
        "starting_theories": starting,
        "replacement": replacement,
    }
    round_dir_path.mkdir(parents=True, exist_ok=True)
    (round_dir_path / "theories.json").write_text(json.dumps(payload, indent=2))

    md_lines = [
        f"# Round {round_idx} — Theories",
        "",
        f"**Verdict:** `{verdict_kind}` "
        f"(slot {round_obj.next_theory_idx} replaced)",
        "",
        "## Starting theories",
        "",
    ]
    for entry in starting:
        marker = "KILLED ✗" if entry["killed"] else "SURVIVED ✓"
        theory_obj = (
            round_obj.observations[0].proposer_theory
            if entry["slot"] == 1
            else round_obj.observations[1].proposer_theory
        )
        md_lines.append(
            f"### slot {entry['slot']} — `{entry['label']}` — {marker}"
        )
        md_lines.append("")
        md_lines.append(_theory_block_md(entry["label"], theory_obj))
        md_lines.append("")
    md_lines.extend(
        [
            "## Replacement",
            "",
            f"### `{replacement['label']}` → slot {replacement['slot']} "
            f"(via `{verdict_kind}`)",
            "",
            _theory_block_md(replacement["label"], round_obj.next_theory),
        ]
    )
    (round_dir_path / "theories.md").write_text("\n".join(md_lines))


def _theory_for_slot(
    pool: Observations,
    *,
    slot_idx: int,
    default_yaml: str,
    fallback_label: str,
    ground_truth: AutoCog,
) -> AutoCog:
    """Resolve which `AutoCog` should occupy `slot_idx` for the next round.

    Walks rounds in reverse to find the most recent `next_theory` admitted
    into this slot; falls back to the seed YAML when none has ever been
    pinned (i.e. on the very first round).
    """
    for r in reversed(pool.rounds):
        if r.next_theory_idx == slot_idx and r.next_theory is not None:
            label = r.next_theory_label or fallback_label
            return AutoCog(
                label=label,
                theory=r.next_theory,
                experiment_class=HDM_EXPERIMENT_CLASS,
                llm_client=ground_truth.llm_client,
            )
    return AutoCog.from_yaml(
        default_yaml,
        label=fallback_label,
        experiment_class=HDM_EXPERIMENT_CLASS,
    )


def run_round(
    *,
    pool: Observations,
    run_dir: Path,
    ground_truth: AutoCog,
    arbiter: Arbiter,
    improver: Improver,
    theory_generator: TheoryGenerator,
    gt_action_noise: float = 0.0,
    gt_noise_rng: "np.random.Generator | None" = None,
    prev_scores: dict[str, float] | None = None,
    initial_seeds: tuple[str, str] = INITIAL_SEEDS,
) -> tuple[int, dict[str, float]]:
    """Run one full adversarial round end-to-end and persist.

    The round body is:
      1. resolve current `pi_1` / `pi_2` from the pool's regeneration chain,
      2. propose this round (or resume an unfinished latest round),
      3. simulate ground-truth subjects to fill `real_value` per observation,
      4. backfill the (observation × theory) prediction matrix,
      5. arbitrate the round,
      6. dispatch on the verdict (improver or theory_generator) under an
         inner critique loop,
      7. pin the regenerated theory back into its slot, backfill again,
         persist, and log.

    `prev_scores` carries the leaderboard from the previous call so the
    pretty log can show per-label deltas across rounds. Returns
    `(round_idx, end_of_round_scores)` so the caller can thread the new
    scores into the next invocation.
    """
    history_path = run_dir / "leaderboard.md"
    label_1, label_2 = replay_slot_labels(pool)
    pi_1 = _theory_for_slot(
        pool,
        slot_idx=1,
        default_yaml=initial_seeds[0],
        fallback_label=label_1,
        ground_truth=ground_truth,
    )
    pi_2 = _theory_for_slot(
        pool,
        slot_idx=2,
        default_yaml=initial_seeds[1],
        fallback_label=label_2,
        ground_truth=ground_truth,
    )

    # Resume vs. start: if the latest round already has `next_theory` pinned
    # (fully arbitrated last time), or the pool is empty, start a new round;
    # otherwise pick up the half-done one. A round with fewer than 2
    # observations is "half-done" from a prior crash — fill the missing slots
    # in place rather than stacking a fresh round on top of a stale one.
    if len(pool) == 0 or pool.latest_round.next_theory is not None:
        round_idx = len(pool)
        rd = round_dir(run_dir, round_idx)
        round_obj = pool.start_round()
    else:
        round_idx = len(pool) - 1
        rd = round_dir(run_dir, round_idx)
        round_obj = pool.latest_round

    # Repopulate any missing observation slots (0 or 1 present).
    if len(round_obj) == 0:
        obs_pi_1 = round_obj.add(
            pi_1.propose_round(adversary=pi_2, pool=pool, workspace=rd / pi_1.label)
        )
        obs_pi_2 = round_obj.add(
            pi_2.propose_round(adversary=pi_1, pool=pool, workspace=rd / pi_2.label)
        )
        pool.save()
    elif len(round_obj) == 1:
        obs_pi_1 = round_obj[0]
        obs_pi_2 = round_obj.add(
            pi_2.propose_round(adversary=pi_1, pool=pool, workspace=rd / pi_2.label)
        )
        pool.save()
    else:
        obs_pi_1, obs_pi_2 = round_obj[0], round_obj[1]

    info(f"[round {round_idx}] slots: 1={pi_1.label}, 2={pi_2.label}")

    # Collect "real" data via the ground-truth theory. `REAL_N_SUBJECTS` is
    # also the N used by `AutoCog.propose_round`'s metric-acceptance Welch
    # test, so the discriminability check uses the exact sample size humans
    # will be run at.
    if obs_pi_1.real_value is None:
        data_p1 = obs_pi_1.experiment.simulate(
            ground_truth.theory,
            n_runs=REAL_N_SUBJECTS,
            action_noise=gt_action_noise,
            rng=gt_noise_rng,
        )
        obs_pi_1.set_data(data_p1)
    if obs_pi_2.real_value is None:
        data_p2 = obs_pi_2.experiment.simulate(
            ground_truth.theory,
            n_runs=REAL_N_SUBJECTS,
            action_noise=gt_action_noise,
            rng=gt_noise_rng,
        )
        obs_pi_2.set_data(data_p2)

    # Backfill the (observation × theory) prediction matrix: any theory ever
    # admitted in this run gets scored on every observation, including the
    # ones just born this round.
    n_added = pool.backfill_predictions()
    if n_added:
        info(f"[round {round_idx}] backfilled {n_added} predictions on new observations")
    mid_scores = _log_leaderboard(
        pool,
        round_idx=round_idx,
        stage="post-data",
        prev_scores=prev_scores,
        history_path=history_path,
    )
    pool.save()

    # Arbitration over the latest round.
    verdict = arbiter.arbitrate(round_obj, pool=pool, workspace=rd / "arbiter")

    target_label = round_obj[verdict.target_theory_idx - 1].proposer_label or (
        label_1 if verdict.target_theory_idx == 1 else label_2
    )
    # Mirror the (THEORY 1, THEORY 2) -> pi-label mapping the arbitration
    # prompt rendered, so downstream prompts can re-emit it as a lookup key
    # above the arbiter's free-text recommendation. Same fallback semantics
    # as `arbitration.render` (which uses "theory_1"/"theory_2" placeholders
    # when proposer_label is None, but in practice the slots' own labels are
    # the better fallback here).
    arbiter_theory_labels = (
        round_obj[0].proposer_label or label_1,
        round_obj[1].proposer_label or label_2,
    )
    seen_labels = _all_labels(pool, initial=(label_1, label_2))

    # Dispatch on the verdict. `"new_model"` keeps the same theory description
    # and asks the improver for a new (predict, policy, parameters) under the
    # killed theory, producing a `pi_<killed_base>_<K+1>` label. `"new_theory"`
    # discards the targeted theory entirely and asks the theory_generator for
    # a brand-new (description + model), producing a fresh `pi_<max_base + 1>`
    # label.
    if verdict.verdict == "new_model":
        # Improver always seeds from the killed theory; the leaderboard
        # provides broader context. Both round theories (the killed one AND
        # the surviving one) are excluded from the leaderboard since their
        # bodies are shown above under `## ROUND THEORIES` (with the killed
        # one tagged TO REVISE).
        killed_theory = round_obj[verdict.target_theory_idx - 1].proposer_theory
        other_idx = 3 - verdict.target_theory_idx
        other_round_theory = round_obj[other_idx - 1].proposer_theory
        other_round_label = arbiter_theory_labels[other_idx - 1]
        next_label = next_model_label(target_label, seen_labels)
        leaderboard = _select_leaderboard(
            pool,
            mode=LEADERBOARD[0],
            n=LEADERBOARD[1],
            exclude={target_label, other_round_label},
        )
        info(
            f"[round {round_idx}] new_model: killed={target_label} "
            f"-> {next_label} | leaderboard={LEADERBOARD[0]}({LEADERBOARD[1]}) "
            f"-> {[L for L, _, _ in leaderboard] or '[]'}"
        )
        new_model = improver.propose_model(
            theory=killed_theory,
            arbiter_guide=verdict.recommendation,
            arbiter_theory_labels=arbiter_theory_labels,
            arbiter_target_idx=verdict.target_theory_idx,
            other_theory=other_round_theory,
            observations=pool.all_observations,
            workspace=rd / f"improver_{next_label}",
            leaderboard=leaderboard,
        )
        next_theory = make_theory(killed_theory, new_model)
    elif verdict.verdict == "new_theory":
        next_label = next_fresh_label(seen_labels)
        leaderboard = _select_leaderboard(
            pool, mode=LEADERBOARD[0], n=LEADERBOARD[1]
        )
        info(
            f"[round {round_idx}] new_theory: -> {next_label} | "
            f"leaderboard={LEADERBOARD[0]}({LEADERBOARD[1]}) -> "
            f"{[L for L, _, _ in leaderboard] or '[]'}"
        )
        next_theory = theory_generator.propose_theory(
            arbiter_guide=verdict.recommendation,
            arbiter_theory_labels=arbiter_theory_labels,
            arbiter_target_idx=verdict.target_theory_idx,
            observations=pool.all_observations,
            workspace=rd / f"theory_generator_{next_label}",
            leaderboard=leaderboard,
        )
    else:
        raise ValueError(f"unknown verdict: {verdict.verdict!r}")

    # Pin the regenerated theory + its label to this round and persist. The
    # next iteration of the loop will read these via `replay_slot_labels` and
    # the per-slot `next_theory` walk above.
    round_obj.set_next_theory(
        next_theory,
        idx=verdict.target_theory_idx,
        label=next_label,
    )
    # Per-round theory snapshot (json + md) — captures the two starting
    # theories, marks the killed one, and pins the replacement so each
    # round dir is self-contained for postmortems.
    _save_round_theories(
        rd,
        round_idx=round_idx,
        round_obj=round_obj,
        pi_1_label=pi_1.label,
        pi_2_label=pi_2.label,
        verdict_kind=verdict.verdict,
    )
    # Backfill again so the freshly admitted theory gets a prediction on
    # every existing observation (dual of the new-observation backfill
    # above; together they keep the matrix dense).
    n_added = pool.backfill_predictions()
    if n_added:
        info(
            f"[round {round_idx}] backfilled {n_added} predictions for newly admitted "
            f"theory {next_label}"
        )
    end_scores = _log_leaderboard(
        pool,
        round_idx=round_idx,
        stage=f"post-admit ({next_label})",
        prev_scores=mid_scores,
        history_path=history_path,
    )
    pool.save()

    info(
        f"[round {round_idx}] verdict={verdict.verdict} "
        f"target_slot={verdict.target_theory_idx} ({target_label}) "
        f"-> next_label={next_label}"
    )
    return round_idx, end_scores


def main(n_rounds: int = N_ROUNDS, run_dir: Path = RUN_DIR) -> None:
    """Run `n_rounds` adversarial rounds end-to-end against `run_dir`.

    Each round extends the same `Observations` pool in place. Re-running
    `main()` simply appends more rounds (with the resume-on-half-finished
    semantics handled inside `run_round`).
    """
    pool_dir = run_dir / "observations"

    # Build shared resources once.
    ground_truth = AutoCog.from_yaml(
        theory_path=GROUND_TRUTH_YAML,
        label="pi_ground_truth",
        experiment_class=HDM_EXPERIMENT_CLASS,
    )
    from src.config import LLMConfig
    from src.llm import make_client
    ground_truth.llm_client = make_client(
        LLMConfig(provider=LLM_PROVIDER, model=LLM_MODEL)
        )
    pool = Observations.load(pool_dir, experiment_class=HDM_EXPERIMENT_CLASS)
    arbiter = Arbiter.from_config(experiment_class=HDM_EXPERIMENT_CLASS)
    improver = Improver.from_config(experiment_class=HDM_EXPERIMENT_CLASS)
    theory_generator = TheoryGenerator.from_config(
        experiment_class=HDM_EXPERIMENT_CLASS
    )

    # Single RNG for all ground-truth action-noise draws across this run.
    # Only materialised when gt_epsilon > 0 so a noise-free run stays a
    # byte-for-byte passthrough of the prior simulate() code path.
    gt_noise_rng = (
        np.random.default_rng(args.gt_seed) if args.gt_epsilon > 0.0 else None
    )
    info(
        f"[main] starting loop: n_rounds={n_rounds} "
        f"existing_rounds={len(pool)} run_dir={run_dir} "
        f"gt_epsilon={args.gt_epsilon} gt_seed={args.gt_seed}"
    )
    # Carries the end-of-round scores between iterations so each round's
    # leaderboard log can show per-label deltas vs. the previous round.
    prev_scores: dict[str, float] | None = pool.theory_scores() or None
    for i in range(n_rounds):
        info(f"[main] === iteration {i + 1}/{n_rounds} ===")
        try:
            round_idx, prev_scores = run_round(
                pool=pool,
                run_dir=run_dir,
                ground_truth=ground_truth,
                arbiter=arbiter,
                improver=improver,
                theory_generator=theory_generator,
                gt_action_noise=args.gt_epsilon,
                gt_noise_rng=gt_noise_rng,
                prev_scores=prev_scores,
            )
            info(f"[main] iteration {i + 1}/{n_rounds} done (round_idx={round_idx})")
        except Exception as e:
            # Persist whatever we managed before re-raising so the next
            # invocation can resume from the partial state.
            info(
                f"[main] iteration {i + 1}/{n_rounds} crashed "
                f"({type(e).__name__}: {e}); saving pool and aborting loop."
            )
            pool.save()
            raise
    info(f"[main] loop done: final pool has {len(pool)} rounds")


if __name__ == "__main__":
    main()
