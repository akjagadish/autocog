"""
Binary decision-making Centaur simulation entry point.

Same adversarial-discovery orchestrator as the (cardinal) heuristic Centaur
runner, but pinned to the binary-feature paradigm: it runs on
`DecisionMakingBinaryExperiment`, where every expert rating is 0 or 1. There
is no `rating_max` knob — the design space is binary by construction — so the
cardinal-feature machinery (`--force_rating_max` and the dynamically
rating-max-locked subclass) is gone. Centaur stands in for human subjects, so
there is no ground-truth theory; Centaur's own next-token sampling is the
implicit noise source.

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
import random
from pathlib import Path
from typing import Literal

import numpy as np

from src.arbiter import Arbiter
from src.decision_making_binary_features.experiment import (
    DecisionMakingBinaryExperiment,
)
from src.improver import Improver, make_theory
from src.logger import info
from src.observation import Observations
from src.online_config import OnlineConfig  # noqa: F401  (used in commented snippet)
from src.pi import AutoPi
from src.run_config import REAL_N_SUBJECTS
from src.theory import Theory
from src.theory_generator import TheoryGenerator

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--n_rounds", type=int, default=1)
parser.add_argument("--llm_provider", type=str, default="gemini")
parser.add_argument("--llm_model", type=str, default="gemini-3.1-pro-preview")
parser.add_argument('--run_id', type=str, help="Unique tag for this run (e.g. timestamp or short hash)")
parser.add_argument(
    "--centaur_model",
    type=str,
    default="marcelbinz/Llama-3.1-Centaur-70B-adapter",
    help=(
        "HuggingFace model name for the Centaur adapter loaded via UnslothAgent. "
        "Only used when --centaur_mode=real."
    ),
)
parser.add_argument(
    "--centaur_mode",
    type=str,
    default="real",
    choices=("real", "stub"),
    help=(
        "real = load UnslothAgent (requires GPU + unsloth, intended for Della); "
        "stub = use a uniform-random letter generator for local prototyping."
    ),
)
parser.add_argument(
    "--centaur_local_files_only",
    action="store_true",
    help=(
        "Pass --local-files-only=True into UnslothAgent (loads from "
        "Della scratch cache without hitting HuggingFace Hub)."
    ),
)
parser.add_argument(
    "--batched",
    action="store_true",
    help=(
        "Run Centaur subjects with one batched forward pass per trial step "
        "(simulate_centaur_batched). Off by default — keeps the serial "
        "per-(subject, trial) loop."
    ),
)
parser.add_argument(
    "--seed_theories",
    type=str,
    default="ttb,wadd",
    help=(
        "Comma-separated YAML basenames in `theories/heuristic_decision_making/` "
        "to seed the two pi slots on round 0 (e.g. 'ttb,wadd'). Centaur stands "
        "in for human subjects, so there is no ground-truth theory — the seeds "
        "are the only theories the adversarial pair starts from. Must list "
        "exactly 2 names."
    ),
)
args = parser.parse_args()

_seed_names = tuple(s.strip() for s in args.seed_theories.split(",") if s.strip())
if len(_seed_names) != 2:
    parser.error(
        f"--seed_theories must list exactly 2 names; got {_seed_names!r}"
    )

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
# have no prior `next_theory` to inherit from). Centaur stands in as the
# subject, so there is no ground-truth theory to exclude — the user picks
# both seeds via --seed_theories. The theory definitions are shared with the
# cardinal runner (stimulus-agnostic predict/policy functions that operate
# fine on binary [0, 1] features), so they are reused in place.
THEORIES_DIR = "theories/heuristic_decision_making"
INITIAL_SEEDS: tuple[str, str] = (
    f"{THEORIES_DIR}/{_seed_names[0]}.yaml",
    f"{THEORIES_DIR}/{_seed_names[1]}.yaml",
)

# The design space is binary by construction (ratings ∈ [0, 1]); there is no
# rating_max to pin, so the experiment class is used directly. Every
# downstream `experiment_class=` goes through EXPERIMENT_CLASS so the choice is
# honoured end-to-end (pool loading, AutoPi seeding, Arbiter/Improver/Generator).
EXPERIMENT_CLASS: type[DecisionMakingBinaryExperiment] = DecisionMakingBinaryExperiment

# Where the run lives on disk.
# Sibling layout: results/decision_making_binary/{synthetic,humans,centaur}/...
# Unlike `synthetic/` and `humans/`, centaur has no ground-truth theory and no
# explicit action noise — Centaur's own sampling is the implicit noise source.
# Results are keyed by the seed-theory pair (the two YAMLs that bootstrap pi_1
# and pi_2 on round 0). Order matters and is preserved verbatim from
# --seed_theories so reruns with the same flag land in the same dir.
_seeds_tag = "_".join(_seed_names)
RUN_DIR = Path(
    f'results/decision_making_binary/centaur_corrected_theories/seeds_{_seeds_tag}_rounds_{args.n_rounds}/'
    f'dmb_seeds_{_seeds_tag}_{LLM_MODEL}_run{args.run_id}'
)


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
    llm_client,
) -> AutoPi:
    """Resolve which `AutoPi` should occupy `slot_idx` for the next round.

    Walks rounds in reverse to find the most recent `next_theory` admitted
    into this slot; falls back to the seed YAML when none has ever been
    pinned (i.e. on the very first round). The shared `llm_client` is the
    one built in `main()` from `--llm_provider` / `--llm_model`, so every
    regenerated theory talks to the run's chosen model rather than the
    YAML-default one.
    """
    for r in reversed(pool.rounds):
        if r.next_theory_idx == slot_idx and r.next_theory is not None:
            label = r.next_theory_label or fallback_label
            return AutoPi(
                label=label,
                theory=r.next_theory,
                experiment_class=EXPERIMENT_CLASS,
                llm_client=llm_client,
            )
    return AutoPi.from_yaml(
        default_yaml,
        label=fallback_label,
        experiment_class=EXPERIMENT_CLASS,
    )


def run_round(
    *,
    pool: Observations,
    run_dir: Path,
    llm_client,
    arbiter: Arbiter,
    improver: Improver,
    theory_generator: TheoryGenerator,
    centaur_generator,
    batched: bool = False,
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
        llm_client=llm_client,
    )
    pi_2 = _theory_for_slot(
        pool,
        slot_idx=2,
        default_yaml=initial_seeds[1],
        fallback_label=label_2,
        llm_client=llm_client,
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

    # Collect "real" data by simulating Centaur as the subject. `REAL_N_SUBJECTS`
    # is also the N used by `AutoPi.propose_round`'s metric-acceptance Welch
    # test, so the discriminability check uses the exact sample size humans
    # will be run at. Centaur's own next-token sampling is the implicit noise
    # source — no explicit action-noise / RNG knob.
    def _simulate_subjects(experiment):
        if batched:
            return experiment.simulate_centaur_batched(
                n_runs=REAL_N_SUBJECTS,
                batch_generator=centaur_generator,
            )
        return experiment.simulate_centaur(
            n_runs=REAL_N_SUBJECTS,
            generator=centaur_generator,
        )

    # Save after each Centaur simulation: each call is expensive (REAL_N_SUBJECTS
    # × n_trials × 70B forward passes), so a mid-round crash on pi_2 must not
    # discard pi_1's just-completed data. The resume path will see real_value
    # populated and skip already-simulated slots.
    if obs_pi_1.real_value is None:
        obs_pi_1.set_data(_simulate_subjects(obs_pi_1.experiment))
        pool.save()
    if obs_pi_2.real_value is None:
        obs_pi_2.set_data(_simulate_subjects(obs_pi_2.experiment))
        pool.save()

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


def build_centaur_generator(
    mode: str,
    model_name: str,
    local_files_only: bool,
    *,
    batched: bool = False,
):
    """Construct the Centaur next-token generator once for this run.

    `mode='real'` lazily imports `UnslothAgent` (needs GPU + unsloth +
    transformers) and loads the Llama-3.1-Centaur-70B adapter; intended for
    the Della SBATCH path. `mode='stub'` returns a uniform-random valid-letter
    generator that doesn't load any model — useful for prototyping the
    pipeline locally without GPU.

    When `batched=False` (default) the returned object is callable as
    `generator(prompt: str) -> str` — matches `simulate_centaur`. When
    `batched=True` the returned object is callable as
    `generator(prompts: list[str]) -> list[str]` — matches
    `simulate_centaur_batched`, which runs all subjects in lockstep through
    a single batched forward pass per trial step.
    """
    if mode == "stub":
        info(
            f"[main] centaur_mode=stub batched={batched} — using uniform-random "
            "letter generator (no model load)."
        )
        rng = np.random.default_rng()
        import re as _re

        def _stub_one(prompt: str) -> str:
            m = _re.search(r"labeled (\w) and (\w)\.", prompt)
            if m is None:
                return ""
            return str(rng.choice([m.group(1), m.group(2)]))

        if not batched:
            return _stub_one

        def _stub_batch(prompts: list[str]) -> list[str]:
            return [_stub_one(p) for p in prompts]

        return _stub_batch

    if mode != "real":
        raise ValueError(f"unknown centaur_mode={mode!r}; expected 'real' or 'stub'.")

    info(
        f"[main] centaur_mode=real batched={batched} — loading "
        f"UnslothAgent(model={model_name!r}, local_files_only={local_files_only}). "
        "This can take several minutes."
    )
    # Lazy import: unsloth + torch are heavyweight and only available on the
    # cluster. Importing at module top would break local dev / tests.
    import sys as _sys
    centaur_dir = str(Path(__file__).resolve().parent / "centaur")
    if centaur_dir not in _sys.path:
        _sys.path.insert(0, centaur_dir)
    from agents import UnslothAgent  # type: ignore

    agent = UnslothAgent(
        model_name=model_name,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
        logprob=False,
        local_files_only=local_files_only,
    )

    if not batched:
        return agent

    # Wrap the underlying transformers pipeline so it processes all subject
    # prompts as a single GPU batch per trial step. We pass batch_size=len(prompts)
    # so the pipeline actually batches on the GPU rather than serializing.
    # Tune (e.g. via a smaller batch_size) if VRAM becomes a constraint.
    def _real_batch(prompts: list[str]) -> list[str]:
        prompts_list = list(prompts)
        out = agent.pipeline(prompts_list, batch_size=len(prompts_list))
        results: list[str] = []
        for p, o in zip(prompts_list, out):
            item = o[0] if isinstance(o, list) else o
            results.append(item["generated_text"][len(p):])
        return results

    return _real_batch


def main(n_rounds: int = N_ROUNDS, run_dir: Path = RUN_DIR) -> None:
    """Run `n_rounds` adversarial rounds end-to-end against `run_dir`.

    Each round extends the same `Observations` pool in place. Re-running
    `main()` simply appends more rounds (with the resume-on-half-finished
    semantics handled inside `run_round`).
    """
    pool_dir = run_dir / "observations"

    # Build shared resources once. There's no ground-truth theory in the
    # centaur pipeline (Centaur is the subject), so we only need the shared
    # `llm_client` that every regenerated AutoPi will talk to.
    from src.config import LLMConfig
    from src.llm import make_client
    llm_client = make_client(LLMConfig(provider=LLM_PROVIDER, model=LLM_MODEL))
    pool = Observations.load(pool_dir, experiment_class=EXPERIMENT_CLASS)
    arbiter = Arbiter.from_config(experiment_class=EXPERIMENT_CLASS)
    improver = Improver.from_config(experiment_class=EXPERIMENT_CLASS)
    theory_generator = TheoryGenerator.from_config(
        experiment_class=EXPERIMENT_CLASS
    )

    # Build the Centaur generator once for this whole run. On Della (centaur_mode=real)
    # this loads the 70B adapter into GPU memory; we want exactly one load per
    # SBATCH execution and reuse it across rounds and pi slots.
    centaur_generator = build_centaur_generator(
        mode=args.centaur_mode,
        model_name=args.centaur_model,
        local_files_only=args.centaur_local_files_only,
        batched=args.batched,
    )
    info(
        f"[main] starting loop: n_rounds={n_rounds} "
        f"existing_rounds={len(pool)} run_dir={run_dir}"
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
                llm_client=llm_client,
                arbiter=arbiter,
                improver=improver,
                theory_generator=theory_generator,
                centaur_generator=centaur_generator,
                batched=args.batched,
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
