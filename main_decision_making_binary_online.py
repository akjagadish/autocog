"""
HeuristicDecisionMaking — ONLINE adversarial loop.

This script ALWAYS collects data from real participants (Firebase, or
Firebase + Prolific). There is no synthetic ground-truth theory and no
action-noise simulation in this code path: the data IS the ground truth.
The CLI exposes `--seeds SEED_1 SEED_2` to pick the two heuristic YAMLs
that initialise slot 1 and slot 2; nothing is "withheld" or "discovered
from scratch", because real human responses are doing the supervising.

For the simulate-mode (synthetic ground-truth) version, see
`main_heuristic_decision_making.py`.

Run-dir layout (encoded into the path itself for at-a-glance triage):

  results/online/dmb/<backend>/<run_id>/<seed1>+<seed2>/
    run_meta.json        # source/backend/seeds/model/etc. — self-documenting
    observations/        # the shared evidence pool (state.json + data/)
    leaderboard.md       # per-round, per-stage scores (appended)
    rounds/round_NNN/    # all artifacts for one round, regardless of which
      pi_1/prompts/...   # pi was active. When a pi gets replaced (gecco),
      pi_1_1/prompts/... # the next round just gets a fresh subdir — no
      arbiter/prompts/   # stale workspace tied to a now-gone theory.

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
from typing import Any, Literal, cast

import yaml
from pydantic import Field

from src.arbiter import Arbiter
from src.decision_making_binary_features.experiment import (
    DecisionMakingBinaryExperiment,
)
from src.improver import Improver, make_theory
from src.llm import LLMClient


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
    "--real_n_subjects",
    type=int,
    default=REAL_N_SUBJECTS,
    help=(
        "Number of online subject slots per proposed experiment (and the "
        "same N used by propose_round discriminability checks)."
    ),
)
parser.add_argument(
    "--proposal_alpha",
    type=float,
    default=0.05,
    help=(
        "Welch-test significance threshold used by theorist proposal "
        "acceptance (`AutoCog.propose_round(alpha=...)`)."
    ),
)
parser.add_argument(
    "--proposal_n_runs",
    type=int,
    default=None,
    help=(
        "Synthetic subjects per theory used by the proposer's pre-flight "
        "discriminability simulation (`AutoCog.propose_round(n_runs=...)`). "
        "These runs estimate the predicted metric mean & between-subject "
        "variance under each competing theory; those estimates are then "
        "plugged into a Welch t-test with N=--real_n_subjects to decide "
        "whether the proposed (experiment, metric) is worth uploading. "
        "Defaults to --real_n_subjects (i.e. the same N as the real data); "
        "raise it for tighter variance estimates at the cost of more LLM "
        "policy calls per proposal attempt."
    ),
)
parser.add_argument(
    "--seeds",
    type=str,
    nargs=2,
    metavar=("SEED_1", "SEED_2"),
    required=True,
    choices=("ttb", "ew", "tallying", "wadd"),
    help=(
        "Two heuristic YAMLs used to seed slot 1 and slot 2 on the very "
        "first round of a run (when no `next_theory` has been pinned yet). "
        "ONLINE runs have no synthetic ground-truth theory; the data IS the "
        "ground truth. Pick whichever pair you want the discovery loop to "
        "start from. Order matters: SEED_1 -> slot 1, SEED_2 -> slot 2. "
        "Must be two distinct values."
    ),
)
parser.add_argument(
    "--online_backend",
    type=str,
    default="firebase",
    choices=("firebase", "firebase_prolific"),
    help=(
        "Online collection backend. `firebase` runs only Firebase recruitment; "
        "`firebase_prolific` runs Firebase and Prolific collections and merges "
        "their data for each experiment."
    ),
)
parser.add_argument(
    "--firebase_credentials_path",
    type=str,
    default="firebase_credentials.json",
    help="Path to Firebase credentials JSON.",
)
parser.add_argument(
    "--prolific_settings_path",
    type=str,
    default="prolific_settings.json",
    help=(
        "Path to Prolific settings JSON. Required only when "
        "--online_backend=firebase_prolific."
    ),
)
parser.add_argument(
    "--study_url",
    type=str,
    default="",
    help="Participant-facing study URL. Required for Prolific runs.",
)
parser.add_argument(
    "--completion_code",
    type=str,
    default="",
    help="Prolific completion code (used only in Prolific mode).",
)
parser.add_argument(
    "--online_duration",
    type=int,
    default=20,
    help="Online study timeout in minutes.",
)
parser.add_argument(
    "--online_max_trials",
    type=int,
    default=None,
    help=(
        "Optional cap for DecisionMakingBinaryExperiment.MAX_TRIALS during "
        "online runs. Useful for Firebase smoke tests to keep payloads small."
    ),
)
parser.add_argument(
    "--min_rt_ms",
    type=int,
    default=1500,
    help=(
        "Per-trial response gate in ms. While > 0, each HDM choice trial "
        "swallows keypresses for the first --min_rt_ms after it renders, "
        "and reveals the A/B answer prompt only once responses are active. "
        "Set to 0 to disable. Default 1500."
    ),
)
parser.add_argument(
    "--consent_yaml_path",
    type=str,
    default="consent.yaml",
    help="Consent YAML path used when building online experiments.",
)
parser.add_argument("--llm_provider", type=str, default="gemini")
parser.add_argument("--llm_model", type=str, default="gemini-3.1-pro-preview")
parser.add_argument('--run_id', type=str, required=True, help="Unique tag for this run (e.g. timestamp or short hash)")
args = parser.parse_args()

if args.real_n_subjects <= 0:
    parser.error(f"--real_n_subjects must be > 0; got {args.real_n_subjects!r}")
if not (0.0 < args.proposal_alpha < 1.0):
    parser.error(
        f"--proposal_alpha must be in (0, 1); got {args.proposal_alpha!r}"
    )
if args.online_backend == "firebase_prolific" and not args.study_url.strip():
    parser.error(
        "--study_url is required when --online_backend=firebase_prolific."
    )
if args.online_max_trials is not None and args.online_max_trials <= 0:
    parser.error(
        f"--online_max_trials must be > 0; got {args.online_max_trials!r}"
    )
if args.min_rt_ms < 0:
    parser.error(
        f"--min_rt_ms must be >= 0; got {args.min_rt_ms!r}"
    )
if args.proposal_n_runs is not None and args.proposal_n_runs <= 0:
    parser.error(
        f"--proposal_n_runs must be > 0; got {args.proposal_n_runs!r}"
    )
if args.seeds[0] == args.seeds[1]:
    parser.error(
        f"--seeds must be two distinct heuristics; got {args.seeds!r}. "
        "Slot 1 and slot 2 cannot start from the same theory."
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
# have no prior `next_theory` to inherit from). The user picks the seed pair
# explicitly via `--seeds`; ONLINE runs have no synthetic ground-truth theory
# (real participants on Firebase / Prolific produce the data), so there is
# nothing to "withhold" — the seeds are just where the discovery loop starts.
THEORIES_DIR = "theories/heuristic_decision_making"
SEEDS: tuple[str, str] = (args.seeds[0], args.seeds[1])
INITIAL_SEEDS: tuple[str, str] = (
    f"{THEORIES_DIR}/{SEEDS[0]}.yaml",
    f"{THEORIES_DIR}/{SEEDS[1]}.yaml",
)

# The design space is binary by construction (ratings in [0, 1]); there is no
# rating_max to pin, so the experiment class is used directly. Every downstream
# `experiment_class=` goes through EXPERIMENT_CLASS so the choice is honoured
# end-to-end (pool loading, AutoCog seeding, Arbiter/Improver/Generator).
EXPERIMENT_CLASS: type[DecisionMakingBinaryExperiment] = DecisionMakingBinaryExperiment
if args.online_max_trials is not None:
    EXPERIMENT_CLASS.MAX_TRIALS = int(args.online_max_trials)

# Where the run lives on disk. Layout is:
#
#   results/online/<task>/<backend>/<run_id>/<seeds>/
#
# i.e. `online/` makes the data source unmistakable (no synthetic ground
# truth — real participants on Firebase / Prolific), `<task>` pins the
# domain (`dmb` here), `<backend>` records firebase vs firebase_prolific,
# `<run_id>` is the user's tag, and the leaf is the seed pair (slot1+slot2)
# that defined the initial adversarial state. Model / rating_max / sample
# size / etc. are written into `run_meta.json` at the run-dir root.
_SEEDS_TAG = f"{SEEDS[0]}+{SEEDS[1]}"
RUN_DIR = Path(
    f"results/online/dmb/{args.online_backend}/{args.run_id}/{_SEEDS_TAG}"
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
    llm_client: LLMClient,
) -> AutoCog:
    """Resolve which `AutoCog` should occupy `slot_idx` for the next round.

    Walks rounds in reverse to find the most recent `next_theory` admitted
    into this slot; falls back to the seed YAML when none has ever been
    pinned (i.e. on the very first round). `llm_client` is shared across
    every spawned `AutoCog` so the whole run talks to the same model.
    """
    for r in reversed(pool.rounds):
        if r.next_theory_idx == slot_idx and r.next_theory is not None:
            label = r.next_theory_label or fallback_label
            return AutoCog(
                label=label,
                theory=r.next_theory,
                experiment_class=EXPERIMENT_CLASS,
                llm_client=llm_client,
            )
    return AutoCog.from_yaml(
        default_yaml,
        label=fallback_label,
        experiment_class=EXPERIMENT_CLASS,
    )


def _collect_online_batch(
    experiments: list[DecisionMakingBinaryExperiment],
    *,
    backend: Literal["firebase", "firebase_prolific"],
    n_subjects: int,
) -> list:
    """Run all proposed experiments online in a **single** upload/wait cycle.

    Conditions are flattened across (experiment_idx × subject_slot) and
    uploaded together, so participants on Firebase / Prolific can pick up
    any condition in parallel — all `n_experiments * n_subjects` slots
    fill concurrently rather than waiting one experiment at a time.

    The returned observations preserve input order, so we split them back
    into per-experiment groups and decode each via the originating
    experiment's `_observations_to_df`. `backend` selects the runner
    (`firebase` vs `firebase_prolific`).

    Each experiment compiles its own JS (``share_template=False``) because
    HDM experiments differ in `validities` / `rating_max`, which are
    baked into the Sweetbean stimulus at compile time and would otherwise
    be silently frozen to the first experiment's values.
    """
    if not experiments:
        return []

    from src.online_workflow import run_online as _run_online

    repo_root = Path(__file__).resolve().parent

    def _resolve(p: str | None) -> str | None:
        if not p:
            return None
        path = Path(p)
        return str(path if path.is_absolute() else repo_root / path)

    consent_path = _resolve(args.consent_yaml_path)
    consent_config: dict = {}
    if consent_path and Path(consent_path).exists():
        with open(consent_path, encoding="utf-8") as f:
            consent_config = yaml.safe_load(f) or {}

    n_slots = max(1, int(n_subjects))
    credentials_path = _resolve(args.firebase_credentials_path)
    if credentials_path is None:
        raise ValueError("firebase_credentials_path is required to run online.")

    payload_experiments: list = []
    payload_conditions: list[dict] = []
    for exp_idx, exp in enumerate(experiments):
        js_exp, _timelines = exp.build_online_experiment(
            consent_config=consent_config,
            min_rt_ms=int(args.min_rt_ms),
        )
        for slot_idx in range(n_slots):
            payload_experiments.append(js_exp)
            payload_conditions.append({
                "experiment_idx": exp_idx,
                "subject_slot": slot_idx,
            })

    expected = len(experiments) * n_slots
    print(
        f"[online] uploading {len(experiments)} experiments × {n_slots} subjects "
        f"= {expected} conditions in a single batch ({backend})",
        flush=True,
    )

    raw_observations = _run_online(
        payload_experiments,
        payload_conditions,
        credentials_path=credentials_path,
        is_prolific=(backend == "firebase_prolific"),
        study_completion_time=args.online_duration,
        prolific_settings_path=_resolve(args.prolific_settings_path),
        study_url=args.study_url.strip(),
        completion_code=args.completion_code,
        share_template=False,
    )
    if len(raw_observations) != expected:
        raise RuntimeError(
            f"runner returned {len(raw_observations)} observations for "
            f"{expected} batched conditions ({len(experiments)} exps × {n_slots} subjects)"
        )

    # Demux by the autora `slot_key` (the true upload index), NOT by list
    # position. The runner returns observations ordered by `sorted(str(key))`
    # over keys "0".."N-1", which is LEXICOGRAPHIC ("10" sorts before "2"), so
    # positional slicing scrambles experiments the moment there are >=11
    # conditions (e.g. 2 exps x 25 subjects = 50). We uploaded condition i at
    # upload index i = exp_idx * n_slots + subject_slot, so the originating
    # experiment is exactly `slot_key // n_slots`. Grouping on it guarantees
    # each experiment is decoded with its own participants — and therefore its
    # own generator / metric / theory — regardless of completion order.
    from collections import defaultdict

    buckets: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for o in raw_observations:
        slot = int(o["slot_key"])
        buckets[slot // n_slots].append((slot % n_slots, o))

    out = []
    for exp_idx, exp in enumerate(experiments):
        # Sort within the experiment by subject_slot so per-experiment subject
        # ordering is deterministic and matches the upload order.
        chunk = [o for _subject_slot, o in sorted(buckets.get(exp_idx, []))]
        df = exp._observations_to_df(chunk)
        # Keep explicit source-condition tags so downstream JSON/data handling
        # can always untangle which online condition produced each row.
        df["online_experiment_idx"] = exp_idx
        if "subject_id" in df.columns:
            df["online_subject_slot"] = df["subject_id"].astype(int)
        out.append(df)
    return out


def run_round(
    *,
    pool: Observations,
    run_dir: Path,
    llm_client: LLMClient,
    arbiter: Arbiter,
    improver: Improver,
    theory_generator: TheoryGenerator,
    real_n_subjects: int = REAL_N_SUBJECTS,
    proposal_alpha: float = 0.05,
    proposal_n_runs: int = REAL_N_SUBJECTS,
    prev_scores: dict[str, float] | None = None,
    initial_seeds: tuple[str, str] = INITIAL_SEEDS,
) -> tuple[int, dict[str, float]]:
    """Run one full adversarial round end-to-end and persist.

    The round body is:
      1. resolve current `pi_1` / `pi_2` from the pool's regeneration chain,
      2. propose this round (or resume an unfinished latest round),
      3. collect ONLINE data from real participants (Firebase / Prolific)
         to fill `real_value` per observation — there is no synthetic
         ground-truth simulation here, the data IS the ground truth,
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
            pi_1.propose_round(
                adversary=pi_2,
                pool=pool,
                workspace=rd / pi_1.label,
                real_n_subjects=real_n_subjects,
                alpha=proposal_alpha,
                n_runs=proposal_n_runs,
            )
        )
        obs_pi_2 = round_obj.add(
            pi_2.propose_round(
                adversary=pi_1,
                pool=pool,
                workspace=rd / pi_2.label,
                real_n_subjects=real_n_subjects,
                alpha=proposal_alpha,
                n_runs=proposal_n_runs,
            )
        )
        pool.save()
    elif len(round_obj) == 1:
        obs_pi_1 = round_obj[0]
        obs_pi_2 = round_obj.add(
            pi_2.propose_round(
                adversary=pi_1,
                pool=pool,
                workspace=rd / pi_2.label,
                real_n_subjects=real_n_subjects,
                alpha=proposal_alpha,
                n_runs=proposal_n_runs,
            )
        )
        pool.save()
    else:
        obs_pi_1, obs_pi_2 = round_obj[0], round_obj[1]

    info(f"[round {round_idx}] slots: 1={pi_1.label}, 2={pi_2.label}")

    # Collect "real" data ONLINE from real participants (Firebase /
    # Prolific). There is no synthetic ground-truth theory in this script —
    # the human data IS the ground truth. `real_n_subjects` is also the N
    # used by `AutoCog.propose_round`'s metric-acceptance Welch test, so the
    # discriminability check uses the exact sample size humans will be run
    # at.
    pending_obs: list = []
    pending_experiments: list[DecisionMakingBinaryExperiment] = []
    if obs_pi_1.real_value is None:
        pending_obs.append(obs_pi_1)
        pending_experiments.append(
            cast(DecisionMakingBinaryExperiment, obs_pi_1.experiment)
        )
    if obs_pi_2.real_value is None:
        pending_obs.append(obs_pi_2)
        pending_experiments.append(
            cast(DecisionMakingBinaryExperiment, obs_pi_2.experiment)
        )
    if pending_experiments:
        online_data = _collect_online_batch(
            pending_experiments,
            backend=args.online_backend,
            n_subjects=real_n_subjects,
        )
        for obs, data in zip(pending_obs, online_data):
            obs.set_data(data)

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


def _write_run_meta(run_dir: Path) -> None:
    """Drop a `run_meta.json` next to `observations/` documenting the run.

    Makes a run dir self-describing: anyone reading it later can see the
    data source (always ``"online"`` here), backend, seed pair, model, and
    every other knob that shaped the run — without having to reverse the
    path. Idempotent: rewritten on every `main()` call so `run_meta` always
    matches the latest CLI invocation.
    """
    meta = {
        "source": "online",
        "task": "decision_making_binary",
        "backend": args.online_backend,
        "seeds": list(SEEDS),
        "run_id": args.run_id,
        "n_rounds": int(args.n_rounds),
        "real_n_subjects": int(args.real_n_subjects),
        "proposal_alpha": float(args.proposal_alpha),
        # Resolved value actually used (defaults to real_n_subjects when
        # the CLI override is unset). Persisted so a later read of the run
        # dir can tell whether the discriminability simulation was sized
        # to match the human N or budgeted higher.
        "proposal_n_runs": int(
            args.proposal_n_runs
            if args.proposal_n_runs is not None
            else args.real_n_subjects
        ),
        "online_max_trials": (
            int(args.online_max_trials)
            if args.online_max_trials is not None
            else None
        ),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "consent_yaml_path": args.consent_yaml_path,
        "study_url": args.study_url,
        "online_duration_minutes": int(args.online_duration),
        # Per-trial response gate (ms). 0 = no gate. Persisted so the
        # subject pool's effective response floor is recoverable from the
        # run dir alone, e.g. when post-hoc filtering for unrealistically
        # short RTs.
        "min_rt_ms": int(args.min_rt_ms),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def main(n_rounds: int = N_ROUNDS, run_dir: Path = RUN_DIR) -> None:
    """Run `n_rounds` adversarial rounds end-to-end against `run_dir`.

    Each round extends the same `Observations` pool in place. Re-running
    `main()` simply appends more rounds (with the resume-on-half-finished
    semantics handled inside `run_round`).

    Data source is always ONLINE here (Firebase or Firebase + Prolific) —
    real participants produce every `real_value`, so there is no synthetic
    ground-truth theory anywhere in this script.
    """
    pool_dir = run_dir / "observations"

    # One LLM client shared by every spawned `AutoCog` (slot 1, slot 2, the
    # improver, the theory-generator) so the whole run talks to the same
    # model. There is no longer a "ground truth" `AutoCog` instance — the
    # data comes from real participants, not from a stand-in theory.
    from src.config import LLMConfig
    from src.llm import make_client
    llm_client: LLMClient = make_client(
        LLMConfig(provider=LLM_PROVIDER, model=LLM_MODEL)
    )
    pool = Observations.load(pool_dir, experiment_class=EXPERIMENT_CLASS)
    arbiter = Arbiter.from_config(experiment_class=EXPERIMENT_CLASS)
    improver = Improver.from_config(experiment_class=EXPERIMENT_CLASS)
    theory_generator = TheoryGenerator.from_config(
        experiment_class=EXPERIMENT_CLASS
    )

    _write_run_meta(run_dir)
    # Resolve the proposer's pre-flight discriminability sim size: defaults
    # to real_n_subjects so the synthetic-prediction sample mirrors the
    # human sample by default; CLI override decouples the two.
    proposal_n_runs = (
        int(args.proposal_n_runs)
        if args.proposal_n_runs is not None
        else int(args.real_n_subjects)
    )
    info(
        f"[main] DATA SOURCE: ONLINE (backend={args.online_backend}) "
        f"task=dmb seeds={SEEDS[0]}+{SEEDS[1]} "
        f"model={LLM_MODEL} run_id={args.run_id}"
    )
    info(
        f"[main] starting loop: n_rounds={n_rounds} "
        f"existing_rounds={len(pool)} run_dir={run_dir} "
        f"real_n_subjects={args.real_n_subjects} "
        f"proposal_alpha={args.proposal_alpha} "
        f"proposal_n_runs={proposal_n_runs}"
        + ("" if args.proposal_n_runs is not None else " (default = real_n_subjects)")
        + f" min_rt_ms={int(args.min_rt_ms)}"
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
                real_n_subjects=args.real_n_subjects,
                proposal_alpha=args.proposal_alpha,
                proposal_n_runs=proposal_n_runs,
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
