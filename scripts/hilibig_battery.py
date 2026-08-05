"""Hilbig heuristic-decision-making battery: base, surfaced, and
ground-truth models.

This is the decision-making analog of `recovery_battery.py` / `shepard_battery.py`.
For an AutoPi run directory (e.g.
`results/wadd/noise=0.3/hdm_ground_truth_wadd_noise=0.3_gemini-3.1-pro-preview_run2`),
it loads the models that surfaced during discovery and replays them on the
EXACT trial pairs the subjects saw, alongside the paper-faithful
ground-truth heuristic (TTB / WADD / Tallying / Equal-Weight).

The hilbig heuristics are stateless (no trial-to-trial learning) so `Theory.from_yaml`
is enough.

Outputs (under `--out`, default `<run-dir>/analysis/hilibig_battery/`):
    hilibig_battery.csv              — per-subject probs/argmax per (model, trial)
    choice_agreement.png             — bar chart: mean argmax-match to GT per model
    distance_to_ground_truth.csv     — per-model MAE/MSE (subject detail + aggregates)
    distance_to_ground_truth.png     — 1x2 bar chart: MAE (left), MSE (right)
"""

from __future__ import annotations

import argparse
import csv as _csv_mod
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from src.theory import Theory  # noqa: E402
from scripts.recovery_battery import INT_PARAM_KEYS  # noqa: E402


YAML_DIR = _REPO_ROOT / "theories" / "heuristic_decision_making"

# Map from --ground-truth CLI name to YAML filename stem.
GROUND_TRUTH_YAML: dict[str, str] = {
    "ttb": "ttb",
    "wadd": "wadd",
    "tallying": "tallying",
    "ew": "ew",
    "eqw": "ew",  # alias
}


# ---------------------------------------------------------------------------
# Run metadata: validities, rating_max, and the unique trial pairs seen.
# ---------------------------------------------------------------------------


def _experiment_from_round(round_obj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the experiment dict out of one `rounds[*]` entry.

    The state.json layout is `rounds[*].observations[*].experiment`. We
    take the first observation's experiment — all observations in the
    same round share the same design.
    """
    obs_list = round_obj.get("observations") or []
    for obs in obs_list:
        exp = obs.get("experiment")
        if isinstance(exp, dict):
            return exp
    return None


def load_run_metadata(
    run_dir: Path, *, round_idx: int | str = "last",
) -> dict[str, Any]:
    """Read `observations/state.json` and return the HDM design for ONE round.

    Returns a dict with:
        validities:   list[float]        — per-expert validities for this round
        rating_max:   int                — rating upper bound (1 for binary)
        trial_pairs:  list[tuple[a, b]]  — unique (option_a, option_b) pairs
                                           for this round, order-preserving dedupe
        round_idx:    int                — the resolved round index used

    We evaluate ONE round at a time because the LLM arbiter may propose
    different designs — different validity vectors, different rating_max,
    different trial pairs — across rounds. Mixing designs would mean each
    model is being evaluated on a different distribution of problems,
    which is not a meaningful comparison. The default `round_idx="last"`
    picks the final round because that's the design the surfaced theories
    were most recently tested on.
    """
    state = json.loads((run_dir / "observations" / "state.json").read_text())
    rounds = state.get("rounds") or []
    if not rounds:
        raise ValueError(
            f"{run_dir}/observations/state.json has no 'rounds' entries."
        )

    if round_idx == "last":
        resolved = len(rounds) - 1
    elif round_idx == "first":
        resolved = 0
    elif isinstance(round_idx, int):
        if round_idx < 0 or round_idx >= len(rounds):
            raise ValueError(
                f"round_idx={round_idx} out of range [0, {len(rounds) - 1}]."
            )
        resolved = round_idx
    else:
        raise ValueError(
            f"round_idx must be int or 'first'/'last'; got {round_idx!r}."
        )

    exp = _experiment_from_round(rounds[resolved])
    if exp is None:
        raise ValueError(
            f"No experiment block in round {resolved} of {run_dir}."
        )

    validities = [float(v) for v in exp["validities"]]
    rating_max = int(exp["rating_max"])
    n_features = len(validities)

    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    trial_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for a, b in zip(exp["trial_a_ratings"], exp["trial_b_ratings"]):
        a_t = tuple(int(x) for x in a)
        b_t = tuple(int(x) for x in b)
        if len(a_t) != n_features or len(b_t) != n_features:
            raise ValueError(
                f"round {resolved} trial has wrong feature count: "
                f"expected {n_features}, got ({len(a_t)}, {len(b_t)})."
            )
        key = (a_t, b_t)
        if key not in seen:
            seen.add(key)
            trial_pairs.append(key)

    return {
        "validities": validities,
        "rating_max": rating_max,
        "trial_pairs": trial_pairs,
        "round_idx": resolved,
    }


# ---------------------------------------------------------------------------
# Trajectory: shuffled per-subject sequence of (option_a, option_b) pairs.
# ---------------------------------------------------------------------------


@dataclass
class HilibigTrajectory:
    """One subject's trial sequence.

    `stimuli[t]` has shape (2, n_features): row 0 is option A's ratings,
    row 1 is option B's. Each entry is an integer rating in
    [0, rating_max], but we store as int (the YAML predict functions
    will cast to float).
    """

    stimuli: np.ndarray  # (T, 2, n_features), int


def sample_trajectory(
    rng: np.random.Generator,
    *,
    trial_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_blocks: int = 1,
) -> HilibigTrajectory:
    """Repeat `trial_pairs` `n_blocks` times and shuffle the concatenation.

    The block boundary isn't meaningful for stateless heuristics, so we do
    NOT shuffle per-block — one single shuffle over the full concatenation
    gives maximum variety of per-subject orderings.
    """
    if not trial_pairs:
        raise ValueError("trial_pairs must be non-empty.")
    repeated = list(trial_pairs) * int(n_blocks)
    idx = rng.permutation(len(repeated))
    stacked = np.array(
        [np.stack([np.asarray(repeated[i][0], dtype=int),
                   np.asarray(repeated[i][1], dtype=int)], axis=0)
         for i in idx],
        dtype=int,
    )
    return HilibigTrajectory(stimuli=stacked)


# ---------------------------------------------------------------------------
# Replay: per-trial prediction. Heuristics are stateless but we still pass
# the cumulative-history dict in case a surfaced theory chose to read it.
# ---------------------------------------------------------------------------


PredictFn = Callable[[dict[str, Any], np.ndarray, dict[str, Any]], np.ndarray]


def predict_trajectory(
    predict_fn: PredictFn,
    params: dict[str, Any],
    trajectory: HilibigTrajectory,
) -> np.ndarray:
    """Replay `trajectory` through `predict_fn` and return (T, 2) probs.

    Keeps both history-key conventions populated (`stimulus` / `label` and
    legacy `previous_stimuli` / `previous_labels`) to mirror
    `recovery_battery.predict_trajectory`. There's no feedback signal in
    Hilbig, so `label` records the model's ARGMAX choice on the trial —
    this is the honest stand-in for "what the subject saw in their own
    head as their answer" and is what any history-reading heuristic
    would want anyway (e.g. win-stay / lose-shift).
    """
    history: dict[str, list[Any]] = {
        "stimulus": [],
        "label": [],
        "previous_stimuli": [],
        "previous_labels": [],
    }
    preds: list[np.ndarray] = []
    for t in range(trajectory.stimuli.shape[0]):
        stim_t = np.asarray(trajectory.stimuli[t], dtype=float)  # (2, n_features)
        probs = np.asarray(predict_fn(params, stim_t, history), dtype=float)
        preds.append(probs)
        stim_list = trajectory.stimuli[t].tolist()
        choice = int(np.argmax(probs))
        history["stimulus"].append(stim_list)
        history["previous_stimuli"].append(stim_list)
        history["label"].append(choice)
        history["previous_labels"].append(choice)
    return np.asarray(preds, dtype=float)


# ---------------------------------------------------------------------------
# Parameter sampling: reuse recovery_battery's _sample_params, then pin the
# `validities` parameter to the run's actual validities. Subjects were TOLD
# the validities, so honest ground-truth / surfaced-model comparisons must
# all use the same vector; random draws from `[(0.0, 1.0)] * n_features`
# would give degenerate (often reversed-sign) TTB / WADD behavior.
# ---------------------------------------------------------------------------


def sample_hilibig_params(
    theory: Theory,
    *,
    validities: list[float],
    n_features: int,
    rating_max: int | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Sample per-subject parameters for a Hilbig theory.

    The `validities` and `rating_max` task constants are passed both
    into the `sample_parameters` context (so theories that declare
    passthrough-style parameters like `"validities": "validities"` or
    `"rating_max": "rating_max"` resolve correctly via
    `src/sample_parameters.py`) AND pinned post-hoc (so theories that
    declared them as ranges don't get random draws that disagree with
    what subjects were told).
    """
    context: dict[str, Any] = {
        "n_features": int(n_features),
        "n_labels": 2,
        "validities": list(validities),
    }
    if rating_max is not None:
        context["rating_max"] = int(rating_max)

    raw = theory.sample_parameters(context)
    p: dict[str, Any] = dict(raw)
    for k in INT_PARAM_KEYS:
        if k in p and p[k] is not None:
            p[k] = int(p[k])
    if seed_override is not None and "seed" in p:
        p["seed"] = int(seed_override)
    if "validities" in p:
        p["validities"] = list(validities)
    if rating_max is not None and "rating_max" in p:
        p["rating_max"] = int(rating_max)
    return p


# ---------------------------------------------------------------------------
# Theory resolution from a run-dir. Same structure as shepard_battery.
# ---------------------------------------------------------------------------


def _round_dirs(run_dir: Path) -> list[Path]:
    rounds = run_dir / "rounds"
    if not rounds.is_dir():
        raise FileNotFoundError(f"Missing {rounds!s}")
    out: list[tuple[int, Path]] = []
    for d in rounds.iterdir():
        if not (d.is_dir() and d.name.startswith("round_")):
            continue
        if not (d / "theories.json").is_file():
            continue
        try:
            idx = int(d.name.removeprefix("round_"))
        except ValueError:
            continue
        out.append((idx, d))
    if not out:
        raise FileNotFoundError(f"No round_NNN/theories.json under {rounds!s}")
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def _theory_from_entry(entry: dict[str, Any]) -> Theory:
    return Theory.model_validate(entry)


def resolve_base_theories(run_dir: Path) -> dict[str, Theory]:
    first = _round_dirs(run_dir)[0]
    data = json.loads((first / "theories.json").read_text())
    out: dict[str, Theory] = {}
    for s in data.get("starting_theories", []):
        out[s["label"]] = _theory_from_entry(s["theory"])
    return out


def resolve_surfaced_theories(run_dir: Path) -> dict[str, Theory]:
    last = _round_dirs(run_dir)[-1]
    data = json.loads((last / "theories.json").read_text())
    out: dict[str, Theory] = {}
    for s in data.get("starting_theories", []):
        if not s.get("killed", False):
            out[s["label"]] = _theory_from_entry(s["theory"])
    repl = data.get("replacement")
    if repl is not None:
        out[repl["label"]] = _theory_from_entry(repl["theory"])
    return out


# ---------------------------------------------------------------------------
# Simulation: each model replayed on the SAME per-subject trajectories so
# per-trial MAE / MSE to ground truth are paired.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Model:
    display_name: str
    role: str                  # "base" | "surfaced" | "ground-truth" | "gt-floor"
    predict_fn: PredictFn
    params_per_subject: list[dict[str, Any]]


def _build_theory_subjects(
    theory: Theory,
    *,
    validities: list[float],
    rating_max: int,
    n_features: int,
    n_subjects: int,
    base_seed: int,
    seed_offset: int,
) -> list[dict[str, Any]]:
    return [
        sample_hilibig_params(
            theory,
            validities=validities,
            rating_max=rating_max,
            n_features=n_features,
            seed_override=base_seed + seed_offset + s,
        )
        for s in range(n_subjects)
    ]


def build_model_plan(
    *,
    run_dir: Path,
    ground_truth: str,
    validities: list[float],
    rating_max: int,
    n_subjects: int,
    base_seed: int,
) -> list[_Model]:
    """Assemble the models to simulate: base → surfaced → GT → gt-floor."""
    random.seed(base_seed)  # Theory.sample_parameters reaches stdlib random.

    n_features = len(validities)
    plan: list[_Model] = []

    def _subjects(theory: Theory, seed_offset: int) -> list[dict[str, Any]]:
        return _build_theory_subjects(
            theory, validities=validities, rating_max=rating_max,
            n_features=n_features, n_subjects=n_subjects,
            base_seed=base_seed, seed_offset=seed_offset,
        )

    base = resolve_base_theories(run_dir)
    for i, (label, theory) in enumerate(base.items()):
        plan.append(_Model(
            display_name=label, role="base",
            predict_fn=theory.predict,
            params_per_subject=_subjects(theory, 100_000 + 10_000 * i),
        ))

    surfaced = resolve_surfaced_theories(run_dir)
    for i, (label, theory) in enumerate(surfaced.items()):
        plan.append(_Model(
            display_name=label, role="surfaced",
            predict_fn=theory.predict,
            params_per_subject=_subjects(theory, 300_000 + 10_000 * i),
        ))

    gt_stem = GROUND_TRUTH_YAML[ground_truth]
    gt_theory = Theory.from_yaml(YAML_DIR / f"{gt_stem}.yaml")

    for role, display, seed_offset in [
        ("ground-truth", f"ground-truth ({ground_truth})", 500_000),
        ("gt-floor", f"gt-floor ({ground_truth}, paired draw)", 700_000),
    ]:
        plan.append(_Model(
            display_name=display, role=role,
            predict_fn=gt_theory.predict,
            params_per_subject=_subjects(gt_theory, seed_offset),
        ))

    return plan


def simulate_probs(
    predict_fn: PredictFn,
    *,
    params_per_subject: list[dict[str, Any]],
    trial_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_blocks: int,
    base_seed: int,
) -> np.ndarray:
    """Replay `predict_fn` for every subject. Returns (n_subjects, T, 2).

    Per-subject trajectories are keyed on `(base_seed, subject_idx)` so
    different models driven by the same `base_seed` see the SAME
    shuffled trial order — a necessary condition for paired per-trial
    MAE / MSE between two models.
    """
    n_subjects = len(params_per_subject)
    T = len(trial_pairs) * int(n_blocks)
    out = np.empty((n_subjects, T, 2), dtype=float)
    for s, params in enumerate(params_per_subject):
        rng = np.random.default_rng(base_seed + 1000 * s)
        traj = sample_trajectory(
            rng, trial_pairs=trial_pairs, n_blocks=n_blocks)
        out[s] = predict_trajectory(predict_fn, params, traj)
    return out


# ---------------------------------------------------------------------------
# Metrics. `per_trial_mae_mse` returns per-subject arrays so SEM bars work.
# ---------------------------------------------------------------------------


def per_trial_mae_mse(
    probs_gt: np.ndarray, probs_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-level MAE and MSE of choice probabilities vs ground truth.

    Inputs are (n_subjects, T, 2). Returns two length-n_subjects arrays.
    """
    gt = np.asarray(probs_gt, dtype=float)
    m = np.asarray(probs_model, dtype=float)
    if gt.shape != m.shape:
        raise ValueError(
            f"probs_gt and probs_model must share shape; got "
            f"{gt.shape} vs {m.shape}."
        )
    diff = m - gt
    return (
        np.mean(np.abs(diff), axis=(1, 2)),
        np.mean(diff * diff, axis=(1, 2)),
    )


def per_subject_argmax_agreement(
    probs_gt: np.ndarray, probs_model: np.ndarray,
) -> np.ndarray:
    """Per-subject mean argmax-match between model and ground-truth preds.

    Ties break to index 0 (numpy default). Returns a length-n_subjects array.
    """
    gt = np.argmax(probs_gt, axis=-1)
    m = np.argmax(probs_model, axis=-1)
    return (gt == m).astype(float).mean(axis=-1)


# ---------------------------------------------------------------------------
# CSV + plots.
# ---------------------------------------------------------------------------


def write_per_trial_csv(
    sims: dict[str, np.ndarray],
    roles: dict[str, str],
    path: Path,
) -> None:
    """Long-format CSV: one row per (model, subject, trial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv_mod.writer(f)
        w.writerow(["model", "role", "subject", "trial", "p_a", "p_b", "argmax"])
        for name, probs in sims.items():
            role = roles[name]
            for s in range(probs.shape[0]):
                for t in range(probs.shape[1]):
                    p_a = float(probs[s, t, 0])
                    p_b = float(probs[s, t, 1])
                    w.writerow([
                        name, role, s, t,
                        f"{p_a:.6f}", f"{p_b:.6f}", int(np.argmax(probs[s, t])),
                    ])


def write_distance_csv(
    distances: dict[str, tuple[np.ndarray, np.ndarray]],
    roles: dict[str, str],
    path: Path,
) -> None:
    """Aggregate + per-subject MAE/MSE to ground truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv_mod.writer(f)
        w.writerow([
            "model", "role",
            "mae_mean", "mae_sem", "mse_mean", "mse_sem",
            "n_subjects", "mae_subjects", "mse_subjects",
        ])
        for name, (mae_s, mse_s) in distances.items():
            n = int(len(mae_s))
            mae_sem = float(mae_s.std(ddof=1) /
                            np.sqrt(n)) if n >= 2 else 0.0
            mse_sem = float(mse_s.std(ddof=1) /
                            np.sqrt(n)) if n >= 2 else 0.0
            w.writerow([
                name, roles[name],
                f"{float(mae_s.mean()):.6f}", f"{mae_sem:.6f}",
                f"{float(mse_s.mean()):.6f}", f"{mse_sem:.6f}",
                n,
                ";".join(f"{x:.6f}" for x in mae_s),
                ";".join(f"{x:.6f}" for x in mse_s),
            ])


def write_agreement_csv(
    agreements: dict[str, np.ndarray],
    roles: dict[str, str],
    path: Path,
) -> None:
    """Aggregate + per-subject argmax-agreement with ground truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv_mod.writer(f)
        w.writerow([
            "model", "role",
            "agreement_mean", "agreement_sem",
            "n_subjects", "agreement_subjects",
        ])
        for name, arr in agreements.items():
            n = int(len(arr))
            sem = float(arr.std(ddof=1) /
                        np.sqrt(n)) if n >= 2 else 0.0
            w.writerow([
                name, roles[name],
                f"{float(arr.mean()):.6f}", f"{sem:.6f}",
                n,
                ";".join(f"{x:.6f}" for x in arr),
            ])


def plot_choice_agreement(
    agreements: dict[str, np.ndarray],
    roles: dict[str, str],
    path: Path,
    *,
    ground_truth_display_name: str,
    color_of: dict[str, Any] | None = None,
    title: str | None = None,
) -> None:
    """Bar chart: mean argmax-agreement with ground truth per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agreements.keys())
    if color_of is None:
        cmap = matplotlib.colormaps["tab10"]
        color_of = {name: cmap(i % 10) for i, name in enumerate(names)}

    means = np.array([agreements[n].mean() for n in names])
    sems = np.array([
        float(agreements[n].std(ddof=1) / np.sqrt(len(agreements[n])))
        if len(agreements[n]) >= 2 else 0.0
        for n in names
    ])

    fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 0.8 * len(names) + 2.0), 4.6))
    x = np.arange(len(names))
    ax.bar(
        x, means, width=0.7, yerr=sems, capsize=3,
        color=[color_of[n] for n in names],
        edgecolor="black", linewidth=0.4,
    )
    ax.axhline(0.5, color="0.7", linestyle=":", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n[{roles[n]}]" for n in names],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"argmax-agreement with {ground_truth_display_name}")
    ax.set_title(title or "Choice agreement with ground truth")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_to_ground_truth(
    distances: dict[str, tuple[np.ndarray, np.ndarray]],
    roles: dict[str, str],
    path: Path,
    *,
    ground_truth_display_name: str,
    color_of: dict[str, Any] | None = None,
    title: str | None = None,
) -> None:
    """1x2 bar chart: MAE (left) and MSE (right) per model vs ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(distances.keys())
    if not names:
        raise ValueError("distances is empty.")
    if color_of is None:
        cmap = matplotlib.colormaps["tab10"]
        color_of = {name: cmap(i % 10) for i, name in enumerate(names)}

    fig, (ax_mae, ax_mse) = plt.subplots(1, 2, figsize=(max(10.0, 0.8 * len(names) + 4.0), 4.6))

    def _draw(ax, idx: int, metric_name: str) -> None:
        x = np.arange(len(names))
        means = np.array([distances[n][idx].mean() for n in names])
        sems = np.array([
            float(distances[n][idx].std(ddof=1) / np.sqrt(len(distances[n][idx])))
            if len(distances[n][idx]) >= 2 else 0.0
            for n in names
        ])
        ax.bar(
            x, means, width=0.7, yerr=sems, capsize=3,
            color=[color_of[n] for n in names],
            edgecolor="black", linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n[{roles[n]}]" for n in names],
            rotation=30, ha="right", fontsize=9,
        )
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} to {ground_truth_display_name}")
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _draw(ax_mae, 0, "MAE")
    _draw(ax_mse, 1, "MSE")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95 if title else 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Replay base / surfaced / ground-truth models on the Hilbig "
            "heuristic-decision-making trials from an autopi run-dir."
        )
    )
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to results/<run> (must contain rounds/ and observations/state.json).",
    )
    p.add_argument(
        "--ground-truth", choices=sorted(GROUND_TRUTH_YAML), required=True,
        help="Ground-truth heuristic for this run (ttb / wadd / tallying / ew|eqw).",
    )
    p.add_argument("--n-subjects", type=int, default=50)
    p.add_argument(
        "--n-blocks", type=int, default=1,
        help="Number of full repetitions of the unique trial-pair set.",
    )
    p.add_argument(
        "--round-idx", default="last",
        help=(
            "Which round's design to use as the battery: 'first', 'last' "
            "(default), or a 0-based integer index. Evaluating ONE round "
            "at a time because the LLM arbiter may propose different "
            "designs (validities / trial pairs) across rounds."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory. Defaults to <run-dir>/analysis/hilibig_battery/.",
    )
    args = p.parse_args(argv)

    out = args.out or (args.run_dir / "analysis" / "hilibig_battery")
    out.mkdir(parents=True, exist_ok=True)

    round_arg: int | str
    if args.round_idx in ("first", "last"):
        round_arg = args.round_idx
    else:
        try:
            round_arg = int(args.round_idx)
        except ValueError:
            p.error(f"--round-idx must be 'first', 'last', or an integer; "
                    f"got {args.round_idx!r}.")

    meta = load_run_metadata(args.run_dir, round_idx=round_arg)
    validities = meta["validities"]
    trial_pairs = meta["trial_pairs"]

    print(
        f"[hilibig_battery] run_dir={args.run_dir} "
        f"ground_truth={args.ground_truth} n_subjects={args.n_subjects} "
        f"n_blocks={args.n_blocks} seed={args.seed} "
        f"round_idx={meta['round_idx']}"
    )
    print(
        f"[hilibig_battery] validities={validities} "
        f"rating_max={meta['rating_max']} "
        f"unique_trial_pairs={len(trial_pairs)}"
    )

    plan = build_model_plan(
        run_dir=args.run_dir,
        ground_truth=args.ground_truth,
        validities=validities,
        rating_max=meta["rating_max"],
        n_subjects=args.n_subjects,
        base_seed=args.seed,
    )

    print(
        "[hilibig_battery] models:",
        ", ".join(f"{m.display_name} [{m.role}]" for m in plan),
    )

    sims: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    gt_display: str | None = None
    for m in plan:
        print(f"[hilibig_battery]   simulating {m.display_name} ...")
        sims[m.display_name] = simulate_probs(
            m.predict_fn,
            params_per_subject=m.params_per_subject,
            trial_pairs=trial_pairs,
            n_blocks=args.n_blocks,
            base_seed=args.seed,
        )
        roles[m.display_name] = m.role
        if m.role == "ground-truth":
            gt_display = m.display_name

    assert gt_display is not None, "plan must include a ground-truth model."

    per_trial_csv = out / "hilibig_battery.csv"
    write_per_trial_csv(sims, roles, per_trial_csv)
    print(f"[hilibig_battery] wrote {per_trial_csv}")

    # Choice agreement with ground truth (argmax match), per model.
    agreements: dict[str, np.ndarray] = {
        name: per_subject_argmax_agreement(sims[gt_display], probs)
        for name, probs in sims.items() if name != gt_display
    }

    # Paired MAE/MSE to ground truth, per model.
    distances: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: per_trial_mae_mse(sims[gt_display], probs)
        for name, probs in sims.items() if name != gt_display
    }

    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    model_names = list(sims.keys())
    color_of = {name: cmap(i % 10) for i, name in enumerate(model_names)}

    dist_csv = out / "distance_to_ground_truth.csv"
    dist_png = out / "distance_to_ground_truth.png"
    agree_csv = out / "choice_agreement.csv"
    agree_png = out / "choice_agreement.png"

    write_distance_csv(distances, roles, dist_csv)
    write_agreement_csv(agreements, roles, agree_csv)
    plot_distance_to_ground_truth(
        distances, roles, dist_png,
        ground_truth_display_name=gt_display,
        color_of={n: color_of[n] for n in distances},
        title=(
            f"Distance to {gt_display} (paired trajectories) — "
            f"run: {args.run_dir.name} — "
            f"N={args.n_subjects} subj × {args.n_blocks} blocks"
        ),
    )
    plot_choice_agreement(
        agreements, roles, agree_png,
        ground_truth_display_name=gt_display,
        color_of={n: color_of[n] for n in agreements},
        title=(
            f"Choice agreement with {gt_display} — run: {args.run_dir.name}"
        ),
    )
    print(
        f"[hilibig_battery] wrote {dist_csv}, {dist_png}, "
        f"{agree_csv}, {agree_png}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
