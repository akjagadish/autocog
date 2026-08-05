"""Behavioural-cloning JSD on the Shepard I-VI battery.

Headline recovery metric for the AutoPi paper: how close is the discovered
winning theory to the ground-truth model family — measured trial-by-trial
in distribution space, on the canonical Shepard / SHJ category structures
that were designed to discriminate exemplar / clustering / rule learners.

Outputs (under `--out`):
- `recovery_battery_<gt>.png` and `recovery_battery_<gt>.pdf` — the paper
  figure: per-Shepard-type bar chart of JSD_gt vs JSD_floor vs
  JSD_baseline, error bars are SEM across simulated subjects.
- `recovery_battery_<gt>.json` — the underlying numbers (per-subject and
  per-type) so the plot can be regenerated or re-styled later.

Design choices (kept honest and faithful to the spec):
- Ground truth is the `ReferenceXxx` Python implementation in
  `domains/category_learning/reference_models.py`, NOT the YAML — this
  short-circuits any worry about YAML-side bugs in the comparison
  baseline (the parity test in `tests/test_yaml_reference_parity.py`
  verifies the YAML matches the reference bit-for-bit anyway).
- Discovered theory is loaded from a YAML via `Theory.from_yaml(...)`,
  i.e. exactly the artifact the AutoPi loop hands you at end of run.
- Each "subject" gets a fresh trial ordering (per Shepard type) and a
  fresh parameter draw from the relevant YAML prior; ground truth is
  drawn twice independently per subject so JSD_floor captures the
  natural variability of the model family under that prior.
- Per-trial JSD is in nats, bounded by ln 2.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import jensenshannon

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domains.category_learning.reference_models import (  # noqa: E402
    ReferenceGCM,
    ReferenceRULEX,
    ReferenceSUSTAIN,
)
from src.theory import Theory  # noqa: E402


YAML_DIR = _REPO_ROOT / "theories" / "category_learning"


# ---------------------------------------------------------------------------
# Shepard battery: 8 binary-feature stimuli, 6 logical category structures.
# ---------------------------------------------------------------------------

SHEPARD_STIMULI: np.ndarray = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=int,
)

SHJ_TYPE_LABELS: dict[str, np.ndarray] = {
    "I":   np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
    "II":  np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=int),
    "III": np.array([0, 1, 0, 0, 1, 0, 1, 1], dtype=int),
    "IV":  np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=int),
    "V":   np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int),
    "VI":  np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=int),
}

REFERENCE_REGISTRY: dict[str, type] = {
    "gcm": ReferenceGCM,
    "sustain": ReferenceSUSTAIN,
    "rulex": ReferenceRULEX,
}

INT_PARAM_KEYS: tuple[str, ...] = (
    "n_categories",
    "n_simulations",
    "max_search_steps",
    "seed",
)


# ---------------------------------------------------------------------------
# Trial-sequence sampling.
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One subject's full Shepard timeline.

    `stimuli[t]` is the binary feature vector shown on trial t and
    `labels[t]` is the correct category for that trial. Trials are
    grouped into blocks of 8 (one of each unique stimulus) and randomly
    permuted within each block.
    """

    stimuli: np.ndarray  # (T, n_features), int
    labels: np.ndarray   # (T,), int


def sample_trajectory(
    rng: np.random.Generator, *, type_name: str, n_blocks: int = 4
) -> Trajectory:
    if type_name not in SHJ_TYPE_LABELS:
        raise ValueError(
            f"Unknown SHJ type {type_name!r}; expected one of "
            f"{list(SHJ_TYPE_LABELS)}."
        )
    labels = SHJ_TYPE_LABELS[type_name]
    n_stim = SHEPARD_STIMULI.shape[0]
    stim_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for _ in range(int(n_blocks)):
        order = rng.permutation(n_stim)
        stim_blocks.append(SHEPARD_STIMULI[order])
        label_blocks.append(labels[order])
    return Trajectory(
        stimuli=np.concatenate(stim_blocks, axis=0).astype(int),
        labels=np.concatenate(label_blocks, axis=0).astype(int),
    )


# ---------------------------------------------------------------------------
# Replay: walk a trajectory and record per-trial choice probabilities.
# ---------------------------------------------------------------------------


PredictFn = Callable[[dict[str, Any], np.ndarray, dict[str, Any]], np.ndarray]


def predict_trajectory(
    predict_fn: PredictFn, params: dict[str, Any], trajectory: Trajectory
) -> np.ndarray:
    """Replay `trajectory` through `predict_fn`, growing the history one
    trial at a time. Returns a (T, n_categories) probability matrix.

    The history dict carries BOTH key conventions used in this codebase:
    - production (`stimulus` / `label`) — read by YAML predict functions.
    - legacy (`previous_stimuli` / `previous_labels`) — read by the
      paper-faithful reference models.
    Either implementation can therefore be driven by this single helper.
    """
    history: dict[str, list[Any]] = {
        "stimulus": [],
        "label": [],
        "previous_stimuli": [],
        "previous_labels": [],
    }
    preds: list[np.ndarray] = []
    for t in range(len(trajectory.labels)):
        stim_t = np.asarray(trajectory.stimuli[t], dtype=float)
        probs = np.asarray(predict_fn(params, stim_t, history), dtype=float)
        preds.append(probs)
        stim_list = list(map(int, trajectory.stimuli[t].tolist()))
        lab_int = int(trajectory.labels[t])
        history["stimulus"].append(stim_list)
        history["previous_stimuli"].append(stim_list)
        history["label"].append(lab_int)
        history["previous_labels"].append(lab_int)
    return np.asarray(preds, dtype=float)


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence (scipy returns the DISTANCE; we square it to
# get the divergence, which is what the spec asks for).
# ---------------------------------------------------------------------------


def trajectory_jsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean per-trial Jensen-Shannon divergence in nats. Bounded by ln 2."""
    P_arr = np.asarray(P, dtype=float)
    Q_arr = np.asarray(Q, dtype=float)
    if P_arr.shape != Q_arr.shape:
        raise ValueError(
            f"P and Q must share shape; got {P_arr.shape} vs {Q_arr.shape}."
        )
    if P_arr.ndim == 1:
        P_arr = P_arr[None, :]
        Q_arr = Q_arr[None, :]
    per_trial: list[float] = []
    for p, q in zip(P_arr, Q_arr):
        # `jensenshannon` returns the JS distance (sqrt of divergence).
        # base=None gives natural log, so distance**2 ∈ [0, ln 2].
        d = jensenshannon(p, q, base=None)
        if np.isnan(d):
            d = 0.0
        per_trial.append(float(d) * float(d))
    return float(np.mean(per_trial))


# ---------------------------------------------------------------------------
# Parameter sampling: one Theory.sample_parameters call, then int coercion
# and (optionally) overrides for RNG seed and RULEX n_simulations.
# ---------------------------------------------------------------------------


def _sample_params(
    theory: Theory,
    *,
    n_features: int,
    n_categories: int,
    seed_override: int | None = None,
    rulex_n_simulations: int | None = None,
) -> dict[str, Any]:
    raw = theory.sample_parameters(
        {"n_features": n_features, "n_labels": n_categories}
    )
    p: dict[str, Any] = dict(raw)
    for k in INT_PARAM_KEYS:
        if k in p and p[k] is not None:
            p[k] = int(p[k])
    if rulex_n_simulations is not None and "n_simulations" in p:
        p["n_simulations"] = int(rulex_n_simulations)
    if seed_override is not None and "seed" in p:
        p["seed"] = int(seed_override)
    return p


# ---------------------------------------------------------------------------
# Score container.
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    """Per-Shepard-type recovery scores.

    `*_subjects` are length-`n_subjects` arrays of subject-level mean
    JSDs; the scalar means/SEMs are derived from them.
    """

    type_name: str
    jsd_gt_subjects: np.ndarray
    jsd_floor_subjects: np.ndarray
    jsd_baseline_subjects: np.ndarray
    jsd_baseline_per_family_subjects: dict[str, np.ndarray] = field(
        default_factory=dict
    )

    @property
    def jsd_gt(self) -> float:
        return float(self.jsd_gt_subjects.mean())

    @property
    def jsd_floor(self) -> float:
        return float(self.jsd_floor_subjects.mean())

    @property
    def jsd_baseline(self) -> float:
        return float(self.jsd_baseline_subjects.mean())

    def sem(self, attr: str) -> float:
        arr = getattr(self, attr + "_subjects")
        n = len(arr)
        if n < 2:
            return 0.0
        return float(arr.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
# The orchestrator.
# ---------------------------------------------------------------------------


def score_recovery(
    *,
    discovered_yaml: str | Path,
    ground_truth_name: str,
    n_subjects: int = 20,
    n_blocks: int = 4,
    base_seed: int = 0,
    type_names: list[str] | None = None,
    rulex_n_simulations: int | None = None,
) -> dict[str, TaskScore]:
    """Run the recovery battery and return one `TaskScore` per Shepard type.

    Parameters
    ----------
    discovered_yaml : path to a Theory YAML (the AutoPi run winner).
    ground_truth_name : one of {"gcm", "sustain", "rulex"}.
    n_subjects : how many independent simulated subjects per type. Each
        subject contributes one mean-per-trial JSD value.
    n_blocks : number of training blocks per subject (×8 trials per block).
    base_seed : reproducibility seed; the same value yields identical scores.
    type_names : subset of SHJ types to score; defaults to all six.
    rulex_n_simulations : if set, overrides any sampled `n_simulations`
        param when a RULEX model is used (caller controls Monte-Carlo
        cost without changing the YAML).
    """
    if ground_truth_name not in REFERENCE_REGISTRY:
        raise ValueError(
            f"ground_truth_name must be one of {list(REFERENCE_REGISTRY)}; "
            f"got {ground_truth_name!r}."
        )

    type_names = list(type_names or SHJ_TYPE_LABELS.keys())
    n_features = int(SHEPARD_STIMULI.shape[1])
    n_categories = 2

    yaml_priors: dict[str, Theory] = {
        name: Theory.from_yaml(YAML_DIR / f"{name}.yaml")
        for name in REFERENCE_REGISTRY
    }
    discovered = Theory.from_yaml(Path(discovered_yaml))

    other_names = [n for n in REFERENCE_REGISTRY if n != ground_truth_name]
    gt_cls = REFERENCE_REGISTRY[ground_truth_name]
    other_cls = {n: REFERENCE_REGISTRY[n] for n in other_names}

    # Theory.sample_parameters reaches the stdlib `random` module under the
    # hood; seeding once makes the entire battery reproducible per
    # (base_seed, n_subjects, n_blocks).
    random.seed(base_seed)

    out: dict[str, TaskScore] = {}
    for type_name in type_names:
        gt_arr = np.empty(n_subjects)
        floor_arr = np.empty(n_subjects)
        baseline_arr = np.empty(n_subjects)
        baseline_per_family = {n: np.empty(n_subjects) for n in other_names}

        for s in range(n_subjects):
            traj = sample_trajectory(
                np.random.default_rng(base_seed + 1000 * s + hash(type_name) % 997),
                type_name=type_name,
                n_blocks=n_blocks,
            )

            gt_params = _sample_params(
                yaml_priors[ground_truth_name],
                n_features=n_features,
                n_categories=n_categories,
                seed_override=base_seed + 100_000 + s,
                rulex_n_simulations=rulex_n_simulations,
            )
            gt_params_alt = _sample_params(
                yaml_priors[ground_truth_name],
                n_features=n_features,
                n_categories=n_categories,
                seed_override=base_seed + 200_000 + s,
                rulex_n_simulations=rulex_n_simulations,
            )
            disc_params = _sample_params(
                discovered,
                n_features=n_features,
                n_categories=n_categories,
                seed_override=base_seed + 300_000 + s,
                rulex_n_simulations=rulex_n_simulations,
            )

            gt_pred = predict_trajectory(
                gt_cls(parameters=dict(gt_params)).predict, gt_params, traj
            )
            gt_pred_alt = predict_trajectory(
                gt_cls(parameters=dict(gt_params_alt)).predict,
                gt_params_alt,
                traj,
            )
            disc_pred = predict_trajectory(discovered.predict, disc_params, traj)

            other_jsds_this_subj: dict[str, float] = {}
            for n in other_names:
                op = _sample_params(
                    yaml_priors[n],
                    n_features=n_features,
                    n_categories=n_categories,
                    seed_override=base_seed + 400_000 + s + 13 * (hash(n) % 100),
                    rulex_n_simulations=rulex_n_simulations,
                )
                other_pred = predict_trajectory(
                    other_cls[n](parameters=dict(op)).predict, op, traj
                )
                other_jsds_this_subj[n] = trajectory_jsd(disc_pred, other_pred)
                baseline_per_family[n][s] = other_jsds_this_subj[n]

            gt_arr[s] = trajectory_jsd(disc_pred, gt_pred)
            floor_arr[s] = trajectory_jsd(gt_pred, gt_pred_alt)
            baseline_arr[s] = float(min(other_jsds_this_subj.values()))

        out[type_name] = TaskScore(
            type_name=type_name,
            jsd_gt_subjects=gt_arr,
            jsd_floor_subjects=floor_arr,
            jsd_baseline_subjects=baseline_arr,
            jsd_baseline_per_family_subjects=baseline_per_family,
        )

    return out


# ---------------------------------------------------------------------------
# Verdict + plot.
# ---------------------------------------------------------------------------


def recovery_verdict(
    scores: dict[str, TaskScore], *, eps: float = 0.05
) -> dict[str, bool]:
    """Per-type pass/fail according to the headline rule from the spec:

        recovery succeeded iff JSD_gt ≤ JSD_floor + eps  AND  JSD_gt < JSD_baseline.
    """
    return {
        name: bool(
            s.jsd_gt <= s.jsd_floor + eps and s.jsd_gt < s.jsd_baseline
        )
        for name, s in scores.items()
    }


def plot_recovery_battery(
    scores: dict[str, TaskScore],
    *,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Per-type bar chart with SEM error bars. Saves PNG and a sibling PDF
    so the figure can be embedded directly in the LaTeX paper."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    types = list(scores.keys())
    n = len(types)
    x = np.arange(n)
    width = 0.27

    def stats(attr: str) -> tuple[np.ndarray, np.ndarray]:
        means = np.array([getattr(scores[t], attr) for t in types])
        sems = np.array([scores[t].sem(attr) for t in types])
        return means, sems

    gt_m, gt_e = stats("jsd_gt")
    fl_m, fl_e = stats("jsd_floor")
    bl_m, bl_e = stats("jsd_baseline")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.titlesize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    bars_gt = ax.bar(
        x - width, gt_m, width, yerr=gt_e, capsize=3,
        label=r"JSD$_{\mathrm{gt}}$ (discovered $\Vert$ ground-truth)",
        color="#3878c4", edgecolor="black", linewidth=0.4,
    )
    bars_fl = ax.bar(
        x, fl_m, width, yerr=fl_e, capsize=3,
        label=r"JSD$_{\mathrm{floor}}$ (ground truth, paired draws)",
        color="#9aa2ad", edgecolor="black", linewidth=0.4,
    )
    bars_bl = ax.bar(
        x + width, bl_m, width, yerr=bl_e, capsize=3,
        label=r"JSD$_{\mathrm{baseline}}$ (closest wrong family)",
        color="#cc4135", edgecolor="black", linewidth=0.4,
    )

    # Mark types where the verdict passes (JSD_gt ≤ JSD_floor + 0.05 AND
    # < JSD_baseline) with a small tick in the x-axis label.
    verdict = recovery_verdict(scores)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Type {t}{'  ✓' if verdict[t] else ''}" for t in types]
    )
    ax.set_ylabel("Mean per-trial Jensen–Shannon divergence (nats)")
    ax.set_xlabel("Shepard / SHJ category structure")

    # Auto-scale y to the data (with headroom for the legend); annotate
    # the theoretical ln 2 cap in the caption instead of forcing the
    # axis up to it (otherwise sub-0.1 nat differences get squashed).
    data_top = float(np.max(np.concatenate([gt_m + gt_e, fl_m + fl_e, bl_m + bl_e])))
    upper = max(0.05, data_top * 1.6)
    ax.set_ylim(0, upper)
    ax.set_yticks(np.linspace(0, upper, 6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Numeric labels above each bar (rounded to 3 dp; empirical JSDs on
    # this battery are O(0.01-0.1)).
    for bars, vals in ((bars_gt, gt_m), (bars_fl, fl_m), (bars_bl, bl_m)):
        for rect, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color="0.25",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=1)

    if title:
        ax.set_title(title)

    # Caption-style note pinned to the figure (not the axes), so it sits
    # below the x-axis label without overlapping. constrained_layout
    # reserves space for it because we add it before saving.
    fig.suptitle("")  # ensure suptitle slot is empty
    ax.text(
        1.0, -0.18,
        r"Per-trial JSD bounded by $\ln 2 \approx 0.693$ nats;  "
        r"error bars are SEM across simulated subjects;  "
        r"$\checkmark$ marks recovery verdict pass.",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color="0.4",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Serialization (so the paper plot is reproducible from cached numbers).
# ---------------------------------------------------------------------------


def scores_to_json(scores: dict[str, TaskScore]) -> dict[str, Any]:
    return {
        name: {
            "jsd_gt": s.jsd_gt,
            "jsd_floor": s.jsd_floor,
            "jsd_baseline": s.jsd_baseline,
            "sem_gt": s.sem("jsd_gt"),
            "sem_floor": s.sem("jsd_floor"),
            "sem_baseline": s.sem("jsd_baseline"),
            "jsd_gt_subjects": s.jsd_gt_subjects.tolist(),
            "jsd_floor_subjects": s.jsd_floor_subjects.tolist(),
            "jsd_baseline_subjects": s.jsd_baseline_subjects.tolist(),
            "jsd_baseline_per_family_subjects": {
                k: v.tolist() for k, v in s.jsd_baseline_per_family_subjects.items()
            },
        }
        for name, s in scores.items()
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _format_summary_table(
    scores: dict[str, TaskScore], verdict: dict[str, bool]
) -> str:
    rows = [
        f"{'Type':<6} {'JSD_gt':>9} {'JSD_floor':>10} {'JSD_baseline':>13} "
        f"{'verdict':>9}",
        "-" * 55,
    ]
    for t, s in scores.items():
        rows.append(
            f"{t:<6} {s.jsd_gt:>9.4f} {s.jsd_floor:>10.4f} "
            f"{s.jsd_baseline:>13.4f} {('PASS' if verdict[t] else 'fail'):>9}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Behavioural-cloning JSD on the Shepard I-VI battery: how well "
            "does a discovered theory recover the ground-truth model family?"
        )
    )
    p.add_argument(
        "--discovered",
        required=True,
        type=Path,
        help="Path to the discovered theory YAML (the AutoPi run winner).",
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        choices=sorted(REFERENCE_REGISTRY.keys()),
        help="Ground-truth model family for this run.",
    )
    p.add_argument("--n-subjects", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rulex-n-simulations",
        type=int,
        default=100,
        help="Override RULEX `n_simulations` for cost control (default 100).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results" / "recovery_battery",
        help="Output directory for the plot and JSON.",
    )
    p.add_argument("--eps", type=float, default=0.05, help="Floor slack for verdict.")
    args = p.parse_args(argv)

    print(
        f"[recovery_battery] discovered={args.discovered} "
        f"ground_truth={args.ground_truth} "
        f"n_subjects={args.n_subjects} n_blocks={args.n_blocks} seed={args.seed}"
    )

    scores = score_recovery(
        discovered_yaml=args.discovered,
        ground_truth_name=args.ground_truth,
        n_subjects=args.n_subjects,
        n_blocks=args.n_blocks,
        base_seed=args.seed,
        rulex_n_simulations=args.rulex_n_simulations,
    )
    verdict = recovery_verdict(scores, eps=args.eps)

    print(_format_summary_table(scores, verdict))
    n_pass = sum(verdict.values())
    print(
        f"\n[recovery_battery] {n_pass}/{len(verdict)} Shepard types pass "
        f"(JSD_gt ≤ JSD_floor + {args.eps:g} and JSD_gt < JSD_baseline)."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    plot_path = args.out / f"recovery_battery_{args.ground_truth}.png"
    plot_recovery_battery(
        scores,
        out_path=plot_path,
        title=(
            f"Behavioural-cloning JSD on Shepard I–VI  ·  "
            f"ground truth: {args.ground_truth.upper()}  ·  "
            f"discovered: {Path(args.discovered).stem}  ·  "
            f"N={args.n_subjects} subj × {args.n_blocks} blocks"
        ),
    )
    json_path = args.out / f"recovery_battery_{args.ground_truth}.json"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "discovered": str(args.discovered),
                    "ground_truth": args.ground_truth,
                    "n_subjects": args.n_subjects,
                    "n_blocks": args.n_blocks,
                    "seed": args.seed,
                    "rulex_n_simulations": args.rulex_n_simulations,
                    "eps": args.eps,
                },
                "verdict": verdict,
                "scores": scores_to_json(scores),
            },
            indent=2,
        )
    )
    print(f"[recovery_battery] wrote {plot_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
