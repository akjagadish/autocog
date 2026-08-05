"""Evaluate base + surfaced models from an autopi run against humans on
Hilbig (2014) Exp 1, in stimulus choice-proportion space.

For each unique (option_a, option_b) pair the human dataset contains, we
compute two empirical proportions of "choose B":

  p_b_pooled            — pool every trial across all participants.
  p_b_within_subj_mean  — average each participant's per-pair P(B), then
                          mean across participants.

The two diverge whenever participants are unbalanced (different numbers
of trials on the same pair, different per-pair preferences). We report
both because the right one to compare against depends on the question
being asked.

Each base / surfaced theory from `--run-dir` is replayed on the same
unique stimulus pairs:

  - validities are pinned to the human-task vector ([0.9, 0.8, 0.7, 0.6])
    regardless of what the autopi run was tuned on, because that is the
    cue ordering subjects were told.
  - rating_max=1 because the human ratings are binary; theories whose
    sample_parameters declare `rating_max` get pinned too so any
    parameter that scales with rating_max stays consistent with the
    inputs they're being fed.
  - n_features must match (4) — replaying e.g. a 3-feature surfaced
    theory on 4-feature stimuli is meaningless. The script aborts with
    a clear message rather than silently truncating.

For each replayed theory we compute its predicted P(B) per stimulus
pair (mean across `--n-subjects` parameter draws), then summarise the
fit to humans via MAE, MSE, and Pearson r over the per-stimulus
vectors. With `--all-explored`, every unique theory the run ever
entertained across all rounds is replayed (deduped by label) and
partitioned into base / intermediate / surfaced by its final fate,
instead of just round-0 base + last-round surfaced. With
`--include-canonical`, every YAML in
`theories/heuristic_decision_making/` (TTB / WADD / Tallying / EW)
is replayed alongside as a fixed-baseline reference under the role
`canonical`. With `--extra-yaml PATH [PATH ...]`, additional YAML
files and/or directories of YAMLs are loaded as theories under the
role `extra` — useful for comparing against ad-hoc baselines or
theories from other runs. Outputs (under `--out`, default
`<run-dir>/analysis/eval_hilbig/`):

  eval_hilbig_per_stimulus.csv   — long-format (model, role, stimulus, p_b_human, p_b_model)
  eval_hilbig_summary.csv        — per-model MAE/MSE/Pearson r vs humans
  eval_hilbig_scatter.png        — multi-panel human-vs-model P(B) scatter grid
  eval_hilbig_scatter_<model>.png — one standalone scatter per model
  eval_hilbig_bars_<metric>.png  — bar chart per metric (mae, mse, pearson_r)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.theory import Theory  # noqa: E402
from scripts.figure_style import (  # noqa: E402
    CYCLE,
    FONTSIZE,
    GRAY,
    save_figure,
    style_axes,
)


HUMAN_VALIDITIES: list[float] = [0.9, 0.8, 0.7, 0.6]
HUMAN_RATING_MAX: int = 1
HUMAN_DATA_DEFAULT: Path = _REPO_ROOT / "results" / "heuristic_decision_making" / "hilbig2014"/ "exp1.txt"
CANONICAL_YAML_DIR: Path = _REPO_ROOT / "theories" / "heuristic_decision_making"

# Mirrors `scripts.recovery_battery.INT_PARAM_KEYS`. Inlined so this script
# stays importable when recovery_battery's heavier deps (e.g. `domains.*`)
# aren't installed — the rest of recovery_battery's surface isn't needed here.
_INT_PARAM_KEYS: tuple[str, ...] = (
    "n_categories",
    "n_simulations",
    "max_search_steps",
    "seed",
)


# ---------------------------------------------------------------------------
# Inlined hilibig_battery / recovery_battery helpers (Theory parameter
# sampling, autopi run-dir traversal). Inlined to keep this script
# self-contained in environments where the wider battery deps don't
# install cleanly. Behavior matches `scripts.hilibig_battery` 1:1.
# ---------------------------------------------------------------------------


def _sample_hilibig_params(
    theory: Theory,
    *,
    validities: list[float],
    n_features: int,
    rating_max: int | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Sample one subject's parameters; pin `validities` and (if declared)
    `rating_max` to the task constants subjects were told."""
    context: dict[str, Any] = {
        "n_features": int(n_features),
        "n_labels": 2,
        "validities": list(validities),
    }
    if rating_max is not None:
        context["rating_max"] = int(rating_max)

    raw = theory.sample_parameters(context)
    p: dict[str, Any] = dict(raw)
    for k in _INT_PARAM_KEYS:
        if k in p and p[k] is not None:
            p[k] = int(p[k])
    if seed_override is not None and "seed" in p:
        p["seed"] = int(seed_override)
    if "validities" in p:
        p["validities"] = list(validities)
    if rating_max is not None and "rating_max" in p:
        p["rating_max"] = int(rating_max)
    return p


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


def resolve_base_theories(run_dir: Path) -> dict[str, Theory]:
    """Theories the run started with, keyed by their stable label."""
    first = _round_dirs(run_dir)[0]
    data = json.loads((first / "theories.json").read_text())
    out: dict[str, Theory] = {}
    for s in data.get("starting_theories", []):
        out[s["label"]] = Theory.model_validate(s["theory"])
    return out


def resolve_surfaced_theories(run_dir: Path) -> dict[str, Theory]:
    """Theories that survived to the last round (un-killed) plus the
    final replacement, if any."""
    last = _round_dirs(run_dir)[-1]
    data = json.loads((last / "theories.json").read_text())
    out: dict[str, Theory] = {}
    for s in data.get("starting_theories", []):
        if not s.get("killed", False):
            out[s["label"]] = Theory.model_validate(s["theory"])
    repl = data.get("replacement")
    if repl is not None:
        out[repl["label"]] = Theory.model_validate(repl["theory"])
    return out


def resolve_all_explored_theories(
    run_dir: Path,
) -> tuple[dict[str, Theory], dict[str, Theory], dict[str, Theory]]:
    """Every theory the run ever entertained, deduped by label and
    partitioned by their final fate:

      base         — present in round 0 and *did not* survive to the end
      intermediate — proposed mid-run as a replacement but later killed
      surfaced     — un-killed in the last round, or the last replacement

    A theory present from round 0 that survives to the end is tagged
    `surfaced` rather than `base`: the fact it weathered every round is
    the more informative label. Each label therefore appears in exactly
    one of the three returned dicts.
    """
    rounds = _round_dirs(run_dir)
    all_theories: dict[str, Theory] = {}
    for r in rounds:
        data = json.loads((r / "theories.json").read_text())
        for s in data.get("starting_theories", []):
            all_theories.setdefault(
                s["label"], Theory.model_validate(s["theory"]),
            )
        repl = data.get("replacement")
        if repl is not None:
            all_theories.setdefault(
                repl["label"], Theory.model_validate(repl["theory"]),
            )

    first_data = json.loads((rounds[0] / "theories.json").read_text())
    last_data = json.loads((rounds[-1] / "theories.json").read_text())
    base_labels = {s["label"] for s in first_data.get("starting_theories", [])}
    surfaced_labels: set[str] = set()
    for s in last_data.get("starting_theories", []):
        if not s.get("killed", False):
            surfaced_labels.add(s["label"])
    repl = last_data.get("replacement")
    if repl is not None:
        surfaced_labels.add(repl["label"])

    base_only: dict[str, Theory] = {}
    intermediate: dict[str, Theory] = {}
    surfaced: dict[str, Theory] = {}
    for label, th in all_theories.items():
        if label in surfaced_labels:
            surfaced[label] = th
        elif label in base_labels:
            base_only[label] = th
        else:
            intermediate[label] = th
    return base_only, intermediate, surfaced


def load_canonical_theories(
    yaml_dir: Path = CANONICAL_YAML_DIR,
) -> dict[str, Theory]:
    """Load every `*.yaml` under `yaml_dir` as a `Theory`, keyed by the
    file stem (e.g. `ttb`, `wadd`, `tallying`, `ew`). These are the
    paper-faithful canonical heuristics — useful as a fixed reference
    baseline alongside whatever the autopi run discovered."""
    if not yaml_dir.is_dir():
        raise FileNotFoundError(f"No canonical YAML dir at {yaml_dir!s}")
    out: dict[str, Theory] = {}
    for path in sorted(yaml_dir.glob("*.yaml")):
        out[path.stem] = Theory.from_yaml(path)
    return out


def load_yaml_theories(paths: list[Path]) -> dict[str, Theory]:
    """Load theories from a mix of YAML files and/or directories. Files
    are loaded individually; directories contribute every `*.yaml` they
    contain. Each theory is keyed by its filename stem. Duplicate stems
    across the inputs are an error — the caller would silently lose one
    of the colliding theories otherwise."""
    out: dict[str, Theory] = {}
    for p in paths:
        if p.is_dir():
            yaml_files = sorted(p.glob("*.yaml"))
            if not yaml_files:
                raise FileNotFoundError(f"No *.yaml files under {p!s}")
        elif p.is_file():
            yaml_files = [p]
        else:
            raise FileNotFoundError(f"Path does not exist: {p!s}")
        for yp in yaml_files:
            stem = yp.stem
            if stem in out:
                raise ValueError(
                    f"Duplicate theory label {stem!r} from {yp!s}; "
                    f"already loaded from a previous --extra-yaml input."
                )
            out[stem] = Theory.from_yaml(yp)
    return out


def _last_round_design(run_dir: Path) -> dict[str, Any] | None:
    """Pull (validities, rating_max) from the run's last round, if present.
    Returns None when the run-dir has no observations/state.json."""
    state_path = run_dir / "observations" / "state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text())
    rounds = state.get("rounds") or []
    if not rounds:
        return None
    last = rounds[-1]
    obs_list = last.get("observations") or []
    for obs in obs_list:
        exp = obs.get("experiment")
        if isinstance(exp, dict) and "validities" in exp:
            return {
                "validities": [float(v) for v in exp["validities"]],
                "rating_max": int(exp.get("rating_max", 1)),
            }
    return None


# ---------------------------------------------------------------------------
# Human data: parse exp1.txt and aggregate stimulus-level choice proportions.
# ---------------------------------------------------------------------------


def _parse_array_string(s: str) -> tuple[int, ...]:
    """Parse `'[0 1 1 1]'` (numpy str-formatted array) into `(0, 1, 1, 1)`."""
    cleaned = s.strip().strip("[]").strip()
    if not cleaned:
        return ()
    return tuple(int(x) for x in cleaned.split())


def load_human_choices(path: Path = HUMAN_DATA_DEFAULT) -> pd.DataFrame:
    """Return a DataFrame with `participant`, `trial`, `choice` (0=A / 1=B),
    `option_a` and `option_b` (each a tuple of ints)."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "participant": df["participant"].astype(int),
        "trial": df["trial"].astype(int),
        "choice": df["choice"].astype(int),
        "option_a": df["stimulus_0"].map(_parse_array_string),
        "option_b": df["stimulus_1"].map(_parse_array_string),
    })
    return out


def compute_human_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Per (option_a, option_b) pair, both pooled and within-participant
    "choose B" proportions. Returned columns:

        option_a, option_b
        n_trials             — total trials across all participants
        n_choose_b           — total choose-B responses
        n_participants       — distinct participants who saw the pair
        p_b_pooled           — n_choose_b / n_trials
        p_b_within_subj_mean — mean over participants of their per-pair P(B)
    """
    rows = []
    for (a, b), g in df.groupby(["option_a", "option_b"], sort=True):

        per_subj = g.groupby("participant")["choice"].mean()
        rows.append({
            "option_a": a,
            "option_b": b,
            "n_trials": int(len(g)),
            "n_choose_b": int(g["choice"].sum()),
            "n_participants": int(per_subj.size),
            "p_b_pooled": float(g["choice"].mean()),
            "p_b_within_subj_mean": float(per_subj.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model replay: predict P(B) per stimulus pair, averaged across subjects.
# ---------------------------------------------------------------------------


def _empty_history() -> dict[str, list[Any]]:
    """Both legacy and current history-key conventions, both empty.
    Stateless heuristics ignore this; surfaced theories that read
    history get well-defined empty inputs."""
    return {
        "stimulus": [], "label": [],
        "previous_stimuli": [], "previous_labels": [],
    }


def _predict_p_b(
    theory: Theory,
    params: dict[str, Any],
    option_a: tuple[int, ...],
    option_b: tuple[int, ...],
) -> float:
    stim = np.asarray([list(option_a), list(option_b)], dtype=float)
    p = np.asarray(theory.predict(params, stim, _empty_history()), dtype=float)
    if p.shape != (2,):
        raise ValueError(
            f"Theory.predict returned shape {p.shape}; expected (2,)."
        )
    return float(p[1])


def model_proportions_for_theory(
    theory: Theory,
    *,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    validities: list[float],
    rating_max: int,
    n_subjects: int,
    base_seed: int,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """Across `n_subjects` parameter draws, return mean predicted P(B)
    per unique stimulus pair. Determinism: stdlib `random` is seeded
    once before sampling all subjects, mirroring `hilibig_battery.py`."""
    n_features = len(validities)
    random.seed(base_seed)

    probs = np.empty((n_subjects, len(stimulus_pairs)), dtype=float)
    for s in range(n_subjects):
        params = _sample_hilibig_params(
            theory,
            validities=validities,
            rating_max=rating_max,
            n_features=n_features,
            seed_override=base_seed + s,
        )
        for j, (a, b) in enumerate(stimulus_pairs):
            probs[s, j] = _predict_p_b(theory, params, a, b)

    means = probs.mean(axis=0)
    return {pair: float(m) for pair, m in zip(stimulus_pairs, means)}


# ---------------------------------------------------------------------------
# Top-level evaluation: combine humans + base + surfaced into per-stimulus
# and per-model summary tables.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelSpec:
    label: str
    role: str          # "base" | "intermediate" | "surfaced" | "canonical" | "extra"
    theory: Theory


def evaluate_models(
    *,
    base: dict[str, Theory],
    surfaced: dict[str, Theory],
    human_props: pd.DataFrame,
    validities: list[float],
    rating_max: int,
    n_subjects: int,
    base_seed: int,
    target_col: str = "p_b_pooled",
    intermediate: dict[str, Theory] | None = None,
    canonical: dict[str, Theory] | None = None,
    extra: dict[str, Theory] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every model on the human stimulus pairs, return
    (per_stimulus_df, summary_df). `target_col` selects which human
    aggregate to compare against (`p_b_pooled` or `p_b_within_subj_mean`).
    `intermediate` is an optional dict of theories that were explored
    mid-run but later killed (populated by `--all-explored`) — included
    with role='intermediate'. `canonical` is an optional dict of
    paper-faithful baseline theories (e.g. TTB / WADD / Tallying / EW) —
    included with role='canonical'. `extra` is an optional dict of
    additional user-supplied theories (loaded via `--extra-yaml`) —
    included with role='extra'."""
    if target_col not in {"p_b_pooled", "p_b_within_subj_mean"}:
        raise ValueError(f"target_col {target_col!r} not supported.")

    pairs = list(zip(human_props["option_a"], human_props["option_b"]))
    human_p = human_props[target_col].to_numpy(dtype=float)

    plan: list[_ModelSpec] = []
    for label, th in base.items():
        plan.append(_ModelSpec(label=label, role="base", theory=th))
    for label, th in (intermediate or {}).items():
        plan.append(_ModelSpec(label=label, role="intermediate", theory=th))
    for label, th in surfaced.items():
        plan.append(_ModelSpec(label=label, role="surfaced", theory=th))
    for label, th in (canonical or {}).items():
        plan.append(_ModelSpec(label=label, role="canonical", theory=th))
    for label, th in (extra or {}).items():
        plan.append(_ModelSpec(label=label, role="extra", theory=th))

    per_stim_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for m in plan:
        # Every model uses the SAME base_seed for parameter sampling so
        # the same theory (e.g. a seed that survives untouched and shows
        # up in both `base` and `surfaced`) produces identical
        # predictions in both rows — no fabricated Monte-Carlo gap. The
        # tradeoff: the s-th "subject" across two different theories
        # shares the seed_override, but stdlib `random.seed(base_seed)`
        # at the start of `model_proportions_for_theory` resets the RNG
        # state per call, so each theory sees its own prior-faithful
        # parameter draws.
        preds = model_proportions_for_theory(
            m.theory,
            stimulus_pairs=pairs,
            validities=validities,
            rating_max=rating_max,
            n_subjects=n_subjects,
            base_seed=base_seed,
        )
        model_p = np.asarray([preds[p] for p in pairs], dtype=float)

        for (a, b), p_h, p_m, n_h in zip(
            pairs, human_p, model_p, human_props["n_trials"].to_numpy(),
        ):
            per_stim_rows.append({
                "model": m.label, "role": m.role,
                "option_a": str(list(a)), "option_b": str(list(b)),
                "p_b_human": float(p_h), "p_b_model": float(p_m),
                "n_trials_human": int(n_h),
            })

        diff = model_p - human_p
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff * diff))
        if model_p.std() > 0 and human_p.std() > 0:
            r = float(np.corrcoef(model_p, human_p)[0, 1])
        else:
            r = float("nan")
        summary_rows.append({
            "model": m.label, "role": m.role,
            "n_stimuli": len(pairs),
            "mae": mae, "mse": mse, "pearson_r": r,
        })

    per_stim_df = pd.DataFrame(per_stim_rows)
    summary_df = pd.DataFrame(summary_rows)
    return per_stim_df, summary_df


# ---------------------------------------------------------------------------
# Theory display names (for readable figure labels).
# ---------------------------------------------------------------------------

# Keyword signatures for the canonical seed heuristics, whose descriptions are
# prose without an explicit name. Checked only as a fallback.
_CANONICAL_NAME_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("one at a time in order of validity", "Take the Best"),
    ("counting, across all features", "Tallying"),
    ("weighted sum of its feature values", "Weighted Additive (WADD)"),
)


def theory_display_name(description: str, fallback: str) -> str:
    """Best-effort short, human-readable theory name from its description.

    Handles the common shapes the LLM produces:
      "Name (ACRONYM) theory posits ..."   -> "Name (ACRONYM)"
      "Name (Subtitle): rest ..."          -> "Name (Subtitle)"
      "Name posits/theory ..."             -> "Name"
    Falls back to canonical-heuristic keyword matching (TTB/Tallying), then to
    `fallback` (typically the pi_N label) when nothing matches.
    """
    import re

    desc = (description or "").strip()
    if not desc:
        return fallback

    # 1. "Heading: rest" where the heading reads like a title (no sentence
    #    punctuation, reasonable length).
    m = re.match(r"^([A-Z][^:.]{2,90}):\s", desc)
    if m:
        return m.group(1).strip()
    # 2. "Name (ACRONYM) ..." — keep the parenthetical acronym.
    m = re.match(r"^([A-Z][A-Za-z0-9 \-]{1,45}\([A-Za-z0-9 \-]{1,35}\))", desc)
    if m:
        return m.group(1).strip()
    # 3. "Name posits/theory ..."
    m = re.match(r"^([A-Z][A-Za-z0-9 \-]{2,45}?)\s+(?:theory\b|posits\b)", desc)
    if m:
        return m.group(1).strip()
    # 4. Canonical seed heuristics (prose descriptions, no explicit name).
    low = desc.lower()
    for needle, name in _CANONICAL_NAME_KEYWORDS:
        if needle in low:
            return name
    return fallback


# Display name for the data's role tags (the round-0 starting theories are
# the "seed" theories, stored with role "base").
_ROLE_DISPLAY: dict[str, str] = {"base": "seed"}


def _theory_label(name: str, role: str, *, width: int = 22) -> str:
    """Wrapped axis/title label: the (word-wrapped) theory name above a
    `[role]` tag, so long names stack vertically instead of running off."""
    import textwrap

    wrapped = textwrap.fill(name, width=width)
    return f"{wrapped}\n[{_ROLE_DISPLAY.get(role, role)}]"


def build_name_map(run_dir: Path) -> dict[str, str]:
    """Map each theory label (pi_N) in a run to a readable display name,
    reading every `rounds/round_*/theories.json` (first occurrence wins)."""
    import json

    name_map: dict[str, str] = {}
    rounds = sorted((run_dir / "rounds").glob("round_*/theories.json"))
    for rj in rounds:
        try:
            d = json.loads(rj.read_text())
        except (OSError, ValueError):
            continue
        items = list(d.get("starting_theories", []))
        if isinstance(d.get("replacement"), dict):
            items.append(d["replacement"])
        for it in items:
            label = it.get("label")
            desc = (it.get("theory") or {}).get("description", "")
            if label and label not in name_map:
                name_map[label] = theory_display_name(desc, label)
    return name_map


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------


def role_color_map(
    models: list[str], lineage_colors: dict[str, str],
) -> dict[str, Any]:
    """Map each model label to its lineage colour so every human-eval figure
    matches the convergence & trajectory palette: seeds → slate (indigo),
    the winning (headline) survivor → sandy (gold), other final survivor →
    terracotta, transient → gray. Labels absent from `lineage_colors` (e.g.
    canonical/extra baselines that never entered the run) fall back to the
    categorical `CYCLE` in enumeration order."""
    return {
        m: lineage_colors.get(m, CYCLE[i % len(CYCLE)])
        for i, m in enumerate(models)
    }


def _lineage_color_map(run_dir: Path) -> dict[str, str]:
    """`{pi_label: hex}` from the run's lineage (single source of truth in
    `plot_autopi_convergence.theory_colors`). Returns `{}` if the lineage
    can't be parsed, so callers degrade to the categorical `CYCLE`."""
    try:
        from scripts.plot_autopi_convergence import theory_colors
        return theory_colors(run_dir)
    except Exception as e:  # lineage parsing is best-effort for colouring
        print(
            f"[eval_hilbig] WARNING: could not derive lineage colours "
            f"({e}); falling back to categorical palette.",
            file=sys.stderr,
        )
        return {}


def _draw_scatter(
    ax, sub: pd.DataFrame, color: Any, disp: str, role: str, *,
    show_title: bool = True,
) -> None:
    """Draw one model's human-vs-model P(B) scatter (plus the y=x line) onto
    `ax`. Shared by the multi-panel grid and the per-model figures so they
    stay visually identical. `show_title=False` drops the per-panel model name
    (the colour already identifies the model)."""
    ax.scatter(
        sub["p_b_human"], sub["p_b_model"],
        color=color, alpha=1.0,
        edgecolor="white", linewidth=0.5, s=110,
    )
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax, xlabel=r"Human $\hat{p}(B)$", ylabel=r"Model $\hat{p}(B)$")
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE + 4)
    if show_title:
        ax.set_title(_theory_label(disp, role), fontsize=FONTSIZE - 4)


def plot_scatter(
    per_stim_df: pd.DataFrame, path: Path, *,
    title: str | None = None,
    color_map: dict[str, Any] | None = None,
    name_map: dict[str, str] | None = None,
    show_title: bool = True,
) -> None:
    """One subplot per model: human P(B) on x, model P(B) on y, plus y=x.
    `color_map` (model label → matplotlib color) is shared with `plot_bars`
    so the same model gets the same color in both figures. `show_title=False`
    drops the per-panel model-name titles (the colour identifies the model)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = per_stim_df["model"].drop_duplicates().tolist()
    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False,
    )
    if color_map is None:
        color_map = {m: CYCLE[i % len(CYCLE)] for i, m in enumerate(models)}

    for i, m in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = per_stim_df[per_stim_df["model"] == m]
        _draw_scatter(
            ax, sub, color_map.get(m, CYCLE[i % len(CYCLE)]),
            (name_map or {}).get(m, m), sub["role"].iloc[0],
            show_title=show_title,
        )

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_scatter_per_model(
    per_stim_df: pd.DataFrame, out_dir: Path, *,
    color_map: dict[str, Any] | None = None,
    name_map: dict[str, str] | None = None,
    stem: str = "eval_hilbig_scatter",
    show_title: bool = True,
) -> list[Path]:
    """Save each model's human-vs-model scatter as its own standalone figure
    (`{stem}_{model}.svg/.png` under `out_dir`), colour-matched to the grid
    and bar charts. `show_title=False` drops the per-panel model-name title.
    Returns the list of base paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = per_stim_df["model"].drop_duplicates().tolist()
    written: list[Path] = []
    for i, m in enumerate(models):
        sub = per_stim_df[per_stim_df["model"] == m]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        _draw_scatter(
            ax, sub, (color_map or {}).get(m, CYCLE[i % len(CYCLE)]),
            (name_map or {}).get(m, m), sub["role"].iloc[0],
            show_title=show_title,
        )
        fig.tight_layout()
        base = out_dir / f"{stem}_{m}.png"
        save_figure(fig, base)
        plt.close(fig)
        written.append(base)
    return written


def _bars_frame(
    summary_df: pd.DataFrame, metric: str, *, ascending: bool,
    exclude_models: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Rows to plot for a bar chart: drop `exclude_models`, then sort by
    `metric` (ascending for lower-is-better). Reset index so bar positions
    are contiguous."""
    df = summary_df[~summary_df["model"].isin(exclude_models)]
    return df.sort_values(metric, ascending=ascending).reset_index(drop=True)


def plot_bars(
    summary_df: pd.DataFrame, path: Path, *,
    metric: str = "mae", title: str | None = None,
    color_map: dict[str, Any] | None = None,
    name_map: dict[str, str] | None = None,
    ymin: float = 0.0, ymax: float | None = None,
    ascending: bool = True, exclude_models: tuple[str, ...] = (),
    show_model_labels: bool = True,
    yerr: dict[str, float] | None = None,
) -> None:
    """Bar chart: per-model `metric` (default MAE) vs humans.
    `color_map` (model label → matplotlib color) is shared with
    `plot_scatter` so the same model gets the same color in both figures.
    `ymin`/`ymax` fix the y-axis range (e.g. to zoom in on a band or to
    compare across runs); `ymax=None` keeps autoscaling. `ascending` orders
    the bars left-to-right (set `False` for higher-is-better metrics like
    Pearson r so the best model leads). `exclude_models` drops bars (e.g. a
    model not wanted in this chart); `show_model_labels=False` hides the
    per-bar name x-ticks (the colour identifies the model). `yerr`
    (model label → error) draws symmetric error bars (e.g. SEM)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _bars_frame(
        summary_df, metric, ascending=ascending, exclude_models=exclude_models,
    )
    if color_map is None:
        color_map = {m: CYCLE[i % len(CYCLE)] for i, m in enumerate(df["model"])}
    # Theory display names wrap to several lines; give each bar enough width
    # and a steep tick rotation so adjacent long labels don't collide.
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(df) + 2.5), 5.0))
    x = np.arange(len(df))
    ax.bar(
        x, df[metric].to_numpy(),
        color=[color_map.get(m, CYCLE[i % len(CYCLE)])
               for i, m in enumerate(df["model"])],
        yerr=(
            [yerr.get(m, float("nan")) for m in df["model"]]
            if yerr else None
        ),
        capsize=4, ecolor="black",
        error_kw=dict(elinewidth=1.2, capthick=1.2),
    )
    nm = name_map or {}
    ax.set_xticks(x)
    if show_model_labels:
        ax.set_xticklabels(
            [_theory_label(nm.get(m, m), r)
             for m, r in zip(df["model"], df["role"])],
            rotation=45, ha="right", fontsize=FONTSIZE - 6,
        )
    else:
        ax.set_xticklabels([])
    if ymax is not None:
        ax.set_ylim(ymin, ymax)
    style_axes(
        ax,
        ylabel=r"$MSE_{\hat{p}(B)}$" if metric == "mse" else metric.upper(),
    )
    if metric == "mse":  # only the MSE chart gets the enlarged fonts
        ax.yaxis.label.set_size(FONTSIZE + 8)
        ax.tick_params(axis="y", which="major", labelsize=FONTSIZE + 6)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def per_pair_squared_error(per_stim_df: pd.DataFrame) -> pd.DataFrame:
    """The input rows plus a `sq_err` column: each unique stimulus pair's
    (p_b_model - p_b_human)^2. Grouping by model and taking the mean
    reproduces the per-model MSE exactly (same definition as the summary:
    mean of diff^2)."""
    out = per_stim_df.copy()
    out["sq_err"] = (out["p_b_model"] - out["p_b_human"]) ** 2
    return out


def per_model_mse_sem(per_stim_df: pd.DataFrame) -> pd.Series:
    """Standard error of the mean of each model's per-pair squared errors,
    std(ddof=1) / sqrt(n_pairs) — the SEM of the MSE estimate across the
    unique stimulus pairs. Index = model."""
    g = per_pair_squared_error(per_stim_df).groupby("model")["sq_err"]
    return g.std(ddof=1) / np.sqrt(g.count())


def plot_mse_swarm(
    per_stim_df: pd.DataFrame, path: Path, *,
    color_map: dict[str, Any] | None = None,
    name_map: dict[str, str] | None = None,
    exclude_models: tuple[str, ...] = (),
    show_model_labels: bool = True,
) -> None:
    """`plot_bars` for MSE, but with the underlying per-pair squared errors
    overlaid as a swarm: each dot is one unique stimulus pair's
    (p_b_model - p_b_human)^2, and the bar height (their mean) is the MSE.
    Same lineage colours, ascending-MSE ordering, and large fonts as the MSE
    bar so the two figures stay visually matched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = per_pair_squared_error(
        per_stim_df[~per_stim_df["model"].isin(exclude_models)]
    )
    means = d.groupby("model")["sq_err"].mean().sort_values()  # lower is better
    order = means.index.tolist()
    roles = d.drop_duplicates("model").set_index("model")["role"].to_dict()
    if color_map is None:
        color_map = {m: CYCLE[i % len(CYCLE)] for i, m in enumerate(order)}
    colors = {
        m: color_map.get(m, CYCLE[i % len(CYCLE)]) for i, m in enumerate(order)
    }

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(order) + 2.5), 5.0))
    x = np.arange(len(order))
    ax.bar(
        x, [means[m] for m in order],
        color=[colors[m] for m in order], alpha=0.30, zorder=1,
    )
    sns.swarmplot(
        data=d, x="model", y="sq_err", order=order, hue="model",
        hue_order=order, palette=colors, legend=False, ax=ax,
        size=4.5, edgecolor="white", linewidth=0.5, zorder=2,
    )
    nm = name_map or {}
    ax.set_xticks(x)
    if show_model_labels:
        ax.set_xticklabels(
            [_theory_label(nm.get(m, m), roles.get(m, "")) for m in order],
            rotation=45, ha="right", fontsize=FONTSIZE - 6,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    style_axes(ax, ylabel=r"$MSE_{\hat{p}(B)}$")
    ax.yaxis.label.set_size(FONTSIZE + 8)
    ax.tick_params(axis="y", which="major", labelsize=FONTSIZE + 6)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run-design compatibility check.
# ---------------------------------------------------------------------------


def _check_design_compatibility(
    run_dir: Path, *, expected_n_features: int,
) -> None:
    """Raise if the autopi run's last-round design has a different feature
    count than the human task. Theories tuned on a different n_features
    can't be replayed on human stimuli without redefining what the
    parameters mean. Differences in `rating_max` / specific validities
    are tolerated (we pin them to the human values when sampling)."""
    design = _last_round_design(run_dir)
    if design is None:
        print(
            f"[eval_hilbig] WARNING: could not read run design from "
            f"{run_dir} (no observations/state.json or empty rounds). "
            f"Proceeding without compatibility check.",
            file=sys.stderr,
        )
        return

    # run_n_features = len(design["validities"])
    # if run_n_features != expected_n_features:
    #     raise SystemExit(
    #         f"[eval_hilbig] Design mismatch: autopi run uses "
    #         f"{run_n_features} features {design['validities']} but "
    #         f"humans saw {expected_n_features}. Theories tuned on a "
    #         f"different feature count cannot be replayed meaningfully "
    #         f"— pick a run with {expected_n_features} features."
    #     )

    if (
        list(design["validities"]) != HUMAN_VALIDITIES
        or int(design["rating_max"]) != HUMAN_RATING_MAX
    ):
        print(
            f"[eval_hilbig] NOTE: autopi run validities="
            f"{design['validities']} rating_max={design['rating_max']} "
            f"differ from human task (validities={HUMAN_VALIDITIES}, "
            f"rating_max={HUMAN_RATING_MAX}). Replaying with human "
            f"values pinned; surfaced theories may have been tuned on "
            f"a slightly different design.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate base + surfaced models from an autopi run vs human "
            "stimulus choice proportions on Hilbig (2014) Exp 1."
        ),
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-subjects", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target", choices=["p_b_pooled", "p_b_within_subj_mean"],
        default="p_b_pooled",
        help="Which human aggregate to score models against.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory. Default: <run-dir>/analysis/eval_hilbig/.",
    )
    p.add_argument(
        "--include-canonical", action="store_true",
        help=(
            "Also evaluate every YAML theory under "
            "theories/heuristic_decision_making/ (TTB / WADD / Tallying "
            "/ EW) as a fixed-baseline reference, alongside the run's "
            "base / surfaced models."
        ),
    )
    p.add_argument(
        "--canonical-dir", type=Path, default=CANONICAL_YAML_DIR,
        help=f"Override canonical YAML dir. Default: {CANONICAL_YAML_DIR}",
    )
    p.add_argument(
        "--extra-yaml", type=Path, nargs="+", default=None, metavar="PATH",
        help=(
            "One or more YAML files and/or directories of YAMLs to "
            "include alongside the run's base/surfaced models. Files are "
            "loaded individually; directories load every *.yaml under "
            "them. Each theory is labelled by its filename stem and "
            "tagged with role='extra'."
        ),
    )
    p.add_argument(
        "--all-explored", action="store_true",
        help=(
            "Evaluate every unique theory the run ever entertained "
            "(across all rounds), not just round-0 base / final-round "
            "surfaced models. Theories are deduped by label and "
            "partitioned into role='base' (round-0 starts that didn't "
            "survive), role='intermediate' (proposed mid-run then "
            "killed), and role='surfaced' (un-killed in the last round)."
        ),
    )
    p.add_argument(
        "--hide-model-labels", action="store_true",
        help=(
            "Drop the per-panel scatter titles and the per-bar model-name "
            "x-tick labels; rely on the shared colour scheme to identify "
            "models (publication-figure styling)."
        ),
    )
    p.add_argument(
        "--exclude-mse", nargs="+", default=(), metavar="LABEL",
        help=(
            "Model labels (e.g. pi_7) to drop from the MSE bar chart only. "
            "Other charts and the scatter keep every model."
        ),
    )
    args = p.parse_args(argv)

    out = args.out or (args.run_dir / "analysis" / "eval_hilbig")
    out.mkdir(parents=True, exist_ok=True)

    df_human = load_human_choices(args.human_data)
    n_features = len(df_human["option_a"].iloc[0])
    if n_features != len(HUMAN_VALIDITIES):
        raise SystemExit(
            f"[eval_hilbig] Human data has {n_features} features but "
            f"HUMAN_VALIDITIES has {len(HUMAN_VALIDITIES)}; check "
            f"--human-data."
        )

    _check_design_compatibility(args.run_dir, expected_n_features=n_features)

    human_props = compute_human_proportions(df_human)
    print(
        f"[eval_hilbig] human: n_participants="
        f"{df_human['participant'].nunique()} "
        f"n_unique_stimuli={len(human_props)} "
        f"target={args.target}"
    )

    intermediate: dict[str, Theory] = {}
    if args.all_explored:
        base, intermediate, surfaced = resolve_all_explored_theories(
            args.run_dir,
        )
    else:
        base = resolve_base_theories(args.run_dir)
        surfaced = resolve_surfaced_theories(args.run_dir)
    canonical: dict[str, Theory] = {}
    if args.include_canonical:
        canonical = load_canonical_theories(args.canonical_dir)
    extra: dict[str, Theory] = {}
    if args.extra_yaml:
        extra = load_yaml_theories(args.extra_yaml)
    print(
        f"[eval_hilbig] base={list(base)} "
        f"intermediate={list(intermediate)} surfaced={list(surfaced)} "
        f"canonical={list(canonical)} extra={list(extra)} "
        f"n_subjects={args.n_subjects} seed={args.seed}"
    )

    per_stim_df, summary_df = evaluate_models(
        base=base, surfaced=surfaced,
        human_props=human_props,
        validities=HUMAN_VALIDITIES,
        rating_max=HUMAN_RATING_MAX,
        n_subjects=args.n_subjects,
        base_seed=args.seed,
        target_col=args.target,
        intermediate=intermediate,
        canonical=canonical,
        extra=extra,
    )

    per_stim_csv = out / "eval_hilbig_per_stimulus.csv"
    summary_csv = out / "eval_hilbig_summary.csv"
    per_stim_df.to_csv(per_stim_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # Plots are nice-to-have; CSVs are the source of truth. Don't fail
    # the whole run when matplotlib isn't installed — research envs vary.
    try:
        # Shared label→color map so a model has the same color in every
        # figure (scatter grid, per-model scatters, bar charts). Colour by
        # lineage role — seeds → slate (indigo), winning survivor → sandy
        # (gold), other survivor → terracotta — to match the convergence and
        # trajectory plots; baselines outside the run fall back to CYCLE.
        models = list(summary_df["model"])
        color_map = role_color_map(models, _lineage_color_map(args.run_dir))
        # Readable theory names (pi_N -> "Weighted Additive (WADD)" etc.).
        name_map = build_name_map(args.run_dir)
        show_title = not args.hide_model_labels
        plot_scatter(
            per_stim_df, out / "eval_hilbig_scatter.png",
            title=(
                f"Model vs human P(B) — run: {args.run_dir.name} — "
                f"target={args.target}"
            ),
            color_map=color_map, name_map=name_map, show_title=show_title,
        )
        plot_scatter_per_model(
            per_stim_df, out,
            color_map=color_map, name_map=name_map, show_title=show_title,
        )
        # One bar chart per metric. MAE/MSE are lower-is-better; Pearson r is
        # higher-is-better, so sort it descending so the best model leads.
        for metric, ascending in (
            ("mae", True), ("mse", True), ("pearson_r", False),
        ):
            # --exclude-mse drops named models from the MSE chart only.
            exclude = tuple(args.exclude_mse) if metric == "mse" else ()
            plot_bars(
                summary_df, out / f"eval_hilbig_bars_{metric}.png",
                metric=metric, ascending=ascending,
                title=(
                    f"{metric.upper()} vs humans ({args.target}) — "
                    f"run: {args.run_dir.name}"
                ),
                color_map=color_map, name_map=name_map,
                exclude_models=exclude,
                show_model_labels=not args.hide_model_labels,
            )
        plot_msg = f"and PNGs under {out}/"
    except ImportError as e:
        print(
            f"[eval_hilbig] WARNING: skipping PNG plots ({e}). "
            f"Install matplotlib to enable.",
            file=sys.stderr,
        )
        plot_msg = "(plots skipped — matplotlib missing)"

    print(summary_df.to_string(index=False))
    print(
        f"[eval_hilbig] wrote {per_stim_csv}, {summary_csv} {plot_msg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
