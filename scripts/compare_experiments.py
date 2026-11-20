"""Compare the experiments autocog proposed, and embed experiments and theories in 2D.

Every round, each PI theory proposes an experiment: a set of cue validities
and a list of (option A, option B) binary cue patterns, plus a prose
rationale and a summary-statistic metric. Designs differ in size (4-50
cues, 3-32 pairs), so they are compared through size-free representations:

1. **Discrimination profile** (main measure). A fixed panel of reference
   heuristics (`PANEL_NAMES`) predicts P(choose B) on every pair of the
   experiment (prior-predictive mean over parameter draws). For each pair
   of panel models, the profile holds their mean |P_a - P_b| over the
   experiment's pairs. Two experiments are close if they separate the same
   heuristics, by the same amount — what matters for model discrimination.
   Euclidean distance between profiles.

2. **Structural features.** Size-free summaries of the design: number of
   cues / pairs, validity mean / SD / range, share of non-discriminating
   pairs, discriminating cues per pair, TTB-vs-tally conflict, tally ties,
   rank of the first discriminating cue, tally gap, A/B balance, cue
   density. z-scored, Euclidean distance.

3. **Rationale text.** TF-IDF cosine distance between the prose rationales:
   what the LLM says the experiment tests.

4. **Joint theory x experiment space** (needs `--models-dir`, the output of
   `compare_models.py` on the same runs). Every discovered theory predicts
   every experiment at its best-fit free parameters (fit on the Hilbig &
   Moshagen stimuli); context values such as validities come from the
   experiment. From that matrix:
     - theory distance  = RMS difference in P(B), experiments weighted
       equally — a much broader test of "do they behave alike" than the
       24 Hilbig pairs;
     - experiment distance = Euclidean distance between the experiments'
       disagreement profiles over *all* theory pairs (the discrimination
       profile with the discovered theories as the panel).
   Each family's ground-truth model (prior-predictive mean, the fit target)
   is placed in the theory map, and `transfer_rmse` is each theory's RMS
   distance from its own ground truth across all proposed experiments —
   generalization beyond the stimuli it was fit on. For a human run (no
   ground-truth model) the reference is the participants' observed share
   choosing B on each pair of that run's experiments.

   Two more theory spaces sit beside the joint one: best-fit predictions on
   the Hilbig & Moshagen pairs alone, and **code** — symmetrised normalized
   compression distance between `predict()` sources with comments and
   docstrings stripped (ground truths enter with their canonical code). The
   code map shows how theories are written; the others, how they behave.

Every distance matrix is embedded with classical (Torgerson) MDS; each map
reports the share of positive eigenvalue mass in its 2 dimensions and
Kruskal's stress-1. Mantel tests (Spearman, permutation p) compare spaces.
The Hilbig & Moshagen (2014) design is added to every experiment map as a
reference point.

Outputs (under `--out`):

  experiments.csv        one row per experiment: run meta, features, profile
  experiment_coords.csv  long: exp_id, space, x, y
  theories.csv           joint-space theories: fit_mse (from --models-dir),
                         transfer_rmse, n experiments failed
  theory_coords.csv      long: model, space (joint | hilbig | code), x, y
  theory_distances_<space>.csv  full theory x theory distance matrices
  joint_predictions.npz  P(B) of every theory (incl. references) on every
                         experiment's pairs: arrays e0..e<n-1>, theory_ids, exp_ids
  mds_summary.csv        per space: n, 2-D variance share, stress
  mantel.csv             Spearman r and permutation p between spaces
  run_diversity.csv      per run: mean pairwise distance among its
                         experiments (profile space), best fit_mse
  experiment_map_<space>.png/svg, theory_map_<space>.png/svg

Run from the repo root:

  python scripts/compare_experiments.py --max-rounds 5 --workers 12 \\
      --run-dir results/recovery*/{single_cue,anti_majority,take_the_worst}/noise=0.0/*_run* \\
      --models-dir <compare_models out-dir> --out <out-dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sample_parameters import draw_parameter_samples, resolve_parameters  # noqa: E402
from src.theory import Theory  # noqa: E402
from scripts.compare_models import (  # noqa: E402
    HUMAN_DATA_DEFAULT,
    HUMAN_VALIDITIES,
    Pair,
    _Budget,
    _OutOfTime,
    _ordered,
    _run_name,
    _slug,
    _strip_comments_and_docstrings,
    family_label,
    FAMILY_LABELS,
    human_counts,
    llm_label,
    llm_styles,
    LLM_LABELS,
    load_models,
    ncd,
    parameter_space,
    parse_run_dir,
    predict_p_b,
    task_context,
)
from scripts.eval_hilbig import CANONICAL_YAML_DIR  # noqa: E402

# Equal-weight is left out: on binary cues it predicts exactly what tallying does.
PANEL_NAMES: tuple[str, ...] = ("ttb", "wadd", "tallying", "single_cue",
                                "anti_majority", "take_the_worst")
HILBIG_ID = "hilbig_moshagen_2014"
FEATURE_COLS: tuple[str, ...] = (
    "n_cues", "n_pairs", "validity_mean", "validity_sd", "validity_range",
    "frac_no_discrimination", "mean_discriminating_cues", "frac_ttb_tally_conflict",
    "frac_tally_tie", "first_disc_rank", "tally_gap", "ab_imbalance", "cue_density",
)


# ---------------------------------------------------------------------------
# Loading experiments.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    run: str
    llm: str
    family: str
    replicate: str
    round: int
    index: int
    proposer: str
    proposer_model: str
    validities: tuple[float, ...]
    pairs: tuple[Pair, ...]
    rationale: str = ""
    metric_source: str = ""
    human: bool = False  # run with real participants (no ground-truth model)
    observed_p_b: tuple[float, ...] = ()  # human runs: share choosing B, per pair
    observed_n: tuple[int, ...] = ()


def _observed_choices(path: Path, pairs: tuple[Pair, ...]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Share of trials choosing B (response 1) on each of `pairs`, from a
    run's per-trial JSONL (dropped participants are already excluded)."""
    k = {p: 0 for p in pairs}
    n = {p: 0 for p in pairs}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        key = (tuple(int(x) for x in t["option_a_ratings"]),
               tuple(int(x) for x in t["option_b_ratings"]))
        if key in n:
            n[key] += 1
            k[key] += int(t["response"])
    return (tuple(k[p] / n[p] if n[p] else float("nan") for p in pairs),
            tuple(n[p] for p in pairs))


def _as_pairs(a_rows: list[list[int]], b_rows: list[list[int]]) -> tuple[Pair, ...]:
    return tuple((tuple(int(x) for x in a), tuple(int(x) for x in b))
                 for a, b in zip(a_rows, b_rows))


def load_experiments(run_dirs: list[Path], *, max_rounds: int | None = None) -> list[Experiment]:
    """Every experiment proposed in each run's first `max_rounds` rounds.
    Round r's experiments were proposed by the theories alive in round r,
    so they pair with `compare_models.run_theories(max_rounds=r + 1)`."""
    out: list[Experiment] = []
    for rd in run_dirs:
        path = rd / "observations" / "state.json"
        if not path.is_file():
            print(f"  skipping {rd}: no observations/state.json")
            continue
        info = parse_run_dir(rd)
        human = info["noise"] is None
        short = f"{info['llm']}/{info['family']}/run{info['replicate']}"
        rounds = json.loads(path.read_text())["rounds"]
        if max_rounds is not None:
            rounds = rounds[:max_rounds]
        for r, rnd in enumerate(rounds):
            for i, ob in enumerate(rnd["observations"]):
                e = ob["experiment"]
                pairs = _as_pairs(e["trial_a_ratings"], e["trial_b_ratings"])
                obs_p, obs_n = (), ()
                data = rd / "observations" / ob.get("data_path", "")
                if human and ob.get("data_path") and data.is_file():
                    obs_p, obs_n = _observed_choices(data, pairs)
                out.append(Experiment(
                    exp_id=f"{short}/r{r}e{i}", run=_run_name(rd), llm=info["llm"],
                    family=info["family"], replicate=info["replicate"], round=r, index=i,
                    proposer=ob.get("proposer_label", ""),
                    proposer_model=f"{short}/{ob.get('proposer_label', '')}",
                    validities=tuple(float(v) for v in e["validities"]),
                    pairs=pairs,
                    rationale=e.get("rationale", ""),
                    metric_source=(ob.get("metric") or {}).get("metric_source", ""),
                    human=human, observed_p_b=obs_p, observed_n=obs_n,
                ))
    return out


def hilbig_experiment(human_data: Path = HUMAN_DATA_DEFAULT) -> Experiment:
    """The Hilbig & Moshagen (2014) design, as a reference point."""
    hc = human_counts(human_data)
    pairs = tuple(zip((tuple(map(int, a)) for a in hc["option_a"]),
                      (tuple(map(int, b)) for b in hc["option_b"])))
    return Experiment(HILBIG_ID, "reference", "reference", "reference", "", -1, 0, "", "",
                      tuple(HUMAN_VALIDITIES), pairs)


# ---------------------------------------------------------------------------
# 2. Structural features.
# ---------------------------------------------------------------------------


def structural_features(validities: Any, pairs: Any) -> dict[str, float]:
    """Size-free summaries of a binary-cue design. Pair-level quantities
    other than `frac_no_discrimination` average over discriminating pairs
    only. TTB's direction = the most valid discriminating cue; tally's =
    the sign of (A's wins - B's wins)."""
    v = np.asarray(validities, dtype=float)
    n = len(v)
    order = np.argsort(-v, kind="stable")          # most valid first
    rank = np.empty(n)
    rank[order] = np.arange(n) / max(n - 1, 1)     # 0 = most valid, 1 = least
    A = np.asarray([p[0] for p in pairs], dtype=float)
    B = np.asarray([p[1] for p in pairs], dtype=float)
    diff = A - B
    disc = diff != 0
    any_disc = disc.any(axis=1)
    conflict, tie, first, gap, ttb_a = [], [], [], [], []
    for d, m in zip(diff[any_disc], disc[any_disc]):
        j = next(k for k in order if m[k])
        ttb = np.sign(d[j])
        tally = np.sign(d.sum())
        conflict.append(tally != 0 and tally != ttb)
        tie.append(tally == 0)
        first.append(rank[j])
        gap.append(abs(d.sum()) / n)
        ttb_a.append(ttb > 0)
    mean = (lambda xs: float(np.mean(xs)) if len(xs) else 0.0)
    return {
        "n_cues": n,
        "n_pairs": len(pairs),
        "validity_mean": float(v.mean()),
        "validity_sd": float(v.std()),
        "validity_range": float(v.max() - v.min()),
        "frac_no_discrimination": float(1 - any_disc.mean()),
        "mean_discriminating_cues": mean(disc[any_disc].sum(axis=1)),
        "frac_ttb_tally_conflict": mean(conflict),
        "frac_tally_tie": mean(tie),
        "first_disc_rank": mean(first),
        "tally_gap": mean(gap),
        "ab_imbalance": abs(mean(ttb_a) - 0.5) if ttb_a else 0.0,
        "cue_density": float(np.concatenate([A, B]).mean()),
    }


# ---------------------------------------------------------------------------
# 1. Reference panel and discrimination profiles.
# ---------------------------------------------------------------------------


def load_panel(names: tuple[str, ...] = PANEL_NAMES) -> dict[str, Theory]:
    return {n: Theory.from_yaml(CANONICAL_YAML_DIR / f"{n}.yaml") for n in names}


def prior_mean(theory: Theory, validities: Any, pairs: Any, *, n_draws: int,
               seed: int) -> np.ndarray:
    """Prior-predictive mean P(B) per pair. Unlike
    `compare_models.prior_predictive`, every leaf (including per-cue weight
    vectors sized by n_features) is redrawn on every draw."""
    ctx = task_context(list(validities))
    resolved = resolve_parameters(theory.parameters, ctx)
    random.seed(seed)
    rows = []
    for _ in range(n_draws):
        params = draw_parameter_samples(resolved)
        for key in ("validities", "rating_max"):
            if key in params:
                params[key] = ctx[key]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            rows.append(predict_p_b(theory, params, list(pairs)))
    return np.vstack(rows).mean(axis=0)


def panel_predictions(panel: dict[str, Theory], validities: Any, pairs: Any, *,
                      n_draws: int, seed: int) -> dict[str, np.ndarray]:
    return {name: prior_mean(th, validities, pairs, n_draws=n_draws, seed=seed)
            for name, th in panel.items()}


def discrimination_profile(preds: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean |P_a - P_b| over the experiment's pairs, per pair of models."""
    return {f"{a}|{b}": float(np.mean(np.abs(preds[a] - preds[b])))
            for a, b in itertools.combinations(preds, 2)}


# ---------------------------------------------------------------------------
# Distances, MDS, Mantel.
# ---------------------------------------------------------------------------


def euclidean_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = (X ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


_TOKEN = re.compile(r"[a-z][a-z\-]+")


def tfidf_cosine_distances(texts: list[str]) -> np.ndarray:
    """1 - cosine similarity of TF-IDF vectors (sublinear tf, smooth idf)."""
    docs = [Counter(_TOKEN.findall(t.lower())) for t in texts]
    vocab = sorted(set().union(*docs))
    idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab))
    for d in docs:
        for w in d:
            df[idx[w]] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w, c in d.items():
            X[i, idx[w]] = (1 + math.log(c)) * idf[idx[w]]
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norm > 0, norm, 1.0)
    D = np.clip(1 - X @ X.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2


def classical_mds(D: np.ndarray, k: int = 2) -> tuple[np.ndarray, float]:
    """Torgerson MDS. Returns (n x k coordinates, share of the positive
    eigenvalue mass in the first k dimensions)."""
    D = np.asarray(D, dtype=float)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh((Bm + Bm.T) / 2)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = w[w > 0]
    top = np.clip(w[:k], 0, None)
    Y = V[:, :k] * np.sqrt(top)
    # Fix sign so maps are reproducible: largest-|loading| point positive.
    for j in range(k):
        if Y[np.argmax(np.abs(Y[:, j])), j] < 0:
            Y[:, j] *= -1
    return Y, float(top.sum() / pos.sum()) if pos.size else float("nan")


def kruskal_stress(D: np.ndarray, Y: np.ndarray) -> float:
    iu = np.triu_indices(len(D), 1)
    d = D[iu]
    e = euclidean_distances(Y)[iu]
    den = (d ** 2).sum()
    return float(np.sqrt(((d - e) ** 2).sum() / den)) if den > 0 else 0.0


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def mantel(D1: np.ndarray, D2: np.ndarray, *, n_perm: int = 999,
           seed: int = 0) -> tuple[float, float]:
    """Spearman correlation between the upper triangles of two distance
    matrices over the same objects; one-sided permutation p (rows and
    columns of D2 permuted together)."""
    n = len(D1)
    iu = np.triu_indices(n, 1)
    a = _rank(D1[iu])
    a = (a - a.mean()) / a.std()
    rng = np.random.default_rng(seed)

    def corr(M: np.ndarray) -> float:
        b = _rank(M[iu])
        return float(np.mean(a * (b - b.mean()) / b.std()))

    r = corr(D2)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        hits += corr(D2[np.ix_(p, p)]) >= r
    return r, (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# 4. Joint theory x experiment space.
# ---------------------------------------------------------------------------


def _is_random(v: Any) -> bool:
    """Whether a resolved parameter value still contains a leaf that
    `draw_parameter_samples` would draw at random."""
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]) != float(v[1])
    if isinstance(v, dict):
        if set(v) == {"interval"}:
            iv = v["interval"]
            return not (isinstance(iv, list) and len(iv) == 2 and float(iv[0]) == float(iv[1]))
        if set(v) == {"choices"}:
            return len(v["choices"]) > 1
        return any(_is_random(x) for x in v.values())
    if isinstance(v, list):
        return any(_is_random(x) for x in v)
    return False


def predict_on_experiment(theory: Theory, best_params: dict[str, Any],
                          validities: Any, pairs: Any, *, n_draws: int = 50,
                          seed: int = 0) -> np.ndarray:
    """P(B) per pair on a new design: context-derived values (validities,
    fixed constants) are re-resolved for the experiment and free parameters
    take their best-fit values. Nested random leaves (e.g. per-cue weight
    vectors `[(0, 1)] * n_features`) are not fit and cannot carry over to a
    design with a different number of cues, so they are averaged over
    `n_draws` seeded draws."""
    ctx = task_context(list(validities))
    fixed, free = parameter_space(theory, ctx)
    free_vals = {p.name: best_params[p.name] for p in free if p.name in best_params}
    resolved = resolve_parameters(theory.parameters, ctx)
    nested = [k for k in fixed if k not in ("validities", "rating_max")
              and _is_random(resolved.get(k))]
    rng_state = random.getstate()
    random.seed(seed)
    rows = []
    try:
        for _ in range(n_draws if nested else 1):
            params = dict(fixed)
            params.update({k: draw_parameter_samples(resolved[k]) for k in nested})
            params.update(free_vals)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                rows.append(predict_p_b(theory, params, list(pairs)))
    finally:
        random.setstate(rng_state)
    return np.vstack(rows).mean(axis=0)


def code_distances(sources: list[str]) -> np.ndarray:
    """Symmetrised normalized compression distance between source texts."""
    n = len(sources)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = (ncd(sources[i], sources[j]) + ncd(sources[j], sources[i])) / 2
    return D


def _clean_source(src: str) -> str:
    try:
        return _strip_comments_and_docstrings(src)
    except SyntaxError:
        return src


def complete_theories(P: list[np.ndarray]) -> np.ndarray:
    """Theories (rows) with finite predictions on every experiment."""
    return np.all(np.column_stack([np.isfinite(p).all(axis=1) for p in P]), axis=1)


def joint_distances(P: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """P[e] is a (theories x pairs_e) prediction matrix for experiment e.
    Returns (theory x theory RMS distance with experiments weighted
    equally, experiment x experiment Euclidean distance between the
    experiments' disagreement profiles over all theory pairs)."""
    T = P[0].shape[0]
    iu = np.triu_indices(T, 1)
    msq = np.zeros((T, T))
    prof = np.empty((len(P), len(iu[0])))
    for e, M in enumerate(P):
        diff = M[:, None, :] - M[None, :, :]
        msq += (diff ** 2).mean(axis=2)
        prof[e] = np.abs(diff).mean(axis=2)[iu]
    return np.sqrt(msq / len(P)), euclidean_distances(prof)


# ---------------------------------------------------------------------------
# Workers (process-pool friendly: plain inputs).
# ---------------------------------------------------------------------------


def _experiment_job(job: dict[str, Any]) -> dict[str, Any]:
    panel = load_panel(tuple(job["panel"]))
    v, pairs = job["validities"], [tuple(map(tuple, p)) for p in job["pairs"]]
    preds = panel_predictions(panel, v, pairs, n_draws=job["n_draws"], seed=job["seed"])
    gts = {}
    for fam in job["gt_families"]:
        th = panel[fam] if fam in panel else Theory.from_yaml(CANONICAL_YAML_DIR / f"{fam}.yaml")
        gts[fam] = preds[fam] if fam in preds else prior_mean(
            th, v, pairs, n_draws=job["n_draws"], seed=job["seed"])
    return {"profile": discrimination_profile(preds),
            "gt": {k: g.tolist() for k, g in gts.items()}}


def _theory_job(job: dict[str, Any]) -> dict[str, Any]:
    theory = Theory.model_validate(job["theory"])
    out, n_fail = [], 0
    with _Budget(job["time_budget"]) as budget:
        for v, pairs in job["designs"]:
            budget.check()
            try:
                out.append(predict_on_experiment(
                    theory, job["params"], v, [tuple(map(tuple, p)) for p in pairs]).tolist())
            except _OutOfTime:
                raise
            except Exception:  # noqa: BLE001 - LLM-written code
                out.append([float("nan")] * len(pairs))
                n_fail += 1
    n_fail += len(job["designs"]) - len(out)
    out += [[float("nan")] * len(p) for _, p in job["designs"][len(out):]]
    return {"model": job["model"], "preds": out, "n_failed": n_fail,
            "truncated": budget.expired}


def _run_pool(fn, jobs: list[dict[str, Any]], workers: int, label: str) -> list[Any]:
    results: list[Any] = [None] * len(jobs)
    t0 = time.monotonic()
    step = max(1, len(jobs) // 10)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
            for done, f in enumerate(as_completed(futs), 1):
                results[futs[f]] = f.result()
                if done % step == 0 or done == len(jobs):
                    print(f"  {label}: {done}/{len(jobs)} ({time.monotonic() - t0:.0f}s)",
                          flush=True)
    else:
        for i, j in enumerate(jobs):
            results[i] = fn(j)
    return results


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------


def _plot_map(coords: pd.DataFrame, *, out_base: Path, title: str, kind: str,
              ref: dict[str, tuple[float, float]] | None = None, ref_label: str = "") -> None:
    """One panel per family on shared axes. `coords` has x, y, llm, family,
    run, order (round or generation) and `seed` (hollow marker). Each run's
    points are joined in `order` (per-order centroid). `ref` maps family
    (or "*" for every panel) to a reference point drawn as a black star."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scripts.figure_style import FONTSIZE, GRAY, INK, save_figure, style_axes

    d = coords
    fams = _ordered(d["family"], FAMILY_LABELS)
    styles = llm_styles(d["llm"].unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(max(5.2 * len(fams), 9.0), 5.2),
                             sharex=True, sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        ax.scatter(d["x"], d["y"], s=10, color=GRAY, alpha=0.15, lw=0, zorder=1)
        g = d[d["family"] == fam]
        for run, gr in g.groupby("run"):
            if gr["seed"].all():
                continue
            c = gr[~gr["seed"]].groupby("order")[["x", "y"]].mean().sort_index()
            ax.plot(c["x"], c["y"], color=styles[gr["llm"].iloc[0]][0], lw=0.8,
                    alpha=0.45, zorder=2)
        for llm in _ordered(g["llm"], LLM_LABELS):
            gl = g[g["llm"] == llm]
            col, mk = styles[llm]
            live = gl[~gl["seed"]]
            ax.scatter(live["x"], live["y"], s=34, color=col, marker=mk, alpha=0.85,
                       edgecolor="white", lw=0.5, zorder=3)
        seeds = g[g["seed"]]
        if len(seeds):
            ax.scatter(seeds["x"], seeds["y"], s=40, facecolor="none", edgecolor=INK,
                       lw=1.0, zorder=4)
        if ref:
            for key in (fam, "*"):
                if key in ref:
                    ax.scatter(*ref[key], s=220, marker="*", color=INK, zorder=5,
                               edgecolor="white", lw=0.8)
        style_axes(ax, title=family_label(fam))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("MDS 1", fontsize=FONTSIZE - 2)
    axes[0][0].set_ylabel("MDS 2", fontsize=FONTSIZE - 2)
    handles = [Line2D([], [], ls="", marker=styles[l][1], color=styles[l][0], ms=8,
                      label=llm_label(l)) for l in _ordered(d["llm"], LLM_LABELS)]
    seed_word = "Proposed by seed theories" if kind == "experiment" else "Seed theories"
    if d["seed"].any():
        handles.append(Line2D([], [], ls="", marker="o", mfc="none", mec=INK, ms=8,
                              label=seed_word))
    if ref:
        handles.append(Line2D([], [], ls="", marker="*", color=INK, ms=13, label=ref_label))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, fontsize=FONTSIZE - 2, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, 0.99, title, ha="left", va="top", fontsize=FONTSIZE)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    save_figure(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _space_coords(D: np.ndarray, ids: list[str], space: str, key: str,
                  summary: list[dict[str, Any]]) -> pd.DataFrame:
    Y, share = classical_mds(D)
    summary.append({"objects": "theories" if key == "model" else "experiments",
                    "space": space, "n": len(ids), "variance_2d": share,
                    "stress": kruskal_stress(D, Y)})
    return pd.DataFrame({key: ids, "space": space, "x": Y[:, 0], "y": Y[:, 1]})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="compare_models.py output for the same runs (fit_params.json, "
                         "models.csv). Enables the joint theory x experiment space.")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Use only each run's first N rounds (match compare_models).")
    ap.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    ap.add_argument("--n-draws", type=int, default=100,
                    help="Parameter draws for panel / ground-truth prior means.")
    ap.add_argument("--n-perm", type=int, default=999, help="Mantel permutations.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=120.0,
                    help="Seconds per theory for the joint predictions.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    exps = load_experiments(args.run_dir, max_rounds=args.max_rounds)
    ref = hilbig_experiment(args.human_data)
    all_exps = exps + [ref]
    families = sorted({e.family for e in exps})
    # Simulated families are judged against their ground-truth model; human
    # families against the participants' observed choices on each experiment.
    human_fams = sorted({e.family for e in exps if e.human})
    gt_fams = [f for f in families if f not in human_fams]
    print(f"{len(exps)} experiments from {len({e.run for e in exps})} runs "
          f"(+ the Hilbig & Moshagen reference).")

    # Panel profiles and ground-truth predictions, per experiment.
    jobs = [{"validities": list(e.validities), "pairs": [list(map(list, p)) for p in e.pairs],
             "panel": list(PANEL_NAMES), "gt_families": gt_fams,
             "n_draws": args.n_draws, "seed": args.seed} for e in all_exps]
    ex_res = _run_pool(_experiment_job, jobs, args.workers, "panel")

    meta = pd.DataFrame([{
        "exp_id": e.exp_id, "run": e.run, "llm": e.llm, "family": e.family,
        "replicate": e.replicate, "round": e.round, "index": e.index,
        "proposer": e.proposer, "proposer_model": e.proposer_model,
        "rationale_words": len(e.rationale.split())} for e in all_exps])
    feats = pd.DataFrame([structural_features(e.validities, e.pairs) for e in all_exps])
    prof = pd.DataFrame([r["profile"] for r in ex_res]).add_prefix("disc_")
    exp_df = pd.concat([meta, feats, prof], axis=1)
    exp_df.to_csv(args.out / "experiments.csv", index=False)

    ids = [e.exp_id for e in all_exps]
    D = {"profile": euclidean_distances(prof.to_numpy()),
         "features": euclidean_distances(zscore(feats[list(FEATURE_COLS)].to_numpy()))}
    # Rationale text: the reference design has none, so it is left out.
    D_text = tfidf_cosine_distances([e.rationale for e in exps])

    summary: list[dict[str, Any]] = []
    coords = [_space_coords(D[s], ids, s, "exp_id", summary) for s in D]
    coords.append(_space_coords(D_text, ids[:-1], "rationale", "exp_id", summary))

    mantel_rows = []

    def _mantel(name_a: str, name_b: str, A: np.ndarray, B: np.ndarray) -> None:
        r, p = mantel(A, B, n_perm=args.n_perm, seed=args.seed)
        mantel_rows.append({"a": name_a, "b": name_b, "n": len(A), "spearman_r": r, "p": p})
        print(f"  Mantel {name_a} vs {name_b}: r = {r:.2f} (p = {p:.3f}, n = {len(A)})")

    n = len(exps)
    _mantel("experiment:profile", "experiment:features", D["profile"][:n, :n], D["features"][:n, :n])
    _mantel("experiment:profile", "experiment:rationale", D["profile"][:n, :n], D_text)

    # Diversity of each run's experiments in profile space.
    Dp = D["profile"][:n, :n]
    div_rows = []
    for run, idx in exp_df.iloc[:n].groupby("run").groups.items():
        idx = list(idx)
        sub = Dp[np.ix_(idx, idx)]
        div_rows.append({"run": run, "llm": exp_df.at[idx[0], "llm"],
                         "family": exp_df.at[idx[0], "family"], "n_experiments": len(idx),
                         "profile_diversity": float(sub[np.triu_indices(len(idx), 1)].mean()),
                         "profile_dist_to_hilbig": float(D["profile"][idx, -1].mean())})
    div_df = pd.DataFrame(div_rows)

    theory_coords = []
    if args.models_dir is not None:
        models_df = pd.read_csv(args.models_dir / "models.csv")
        params = json.loads((args.models_dir / "fit_params.json").read_text())
        models = load_models(args.run_dir, which="all", max_rounds=args.max_rounds,
                             include_canonical=False)
        # Seeds are shared by every run; keep one copy per (family, code, params).
        # `alias` maps a dropped copy to the kept one (for proposer lookups).
        seen: dict[tuple[str, str, str], str] = {}
        keep, alias = [], {}
        for m in models:
            key = (m.family, m.theory.predict_source, json.dumps(params.get(m.model_id),
                                                                 sort_keys=True, default=str))
            if m.generation == 0 and key in seen:
                alias[m.model_id] = seen[key]
                continue
            seen.setdefault(key, m.model_id)
            keep.append(m)
        designs = [[list(e.validities), [list(map(list, p)) for p in e.pairs]] for e in exps]
        tjobs = [{"model": m.model_id, "theory": m.theory.model_dump(),
                  "params": params.get(m.model_id, {}), "designs": designs,
                  "time_budget": args.time_budget} for m in keep]
        print(f"Joint space: {len(keep)} theories x {len(exps)} experiments.")
        th_res = _run_pool(_theory_job, tjobs, args.workers, "theories")

        # References join as theories: each ground truth's prior-mean
        # predictions, and for human runs the observed share choosing B
        # (defined only on that family's own experiments).
        ref_ids = {f: f"ground_truth/{f}" for f in gt_fams}
        ref_ids |= {f: f"humans/{f}" for f in human_fams}
        P = []
        for e_i, e in enumerate(exps):
            rows = [r["preds"][e_i] for r in th_res]
            rows += [ex_res[e_i]["gt"][f] for f in gt_fams]
            rows += [list(e.observed_p_b) if e.family == f and e.observed_p_b
                     else [float("nan")] * len(e.pairs) for f in human_fams]
            P.append(np.asarray(rows, dtype=float))
        gt_ids = [ref_ids[f] for f in gt_fams + human_fams]
        t_ids = [m.model_id for m in keep] + gt_ids
        ok = complete_theories(P)
        dropped = [t for t, k in zip(t_ids, ok) if not k]
        if dropped:
            print(f"  {len(dropped)} theories failed on some experiment and are left out "
                  f"of the joint space: {', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}")
        P_ok = [p[ok] for p in P]
        # Every theory's prediction on every experiment (rows = theory_ids).
        np.savez_compressed(args.out / "joint_predictions.npz",
                            theory_ids=np.array(t_ids), exp_ids=np.array(ids[:-1]),
                            **{f"e{i}": P[i] for i in range(n)})
        t_ok = [t for t, k in zip(t_ids, ok) if k]
        D_th, D_ex = joint_distances(P_ok)

        th_meta = {m.model_id: m for m in keep}
        gt_pos = {f: t_ok.index(ref_ids[f]) for f in families if ref_ids[f] in t_ok}
        gt_full = {f: t_ids.index(ref_ids[f]) for f in families}
        fam_of = exp_df["family"].to_numpy()[:n]
        rows = []
        for i, t in enumerate(t_ok):
            m = th_meta.get(t)
            row = {"model": t, "llm": m.llm if m else "ground_truth",
                   "family": m.family if m else t.split("/", 1)[1],
                   "run": m.run if m else "ground_truth",
                   "label": m.label if m else t.split("/", 1)[0],
                   "generation": m.generation if m else -1}
            gi = gt_pos.get(row["family"])
            if m is not None:
                if gi is not None:
                    row["transfer_rmse"] = float(D_th[i, gi])
                fi, gf = t_ids.index(t), gt_full[row["family"]]
                row["transfer_rmse_own_family"] = float(np.sqrt(np.mean(
                    [np.nanmean((P[e][fi] - P[e][gf]) ** 2)
                     for e in np.where(fam_of == row["family"])[0]])))
            rows.append(row)
        th_df = pd.DataFrame(rows)
        fails = {r["model"]: (r["n_failed"], r["truncated"]) for r in th_res}
        th_df["n_failed"] = th_df["model"].map(lambda t: fails.get(t, (0, False))[0])
        th_df = th_df.merge(models_df[["model", "role", "fit_mse", "n_free_params",
                                       "code_lines"]], on="model", how="left")
        th_df.to_csv(args.out / "theories.csv", index=False)

        theory_coords.append(_space_coords(D_th, t_ok, "joint", "model", summary))
        pd.DataFrame(D_th, index=t_ok, columns=t_ok).to_csv(args.out / "theory_distances_joint.csv")
        coords.append(_space_coords(D_ex, ids[:-1], "joint", "exp_id", summary))
        _mantel("experiment:profile", "experiment:joint", Dp, D_ex)

        # Theory map on the Hilbig stimuli alone (best-fit predictions), for comparison.
        preds = pd.read_csv(args.models_dir / "predictions.csv",
                            dtype={"option_a": str, "option_b": str})
        wide = preds.pivot_table(index="model", columns=["option_a", "option_b"],
                                 values="p_b_fit")
        tpath = args.models_dir / "targets.csv"
        if tpath.is_file():  # the ground truths' P(B) on these pairs = the fit targets
            tg = pd.read_csv(tpath, dtype={"option_a": str, "option_b": str})
            tg = tg.pivot_table(index="family", columns=["option_a", "option_b"],
                                values="p_b_target")
            tg.index = [f"ground_truth/{f}" for f in tg.index]
            wide = pd.concat([wide, tg[wide.columns]])
        # Human families were fit to the Hilbig & Moshagen choices themselves.
        hc = human_counts(args.human_data)
        hk = pd.Series(hc["k"].to_numpy() / hc["n"].to_numpy(),
                       index=pd.MultiIndex.from_arrays(
                           [["".join(map(str, x)) for x in hc[c]] for c in ("option_a", "option_b")]))
        for f in human_fams:
            if ref_ids[f] not in wide.index:
                wide.loc[ref_ids[f]] = hk.reindex(wide.columns).to_numpy()
        hil_ids = [t for t in t_ok if t in wide.index]
        Xh = wide.loc[hil_ids].to_numpy(dtype=float)
        D_hil = euclidean_distances(Xh) / np.sqrt(Xh.shape[1])
        theory_coords.append(_space_coords(D_hil, hil_ids, "hilbig", "model", summary))
        pd.DataFrame(D_hil, index=hil_ids, columns=hil_ids).to_csv(
            args.out / "theory_distances_hilbig.csv")
        sel = [t_ok.index(t) for t in hil_ids]
        _mantel("theory:joint", "theory:hilbig", D_th[np.ix_(sel, sel)], D_hil)

        # Theory code: compression distance between predict() sources (comments
        # and docstrings stripped); ground truths join with their canonical code.
        src = {m.model_id: m.theory.predict_source for m in keep}
        for f in gt_fams:
            src[ref_ids[f]] = Theory.from_yaml(CANONICAL_YAML_DIR / f"{f}.yaml").predict_source
        code_ids = [t for t in t_ok if t in src]
        D_code = code_distances([_clean_source(src[t]) for t in code_ids])
        theory_coords.append(_space_coords(D_code, code_ids, "code", "model", summary))
        pd.DataFrame(D_code, index=code_ids, columns=code_ids).to_csv(
            args.out / "theory_distances_code.csv")
        sel = [t_ok.index(t) for t in code_ids]
        _mantel("theory:joint", "theory:code", D_th[np.ix_(sel, sel)], D_code)

        # Do experiments resemble each other when their proposers do?
        # (Model ids are bare labels for a single run, so look up by run + label.)
        pos = {t: i for i, t in enumerate(t_ok)}
        by_label = {(m.run, m.label): m.model_id for m in models}
        prop = [alias.get(by_label.get((e.run, e.proposer), ""), by_label.get((e.run, e.proposer), ""))
                for e in exps]
        has = [i for i in range(n) if prop[i] in pos]
        pi = [pos[prop[i]] for i in has]
        _mantel("experiment:profile", "proposer theory:joint",
                Dp[np.ix_(has, has)], D_th[np.ix_(pi, pi)])

        th_df["fit_mse_rank"] = th_df["fit_mse"].rank()
        best = (th_df[th_df["generation"] > 0].groupby("run")
                .agg(best_fit_mse=("fit_mse", "min"),
                     best_transfer_rmse=("transfer_rmse", "min")).reset_index())
        div_df = div_df.merge(best, on="run", how="left")
        ok_rows = th_df[(th_df["generation"] > 0) & th_df["transfer_rmse"].notna()]
        rho = ok_rows[["fit_mse", "transfer_rmse"]].corr(method="spearman").iloc[0, 1]
        print(f"  fit_mse (Hilbig) vs transfer_rmse (proposed experiments), "
              f"non-seed theories: Spearman {rho:.2f} (n = {len(ok_rows)})")
        rho2 = div_df[["profile_diversity", "best_fit_mse"]].corr(method="spearman").iloc[0, 1]
        print(f"  run profile_diversity vs best fit_mse: Spearman {rho2:.2f} "
              f"(n = {len(div_df)} runs)")

    div_df.to_csv(args.out / "run_diversity.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "mds_summary.csv", index=False)
    pd.DataFrame(mantel_rows).to_csv(args.out / "mantel.csv", index=False)
    exp_coords = pd.concat(coords, ignore_index=True)
    exp_coords.to_csv(args.out / "experiment_coords.csv", index=False)
    if theory_coords:
        tc = pd.concat(theory_coords, ignore_index=True)
        tc.to_csv(args.out / "theory_coords.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        names = {"profile": "discrimination profile", "features": "structural features",
                 "rationale": "rationale text (TF-IDF)", "joint": "disagreement among discovered theories"}
        share = {(s["objects"], s["space"]): s for s in summary}
        em = exp_df[["exp_id", "llm", "family", "run", "round"]].rename(columns={"round": "order"})
        em["seed"] = em["order"] == 0
        for space in exp_coords["space"].unique():
            c = exp_coords[exp_coords["space"] == space].merge(em, on="exp_id")
            refpt = c[c["exp_id"] == HILBIG_ID]
            ref_xy = {"*": (refpt["x"].iloc[0], refpt["y"].iloc[0])} if len(refpt) else None
            s = share[("experiments", space)]
            _plot_map(c[c["exp_id"] != HILBIG_ID], out_base=args.out / f"experiment_map_{_slug(space)}",
                      kind="experiment", ref=ref_xy, ref_label="Hilbig & Moshagen (2014)",
                      title=f"Experiments by {names[space]}  (classical MDS; 2-D share "
                            f"{s['variance_2d']:.0%}, stress {s['stress']:.2f})")
        if theory_coords:
            tm = th_df[["model", "llm", "family", "run", "generation"]].rename(
                columns={"generation": "order"})
            tm["seed"] = tm["order"] == 0
            for space, word in (("joint", "predictions on every proposed experiment"),
                                ("hilbig", "best-fit predictions on the Hilbig & Moshagen pairs"),
                                ("code", "code (compression distance between predict() sources)")):
                c = tc[tc["space"] == space].merge(tm, on="model")
                gt = c[c["llm"] == "ground_truth"]
                ref_xy = {r.family: (r.x, r.y) for r in gt.itertuples()}
                s = share[("theories", space)]
                ref_label = ("Human choices" if not gt_fams else
                             "Ground truth" if not human_fams else "Ground truth / human choices")
                _plot_map(c[c["llm"] != "ground_truth"], out_base=args.out / f"theory_map_{space}",
                          kind="theory", ref=ref_xy, ref_label=ref_label,
                          title=f"Theories by {word}  (classical MDS; 2-D share "
                                f"{s['variance_2d']:.0%}, stress {s['stress']:.2f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
