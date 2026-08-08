"""Preregistered confirmatory analysis for the concave value-function study.

This script implements the analysis plan exactly as registered. It consumes the
trial-level data produced by ``download_data.py`` and computes the three
preregistered confirmatory tests on participant-level derived variables.

The stimulus set is fixed in ``battery.json`` (generated once by ``build.py``).
It defines two SEPARATE, between-subjects experiments; each participant is in one
experiment only and completed that experiment's components under one validity
vector drawn from the five strictly-descending, all-distinct validity vectors,
counterbalanced across that experiment's participants. Each experiment is
collected and analyzed independently (``--experiment exp1|exp2``); each
participant-level statistic is a dimensionless choice proportion pooled across
validity vectors.

Hypotheses and tests (alpha = .05, one-tailed, predicted direction):

  Experiment 1 — Model comparison (pairwise)
  H1  Pairwise model discrimination.  The concave account is contrasted against
      each heuristic m in {WADD, tallying, TTB} separately, on m's own
      discriminating subset (stimuli where the concave point prediction is the
      option opposite to m). Per participant, cmr_m = concave-match rate and
      hmr_m = m-match rate on that subset; the difference D_m = cmr_m - hmr_m is
      tested with a one-sided paired t-test (H1: mean > 0). The three p-values
      (one per heuristic) are Holm-corrected.

  Experiment 2 — Value curvature (steep-vs-flat and level-shift)
  H2  Steep-vs-flat.  SteepChoiceRate_j = proportion of steep-vs-flat trials on
      which participant j chose the option whose advantage lies in the lower
      value region. One-sample t-test, H2: mean > 0.5.

  H3  Level-shift / offset.  OffsetEffect_j = TargetChoiceRate_j(lower) -
      TargetChoiceRate_j(shifted-up), where the target option is fixed per pair.
      One-sample t-test, H3: mean > 0.

Inclusion follows the registration: every participant who completed the study is
analyzed; participants with incomplete data are excluded. No response-pattern,
response-time, or performance exclusions are applied.

Usage:
  .venv/bin/python scripts/preregistration/analyse.py --data scripts/preregistration/data/trials.csv
  .venv/bin/python scripts/preregistration/analyse.py --data ... --experiment exp1
  .venv/bin/python scripts/preregistration/analyse.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
RESPONSE_A, RESPONSE_B = 0, 1
MODELS = ("concave", "wadd", "tallying", "ttb")
COMPARES = ("wadd", "tallying", "ttb")   # Exp 1 pairwise: concave vs each heuristic


# --------------------------------------------------------------------------- #
# Frozen design                                                               #
# --------------------------------------------------------------------------- #

def load_design(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "battery.json"), encoding="utf-8") as f:
        return json.load(f)


def build_lookup(design: dict) -> dict:
    """Map (vector_id, option_a, option_b) -> trial record with component labels.

    Each record carries the experiment ("exp1" / "exp2"), the component
    ("steep" / "offset" / "md") and the preregistered labels required to score
    that component. The experiment tag is what keeps the two between-subjects
    experiments separable."""
    # component -> experiment fallback, used only when a stimulus lacks its
    # "experiment" tag. battery.json stores experiments as {exp: {components,
    # reps, ...}}; tolerate the older {exp: [components]} shape too.
    comp_exp = {"md": "exp1", "steep": "exp2", "offset": "exp2"}
    for e, meta in design.get("experiments", {}).items():
        comps = meta["components"] if isinstance(meta, dict) else meta
        for c in comps:
            comp_exp[c] = e
    lookup: dict = {}
    for vid, spec in design["vectors"].items():
        for s in spec["steep"]:
            lookup[(vid, tuple(s["a"]), tuple(s["b"]))] = {
                "experiment": s.get("experiment", comp_exp["steep"]),
                "component": "steep", "steep_opt": s["steep_opt"]}
        for s in spec["offset"]:
            lookup[(vid, tuple(s["a"]), tuple(s["b"]))] = {
                "experiment": s.get("experiment", comp_exp["offset"]),
                "component": "offset", "target": s["target"], "cell": s["cell"]}
        for s in spec["md"]:
            lookup[(vid, tuple(s["a"]), tuple(s["b"]))] = {
                "experiment": s.get("experiment", comp_exp["md"]),
                "component": "md", "compare": s["compare"],
                **{m: s[m] for m in MODELS}}
    return lookup


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def load_trials(path: str) -> pd.DataFrame:
    """Load the trial-level table written by download_data.py. Required columns:
    participant, vector, option_a, option_b (JSON lists), response (0=A, 1=B).
    An optional `experiment` column (exp1/exp2) is preserved when present."""
    df = pd.read_csv(path)
    df["a"] = df["option_a"].apply(lambda s: tuple(json.loads(s)))
    df["b"] = df["option_b"].apply(lambda s: tuple(json.loads(s)))
    df["response"] = df["response"].astype(int)
    df["participant"] = df["participant"].astype(str)
    df["vector"] = df["vector"].astype(str)
    return df


def participant_table(df: pd.DataFrame, lookup: dict,
                      which: str = "both") -> pd.DataFrame:
    """One row per participant with the preregistered derived variables. The two
    experiments are separate between-subjects studies, so inclusion is applied
    per experiment: exp1 requires the model-discrimination trials; exp2 requires
    both steep-vs-flat and (lower & shifted) offset trials. `which` in
    {exp1, exp2, both} selects which experiment's inclusion + variables to build;
    when an `experiment` column is present it is also used to pre-filter rows."""
    need_md = which in ("both", "exp1")
    need_curv = which in ("both", "exp2")
    if which != "both" and "experiment" in df.columns:
        df = df[df["experiment"].astype(str) == which]
    rows = []
    for pid, g in df.groupby("participant"):
        steep_hits = []
        off_low, off_high = [], []
        # Exp 1 is pairwise: per heuristic m, the concave-match and the m-match
        # on m's own discriminating subset (where concave opposes m).
        cmr = {m: [] for m in COMPARES}     # concave-match on m's subset
        hmr = {m: [] for m in COMPARES}     # heuristic-m-match on m's subset
        for vec, a, b, resp in zip(g["vector"], g["a"], g["b"], g["response"]):
            rec = lookup.get((vec, a, b))
            if rec is None:
                continue
            if rec["component"] == "steep":
                steep_hits.append(int(resp == rec["steep_opt"]))
            elif rec["component"] == "offset":
                hit = int(resp == rec["target"])
                (off_low if rec["cell"] == "low" else off_high).append(hit)
            elif rec["component"] == "md":
                m = rec["compare"]
                cmr[m].append(int(resp == rec["concave"]))
                hmr[m].append(int(resp == rec[m]))
        has_md = all(cmr[m] for m in COMPARES)
        if need_md and not has_md:
            continue                       # incomplete exp1 participant -> excluded
        if need_curv and (not steep_hits or not off_low or not off_high):
            continue                       # incomplete exp2 participant -> excluded
        row = {"participant": pid}
        if steep_hits:
            row["steep_rate"] = float(np.mean(steep_hits))
        if off_low and off_high:
            row["offset_lower"] = float(np.mean(off_low))
            row["offset_shifted"] = float(np.mean(off_high))
            row["offset_effect"] = float(np.mean(off_low) - np.mean(off_high))
        if has_md:
            for m in COMPARES:
                row[f"cmr_{m}"] = float(np.mean(cmr[m]))   # concave-match vs m
                row[f"hmr_{m}"] = float(np.mean(hmr[m]))   # m-match
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #

def one_sample_greater(x: np.ndarray, mu: float) -> dict:
    """One-sample, one-sided (greater) t-test plus descriptive statistics."""
    x = np.asarray(x, float)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    res = stats.ttest_1samp(x, mu, alternative="greater")
    t = float(getattr(res, "statistic"))
    p = float(getattr(res, "pvalue"))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return {"n": n, "mean": mean, "se": se, "ci": (mean - tcrit * se, mean + tcrit * se),
            "t": t, "df": n - 1, "p": p, "d": (mean - mu) / sd if sd > 0 else float("nan")}


def holm(pvalues: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm step-down decisions for an ordered family of one-sided tests."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    decided = [False] * m
    for rank, idx in enumerate(order):
        if pvalues[idx] < alpha / (m - rank):
            decided[idx] = True
        else:
            break
    return decided


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def _fmt(r: dict) -> str:
    return (f"mean = {r['mean']:.4f}, SE = {r['se']:.4f}, "
            f"95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}], "
            f"t({r['df']}) = {r['t']:.3f}, p = {r['p']:.3e}, d = {r['d']:.3f}")


def analyse_experiment1(pt: pd.DataFrame) -> None:
    """Experiment 1 — pairwise model comparison (the preregistered FIRST
    experiment). For each heuristic m in {WADD, tallying, TTB}, on m's own
    discriminating subset (where the concave account predicts the option opposite
    to m) we compare the participant-level concave-match rate against the m-match
    rate.

    H1  Pairwise discrimination (one-sided paired t-tests, Holm-corrected;
        H1: ModelMatchRate_concave > ModelMatchRate_m on m's subset, for each m)."""
    print("=== Experiment 1 — Model comparison (pairwise) ===\n")
    if pt.empty or "cmr_wadd" not in pt.columns:
        print("    No model-discrimination data for these participants; "
              "skipping (is this an exp1 dataset?).\n")
        return
    print("H1  Pairwise model discrimination (one-sided paired t-tests, "
          "Holm-corrected;\n    H1: ModelMatchRate_concave > ModelMatchRate_m "
          "on each heuristic's own subset)")
    results = {m: one_sample_greater(
        (pt[f"cmr_{m}"] - pt[f"hmr_{m}"]).to_numpy(), 0.0) for m in COMPARES}
    decisions = holm([results[m]["p"] for m in COMPARES])
    for m, keep in zip(COMPARES, decisions):
        r = results[m]
        print(f"    concave vs {m:>8}: concave-match={pt[f'cmr_{m}'].mean():.4f} "
              f"{m}-match={pt[f'hmr_{m}'].mean():.4f} | {_fmt(r)}  Holm: "
              f"{'significant' if keep else 'not significant'}")
    print(f"    Result: {'supported' if all(decisions) else 'not supported'} "
          f"(all three pairwise comparisons significant after Holm correction).\n")


def analyse_experiment2(pt: pd.DataFrame) -> None:
    """Experiment 2 — value curvature (the preregistered SECOND experiment).

    H2  Steep-vs-flat (one-sample t-test, H2: SteepChoiceRate > 0.5).
    H3  Level-shift / offset (one-sample t-test, H3: OffsetEffect > 0)."""
    print("=== Experiment 2 — Value curvature (steep-vs-flat and level-shift) ===\n")
    if pt.empty or "steep_rate" not in pt.columns or "offset_effect" not in pt.columns:
        print("    No steep-vs-flat / level-shift data for these participants; "
              "skipping (is this an exp2 dataset?).\n")
        return
    print("H2  Steep-vs-flat (one-sample t-test, H2: SteepChoiceRate > 0.5)")
    h2 = one_sample_greater(pt["steep_rate"].to_numpy(), 0.5)
    print("    " + _fmt(h2))
    print(f"    Result: {'supported' if h2['p'] < ALPHA else 'not supported'} "
          f"at alpha = {ALPHA}.\n")

    print("H3  Level-shift / offset (one-sample t-test, H3: OffsetEffect > 0)")
    h3 = one_sample_greater(pt["offset_effect"].to_numpy(), 0.0)
    print("    " + _fmt(h3))
    print(f"    Lower-range target-choice rate = {pt['offset_lower'].mean():.4f}; "
          f"shifted-up target-choice rate = {pt['offset_shifted'].mean():.4f}.")
    print(f"    Result: {'supported' if h3['p'] < ALPHA else 'not supported'} "
          f"at alpha = {ALPHA}.\n")


def analyse(pt: pd.DataFrame, which: str = "both") -> None:
    """Run the preregistered analyses. ``which`` selects the experiment:
    'exp1' (model comparison), 'exp2' (value curvature), or 'both'. The two are
    separate between-subjects studies; pass the matching experiment so inclusion
    and the reported tests line up with the data."""
    print(f"Participants analyzed: {len(pt)}\n")
    if which in ("both", "exp1"):
        analyse_experiment1(pt)
    if which in ("both", "exp2"):
        analyse_experiment2(pt)


# --------------------------------------------------------------------------- #
# Self-test on synthetic data with known ground truth                         #
# --------------------------------------------------------------------------- #

def _simulate_trials(design: dict, exp: str, *, concave: bool,
                     seed: int) -> pd.DataFrame:
    """Generate trial-level data for ONE experiment under a concave (alpha<1) or
    linear (alpha=1) generating model with a 30% lapse, at that experiment's
    preregistered n_per_vector and per-stimulus reps."""
    rng = np.random.default_rng(seed)
    espec = design["experiments"][exp]
    reps = espec["reps"]; components = espec["components"]
    n_per_vector = espec["n_per_vector"]; lapse = 0.30
    rows = []
    for vid, spec in design["vectors"].items():
        v = np.asarray(spec["validities"], float); vn = v / v.sum()
        for k in range(n_per_vector):
            pid = f"{vid}_{k}"
            alpha = rng.uniform(0.3, 0.7) if concave else 1.0
            shift = rng.uniform(0.5, 3.0); beta = rng.uniform(4.0, 12.0)

            def emit(a, b, r):
                a = np.asarray(a, float); b = np.asarray(b, float)
                u_a = np.power(a + shift, alpha) - shift ** alpha
                u_b = np.power(b + shift, alpha) - shift ** alpha
                diff = float(((u_b - u_a) * vn).sum())
                p_b = (1 - lapse) / (1 + np.exp(-beta * diff)) + lapse * 0.5
                for _ in range(r):
                    resp = RESPONSE_B if rng.random() < p_b else RESPONSE_A
                    rows.append({"participant": pid, "vector": vid,
                                 "experiment": exp,
                                 "option_a": json.dumps(list(map(int, a))),
                                 "option_b": json.dumps(list(map(int, b))),
                                 "response": resp})

            for c in components:
                for s in spec[c]:
                    emit(s["a"], s["b"], reps[c])
    return pd.DataFrame(rows)


def selftest(which: str = "both") -> None:
    design = load_design(); lookup = build_lookup(design)
    exps = ["exp1", "exp2"] if which == "both" else [which]
    seed = 1
    for exp in exps:
        for world in ("concave", "linear"):
            n = design["experiments"][exp]["n_total"]
            print(f"=========== {exp} ({design['experiments'][exp]['label']}), "
                  f"N={n}, synthetic ground truth: {world.upper()} ===========")
            raw = _simulate_trials(design, exp, concave=(world == "concave"),
                                   seed=seed)
            seed += 1
            raw["a"] = raw["option_a"].apply(lambda s: tuple(json.loads(s)))
            raw["b"] = raw["option_b"].apply(lambda s: tuple(json.loads(s)))
            raw["response"] = raw["response"].astype(int)
            raw["participant"] = raw["participant"].astype(str)
            raw["vector"] = raw["vector"].astype(str)
            analyse(participant_table(raw, lookup, exp), exp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="trial-level CSV from download_data.py "
                                    "(one experiment's data/<exp>/trials.csv)")
    ap.add_argument("--experiment", choices=["exp1", "exp2", "both"],
                    default="both",
                    help="which preregistered experiment to analyze: exp1 "
                         "(model comparison), exp2 (value curvature), or both. "
                         "For real data this must match the dataset's experiment.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the analysis on synthetic data with known truth")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.experiment)
    elif args.data:
        design = load_design()
        analyse(participant_table(load_trials(args.data), build_lookup(design),
                                  args.experiment), args.experiment)
    else:
        ap.error("pass one of --data or --selftest")


if __name__ == "__main__":
    main()
