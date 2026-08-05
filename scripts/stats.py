"""Compute every statistic needed to back the claims/figures in main.tex.

This script reproduces the *figures' own* aggregation (it reuses the helpers in
scripts/recovery_correlation.py and reads the committed eval artifacts that the
plotting scripts read), so the numbers here match Figures 3-5 rather than being
re-derived. Each computed quantity is tagged with the main.tex line that carries
the STAT / XX / YY placeholder it fills.

Outputs (under results/stats/):
  stats_results.csv     one row per reported quantity (value, sem, n, source)
  stats_summary.png     the same table rendered as a figure
  stats_summary.pdf

Run:  python stats.py
"""
from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.recovery_correlation import (  # noqa: E402
    _best_surfaced_by_metric,
    summarise,
)

BIN = REPO / "results/recovery"
CS1 = REPO / "results/human_decision_making_binary/ttb+wadd"            # binary human run (NLSWM)
CS2 = REPO / "results/human_decision_making_cardinal/ttb+tallying"  # cardinal human run

OUT = REPO / "results" / "stats"
OUT.mkdir(parents=True, exist_ok=True)

ROWS: list[dict] = []


def add(section: str, line: str, quantity: str, value, sem=None, n=None, source: str = ""):
    """Collect one reported quantity. `value`/`sem` may be floats or pre-formatted strings."""
    ROWS.append(
        dict(section=section, paper_line=line, quantity=quantity,
             value=value, sem=sem, n=n, source=source)
    )


def fmt(v, sem=None, nd=4):
    if isinstance(v, str):
        return v
    s = f"{v:.{nd}f}"
    if sem is not None and not (isinstance(sem, float) and np.isnan(sem)):
        s += f" ± {sem:.{nd}f}"
    return s


# ---------------------------------------------------------------------------
# Pooled two-stage recovery aggregation (matches summarise(), but collapses
# family so we get one mean +/- SEM across all run-dirs at a noise level).
# ---------------------------------------------------------------------------
def pooled_recovery(long_df: pd.DataFrame, noise: float, metric: str = "mse") -> pd.DataFrame:
    df = long_df[np.isclose(long_df["noise"], noise)].copy()
    best = _best_surfaced_by_metric(
        df, metric=metric, higher_is_better=False,
        group_keys=("family", "noise", "run_dir"),
        role_label="surfaced (best per run)",
    )
    aug = pd.concat([df, best], ignore_index=True)
    aug = aug[aug[metric].notna()]
    per_run = aug.groupby(["run_dir", "role"], observed=True)[metric].mean().reset_index()
    g = per_run.groupby("role", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"])
    return g.set_index("role")


# ===========================================================================
# CLUSTER A -- canonical recovery (Fig 3A) + LLM-judge mechanism (Fig 3B)
# ===========================================================================
def cluster_a():
    print("\n" + "=" * 78 + "\nCLUSTER A  canonical recovery + LLM-judge\n" + "=" * 78)
    long = pd.read_csv(BIN / "analysis/recovery_vs_clean_gt/recovery_long.csv")
    src = "recovery_vs_clean_gt/recovery_long.csv  (MSE_pB vs clean GT)"

    # ---- line 218: noise-free, pooled across TTB/WADD/Tallying ----
    g0 = pooled_recovery(long, 0.0, "mse")
    role_map = {
        "seed": "seed (baseline)",
        "surfaced": "surfaced",
        "surfaced (best per run)": "best-surfaced",
        "gt_clean": "ceiling (GT vs itself)",
        "random": "floor (random policy)",
    }
    for role, label in role_map.items():
        if role in g0.index:
            r = g0.loc[role]
            add("A. Canonical recovery (Fig 3A)", "218",
                f"MSE_pB @ noise=0, pooled, {label}",
                float(r["mean"]), float(r["sem"]), int(r["count"]), src)
            print(f"  line218  {label:24s} {fmt(r['mean'], r['sem'])}  (n={int(r['count'])})")

    # ---- line 223: pooled MSE_pB across noise levels ----
    for noise in (0.0, 0.5, 0.75):
        g = pooled_recovery(long, noise, "mse")
        for role, label in (("surfaced", "surfaced"), ("seed", "seed"),
                            ("surfaced (best per run)", "best-surfaced")):
            if role in g.index:
                r = g.loc[role]
                add("A. Canonical recovery vs noise (Fig 3A)", "223",
                    f"MSE_pB @ noise={noise}, pooled, {label}",
                    float(r["mean"]), float(r["sem"]), int(r["count"]), src)
        s = g.loc["surfaced"]
        print(f"  line223  surfaced @ eps={noise}: {fmt(s['mean'], s['sem'])}")

    # ---- line 226: LLM-judge mechanism similarity, pooled (description mode) ----
    rows = []
    for f in sorted(glob(str(BIN / "*_sampling/noise=*/judge_similarity_desc.csv"))):
        noise = float(Path(f).parent.name.split("=")[1])
        d = pd.read_csv(f)
        d = d[["similarity"]].dropna().assign(noise=noise)
        rows.append(d)
    jdf = pd.concat(rows, ignore_index=True)
    jsrc = "*_sampling/noise=*/judge_similarity_desc.csv  (description mode; per-theory mean of 3 votes)"
    for noise in sorted(jdf["noise"].unique()):
        sub = jdf[jdf["noise"] == noise]["similarity"]
        mean, sem, nn = sub.mean(), sub.std(ddof=1) / np.sqrt(len(sub)), len(sub)
        add("A. LLM-judge mechanism similarity (Fig 3B)", "226",
            f"judge similarity (0-1) @ noise={noise}, pooled", float(mean), float(sem), int(nn), jsrc)
        print(f"  line226  judge sim @ eps={noise}: {fmt(mean, sem)}  (n={nn})")


# ===========================================================================
# CLUSTER B -- non-canonical recovery (Fig 3C)
# ===========================================================================
def cluster_b():
    print("\n" + "=" * 78 + "\nCLUSTER B  non-canonical recovery (Fig 3C)\n" + "=" * 78)
    # Use the table the FIGURE was rendered from (per_model/recovery_long.csv),
    # NOT recovery_noncanonical_all.csv: the latter is an EARLY-cycle snapshot
    # (anti_majority from pi_3..pi_7 -> 0.317), whereas the figure uses the
    # EXTENDED-cycle runs (anti_majority up to pi_22 -> 0.214). cue_parity and
    # anti_majority are the only families given extra cycles here.
    long = pd.read_csv(BIN / "analysis/per_model/recovery_long.csv")
    long = long[np.isclose(long["noise"], 0.0)]
    src = "per_model/recovery_long.csv  (figure source; MSE_pB vs clean GT, eps=0)"
    best = _best_surfaced_by_metric(
        long, metric="mse", higher_is_better=False,
        group_keys=("family", "noise", "run_dir"),
        role_label="surfaced (best per run)",
    )
    aug = pd.concat([long, best], ignore_index=True)
    summ = summarise(aug, metric="mse")
    summ = summ[np.isclose(summ["noise"], 0.0)]
    order = ["alternating", "perseveration", "take_the_worst", "single_cue", "anti_majority", "cue_parity"]
    for fam in order:
        sub = summ[summ["family"] == fam]
        if sub.empty:
            continue
        for role, label in (("seed", "seed"), ("surfaced", "surfaced"),
                            ("surfaced (best per run)", "best-surfaced"),
                            ("gt", "ceiling"), ("random", "floor")):
            r = sub[sub["role"] == role]
            if not r.empty:
                r = r.iloc[0]
                add("B. Non-canonical recovery (Fig 3C)", "231-232",
                    f"MSE_pB {fam} / {label}", float(r["mean"]), float(r["sem"]),
                    int(r["n_runs"]), src)
        srow = sub[sub["role"] == "surfaced"]
        if not srow.empty:
            srow = srow.iloc[0]
            print(f"  {fam:16s} surfaced {fmt(srow['mean'], srow['sem'])}  (n={int(srow['n_runs'])})")


def cycle_effect_lme(run_dir: Path, *, section: str, line: str, label: str):
    """Per-(cycle, experiment, model) fit table -> per-cycle best-surfaced
    mean +/- SEM (across the 10 experiments) and a mixed-effects test of the
    cycle effect: metric ~ cycle + (1|experiment), over all non-seed surfaced
    theories. Backs the 'continue to get better over cycles' claim for either
    human run. Negative beta on MSE / positive on r = improvement."""
    pe = pd.read_csv(run_dir / "analysis/eval_run_self/per_experiment_summary_all.csv")
    surf = pe[(pe["role"] == "surfaced") & (~pe["model"].isin(["pi_1", "pi_2"]))].copy()
    src = f"{label}/analysis/eval_run_self/per_experiment_summary_all.csv (non-seed surfaced)"
    for metric, best in (("mse", "min"), ("pearson_r", "max")):
        for c, g in surf.groupby("cutoff_round"):
            means = g.groupby("model")[metric].mean()
            pick = means.idxmin() if best == "min" else means.idxmax()
            v = g[g["model"] == pick][metric]
            add(f"{section} per-cycle best-surfaced", line,
                f"{metric}_pB best-surfaced @ cycle {int(c)} ({pick})",
                float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), int(len(v)), src)
    try:
        import statsmodels.formula.api as smf
        import warnings
        for metric in ("mse", "pearson_r"):
            d = surf.rename(columns={metric: "y", "cutoff_round": "cycle"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm("y ~ cycle", d, groups=d["experiment"]).fit()
            b, se, p = m.params["cycle"], m.bse["cycle"], m.pvalues["cycle"]
            add(f"{section} cycle effect (LME)", line,
                f"{metric}_pB ~ cycle + (1|experiment): beta_cycle",
                f"{b:+.5f} (SE {se:.5f}), z={b / se:+.2f}, p={p:.3g}, n={len(d)}",
                None, int(d["experiment"].nunique()), src)
            print(f"  [{label}] LME {metric:9s}: beta_cycle={b:+.5f} SE={se:.5f} "
                  f"z={b / se:+.2f} p={p:.3g}")
    except Exception as e:
        print(f"  [{label}] LME skipped: {type(e).__name__}: {e}", file=sys.stderr)


# ===========================================================================
# CLUSTER C -- case study 1, binary human run (Fig 4); line 273
# ===========================================================================
def cluster_c():
    print("\n" + "=" * 78 + "\nCLUSTER C  case study 1 (binary, NLSWM) Fig 4\n" + "=" * 78)
    traj = pd.read_csv(CS1 / "analysis/eval_run_self/trajectory_all.csv")
    tsrc = "ttb+wadd/analysis/eval_run_self/trajectory_all.csv  (mean across 10 experiments)"
    # gold survivor = pi_4 (NLSWM), admitted cycle 2, constant thereafter
    for metric in ("mse", "pearson_r"):
        sub = traj[(traj["model"] == "pi_4") & (traj["metric"] == metric)].sort_values("cutoff_round")
        if sub.empty:
            continue
        first, last = sub.iloc[0], sub.iloc[-1]
        add("C. Case study 1 survivor pi_4 NLSWM (Fig 4C/D)", "273",
            f"{metric}_pB pi_4 @ cycle {int(last['cutoff_round'])} (final)",
            float(last["mean_across_exps"]), float(last["sem_across_exps"]),
            int(last["n_experiments"]), tsrc)
        print(f"  pi_4 {metric:9s} final cycle {int(last['cutoff_round'])}: "
              f"{fmt(last['mean_across_exps'], last['sem_across_exps'])}")

    # "gets better over cycles" = best surfaced theory at each cycle
    surf = traj[traj["role"].isin(["survivor", "surfaced"])] if "role" in traj else traj
    for metric, better in (("mse", "min"), ("pearson_r", "max")):
        sub = traj[traj["metric"] == metric]
        # best surfaced per cutoff among non-seed models (pi_3..)
        nonseed = sub[~sub["model"].isin(["pi_1", "pi_2"])]
        agg = nonseed.groupby("cutoff_round")["mean_across_exps"].agg(better).reset_index()
        if len(agg) >= 2:
            c_first, c_last = agg.iloc[0], agg.iloc[-1]
            add("C. Case study 1 best-surfaced trajectory (Fig 4C/D)", "273",
                f"best surfaced {metric}_pB cycle {int(c_first['cutoff_round'])} -> {int(c_last['cutoff_round'])}",
                f"{c_first['mean_across_exps']:.4f} -> {c_last['mean_across_exps']:.4f}",
                None, None, tsrc)
            print(f"  best-surfaced {metric}: {c_first['mean_across_exps']:.4f} -> {c_last['mean_across_exps']:.4f}")

    # per-cycle best-surfaced mean +/- SEM + mixed-effects cycle test
    cycle_effect_lme(CS1, section="C. Case study 1", line="273", label="ttb+wadd")

    # final leaderboard (Fig 4J), pooled-stimulus eval_hilbig
    lb = pd.read_csv(CS1 / "analysis/eval_hilbig/eval_hilbig_summary.csv")
    for _, r in lb.iterrows():
        add("C. Case study 1 final leaderboard (Fig 4J)", "273",
            f"{r['model']} ({r['role']}): MSE / r",
            f"MSE={r['mse']:.4f}, r={r['pearson_r']:.3f}, MAE={r['mae']:.3f}",
            None, int(r["n_stimuli"]), "ttb+wadd/analysis/eval_hilbig/eval_hilbig_summary.csv")
    add("C. Case study 1 reliability floor (Fig 4D)", "273",
        "human-noise MSE floor", *reliability_floor(CS1))


def reliability_floor(run_dir: Path):
    """Irreducible MSE from human sampling noise = mean over final-cutoff stimuli
    of p_b_human_sem**2 (Var of the empirical proportion). Returns (value, sem, n, src)."""
    f = run_dir / "analysis/eval_run_self/calibration_points_all.csv"
    if not f.exists():
        return (np.nan, None, None, "n/a")
    d = pd.read_csv(f)
    last = d[d["cutoff_round"] == d["cutoff_round"].max()]
    stim = last.drop_duplicates(["experiment", "option_a", "option_b"])
    floor = float((stim["p_b_human_sem"] ** 2).mean())
    print(f"  reliability floor MSE ~ {floor:.4f}  (n={len(stim)} stimuli)")
    return (floor, None, len(stim), "calibration_points_all.csv: mean(p_b_human_sem^2)")


# ===========================================================================
# CLUSTER D -- case study 2, cardinal human run (Fig 5); lines 354, 372
# ===========================================================================
def cluster_d():
    print("\n" + "=" * 78 + "\nCLUSTER D  case study 2 (cardinal, DR-WADD) Fig 5\n" + "=" * 78)
    traj = pd.read_csv(CS2 / "analysis/eval_run_self/trajectory_all.csv")
    tsrc = "ttb+tallying/analysis/eval_run_self/trajectory_all.csv  (mean across 10 experiments)"
    survivors = {"pi_6": "Satisficing WADD", "pi_7": "Diminishing-Returns WADD"}

    # per-cycle best-surfaced mean +/- SEM + mixed-effects cycle test (line 354)
    cycle_effect_lme(CS2, section="D. Case study 2", line="354", label="ttb+tallying")

    # line 354: improvement over cycles for the concave/DR slot (WADD pi_3 cycle1 -> pi_7 cycle5)
    for metric in ("mse", "pearson_r"):
        w = traj[(traj["model"] == "pi_3") & (traj["metric"] == metric)].sort_values("cutoff_round")
        f7 = traj[(traj["model"] == "pi_7") & (traj["metric"] == metric)].sort_values("cutoff_round")
        if not w.empty and not f7.empty:
            add("D. Case study 2 improvement over cycles (Fig 5C/D)", "354",
                f"{metric}_pB WADD(c{int(w.iloc[0]['cutoff_round'])}) -> DR-WADD(c{int(f7.iloc[-1]['cutoff_round'])})",
                f"{w.iloc[0]['mean_across_exps']:.4f} -> {f7.iloc[-1]['mean_across_exps']:.4f}",
                None, None, tsrc)
            print(f"  {metric}: WADD {w.iloc[0]['mean_across_exps']:.4f} -> DR-WADD {f7.iloc[-1]['mean_across_exps']:.4f}")

    # line 354: two survivors "almost indistinguishable" at final cycle (mean-across-exps)
    last_cut = traj["cutoff_round"].max()
    for model, name in survivors.items():
        for metric in ("mse", "pearson_r"):
            r = traj[(traj["model"] == model) & (traj["metric"] == metric)
                     & (traj["cutoff_round"] == last_cut)]
            if not r.empty:
                r = r.iloc[0]
                add("D. Case study 2 final survivors (Fig 5C/D)", "354",
                    f"{metric}_pB {model} {name} @ final cycle",
                    float(r["mean_across_exps"]), float(r["sem_across_exps"]),
                    int(r["n_experiments"]), tsrc)

    # lines 372 + final scatter: pooled r/mse/mae over calibration points + ambivalence
    cal = pd.read_csv(CS2 / "analysis/eval_run_self/calibration_points_all.csv")
    cal = cal[cal["cutoff_round"] == cal["cutoff_round"].max()]
    csrc = "ttb+tallying/analysis/eval_run_self/calibration_points_all.csv (final cutoff, pooled stimuli)"
    for model, name in survivors.items():
        sub = cal[cal["model"] == model]
        if sub.empty:
            continue
        h, m = sub["p_b_human"].to_numpy(), sub["p_b_model"].to_numpy()
        pooled_r = float(np.corrcoef(h, m)[0, 1])
        pooled_mse = float(np.mean((h - m) ** 2))
        pooled_mae = float(np.mean(np.abs(h - m)))
        n = len(sub)
        ambiv = int(((m >= 0.45) & (m <= 0.55)).sum())
        add("D. Case study 2 pooled scatter (Fig 5E/F)", "372",
            f"{model} {name}: pooled r", f"{pooled_r:.3f}", None, n, csrc)
        add("D. Case study 2 pooled scatter (Fig 5E/F)", "372",
            f"{model} {name}: pooled MSE / MAE", f"MSE={pooled_mse:.4f}, MAE={pooled_mae:.3f}", None, n, csrc)
        add("D. Case study 2 ambivalence (Fig 5E/F)", "372",
            f"{model} {name}: predictions in p(B) in [0.45,0.55]",
            f"{ambiv}/{n} = {ambiv / n:.1%}", None, n, csrc)
        print(f"  {model} {name:26s} pooled r={pooled_r:.3f} MSE={pooled_mse:.4f} "
              f"MAE={pooled_mae:.3f} ambiv={ambiv}/{n} ({ambiv / n:.0%})")


# ===========================================================================
# Render
# ===========================================================================
def render():
    df = pd.DataFrame(ROWS)
    df["value_str"] = [fmt(v, s) if not isinstance(v, str) else v
                       for v, s in zip(df["value"], df["sem"])]
    csv_path = OUT / "stats_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = df[["section", "paper_line", "quantity", "value_str", "n", "source"]].copy()
    tbl.columns = ["Section", "Line", "Quantity", "Value (mean ± SEM)", "n", "Source"]
    tbl["Source"] = tbl["Source"].str.slice(0, 52)
    tbl["Quantity"] = tbl["Quantity"].str.slice(0, 52)

    nrows = len(tbl)
    fig, ax = plt.subplots(figsize=(20, 0.34 * nrows + 1.2))
    ax.axis("off")
    t = ax.table(cellText=tbl.values, colLabels=tbl.columns, loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.25)
    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2f3e46")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f4f4")
    # shade section boundaries
    prev = None
    for i, sec in enumerate(tbl["Section"], start=1):
        if sec != prev:
            for col in range(len(tbl.columns)):
                t[(i, col)].set_facecolor("#dbe7e4")
            prev = sec
    col_w = [0.20, 0.04, 0.26, 0.18, 0.04, 0.28]
    for (row, col), cell in t.get_celld().items():
        cell.set_width(col_w[col])
    ax.set_title("AutoCog paper: computed statistics for STAT / XX / YY placeholders",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"stats_summary.{ext}", dpi=160, bbox_inches="tight")
        print(f"Wrote {OUT / f'stats_summary.{ext}'}")


# ===========================================================================
# CLUSTER F -- preregistered confirmatory tests H1/H2/H3 (Fig 5 bottom).
# The plot scripts draw bars+SE only; the actual significance tests are
# computed here. These back the "confirmed prospectively in a preregistered
# study" claim (Significance Statement + Discussion line 493), which currently
# reports NO test statistics in the prose.
# ===========================================================================
def cluster_f():
    print("\n" + "=" * 78 + "\nCLUSTER F  preregistration tests H1/H2/H3 (Fig 5 bottom)\n" + "=" * 78)
    from scipy import stats as st
    pre = CS2 / "preregistration_visualization/data"

    # ---- H1: does the discovered model (pi_7) out-predict WADD / Tallying / TTB?
    # On disagreement trials, test whether the human match-rate to pi_7 exceeds
    # chance (0.5). Counts unavailable -> z-test from reported rate & SE.
    h1 = pd.read_csv(pre / "h1_results.csv")
    h1src = "preregistration_visualization/data/h1_results.csv (match-rate +/- SE)"
    for comp in ("wadd", "tallying", "ttb"):
        row = h1[(h1["comparison"] == comp) & (h1["model"] == "pi_7")]
        if row.empty:
            continue
        rate, se = float(row["match_rate"].iloc[0]), float(row["se"].iloc[0])
        z = (rate - 0.5) / se
        p_two = 2 * st.norm.sf(abs(z))
        add("F. Preregistration H1 model discrimination (Fig 5)", "493 / Sig.Stmt",
            f"pi_7 match-rate vs {comp.upper()} (vs 0.5)",
            f"{rate:.3f} ± {se:.3f}, z={z:.2f}, p={p_two:.3g}", None, None, h1src)
        print(f"  H1 vs {comp.upper():9s} rate={rate:.3f}±{se:.3f}  z={z:.2f}  p={p_two:.4f}")

    # ---- H2: steep-vs-flat. Per-participant rate of choosing in the steep region;
    # test against indifference (0.5) across participants (one-sample t-test).
    h2 = pd.read_csv(pre / "h2_participants.csv")
    s = h2["steep_rate"].to_numpy()
    t, p = st.ttest_1samp(s, 0.5)
    d = (s.mean() - 0.5) / s.std(ddof=1)
    add("F. Preregistration H2 steep-vs-flat (Fig 5)", "493 / Sig.Stmt",
        "steep-region choice rate vs 0.5",
        f"mean={s.mean():.3f} ± {s.std(ddof=1) / np.sqrt(len(s)):.3f}, "
        f"t({len(s) - 1})={t:.2f}, p={p:.3g}, d={d:.2f}", None, int(len(s)),
        "h2_participants.csv: one-sample t-test")
    print(f"  H2 steep-rate mean={s.mean():.3f}  t({len(s) - 1})={t:.2f}  p={p:.4g}  d={d:.2f}")

    # ---- H3: diminishing preference. Per-participant level-shift effect; test
    # against zero (one-sample t-test + Wilcoxon).
    h3 = pd.read_csv(pre / "h3_participants.csv")
    e = h3["offset_effect"].to_numpy()
    t3, p3 = st.ttest_1samp(e, 0.0)
    w, pw = st.wilcoxon(e)
    d3 = e.mean() / e.std(ddof=1)
    add("F. Preregistration H3 diminishing preference (Fig 5)", "493 / Sig.Stmt",
        "level-shift effect vs 0",
        f"mean={e.mean():.4f} ± {e.std(ddof=1) / np.sqrt(len(e)):.4f}, "
        f"t({len(e) - 1})={t3:.2f}, p={p3:.3g}, d={d3:.2f}; Wilcoxon p={pw:.3g}",
        None, int(len(e)), "h3_participants.csv: one-sample t-test + Wilcoxon")
    print(f"  H3 effect mean={e.mean():.4f}  t({len(e) - 1})={t3:.2f}  p={p3:.4g}  "
          f"d={d3:.2f}  Wilcoxon p={pw:.4g}")


def main():
    for fn in (cluster_a, cluster_b, cluster_c, cluster_d, cluster_f):
        try:
            fn()
        except Exception as e:  # keep going; report which cluster failed
            print(f"!! {fn.__name__} failed: {type(e).__name__}: {e}", file=sys.stderr)
    render()


if __name__ == "__main__":
    main()
