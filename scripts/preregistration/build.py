"""Provenance/build script: generates the frozen battery (battery.json) and runs
the preregistered power simulation (30% observation noise) to fix N per
experiment. Run once; not part of the run/analyse pipeline.

The study is TWO SEPARATE, between-subjects experiments. Each participant is in
ONE experiment only and sees ONLY that experiment's stimuli, replicated to ~96
trials. Both experiments still use the same five strictly-descending,
all-distinct validity vectors {.5,.6,.7,.8,.9}, counterbalanced across that
experiment's participants (one vector per participant).

  Experiment 1 (model comparison)  : a PAIRWISE battery of 12 model-
      discrimination ("md") stimuli per vector (4 per heuristic: 2 base + 2
      mirrors for concave-vs-WADD, concave-vs-tallying, concave-vs-TTB). List
      multiplicity md x1 -> 12-entry list -> replicated to 96 -> per-stimulus
      reps md=8 (32 trials per pairwise comparison).
  Experiment 2 (value curvature)    : 8 steep + 8 offset stimuli per vector
      (4 steep-vs-flat counterbalanced pairs with distinct gain sizes d=1..4,
      and 4 level-shift pairs across distinct low/high ranges, each pair = a low
      and a shifted-up cell). List multiplicity steep x1, offset x2 -> 24-entry
      list -> replicated to 96 -> per-stimulus reps steep=4, offset=8 (steep 32
      / offset 64 trials; the subtle offset effect keeps more trials).

For each experiment the power simulation reports the smallest multiple of 5 (5
balanced vectors) reaching the target power (>= 99%) for that experiment's
confirmatory test(s) under the concave generating model with 30% lapse. The
PLANNED sample we will actually collect is set in ``N_COLLECT`` (>= that power
minimum, as a conservative margin) and written into battery.json as each
experiment's "n_per_vector"/"n_total"; the power minimum is also stored as
"n_power_min" for provenance. Each stimulus is tagged with its "experiment" and
battery.json's "experiments" map records the components, multiplicity, reps and
N per experiment.
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search

RMAX = 5
TARGET_TRIALS = 96               # trials per participant (per experiment)
TARGET_POWER = 0.99              # per-experiment target for every confirmatory test
N_GRID = (10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100)   # total N (multiples of 5)
# Planned sample to collect per experiment (total, multiple of 5 for the 5
# balanced vectors). Set >= the power-derived minimum; the power minimum is still
# computed and stored as n_power_min for provenance.
N_COLLECT = {"exp1": 50, "exp2": 100}

# Experiment 1 is a PAIRWISE model comparison: the concave account is contrasted
# against each heuristic separately, on that heuristic's own discriminating
# stimuli. COMPARES lists the three alternatives; for each, per validity vector
# we build MD_BASE_PER_MODEL base pairs (+ their A/B mirrors) on which the concave
# account opposes that heuristic on >= MD_MIN_GRID of the (alpha<1, shift) grid
# and the heuristic makes a decisive (non-tie) choice. So md has
# 2 * MD_BASE_PER_MODEL * len(COMPARES) = 12 pairs per vector.
COMPARES = ("wadd", "tallying", "ttb")
MD_BASE_PER_MODEL = 2
MD_MIN_GRID = 0.85
N_MD_PER_VECTOR = 2 * MD_BASE_PER_MODEL * len(COMPARES)   # = 12

# Experiment 2 stimulus variety. Steep-vs-flat: STEEP_N_PAIRS counterbalanced
# pairs, each carrying a DISTINCT gain size d (drawn from STEEP_DELTAS) on a
# distinct cue pair, so the steep/flat trade-off is probed at several points of
# the value scale. Level-shift: pairs are drawn across several (low_max, shift)
# range settings so the absolute low/high value ranges differ across pairs
# (e.g. 0-2 vs 3-5, 0-3 vs 2-5, 0-1 vs 4-5). Each setting is (low_max, shift,
# n_pairs); the n_pairs sum to the number of level-shift pairs per vector.
STEEP_DELTAS = (1, 2, 3, 4)
STEEP_N_PAIRS = 4
OFFSET_SETTINGS = ((2, 3, 2), (3, 2, 1), (1, 4, 1))   # sum n_pairs = 4
# Tied filler values for the two non-trade-off ("context") cues of each
# steep-vs-flat pair, one (c0, c1) per pair. These cues hold the SAME value in
# both options, so they cancel in every model (WADD/tallying/TTB/concave) and
# leave the predictions untouched; varying them just makes the stimuli look less
# degenerate than the all-zero default (e.g. [2,3,..] instead of [0,0,..]).
STEEP_CONTEXTS = ((2, 3), (3, 1), (1, 4), (4, 2))

# Two SEPARATE between-subjects experiments. Each participant does ONE experiment.
# Component list multiplicity is per experiment; per-stimulus reps =
# multiplicity * (TARGET_TRIALS // list_len), via the experiment class's
# replicate-to-MAX_TRIALS=96 mechanism.
EXP_COMPONENTS = {"exp1": ["md"], "exp2": ["steep", "offset"]}
EXP_MULT = {"exp1": {"md": 1}, "exp2": {"steep": 1, "offset": 2}}
EXP_LABEL = {"exp1": "model comparison", "exp2": "value curvature"}
EXP_TESTS = {"exp1": [f"vs_{m}" for m in COMPARES], "exp2": ["steep", "offset"]}
N_STIM_PER_VECTOR = {"steep": 2 * STEEP_N_PAIRS,
                     "offset": 2 * sum(n for *_, n in OFFSET_SETTINGS),
                     "md": N_MD_PER_VECTOR}
COMPONENT_EXPERIMENT = {c: e for e, cs in EXP_COMPONENTS.items() for c in cs}
VALUE_SET = [0.5, 0.6, 0.7, 0.8, 0.9]
VECTORS = [tuple(sorted(c, reverse=True))
           for c in itertools.combinations(VALUE_SET, 4)]            # 5 vectors

LAPSE = 0.30                                     # 30% random responses (obs noise)
ALPHA_RANGE = (0.30, 0.70)
SHIFT_RANGE = (0.5, 3.0)
BETA_RANGE = (4.0, 12.0)


def reps_for(exp: str) -> tuple[dict, int, int]:
    """Per-stimulus reps for one experiment: (reps_by_component, list_len, repl).
    reps[c] = multiplicity[c] * (TARGET_TRIALS // list_len)."""
    mult = EXP_MULT[exp]
    list_len = sum(mult[c] * N_STIM_PER_VECTOR[c] for c in EXP_COMPONENTS[exp])
    repl = TARGET_TRIALS // list_len
    return {c: mult[c] * repl for c in EXP_COMPONENTS[exp]}, list_len, repl


def _p_choose_b(a, b, v, alpha, shift, beta):
    """P(choose B) under P7 with lapse, vectorized over (alpha,shift,beta) arrays."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    vv = np.asarray(v, float); vv = vv / vv.sum()
    alpha = np.asarray(alpha, float)[:, None]; shift = np.asarray(shift, float)[:, None]
    u_a = np.power(a + shift, alpha) - np.power(shift, alpha)
    u_b = np.power(b + shift, alpha) - np.power(shift, alpha)
    diff = ((u_b - u_a) * vv).sum(axis=1)
    core = 1.0 / (1.0 + np.exp(-np.asarray(beta, float) * diff))
    return (1.0 - LAPSE) * core + LAPSE * 0.5


def p7_majority_B(stim, v):
    return float((search.p7_choices_grid(np.array(stim), np.array(v)) == 1).mean())


# ---- per-vector stimulus generation -----------------------------------------

def steep_stimuli(v):
    search.VALIDITY_SUBSETS = [np.array(v)]
    search.STEEP_FLAT_DELTAS = STEEP_DELTAS          # widen gain sizes for variety
    out = []
    for p, (tx, ty, (i, j, d), _vals, _agg) in enumerate(
            search.search_steep_flat(STEEP_N_PAIRS, set())):
        # Fill the two non-trade-off ("context") cues with a varied TIED value so
        # the stimuli aren't all zeros there. Tied across both options -> cancels
        # in every model, so steep_opt and all predictions are unchanged.
        rest = [k for k in range(4) if k not in (i, j)]
        tx = tx.copy(); ty = ty.copy()
        for col, cval in zip(rest, STEEP_CONTEXTS[p % len(STEEP_CONTEXTS)]):
            tx[:, col] = cval
            ty[:, col] = cval
        out.append({"a": tx[0].tolist(), "b": tx[1].tolist(), "steep_opt": 0, "d": int(d)})  # X: steep=A
        out.append({"a": ty[0].tolist(), "b": ty[1].tolist(), "steep_opt": 1, "d": int(d)})  # Y: steep=B
    return out


def offset_stimuli(v, sample, settings=OFFSET_SETTINGS):
    """Level-shift pairs: the SAME (a, b) structure shown once with values in a
    low range and once shifted up by a constant `shift` into a higher range (so
    every linear rule predicts identically for the two cells). To vary the
    absolute low/high ranges across pairs, pairs are drawn from several
    (low_max, shift, n_pairs) settings: within a setting both options' low-range
    values lie in 0..low_max and the shifted cell is +shift (kept <= STIM_MAX).
    Within each setting, pairs are ranked by the realized expected OffsetEffect
    E[P(target|low) - P(target|high)] under the concave generating model and
    deduped by difference vector (shared across settings). target = the
    concave-favored option at the low range."""
    al, sh, be = sample
    out = []; used = set(); k = 0
    for low_max, shift, n_take in settings:
        cands = []
        for a in itertools.product(range(low_max + 1), repeat=4):
            for b in itertools.product(range(low_max + 1), repeat=4):
                if a == b or max(max(a), max(b)) + shift > RMAX:
                    continue
                tgt = 1 if p7_majority_B([a, b], v) > 0.5 else 0
                plo = _p_choose_b(a, b, v, al, sh, be)
                phi = _p_choose_b([x + shift for x in a],
                                  [x + shift for x in b], v, al, sh, be)
                if tgt == 0:
                    plo, phi = 1 - plo, 1 - phi
                eff = float((plo - phi).mean())
                if plo.mean() > 0.55:
                    cands.append((eff, a, b, tgt))
        cands.sort(key=lambda c: -c[0])
        taken = 0
        for eff, a, b, tgt in cands:
            d = tuple(x - y for x, y in zip(a, b))
            if d in used or tuple(-x for x in d) in used:
                continue
            used.add(d)
            out.append({"a": list(a), "b": list(b), "target": tgt,
                        "cell": "low", "pair": k, "shift": shift})
            out.append({"a": [x + shift for x in a], "b": [x + shift for x in b],
                        "target": tgt, "cell": "high", "pair": k, "shift": shift})
            k += 1; taken += 1
            if taken == n_take:
                break
        if taken < n_take:
            raise RuntimeError(f"offset: only {taken}/{n_take} pairs for "
                               f"low_max={low_max} shift={shift}, v={v}.")
    return out


def concave_grid_choice(stim, v):
    """Concave (sub-linear) point prediction for an md pair: the option chosen by
    the majority of the (alpha<1, shift) grid, with the agreeing grid fraction."""
    g = search.p7_choices_grid(np.array(stim), np.array(v))[search.ALPHAS < 1.0 - 1e-9]
    frac1 = float(g.mean())
    return (1, frac1) if frac1 > 0.5 else (0, 1.0 - frac1)


def _heuristic_pred(stim, v, m):
    s = np.array(stim)
    if m == "wadd":
        return search.wadd_choice_exact(s, v)
    if m == "tallying":
        return search.response_tallying(None, s, v)
    return search.response_ttb(None, s, v)


def _md_preds(stim, v):
    """All four model point-predictions for an md stimulus. Concave is the
    grid-majority (alpha<1) choice; heuristic ties are recorded as None."""
    cc, frac = concave_grid_choice(stim, v)
    preds = {"concave": cc, "concave_grid_frac": round(frac, 3)}
    for m in COMPARES:
        h = _heuristic_pred(stim, v, m)
        preds[m] = None if h is None else int(h)
    return preds


def md_stimuli(v):
    """Experiment 1 PAIRWISE battery. For each heuristic m in COMPARES, find
    MD_BASE_PER_MODEL base pairs on which the concave account opposes m on at
    least MD_MIN_GRID of the (alpha<1, shift) grid AND m makes a decisive (non-
    tie) choice; emit each base pair plus its A/B mirror (side balance). Every
    stimulus records all four model predictions, so across the three sub-sets the
    battery contains stimuli where WADD, tallying and TTB disagree with one
    another. Each record carries `compare` = the heuristic it discriminates."""
    rng = np.random.default_rng(1)
    out = []
    for m in COMPARES:
        cands = []; seen = set()
        for _ in range(300000):
            s = rng.integers(0, RMAX + 1, size=(2, 4))
            if tuple(s[0]) == tuple(s[1]):
                continue
            cc, frac = concave_grid_choice(s, v)
            if frac < MD_MIN_GRID:
                continue
            hm = _heuristic_pred(s, v, m)
            if hm is None or int(hm) == cc:       # m decisive AND opposite to concave
                continue
            key = (tuple(s[0]), tuple(s[1]))
            if key in seen:
                continue
            seen.add(key); cands.append((frac, s))
        cands.sort(key=lambda c: -c[0])
        bases = []
        for frac, s in cands:
            if any(np.array_equal(s, p) or np.array_equal(s[::-1], p) for p in bases):
                continue
            bases.append(s)
            if len(bases) == MD_BASE_PER_MODEL:
                break
        if len(bases) < MD_BASE_PER_MODEL:
            raise RuntimeError(f"vector {v}: only {len(bases)} concave-vs-{m} "
                               f"pairs found; widen the sampling budget.")
        for s in bases:
            for stim in (s, s[::-1]):             # base + A<->B mirror
                out.append({"a": stim[0].tolist(), "b": stim[1].tolist(),
                            "compare": m, **_md_preds(stim, v)})
    return out


def build_design(sample):
    vectors = {}
    for k, v in enumerate(VECTORS):
        comps = {"steep": steep_stimuli(v),
                 "offset": offset_stimuli(v, sample),
                 "md": md_stimuli(v)}
        for cname, stims in comps.items():               # tag each stimulus's experiment
            for s in stims:
                s["experiment"] = COMPONENT_EXPERIMENT[cname]
        vectors[f"v{k}"] = {"validities": list(v), **comps}
    experiments = {}
    for e in EXP_COMPONENTS:
        reps, list_len, repl = reps_for(e)
        experiments[e] = {
            "label": EXP_LABEL[e], "components": EXP_COMPONENTS[e],
            "multiplicity": EXP_MULT[e], "reps": reps, "list_len": list_len,
            "trials_per_participant": list_len * repl, "tests": EXP_TESTS[e],
            "n_per_vector": None, "n_total": None,        # filled by the power search
        }
    return {"rating_max": RMAX, "target_trials": TARGET_TRIALS,
            "design": "between_experiments", "experiments": experiments,
            "vectors": vectors}


# ---- power simulation (30% noise), per experiment ---------------------------

def _draw(n, rng, *, concave):
    alpha = rng.uniform(*ALPHA_RANGE, n) if concave else np.ones(n)
    shift = rng.uniform(*SHIFT_RANGE, n)
    beta = rng.uniform(*BETA_RANGE, n)
    return alpha, shift, beta


def _rate(stim, target, reps, v, params, rng):
    """One participant batch's observed choice rate toward `target` for a
    stimulus: noisy P(target) sampled as `reps` Bernoulli trials per participant."""
    alpha, shift, beta = params
    p_b = _p_choose_b(stim["a"], stim["b"], v, alpha, shift, beta)
    p_t = p_b if target == 1 else 1.0 - p_b
    return rng.binomial(reps, np.clip(p_t, 0, 1)) / reps


def power_exp1(design, reps, N, n_sims, rng, *, concave):
    """Experiment 1 power: P(all three pairwise comparisons survive Holm). For
    each heuristic m, on m's own discriminating subset the concave point
    prediction is the option opposite to m, so per participant
    ModelMatchRate(concave) - ModelMatchRate(m) = 2*concave_match - 1 on that
    subset. We Holm-correct the three one-sided tests and count a rejection only
    when all three are significant."""
    from scipy import stats
    vids = list(design["vectors"]); per = N // len(vids)
    rej = 0
    for _ in range(n_sims):
        Dm = {m: [] for m in COMPARES}
        for vid in vids:
            spec = design["vectors"][vid]; v = spec["validities"]
            params = _draw(per, rng, concave=concave)
            for m in COMPARES:
                sub = [s for s in spec["md"] if s["compare"] == m]
                cmr = np.mean([_rate(s, s["concave"], reps["md"], v, params, rng)
                               for s in sub], axis=0)
                Dm[m].append(2.0 * cmr - 1.0)
        ps = [float(getattr(stats.ttest_1samp(np.concatenate(Dm[m]), 0.0,
                                              alternative="greater"), "pvalue"))
              for m in COMPARES]
        order = np.argsort(ps); k = len(ps); ok = [False] * k
        for rank, idx in enumerate(order):
            if ps[idx] < 0.05 / (k - rank):
                ok[idx] = True
            else:
                break
        rej += int(all(ok))
    return rej / n_sims


def power_exp2(design, reps, N, n_sims, rng, *, concave):
    """Experiment 2 power: (P(reject H2 steep), P(reject H3 offset))."""
    from scipy import stats
    vids = list(design["vectors"]); per = N // len(vids)
    rs = ro = 0
    for _ in range(n_sims):
        S, O = [], []
        for vid in vids:
            spec = design["vectors"][vid]; v = spec["validities"]
            params = _draw(per, rng, concave=concave)
            steep = np.mean([_rate(s, s["steep_opt"], reps["steep"], v, params, rng)
                             for s in spec["steep"]], axis=0)
            lo = np.mean([_rate(s, s["target"], reps["offset"], v, params, rng)
                          for s in spec["offset"] if s["cell"] == "low"], axis=0)
            hi = np.mean([_rate(s, s["target"], reps["offset"], v, params, rng)
                          for s in spec["offset"] if s["cell"] == "high"], axis=0)
            S.append(steep); O.append(lo - hi)
        S = np.concatenate(S); O = np.concatenate(O)
        rs += int(float(getattr(stats.ttest_1samp(S, 0.5, alternative="greater"), "pvalue")) < 0.05)
        ro += int(float(getattr(stats.ttest_1samp(O, 0.0, alternative="greater"), "pvalue")) < 0.05)
    return rs / n_sims, ro / n_sims


def _smallest_n(power_min_fn, n_sims, rng):
    """Smallest N in N_GRID whose minimum test power >= TARGET_POWER. Returns
    (chosen_N, [(N, power_tuple), ...]) for reporting; chosen_N is None if the
    grid never reaches the target."""
    sweep = []
    chosen = None
    for N in N_GRID:
        pw = power_min_fn(N, n_sims)
        sweep.append((N, pw))
        if chosen is None and min(pw) >= TARGET_POWER:
            chosen = N
    return chosen, sweep


def main():
    rng = np.random.default_rng(0)
    M = 400
    sample = (rng.uniform(*ALPHA_RANGE, M), rng.uniform(*SHIFT_RANGE, M),
              rng.uniform(*BETA_RANGE, M))
    design = build_design(sample)
    reps1 = design["experiments"]["exp1"]["reps"]
    reps2 = design["experiments"]["exp2"]["reps"]

    print("Two separate between-subjects experiments (30% lapse, 5 vectors).")
    print(f"  Exp1 model comparison: reps={reps1} "
          f"-> {design['experiments']['exp1']['trials_per_participant']} trials/subj")
    print(f"  Exp2 value curvature : reps={reps2} "
          f"-> {design['experiments']['exp2']['trials_per_participant']} trials/subj\n")

    n_sims = 400
    n1, sweep1 = _smallest_n(
        lambda N, ns: (power_exp1(design, reps1, N, ns, rng, concave=True),),
        n_sims, rng)
    n2, sweep2 = _smallest_n(
        lambda N, ns: power_exp2(design, reps2, N, ns, rng, concave=True),
        n_sims, rng)

    print(f"Exp1 power ({n_sims} sims): N(total) /vec | all-3 pairwise (Holm)")
    for N, pw in sweep1:
        print(f"   {N:>4} {N//5:>4} | {pw[0]:>6.0%}")
    print(f"  -> chosen N = {n1} ({(n1 or 0)//5}/vector)\n")
    print(f"Exp2 power ({n_sims} sims): N(total) /vec | steep  offset")
    for N, (ps, po) in sweep2:
        print(f"   {N:>4} {N//5:>4} | {ps:>6.0%} {po:>6.0%}")
    print(f"  -> chosen N = {n2} ({(n2 or 0)//5}/vector)\n")

    for e, n_min in (("exp1", n1), ("exp2", n2)):
        if n_min is None:
            raise SystemExit(f"{e}: N_GRID exhausted before reaching "
                             f"{TARGET_POWER:.0%} power; widen N_GRID.")
        n_plan = N_COLLECT[e]
        if n_plan % len(VECTORS):
            raise SystemExit(f"{e}: N_COLLECT={n_plan} not a multiple of "
                             f"{len(VECTORS)} (needed to balance the vectors).")
        if n_plan < n_min:
            print(f"  WARNING: {e} planned N={n_plan} < power-min N={n_min}; "
                  f"power will be below {TARGET_POWER:.0%}.")
        design["experiments"][e]["n_per_vector"] = n_plan // len(VECTORS)
        design["experiments"][e]["n_total"] = n_plan
        design["experiments"][e]["n_power_min"] = n_min

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "battery.json"), "w") as f:
        json.dump(design, f, indent=1)
    print(f"battery.json written. Planned: Exp1 N={N_COLLECT['exp1']} "
          f"(power-min {n1}), Exp2 N={N_COLLECT['exp2']} (power-min {n2}).\n")

    print("False-positive under LINEAR truth (30% lapse, 800 sims) at planned N:")
    fp1 = power_exp1(design, reps1, N_COLLECT["exp1"], 800, rng, concave=False)
    fps, fpo = power_exp2(design, reps2, N_COLLECT["exp2"], 800, rng, concave=False)
    print(f"  Exp1 all-3 pairwise={fp1:.0%} | Exp2 steep={fps:.0%} offset={fpo:.0%}")


if __name__ == "__main__":
    main()
