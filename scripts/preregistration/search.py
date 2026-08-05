"""Build an 8-trial validation battery for P7 (concave-utility WADD) against
linear baselines.

Both response functions are deterministic (argmax) simplifications -- the
softmax/epsilon noise of the full models is dropped. P7 depends on two free
parameters (alpha in [0.1, 1.0], shift in [0.1, 5.0]); at alpha = 1 the
utility is linear and P7 reduces exactly to WADD, so predictions are checked
across an (alpha, shift) grid.

Trial types (see main()):
  1. curve-ends (2 matched pairs = 4 trials) -- the SAME exactly-WADD-tied
     value change d is applied once inside the low end of the value scale
     (all values in [0, 2]) and once, shifted, inside the high end ([3, 5]).
     Any linear weighter is indifferent on both trials (50/50); P7 prefers
     the same option in both for every grid point with alpha < 1, but with a
     much larger margin at the low (steep) end than at the high (flat) end.
     A preference that is strong at the low end and weak at the high end is
     direct evidence that equal value changes are NOT worth the same, i.e.
     the utility curve is nonlinear at both ends. Pairs additionally prefer
     contrasts where tallying/TTB pick the option P7 rejects.
  2. steep-vs-flat (2 counterbalanced pairs = 4 trials) -- the curve shown
     directly as a trade-off: one option gains d on the STEEP part of the
     scale (0 -> d), the other gains the same d on the FLAT part
     ((5-d) -> 5), on two different cues. The weights are counterbalanced:
     the second trial of a pair swaps which cue (validity) carries the steep
     vs the flat gain and which option holds it, so every linear weighter's
     two margins are exactly equal and opposite -- aggregated over the pair,
     linear models choose the steep option exactly 50% of the time. A
     concave P7 net-prefers the steep gain at EVERY grid point with
     alpha < 1 (u(d) - u(0) > u(5) - u(5-d)). TTB's choices also cancel
     across the pair; tallying is tied throughout.
  3. value-sweep (6 trials) -- the same option pair repeated; only B's value
     on one cue changes, v = 0..5. P(choose B) as a function of v is a
     direct psychometric readout of the utility curve: WADD predicts a
     choice flip at its tie point with equal score steps per unit of v,
     while a concave P7's score steps SHRINK as v grows, shifting the flip
     point and compressing the high end of the curve.
  4. critical (4 trials) -- WADD, tallying and TTB all pick the same option
     and P7 picks the opposite on a maximal fraction of the parameter grid:
     each trial votes P7 against all three baselines at once.

(search_curvature and search_additivity below are retained from the earlier
9-trial battery but are no longer part of main().)

Search space: (2, 4) integer stimuli in [0, 5]; validities are 4 distinct
values from {0.5, ..., 0.9}, always sorted descending (features are presented
in descending-validity order); every option vector is unique across the
whole battery.

Run:  .venv/bin/python validation_online/search.py
"""

import itertools

import numpy as np

N_FEATURES = 4
STIM_MAX = 5
VALIDITY_SET = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
ALPHAS = np.linspace(0.1, 1.0, 10)
SHIFTS = np.linspace(0.1, 5.0, 10)

# All ways to pick 4 distinct validities from the set, kept sorted descending.
VALIDITY_SUBSETS = [
    np.array(sorted(c, reverse=True))
    for c in itertools.combinations(VALIDITY_SET, N_FEATURES)
]

# Contexts for the additivity trials: values shared by both options on the
# two low-validity cues, spanning the utility curve (steep / middle / flat).
ADDITIVITY_CONTEXTS = ((0, 0), (2, 3), (5, 5))

CHOICE_NAMES = {0: "A", 1: "B", None: "indifferent"}


def responde_p7(parameters, stimulus, validities):

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)

    alpha = float(parameters["alpha"])
    shift = float(parameters["shift"])

    # Normalize validities to stabilize softmax across different experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val

    # Apply concave utility transformation (diminishing returns)
    # Shifted by a parameterized value to flexibly smooth infinite marginal utility at zero
    u_a = np.power(a + shift, alpha) - np.power(shift, alpha)
    u_b = np.power(b + shift, alpha) - np.power(shift, alpha)

    score_a = np.sum(u_a * val)
    score_b = np.sum(u_b * val)

    scores = np.array([score_a, score_b])

    return np.argmax(scores)


def response_wadd(_, stimulus, validities):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * 0.5)

    return np.argmax(scores)


def response_tallying(_, stimulus, validities):
    """Deterministic tallying (theories/heuristic_decision_making/tallying.yaml):
    count strict feature-wise wins; None on a tied tally (model must guess)."""
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    b_wins = int(np.sum(b > a))
    if a_wins == b_wins:
        return None
    return 0 if a_wins > b_wins else 1


def response_ttb(_, stimulus, validities):
    """Deterministic Take The Best (theories/heuristic_decision_making/ttb.yaml):
    first discriminating cue in descending-validity order decides; None if no
    cue discriminates (model must guess)."""
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    for j in np.argsort(-np.asarray(validities, dtype=float), kind="stable"):
        if a[j] > b[j]:
            return 0
        if b[j] > a[j]:
            return 1
    return None


def wadd_choice_exact(stimulus, validities):
    """WADD comparison in integer arithmetic (validities x 10), so exact ties
    are detected exactly; returns None on a tie."""
    val10 = np.round(np.asarray(validities) * 10).astype(int)
    scores = np.asarray(stimulus, dtype=int) @ val10
    if scores[0] == scores[1]:
        return None
    return int(np.argmax(scores))


def p7_margins_grid(stimulus, validities, alphas=ALPHAS, shifts=SHIFTS):
    """P7's score(A) - score(B) for every (alpha, shift) on the grid,
    shape (len(alphas), len(shifts))."""
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(validities, dtype=float)
    val = val / val.sum()

    A = alphas[:, None, None]
    S = shifts[None, :, None]
    u_a = np.power(a + S, A) - np.power(S, A)
    u_b = np.power(b + S, A) - np.power(S, A)
    return np.sum((u_a - u_b) * val, axis=-1)


def p7_choices_grid(stimulus, validities, alphas=ALPHAS, shifts=SHIFTS):
    """Vectorized P7 choice per grid point; ties resolve to option 0,
    matching np.argmax in responde_p7."""
    return (p7_margins_grid(stimulus, validities, alphas, shifts) < 0).astype(int)


def p7_unanimous_choice(stimulus, validities, tol=1e-9):
    """P7's choice if it is the same for every grid point with alpha < 1
    (at alpha = 1 P7 is linear, so a WADD-tied pair is tied for P7 too).
    Returns (choice, min_margin) or (None, 0.0)."""
    m = p7_margins_grid(stimulus, validities)[ALPHAS < 1.0 - 1e-12]
    if np.all(m > tol):
        return 0, float(m.min())
    if np.all(m < -tol):
        return 1, float(-m.max())
    return None, 0.0


def _option_rows(stim):
    return {tuple(int(x) for x in stim[0]), tuple(int(x) for x in stim[1])}


def _too_close(stim, kept_stims, min_dist=4):
    swapped = stim[::-1]
    return any(
        min(np.abs(stim - k).sum(), np.abs(swapped - k).sum()) < min_dist
        for k in kept_stims
    )


# ---------------------------------------------------------------------------
# 1) Curvature trials: WADD exactly tied, P7 unanimous.
# ---------------------------------------------------------------------------

def _wadd_null_diffs(val10):
    """Nonzero integer difference vectors d (entries in [-STIM_MAX, STIM_MAX])
    with d @ val10 == 0: adding d to an option leaves its WADD score exactly
    unchanged."""
    rng_vals = range(-STIM_MAX, STIM_MAX + 1)
    return [
        np.array(d)
        for d in itertools.product(rng_vals, repeat=N_FEATURES)
        if any(d) and int(np.dot(d, val10)) == 0
    ]


def search_curvature(n_keep, used_options, rng, n_samples=20000):
    """WADD-tied pairs with a grid-unanimous P7 preference. Ranked first by
    how many of {tallying, TTB} decisively pick the option P7 rejects (the
    collapse bonus), then by P7's worst-case grid margin."""
    candidates = []
    per_subset = n_samples // len(VALIDITY_SUBSETS)
    for vals in VALIDITY_SUBSETS:
        val10 = np.round(vals * 10).astype(int)
        diffs = _wadd_null_diffs(val10)
        for _ in range(per_subset):
            a = rng.integers(0, STIM_MAX + 1, size=N_FEATURES)
            b = a + diffs[rng.integers(len(diffs))]
            if b.min() < 0 or b.max() > STIM_MAX:
                continue
            stim = np.stack([a, b])
            choice, margin = p7_unanimous_choice(stim, vals)
            if choice is None:
                continue
            opposed = 0
            for response in (response_tallying, response_ttb):
                c = response(None, stim, vals)
                if c is not None and c != choice:
                    opposed += 1
            candidates.append(((opposed, margin), stim, vals))
    candidates.sort(key=lambda c: c[0], reverse=True)

    kept = []
    for _, stim, vals in candidates:
        if len(kept) >= n_keep:
            break
        rows = _option_rows(stim)
        if len(rows) < 2 or rows & used_options:
            continue
        if _too_close(stim, [k[0] for k in kept]):
            continue
        kept.append((stim, vals))
        used_options |= rows
    return kept


# ---------------------------------------------------------------------------
# 1b) Curve-ends trials: the same WADD-tied change at the low vs high end.
# ---------------------------------------------------------------------------

LOW_MAX = 2      # low-end trials use values in [0, LOW_MAX]
HIGH_SHIFT = 3   # high-end trial = low-end trial + HIGH_SHIFT (values in [3, 5])


def _mean_margin(stim, vals, choice):
    """Mean grid margin (alpha < 1) in favour of P7's chosen option."""
    m = p7_margins_grid(stim, vals)[ALPHAS < 1.0 - 1e-12]
    return float(np.mean(m if choice == 0 else -m))


def search_curve_ends(n_pairs, used_options):
    """Matched (low-end, high-end) trial pairs sharing one WADD-null change d.

    Both trials are exactly WADD-tied; P7 must prefer the same option in both
    (unanimously over the grid). Ranked first by how many of {tallying, TTB}
    decisively pick the option P7 rejects (the collapse bonus -- tallying and
    TTB are translation-invariant, so the count is identical at both ends),
    then by the low-minus-high mean-margin gap: the bigger the gap, the
    bigger the predicted drop in choice rate from the steep to the flat end."""
    candidates = []
    for vals in VALIDITY_SUBSETS:
        val10 = np.round(vals * 10).astype(int)
        diffs = [d for d in _wadd_null_diffs(val10) if np.abs(d).max() <= LOW_MAX]
        for d in diffs:
            for a in itertools.product(range(LOW_MAX + 1), repeat=N_FEATURES):
                a = np.array(a)
                b = a + d
                if b.min() < 0 or b.max() > LOW_MAX:
                    continue
                low = np.stack([a, b])
                high = low + HIGH_SHIFT
                c_low, _ = p7_unanimous_choice(low, vals)
                c_high, _ = p7_unanimous_choice(high, vals)
                if c_low is None or c_low != c_high:
                    continue
                gap = (_mean_margin(low, vals, c_low)
                       - _mean_margin(high, vals, c_high))
                opposed = 0
                for response in (response_tallying, response_ttb):
                    c = response(None, low, vals)
                    if c is not None and c != c_low:
                        opposed += 1
                candidates.append(((opposed, gap), low, high, vals))
    candidates.sort(key=lambda c: c[0], reverse=True)

    kept = []
    used_diffs = set()
    for _, low, high, vals in candidates:
        if len(kept) >= n_pairs:
            break
        d = tuple(int(x) for x in (low[1] - low[0]))
        # One pair per change vector (up to sign), so the pairs probe the
        # curve with genuinely different trade-offs.
        if d in used_diffs or tuple(-x for x in d) in used_diffs:
            continue
        rows = _option_rows(low) | _option_rows(high)
        if len(rows) < 4 or rows & used_options:
            continue
        prior = [s for pair in kept for s in pair[:2]]
        if _too_close(low, prior) or _too_close(high, prior):
            continue
        kept.append((low, high, vals))
        used_options |= rows
        used_diffs.add(d)
    return kept


# ---------------------------------------------------------------------------
# 2) Steep-vs-flat trade-off pairs, weight-counterbalanced.
# ---------------------------------------------------------------------------

STEEP_FLAT_DELTAS = (1, 2)  # gain size d: steep 0 -> d vs flat (5-d) -> 5


def search_steep_flat(n_pairs, used_options):
    """Counterbalanced trial pairs trading a steep gain against a flat gain.

    Trial X: option A gains d from 0 (steep) on cue i, option B gains d up
    to 5 (flat) on cue j. Trial Y swaps everything: B holds the steep gain
    on cue j, A holds the flat gain on cue i. For ANY linear weighting w the
    two margins toward the steep option are d*(w_i - w_j) and d*(w_j - w_i)
    -- exactly opposite, so over the pair linear models are 50/50 on the
    steep option. A concave utility has u(d) - u(0) > u(5) - u(5-d), so P7's
    AGGREGATE margin toward steep, (val_i + val_j)*(u(d) - (u(5) - u(5-d))),
    is positive at every grid point with alpha < 1.

    Kept pairs must differ in both the gain size d and the validity pair
    carrying the trade-off, so the steep preference is tested under
    different weights. Ranked by the worst-case aggregate margin."""
    candidates = []
    for vals in VALIDITY_SUBSETS:
        for i, j in itertools.combinations(range(N_FEATURES), 2):
            for d in STEEP_FLAT_DELTAS:
                X = np.zeros((2, N_FEATURES), dtype=int)
                X[0, i], X[1, i] = d, 0                      # steep, A
                X[0, j], X[1, j] = STIM_MAX - d, STIM_MAX    # flat, B
                Y = np.zeros((2, N_FEATURES), dtype=int)
                Y[0, j], Y[1, j] = 0, d                      # steep, B
                Y[0, i], Y[1, i] = STIM_MAX, STIM_MAX - d    # flat, A
                # Tied context cues cancel in every model, so the margins do
                # not depend on the context values chosen below.
                m_steep = (p7_margins_grid(X, vals)
                           - p7_margins_grid(Y, vals))      # toward steep
                agg = m_steep[ALPHAS < 1.0 - 1e-12]
                if agg.min() <= 0:
                    continue
                candidates.append((float(agg.min()), X, Y, (i, j, d), vals))
    candidates.sort(key=lambda c: -c[0])

    kept = []
    used_ds, used_val_pairs = set(), set()
    for agg_min, X, Y, (i, j, d), vals in candidates:
        if len(kept) >= n_pairs:
            break
        val_pair = (vals[i], vals[j])
        if d in used_ds or val_pair in used_val_pairs:
            continue
        rest = [k for k in range(N_FEATURES) if k not in (i, j)]
        for c in itertools.product(range(STIM_MAX + 1), repeat=2):
            tx, ty = X.copy(), Y.copy()
            tx[:, rest] = c
            ty[:, rest] = c
            rows = _option_rows(tx) | _option_rows(ty)
            if len(rows) == 4 and not (rows & used_options):
                kept.append((tx, ty, (i, j, d), vals, agg_min))
                used_options |= rows
                used_ds.add(d)
                used_val_pairs.add(val_pair)
                break
    return kept


# ---------------------------------------------------------------------------
# 3) Value-sweep trials: the same pair, one value swept 0..5.
# ---------------------------------------------------------------------------

def _sweep_stims(k, m, w, context, rest):
    stims = []
    for v in range(STIM_MAX + 1):
        stim = np.zeros((2, N_FEATURES), dtype=int)
        stim[0, m] = w        # A's fixed advantage
        stim[1, k] = v        # B's swept value
        stim[:, rest] = context
        stims.append(stim)
    return stims


def search_value_sweep(n_sweeps, used_options):
    """Sweeps of STIM_MAX+1 trials: A is fixed (value w on cue m), B is
    identical except its value on cue k runs v = 0..5. P(choose B) against v
    reads the utility curve out directly: WADD's score margin is linear in v
    (flip at v = w*val_m/val_k, equal steps), while a concave P7's margin
    grows in shrinking steps, displacing the flip point and flattening the
    high end of the choice curve.

    Admissible sweeps must flip WADD's choice strictly inside the sweep;
    ranked by the total |P7 grid choice rate - WADD choice| across v. The
    swept cue's validity regime must differ across kept sweeps (val_k >
    val_m in one, val_k < val_m in the other), so the curve is measured
    under both weight orders."""
    candidates = []
    for vals in VALIDITY_SUBSETS:
        for k in range(N_FEATURES):
            for m in range(N_FEATURES):
                if m == k:
                    continue
                for w in range(1, STIM_MAX + 1):
                    rest = [r for r in range(N_FEATURES) if r not in (k, m)]
                    stims = _sweep_stims(k, m, w, (0, 0), rest)
                    wadd = [wadd_choice_exact(s, vals) for s in stims]
                    # WADD must start at A and have flipped to B by v = 4,
                    # so the sweep has points on both sides of the flip.
                    if wadd[0] != 0 or wadd[-2] != 1:
                        continue
                    lin = np.array([0.5 if c is None else float(c)
                                    for c in wadd])
                    f = np.array([float(np.mean(p7_choices_grid(s, vals)))
                                  for s in stims])
                    score = float(np.abs(f - lin).sum())
                    candidates.append((score, k, m, w, vals))
    candidates.sort(key=lambda c: -c[0])

    kept = []
    used_regimes = set()
    for score, k, m, w, vals in candidates:
        if len(kept) >= n_sweeps:
            break
        regime = vals[k] > vals[m]
        if regime in used_regimes:
            continue
        rest = [r for r in range(N_FEATURES) if r not in (k, m)]
        for c in itertools.product(range(STIM_MAX + 1), repeat=2):
            stims = _sweep_stims(k, m, w, c, rest)
            rows = set().union(*(_option_rows(s) for s in stims))
            # 1 fixed A row + 6 distinct B rows, none seen before.
            if len(rows) == STIM_MAX + 2 and not (rows & used_options):
                kept.append((stims, (k, m, w), vals))
                used_options |= rows
                used_regimes.add(regime)
                break
    return kept


# ---------------------------------------------------------------------------
# (retired) Additivity trials: one contrast, three contexts.
# ---------------------------------------------------------------------------

def search_additivity(used_options):
    """Exhaustively pick the most diagnostic two-cue contrast (on the two
    highest-validity cues, each option winning one cue) and instantiate it
    under ADDITIVITY_CONTEXTS on the two low-validity cues.

    Choice rates are exactly context-invariant under every additive model
    (the tied context cues cancel in P7/WADD, contribute nothing to the
    tally, and are skipped by TTB), so any observed context effect falsifies
    additivity. Diagnostic = P7 opposes WADD's and TTB's shared choice on the
    largest grid fraction."""
    best = None
    for vals in VALIDITY_SUBSETS:
        for a1, a2, b1, b2 in itertools.product(range(STIM_MAX + 1), repeat=4):
            if not ((a1 > b1 and a2 < b2) or (a1 < b1 and a2 > b2)):
                continue
            stims = [
                np.array([[a1, a2, *c], [b1, b2, *c]])
                for c in ADDITIVITY_CONTEXTS
            ]
            rows = set().union(*(_option_rows(s) for s in stims))
            if rows & used_options:
                continue
            # Context-invariant, so evaluating one context suffices.
            w = wadd_choice_exact(stims[0], vals)
            t = response_ttb(None, stims[0], vals)
            if w is None or t is None or w != t:
                continue
            frac = float(np.mean(p7_choices_grid(stims[0], vals) != w))
            if best is None or frac > best[0]:
                best = (frac, stims, vals, rows)

    assert best is not None, "no admissible additivity contrast found"
    frac, stims, vals, rows = best
    used_options |= rows
    return [(stim, vals, frac) for stim in stims]


# ---------------------------------------------------------------------------
# 3) Critical trials: P7 vs WADD + tallying + TTB simultaneously.
# ---------------------------------------------------------------------------

def critical_score(stimulus, validities):
    """Fraction of the grid where P7 opposes the shared choice of WADD,
    tallying and TTB; (0, None) unless all three decisively agree."""
    w = wadd_choice_exact(stimulus, validities)
    t = response_tallying(None, stimulus, validities)
    b = response_ttb(None, stimulus, validities)
    if w is None or t is None or b is None or not (w == t == b):
        return 0.0, None
    frac = float(np.mean(p7_choices_grid(stimulus, validities) != w))
    return frac, w


def _hill_climb_critical(stim, vals, rng, n_iter=300):
    best, _ = critical_score(stim, vals)
    stim = stim.copy()
    for _ in range(n_iter):
        s2 = stim.copy()
        i = rng.integers(2)
        j = rng.integers(N_FEATURES)
        s2[i, j] = np.clip(s2[i, j] + rng.choice([-2, -1, 1, 2]), 0, STIM_MAX)
        score, _ = critical_score(s2, vals)
        if score > best:
            best, stim = score, s2
    return stim


def search_critical(n_keep, used_options, rng, n_samples=60000):
    candidates = []
    for _ in range(n_samples):
        stim = rng.integers(0, STIM_MAX + 1, size=(2, N_FEATURES))
        vals = VALIDITY_SUBSETS[rng.integers(len(VALIDITY_SUBSETS))]
        score, _ = critical_score(stim, vals)
        if score > 0:
            candidates.append((score, stim, vals))
    candidates.sort(key=lambda c: -c[0])

    kept = []
    used_vals = set()
    for score, stim, vals in candidates:
        if len(kept) >= n_keep:
            break
        # One trial per validity subset, so P7's disagreement with the
        # baselines is shown under different weights.
        if tuple(vals) in used_vals:
            continue
        if _too_close(stim, [k[0] for k in kept]):
            continue
        climbed = _hill_climb_critical(stim, vals, rng)
        # Prefer the hill-climbed pair; fall back to the raw one if climbing
        # converged onto an already-used option vector.
        for cand in (climbed, stim):
            rows = _option_rows(cand)
            if len(rows) == 2 and not (rows & used_options):
                kept.append((cand, vals))
                used_options |= rows
                used_vals.add(tuple(vals))
                break
    return kept


# ---------------------------------------------------------------------------


def _sanity_check(rng, n=200):
    """Vectorized grid choices must match the original responde_p7."""
    for _ in range(n):
        stim = rng.integers(0, STIM_MAX + 1, size=(2, N_FEATURES))
        vals = VALIDITY_SUBSETS[rng.integers(len(VALIDITY_SUBSETS))]
        i = rng.integers(len(ALPHAS))
        j = rng.integers(len(SHIFTS))
        params = {"alpha": ALPHAS[i], "shift": SHIFTS[j], "validities": vals}
        assert p7_choices_grid(stim, vals)[i, j] == responde_p7(
            params, stim, vals
        )


def _print_trial(idx, stim, vals, lines):
    print(f"trial {idx}")
    print(f"  A: {stim[0].tolist()}   B: {stim[1].tolist()}   "
          f"validities: {vals.tolist()}")
    for line in lines:
        print(f"  {line}")
    print()


def main():
    _sanity_check(np.random.default_rng(123))
    rng = np.random.default_rng(0)
    used_options = set()

    curve = search_curve_ends(2, used_options)
    steep_flat = search_steep_flat(2, used_options)
    sweeps = search_value_sweep(2, used_options)
    critical = search_critical(4, used_options, rng)

    idx = 0

    print("=== 1) Curve-ends trials: the same WADD-tied value change at the "
          "low vs high end -- preference strength traces the utility "
          "curve ===\n")
    for low, high, vals in curve:
        d = (low[1] - low[0]).tolist()
        for end, stim in (("low", low), ("high", high)):
            idx += 1
            choice, margin = p7_unanimous_choice(stim, vals)
            t = response_tallying(None, stim, vals)
            b = response_ttb(None, stim, vals)
            _print_trial(idx, stim, vals, [
                f"{end} end, B = A + d with d = {d}",
                f"P7: {CHOICE_NAMES[choice]} for every alpha < 1 "
                f"(min margin {margin:.4f}, "
                f"mean {_mean_margin(stim, vals, choice):.4f})",
                f"WADD: exactly tied -> 50/50 | "
                f"Tallying: {CHOICE_NAMES[t]} | TTB: {CHOICE_NAMES[b]}",
            ])

    print("=== 2) Steep-vs-flat pairs (weight-counterbalanced): a 0->d gain "
          "trades against a (5-d)->5 gain; linear preferences cancel "
          "exactly across each pair ===\n")
    for tx, ty, (i, j, d), vals, agg_min in steep_flat:
        for stim, steep_opt, steep_cue, flat_cue in ((tx, 0, i, j),
                                                     (ty, 1, j, i)):
            idx += 1
            m = p7_margins_grid(stim, vals)
            steep_frac = float(np.mean(m > 0 if steep_opt == 0 else m < 0))
            w_c = wadd_choice_exact(stim, vals)
            t = response_tallying(None, stim, vals)
            b = response_ttb(None, stim, vals)
            _print_trial(idx, stim, vals, [
                f"steep 0->{d} on cue {steep_cue + 1} "
                f"(option {CHOICE_NAMES[steep_opt]}, validity "
                f"{vals[steep_cue]}), flat {STIM_MAX - d}->{STIM_MAX} on "
                f"cue {flat_cue + 1} (option {CHOICE_NAMES[1 - steep_opt]}, "
                f"validity {vals[flat_cue]})",
                f"P7: steep option on {steep_frac:.0%} of grid | "
                f"WADD: {CHOICE_NAMES[w_c]} | Tallying: {CHOICE_NAMES[t]} | "
                f"TTB: {CHOICE_NAMES[b]}",
            ])
        print(f"  pair aggregate: every linear weighting -> exactly 50% "
              f"steep; P7 -> steep at every alpha < 1 "
              f"(min aggregate margin {agg_min:.4f})\n")

    print("=== 3) Value-sweep trials: identical pair, only B's value on one "
          "cue changes -- P(choose B) vs v reads the utility curve out "
          "directly ===\n")
    for stims, (k, m, w), vals in sweeps:
        print(f"  sweep: B's cue {k + 1} (validity {vals[k]}) runs v = 0..5 "
              f"against A's fixed {w} on cue {m + 1} (validity {vals[m]})\n")
        for v, stim in enumerate(stims):
            idx += 1
            f = float(np.mean(p7_choices_grid(stim, vals) == 1))
            w_c = wadd_choice_exact(stim, vals)
            t = response_tallying(None, stim, vals)
            b = response_ttb(None, stim, vals)
            _print_trial(idx, stim, vals, [
                f"v = {v} | WADD: {CHOICE_NAMES[w_c]} | "
                f"Tallying: {CHOICE_NAMES[t]} | TTB: {CHOICE_NAMES[b]} | "
                f"P7: B on {f:.0%} of grid",
            ])

    print("=== 4) Critical trials: P7 predicts the opposite of WADD, "
          "tallying and TTB at once ===\n")
    for stim, vals in critical:
        idx += 1
        frac, w = critical_score(stim, vals)
        assert w is not None  # search_critical only keeps decisive trials
        p7 = p7_choices_grid(stim, vals)
        disagree_alphas = ALPHAS[np.any(p7 != w, axis=1)]
        _print_trial(idx, stim, vals, [
            f"WADD, Tallying, TTB: {CHOICE_NAMES[w]}",
            f"P7: {CHOICE_NAMES[1 - w]} on {frac:.0%} of grid "
            f"(alpha in [{disagree_alphas.min():.2f}, "
            f"{disagree_alphas.max():.2f}])",
        ])

    print(f"{idx} trials total")


if __name__ == "__main__":
    main()
