# generation_iter_04_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with a DOUBT-THRESHOLD-GATED MIXTURE theory (working name: DTG-2R, 'Doubt-Triggered One-Reason with compensatory fallback'). Core claims:

(1) ABSOLUTE doubt trigger, not relative argmax. An expert commands one-reason adjudication only if its PRINTED validity falls below an absolute threshold theta (fit range ~.56-.64), i.e. only experts that are essentially coin-flips are 'newsworthy'. Trigger strength should be a steep but smooth function of (theta - v), e.g. g = sigmoid((theta - v)/tau) with tau ~ .01-.04.

(2) When a sub-threshold expert discriminates, choice follows its verdict with very high probability, independent of how many higher-validity rows oppose it: target ~.85-.90 after lapses on 1-vs-1 through 1-vs-5 and on 2-vs-2 / 3-vs-3 ties (this is what Exps 1 and 2 demand: .862 and .826, flat across coalition size). The load lapse must be MILD in these cells — noticeably milder than pi_7's current eps0/rho, which drag these to .72-.76.

(3) When NO sub-threshold expert discriminates, the trigger is off and evidence is compensatory: a divisively-normalized, anti-validity ((1-v)^gamma) and recency-tilted (exp(lam*j)) weighted sum over discriminating rows, so that a lone mid-doubt dissenter (v~.62-.68) opposed by a coalition of 2-3 sits at .45-.56 (Exps 15, 16) and conflict-display metrics land near .42-.60 (Exps 12, 13).

(4) Silent sub-threshold experts do NOT need a special 'capture' mechanism: the near-chance behaviour in silent-anchor displays follows automatically from (3). Drop null-verdict capture and the doubt-gradient leak, or keep at most a small residual — they are now redundant and were the source of pi_7's over-prediction in Exps 15/16.

(5) Preserve the two features already validated across the corpus: the label-degeneracy collapse to an unweighted tally when two printed validities are within .002 (Exps 10, 11 fit at ~.77-.82 via recency among near-equal labels), and a load-dependent lapse that keeps unanimous 6-0 dominance at ~.74 (Exp 7) rather than ceiling.

The new theory differs from pi_7 by making the one-reason rule conditional on an ABSOLUTE near-chance validity rather than on being the relative minimum among discriminating rows, and by being compensatory when no near-chance expert speaks; it differs from pi_9 by denying that coalitions can ever overturn a genuinely near-chance dissenter. Parameter ranges should be chosen so that, at mid-box, the model reproduces: Exp1 ~.86, Exp2 ~.83, Exp15 ~.54, Exp16 ~.47, Exp7 ~.74, Exp13 ~.59, Exp12 ~.42, Exp14 ~.26, Exp6 ~.15.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.907)

**Description**
**MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # MDSC-NV: Maximum-Doubt Single-Cue selection with Null-Verdict capture
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9975)

    # ---------- parameters ----------
    delta = float(parameters['doubt_gain'])
    lam = float(parameters['pos_ratio']) * delta        # weak positional gain
    q = float(np.clip(float(parameters['capture']), 0.0, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kap = float(parameters['tally_kappa'])

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    fav_a = diff[disc] > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (all-or-none, exact repetition) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> unweighted graded row tally
        z = float(np.clip(kap * (n_a - n_b), -60.0, 60.0))
        p_core_a = 1.0 / (1.0 + np.exp(-z))
    else:
        # doubt salience over ALL rows
        logu = delta * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)

        lg = logu - np.max(logu)
        s = np.exp(lg)
        ssum = float(np.sum(s))
        s = s / ssum if (np.isfinite(ssum) and ssum > 0.0) else np.full(n, 1.0 / n)

        ld = logu[disc]
        ld = ld - np.max(ld)
        t = np.exp(ld)
        tsum = float(np.sum(t))
        t = t / tsum if (np.isfinite(tsum) and tsum > 0.0) else np.full(m, 1.0 / m)

        # (i) deliberate re-targeting: doubt competition among disagreeing rows
        p_rest_a = float(np.sum(t[fav_a]))

        # (ii) null-verdict capture: attention grabbed by the most doubtful
        #      expert in the whole display; if it is silent -> guess
        mask = np.zeros(n, dtype=bool)
        mask[disc] = True
        s_nondisc = float(np.sum(s[~mask]))
        s_disc = s[disc]
        p_glob_a = float(np.sum(s_disc[fav_a])) + 0.5 * s_nondisc

        p_core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- load-dependent lapse ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- doubt_gain: [13.0, 17.0]
- pos_ratio: [0.09, 0.14]
- capture: [0.33, 0.49]
- eps0: [0.19, 0.29]
- rho: [0.77, 0.85]
- tally_kappa: [1.00, 1.40]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2820 (var=0.0035) vs this=0.2878 (var=0.0023)
- Experiment 2: real=0.1744 (var=0.0127) vs this=0.2300 (var=0.0149)
- Experiment 3: real=0.2856 (var=0.0160) vs this=0.2778 (var=0.0058)
- Experiment 4: real=0.1500 (var=0.0194) vs this=0.1500 (var=0.0133)
- Experiment 5: real=0.7427 (var=0.0173) vs this=0.7467 (var=0.0048)
- Experiment 6: real=0.5067 (var=0.0112) vs this=0.4817 (var=0.0130)
- Experiment 7: real=0.4383 (var=0.0023) vs this=0.4508 (var=0.0038)
- Experiment 8: real=0.7697 (var=0.0057) vs this=0.7686 (var=0.0034)
- Experiment 9: real=0.8189 (var=0.0074) vs this=0.7714 (var=0.0022)
- Experiment 10: real=0.4178 (var=0.0051) vs this=0.3878 (var=0.0029)
- Experiment 11: real=0.5925 (var=0.0026) vs this=0.6833 (var=0.0038)
- Experiment 12: real=0.2608 (var=0.0079) vs this=0.2592 (var=0.0042)
- Experiment 13: real=0.5460 (var=0.0126) vs this=0.6410 (var=0.0093)
- Experiment 14: real=0.4675 (var=0.0144) vs this=0.6388 (var=0.0149)
- Experiment 15: real=0.8620 (var=0.0155) vs this=0.7610 (var=0.0035)
- Experiment 16: real=0.8260 (var=0.0180) vs this=0.7245 (var=0.0054)


---

### `pi_8` (overall score: 0.821)

**Description**
**DTV-RG/CL — Doubt-Triggered one-reason with Rank-Geometric weighting, under a two-component (load + conflict) leak and a DECISIVE degenerate-label tally.**

The evidence rule is unchanged from the accepted base; two claims about *how the rule degrades* are sharpened.

**(1) Rank-geometric evidence, decided from the LEAST valid dissenting expert.** On each trial the subject lines the *discriminating* rows up in order of printed validity, least valid first, and sums their verdicts with a geometric discount: E = Σ_r s_r · φ^r (r = 0 for the least valid discriminating expert, s_r = ±1 its verdict); p(A) = σ(βE). Only the ORDER of the printed labels enters, never the size of the gaps, which is why the .955/.950/.945 and .958/.954/.950 families are still resolved almost deterministically in favour of the strictly least valid dissenter, and why coalitions of more valid experts normally buy nothing (though a large enough coalition can still overturn one doubtful voice).

**(2) The discount rate is set by the ABSOLUTE doubtfulness of the decisive voice.** One-reason choice is a *reaction to a near-chance expert*: φ = φ_lo + (φ_hi − φ_lo)·σ((v₀ − v_c)/w), with v₀ the validity of the least valid DISCRIMINATING expert, φ_lo ≈ .25 (take-the-least), φ_hi ≈ .88 (near tally), v_c ≈ .575, w ≈ .015. Below the trigger the display is settled by one strange dissent; above it nothing is newsworthy and the rows are weighted nearly equally.

**(3) No null-verdict capture.** Non-discriminating rows are strictly irrelevant whatever their validity; all pull toward chance comes from the leak in (4).

**(4) NEW — the leak has TWO separable sources, and dominance displays leak only through the first.** Losing the thread is driven (i) by sheer display load — how many rows have to be held while the ordering is run — and (ii) independently by *opposition*: every additional row that argues the other way adds a fresh occasion to lose track of which side one was on. p(guess) = 1 − (1−ε₀)·ρ_m^max(m−2,0)·ρ_c^min(n_A,n_B), with ρ_m ≈ .79 (load) and ρ_c ≈ .97 (opposition). The two terms are close to orthogonal in the corpus: unanimous (dominance) displays have min(n_A,n_B)=0 and therefore leak only by load, so they stay well above the balanced-conflict displays of the same size, whereas a 3-vs-3 display of six rows is markedly noisier than a 6-vs-0 display of six rows. This is a direct, testable clash both with a pure-load leak (which charges a 6-0 display exactly what it charges a 3-3 display) and with a pure-opposition leak (which would make unanimous displays near-perfect).

**(5) NEW — when labels are degenerate the fallback tally is DECISIVE, not tepid.** The doubt ordering presupposes that each printed validity picks out one expert. If any label is repeated (gap < .002) the columns are interchangeable and choice collapses to an unweighted row tally — but that tally is used with a steep slope (κ ≈ 1.6, not ≈ 0.85): having nothing else to go on, the subject commits hard to the vote margin. Hence 1-vs-1 splits under duplicated labels are an exact coin flip (no anti-validity, no recency), while 2-vs-0 and 4-vs-0 splits under duplicated labels are chosen at ~.83 and ~.95 before the leak — which is what keeps unanimous displays high and what makes a 2-vs-1 display under duplicated labels clearly favour the majority rather than hovering near chance.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # DTV-RG/CL: Doubt-Triggered one-reason with Rank-Geometric weighting,
    # two-component (load + conflict) leak, decisive degenerate-label tally
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    phi_lo = float(np.clip(float(parameters['phi_lo']), 1e-3, 0.999))
    phi_hi = float(np.clip(float(parameters['phi_hi']), 1e-3, 0.999))
    v_crit = float(parameters['v_crit'])
    v_width = float(max(float(parameters['v_width']), 1e-4))
    beta = float(parameters['beta'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (repeated printed labels) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> decisive unweighted row tally
        p_core_a = sig(kappa * (n_a - n_b))
    else:
        # order discriminating rows by printed validity (ascending);
        # ties broken by reading position (later read = more decisive)
        order = np.lexsort((-disc.astype(float), v[disc]))
        idx_sorted = disc[order]
        s_sorted = signs[order]

        v0 = float(v[idx_sorted[0]])
        # doubt trigger: only a near-chance dissenter buys one-reason choice
        phi = phi_lo + (phi_hi - phi_lo) * sig((v0 - v_crit) / v_width)
        phi = float(np.clip(phi, 1e-4, 0.9999))

        r = np.arange(m, dtype=float)
        w = np.power(phi, r)
        E = float(np.sum(s_sorted * w))
        p_core_a = sig(beta * E)

    # ---------- two-component leak: display load x opposition ----------
    expo_m = max(m - 2, 0)
    k_conf = int(min(n_a, n_b))
    g = 1.0 - (1.0 - eps0) * (rho_m ** expo_m) * (rho_c ** k_conf)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- phi_lo: [0.21, 0.29]
- phi_hi: [0.84, 0.92]
- v_crit: [0.570, 0.580]
- v_width: [0.013, 0.018]
- beta: [4.5, 5.5]
- eps0: [0.20, 0.28]
- rho_m: [0.76, 0.82]
- rho_c: [0.93, 1.00]
- tally_kappa: [1.30, 1.90]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2820 (var=0.0035) vs this=0.2730 (var=0.0019)
- Experiment 2: real=0.1744 (var=0.0127) vs this=0.2117 (var=0.0142)
- Experiment 3: real=0.2856 (var=0.0160) vs this=0.2794 (var=0.0061)
- Experiment 4: real=0.1500 (var=0.0194) vs this=0.1217 (var=0.0081)
- Experiment 5: real=0.7427 (var=0.0173) vs this=0.7487 (var=0.0072)
- Experiment 6: real=0.5067 (var=0.0112) vs this=0.4725 (var=0.0109)
- Experiment 7: real=0.4383 (var=0.0023) vs this=0.4408 (var=0.0032)
- Experiment 8: real=0.7697 (var=0.0057) vs this=0.7611 (var=0.0025)
- Experiment 9: real=0.8189 (var=0.0074) vs this=0.7536 (var=0.0025)
- Experiment 10: real=0.4178 (var=0.0051) vs this=0.4492 (var=0.0019)
- Experiment 11: real=0.5925 (var=0.0026) vs this=0.6367 (var=0.0043)
- Experiment 12: real=0.2608 (var=0.0079) vs this=0.2729 (var=0.0043)
- Experiment 13: real=0.5460 (var=0.0126) vs this=0.2680 (var=0.0082)
- Experiment 14: real=0.4675 (var=0.0144) vs this=0.2313 (var=0.0086)
- Experiment 15: real=0.8620 (var=0.0155) vs this=0.7143 (var=0.0022)
- Experiment 16: real=0.8260 (var=0.0180) vs this=0.6770 (var=0.0061)


---

### `pi_9` (overall score: 0.791)

**Description**
**NAVA-HLp — Normalized Anti-Validity Accumulation with a HALF-LOUDNESS load lapse and a PARTIAL unanimity discount.**

Six claims. The evidence rule is unchanged from the accepted base: graded, compensatory, anti-validity, divisively normalized, with no one-reason step, no argmax, no verdict capture and no doubt trigger. Only the *price charged for holding rival voices* is revised, and only partially.

**(1) Every dissenting row speaks, and doubt is what makes it loud.** A printed validity close to chance makes a disagreement *newsworthy* (it demands adjudication); a printed validity close to certainty makes the same disagreement a mere echo of the impression the subject already has. Each row carries weight w_j = (1 − v_j)^γ · exp(λ·j): doubt raised to a moderate power γ ≈ 1.9–2.5, times a modest reading-position gain λ ≈ 0.28–0.45. All discriminating rows contribute simultaneously, so coalitions matter monotonically, while the global direction of choice stays systematically anti-validity in every design whose labels are usable.

**(2) Divisive normalization relative to the loudest dissent.** Weights enter only as ratios to the loudest discriminating voice, ŵ_j = w_j / max_{k∈disc} w_k, and E = Σ_{j∈disc} s_j ŵ_j / (σ + Σ_{j∈disc} ŵ_j), p(A) = σ(βE). Absolute scale is discarded (dominance displays decided only by high-validity experts are still resolved confidently) while the denominator grows with each speaking row, so a coalition erodes a doubtful dissenter's lead toward, but rarely far past, chance.

**(3) The displacement leak is a DOUBT GRADIENT.** Doubt is a property of the expert, not of the disagreement, so the most doubtful expert *in the whole display* competes for attention before the subject checks whether it discriminates. What costs accuracy is the *doubt gradient* between the loudest silent expert and the loudest speaking one: attention is pulled off the adjudication only when the silent expert is appreciably MORE doubtful than the expert currently carrying the argument. Formally d = clip((v_loudest-disc − v_loudest-silent)/Δ, 0, 1) · w_silent,max/(w_silent,max + w_disc,max) with fixed Δ = .05, and the leak multiplies the retention term by (1 − κ·d), κ ≈ .40–.95. Silent rows never contribute evidence; the term is purely subtractive and can only shrink |p − .5|.

**(4) Losing the thread is charged HALF PER ROW and HALF PER LOUD VOICE.** What has to be held in mind while the comparison is run is neither the raw number of disagreeing rows (which over-charges a display in which one towering doubtful voice drowns out five whispers) nor purely the effective number of loud voices (which under-charges a long unanimous list that still has to be scanned). Both costs are real and roughly equally weighted: L = ½·(m − 2) + ½·c·(Σ_{j∈disc} ŵ_j − 1), and p(guess) = 1 − (1−ε₀)·ρ^max(L,0)·(1 − κ·d) with ρ ≈ .64–.80 and floor ε₀ ≈ .07–.15.

**(5) NEW — unanimity is CHEAPER BUT NOT FREE.** The loud-voice component of the load is an *adjudication* cost: it is the price of holding rival loud voices against one another. When every speaking row agrees there is no adjudication, but the rows still have to be scanned and their loudness still occupies the comparison, so the cost is discounted rather than waived: c = 1 when both options are favoured by at least one row, and c = c₀ ≈ .45–.75 when the display is unanimous. The row-scan half, ½·(m − 2), is charged in full regardless, so a 6-vs-0 dominance display remains well below ceiling while a 3-vs-3 display of the same size is markedly noisier. This is the model's sharp prediction about dominance: accuracy on unanimous displays falls with list length but strictly more slowly than on conflict displays of matched length and matched weight concentration.

**(6) Degenerate printed labels collapse the weighting.** If any two printed validities are within .002 the experts cannot be individuated; all weights become equal and the same normalized rule reduces to an unweighted tally, E = (n_A − n_B)/(σ + m) with the same β and σ — exact chance on 1-1 splits, no anti-validity and no recency signature. The displacement leak is switched off, because no row is 'the doubtful one'.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # NAVA-HLp: Normalized Anti-Validity Accumulation with doubt-gradient displacement leak,
    # HALF-ROW / HALF-LOUD-VOICE load lapse, and a PARTIAL unanimity discount on the loud-voice half
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kappa = float(np.clip(float(parameters['kappa']), 0.0, 1.0))
    c0 = float(np.clip(float(parameters['c0']), 0.0, 1.0))

    DV = 0.05  # fixed doubt-gradient scale (not fitted)

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- label-degeneracy gate (printed labels not individuable) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    # ---------- doubt x reading-position weights ----------
    if degenerate:
        w = np.ones(n, dtype=float)
    else:
        logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(np.clip(logw, -700.0, 0.0))

    wd = w[disc]
    wmax = float(np.max(wd))
    if (not np.isfinite(wmax)) or wmax <= 0.0:
        what = np.ones(m, dtype=float)
        wmax = 1.0
    else:
        what = wd / wmax

    # ---------- divisive normalization ----------
    num = float(np.sum(signs * what))
    den = sigma + float(np.sum(what))
    if den <= 0.0:
        E = 0.0
    else:
        E = num / den
    p_core_a = sig(beta * E)

    # ---------- doubt-gradient displacement leak (bounded, evidence-free) ----------
    d = 0.0
    if (not degenerate) and m < n:
        mask = np.ones(n, dtype=bool)
        mask[disc] = False
        sil_idx = np.nonzero(mask)[0]
        if sil_idx.size > 0:
            k_s = int(sil_idx[int(np.argmax(w[sil_idx]))])
            ws = float(w[k_s])
            k_l = int(disc[int(np.argmax(wd))])
            gap = float(v[k_l] - v[k_s])          # positive iff silent row is MORE doubtful
            mult = float(np.clip(gap / DV, 0.0, 1.0))
            if np.isfinite(ws) and (ws + wmax) > 0.0:
                d = mult * ws / (ws + wmax)
    d = float(min(max(d, 0.0), 1.0))

    # ---------- HALF-ROW / HALF-LOUD-VOICE effective load, with partial unanimity discount ----------
    sum_what = float(np.sum(what))
    if not np.isfinite(sum_what):
        sum_what = float(m)
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    c = 1.0 if min(n_a, n_b) >= 1.0 else c0   # unanimity: adjudication cost discounted, not waived
    expo = 0.5 * float(m - 2) + 0.5 * c * (sum_what - 1.0)
    expo = float(max(expo, 0.0))

    keep = (1.0 - eps0) * (rho ** expo) * (1.0 - kappa * d)
    keep = float(min(max(keep, 0.0), 1.0))
    g = 1.0 - keep

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- gamma: [1.9, 2.5]
- lam: [0.28, 0.45]
- sigma: [0.12, 0.30]
- beta: [2.8, 3.8]
- eps0: [0.07, 0.15]
- rho: [0.64, 0.80]
- kappa: [0.40, 0.95]
- c0: [0.48, 0.75]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2820 (var=0.0035) vs this=0.3068 (var=0.0031)
- Experiment 2: real=0.1744 (var=0.0127) vs this=0.3256 (var=0.0184)
- Experiment 3: real=0.2856 (var=0.0160) vs this=0.2956 (var=0.0037)
- Experiment 4: real=0.1500 (var=0.0194) vs this=0.1067 (var=0.0070)
- Experiment 5: real=0.7427 (var=0.0173) vs this=0.7473 (var=0.0050)
- Experiment 6: real=0.5067 (var=0.0112) vs this=0.5208 (var=0.0068)
- Experiment 7: real=0.4383 (var=0.0023) vs this=0.4628 (var=0.0033)
- Experiment 8: real=0.7697 (var=0.0057) vs this=0.7766 (var=0.0030)
- Experiment 9: real=0.8189 (var=0.0074) vs this=0.7783 (var=0.0025)
- Experiment 10: real=0.4178 (var=0.0051) vs this=0.4247 (var=0.0030)
- Experiment 11: real=0.5925 (var=0.0026) vs this=0.6033 (var=0.0045)
- Experiment 12: real=0.2608 (var=0.0079) vs this=0.2617 (var=0.0055)
- Experiment 13: real=0.5460 (var=0.0126) vs this=0.4660 (var=0.0150)
- Experiment 14: real=0.4675 (var=0.0144) vs this=0.5387 (var=0.0195)
- Experiment 15: real=0.8620 (var=0.0155) vs this=0.4777 (var=0.0043)
- Experiment 16: real=0.8260 (var=0.0180) vs this=0.4215 (var=0.0085)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0707 -> ACCEPTED
- iter 2: loss=0.0760 -> REJECTED
- iter 3: loss=0.0692 -> ACCEPTED
- iter 4: loss=0.0549 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.0549 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Previous candidate values (this loop):**
  - iter 1: 0.2527 (var=0.0020) (Δ vs real -0.0292)
  - iter 2: 0.2335 (var=0.0029) (Δ vs real -0.0485)
  - iter 3: 0.2480 (var=0.0016) (Δ vs real -0.0340)
  - iter 4 (most recent): 0.2435 (var=0.0024) (Δ vs real -0.0385)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8515 (var=0.0078)
- pi_2: 0.1903 (var=0.0042)
- pi_3: 0.2870 (var=0.0019)
- pi_4: 0.2843 (var=0.0034)
- pi_5: 0.2722 (var=0.0027)
- pi_6: 0.2560 (var=0.0021)
- pi_7: 0.2878 (var=0.0023)
- pi_8: 0.2730 (var=0.0019)
- pi_9: 0.3068 (var=0.0031)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Previous candidate values (this loop):**
  - iter 1: 0.2897 (var=0.0103) (Δ vs real +0.1153)
  - iter 2: 0.2744 (var=0.0078) (Δ vs real +0.1000)
  - iter 3: 0.3622 (var=0.0155) (Δ vs real +0.1878)
  - iter 4 (most recent): 0.3214 (var=0.0119) (Δ vs real +0.1469)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3828 (var=0.0220)
- pi_1: -0.7044 (var=0.0395)
- pi_3: 0.1722 (var=0.0132)
- pi_4: 0.1856 (var=0.0146)
- pi_5: 0.1956 (var=0.0184)
- pi_6: 0.1639 (var=0.0160)
- pi_7: 0.2300 (var=0.0149)
- pi_8: 0.2117 (var=0.0142)
- pi_9: 0.3256 (var=0.0184)

### Experiment 3
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Previous candidate values (this loop):**
  - iter 1: 0.2167 (var=0.0063) (Δ vs real -0.0689)
  - iter 2: 0.1583 (var=0.0049) (Δ vs real -0.1272)
  - iter 3: 0.2389 (var=0.0078) (Δ vs real -0.0467)
  - iter 4 (most recent): 0.2733 (var=0.0044) (Δ vs real -0.0122)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9389 (var=0.0016)
- pi_2: 0.5100 (var=0.0066)
- pi_1: 0.8522 (var=0.0137)
- pi_4: 0.3056 (var=0.0052)
- pi_5: 0.2767 (var=0.0068)
- pi_6: 0.2928 (var=0.0057)
- pi_7: 0.2778 (var=0.0058)
- pi_8: 0.2794 (var=0.0061)
- pi_9: 0.2956 (var=0.0037)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Previous candidate values (this loop):**
  - iter 1: 0.1383 (var=0.0104) (Δ vs real -0.0117)
  - iter 2: 0.1050 (var=0.0066) (Δ vs real -0.0450)
  - iter 3: 0.1533 (var=0.0104) (Δ vs real +0.0033)
  - iter 4 (most recent): 0.1733 (var=0.0105) (Δ vs real +0.0233)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4917 (var=0.0165)
- pi_3: 0.9850 (var=0.0010)
- pi_1: 0.8500 (var=0.0219)
- pi_4: 0.1583 (var=0.0128)
- pi_5: 0.1650 (var=0.0115)
- pi_6: 0.1450 (var=0.0086)
- pi_7: 0.1500 (var=0.0133)
- pi_8: 0.1217 (var=0.0081)
- pi_9: 0.1067 (var=0.0070)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Previous candidate values (this loop):**
  - iter 1: 0.8013 (var=0.0070) (Δ vs real +0.0587)
  - iter 2: 0.7840 (var=0.0053) (Δ vs real +0.0413)
  - iter 3: 0.7987 (var=0.0041) (Δ vs real +0.0560)
  - iter 4 (most recent): 0.7847 (var=0.0051) (Δ vs real +0.0420)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9847 (var=0.0005)
- pi_4: 0.7440 (var=0.0064)
- pi_1: 0.8340 (var=0.0138)
- pi_2: 0.8493 (var=0.0126)
- pi_5: 0.7427 (var=0.0074)
- pi_6: 0.7773 (var=0.0049)
- pi_7: 0.7467 (var=0.0048)
- pi_8: 0.7487 (var=0.0072)
- pi_9: 0.7473 (var=0.0050)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Previous candidate values (this loop):**
  - iter 1: 0.5175 (var=0.0075) (Δ vs real +0.0108)
  - iter 2: 0.4858 (var=0.0093) (Δ vs real -0.0208)
  - iter 3: 0.4850 (var=0.0092) (Δ vs real -0.0217)
  - iter 4 (most recent): 0.4983 (var=0.0111) (Δ vs real -0.0083)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8367 (var=0.0052)
- pi_3: 0.0300 (var=0.0018)
- pi_1: 0.1550 (var=0.0169)
- pi_2: 0.4942 (var=0.0103)
- pi_5: 0.5083 (var=0.0072)
- pi_6: 0.5033 (var=0.0076)
- pi_7: 0.4817 (var=0.0130)
- pi_8: 0.4725 (var=0.0109)
- pi_9: 0.5208 (var=0.0068)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4383 (var=0.0023)
**Previous candidate values (this loop):**
  - iter 1: 0.4517 (var=0.0034) (Δ vs real +0.0133)
  - iter 2: 0.4294 (var=0.0033) (Δ vs real -0.0089)
  - iter 3: 0.4328 (var=0.0033) (Δ vs real -0.0056)
  - iter 4 (most recent): 0.4503 (var=0.0035) (Δ vs real +0.0119)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4711 (var=0.0034)
- pi_4: 0.7642 (var=0.0020)
- pi_1: 0.2703 (var=0.0046)
- pi_2: 0.3842 (var=0.0037)
- pi_3: 0.3467 (var=0.0009)
- pi_6: 0.4419 (var=0.0028)
- pi_7: 0.4508 (var=0.0038)
- pi_8: 0.4408 (var=0.0032)
- pi_9: 0.4628 (var=0.0033)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7697 (var=0.0057)
**Previous candidate values (this loop):**
  - iter 1: 0.8097 (var=0.0027) (Δ vs real +0.0400)
  - iter 2: 0.8600 (var=0.0021) (Δ vs real +0.0903)
  - iter 3: 0.8043 (var=0.0027) (Δ vs real +0.0346)
  - iter 4 (most recent): 0.7949 (var=0.0026) (Δ vs real +0.0251)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7326 (var=0.0025)
- pi_5: 0.4900 (var=0.0045)
- pi_1: 0.1511 (var=0.0123)
- pi_2: 0.3354 (var=0.0031)
- pi_3: 0.0803 (var=0.0009)
- pi_6: 0.7540 (var=0.0026)
- pi_7: 0.7686 (var=0.0034)
- pi_8: 0.7611 (var=0.0025)
- pi_9: 0.7766 (var=0.0030)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)
    LAM = 1.5      # reading-order recency gain (shared by both accounts)
    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n == V_DESIGN.size:
            v = V_DESIGN.copy()
        else:
            v = np.linspace(0.95, 0.55, n)
        v = np.clip(v, 0.5, 0.995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        signs = np.sign(diff[disc])
        # keep only CONFLICT displays (both options win at least one row)
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)
        k = int(np.argmax(logw))
        favored_a = signs[k] > 0.0

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        hits += 1.0 if (chose_a == favored_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8189 (var=0.0074)
**Previous candidate values (this loop):**
  - iter 1: 0.7989 (var=0.0017) (Δ vs real -0.0200)
  - iter 2: 0.8453 (var=0.0013) (Δ vs real +0.0264)
  - iter 3: 0.7911 (var=0.0030) (Δ vs real -0.0278)
  - iter 4 (most recent): 0.7969 (var=0.0025) (Δ vs real -0.0219)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4964 (var=0.0034)
- pi_6: 0.7364 (var=0.0031)
- pi_1: 0.1528 (var=0.0126)
- pi_2: 0.3228 (var=0.0037)
- pi_3: 0.0178 (var=0.0002)
- pi_4: 0.7092 (var=0.0034)
- pi_7: 0.7714 (var=0.0022)
- pi_8: 0.7536 (var=0.0025)
- pi_9: 0.7783 (var=0.0025)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (fixed by the experiment)
    v_default = np.array([0.53, 0.72, 0.86, 0.93, 0.99], dtype=float)

    def weights(n):
        if n <= v_default.size:
            v = v_default[:n]
        else:
            v = np.concatenate([v_default, np.full(n - v_default.size, 0.99)])
        v = np.clip(v, 0.5, 0.9995)
        logw = 0.9 * np.log(1.0 - v) + 1.5 * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        return np.exp(logw)

    wcache = {}
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].values,
                                  data['option_b_ratings'].values,
                                  data['response'].values):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        # keep only CONFLICT displays (both options favoured by at least one expert)
        if nA == 0 or nB == 0:
            continue
        if n not in wcache:
            wcache[n] = weights(n)
        w = wcache[n]
        S = float(np.sum(np.sign(d) * w))
        if S == 0.0:
            continue
        fav = 0 if S > 0 else 1  # response coding: 0 = A, 1 = B
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4178 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: 0.4189 (var=0.0032) (Δ vs real +0.0011)
  - iter 2: 0.4192 (var=0.0023) (Δ vs real +0.0014)
  - iter 3: 0.4156 (var=0.0028) (Δ vs real -0.0022)
  - iter 4 (most recent): 0.4036 (var=0.0036) (Δ vs real -0.0142)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7386 (var=0.0031)
- pi_5: 0.6297 (var=0.0048)
- pi_1: 0.6053 (var=0.0037)
- pi_2: 0.5439 (var=0.0033)
- pi_3: 0.7967 (var=0.0021)
- pi_4: 0.6303 (var=0.0046)
- pi_7: 0.3878 (var=0.0029)
- pi_8: 0.4492 (var=0.0019)
- pi_9: 0.4247 (var=0.0030)

### Experiment 11
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed printed validities for this experiment
    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)
    n = v.size
    j = np.arange(n, dtype=float)

    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)
    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j
    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)
    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n or b.size != n:
            continue
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue

        # MDSC-NV predicted winner: verdict of the single most doubtful
        # discriminating expert
        k = disc[int(np.argmax(logu[disc]))]
        pred_mdsc = 0 if d[k] > 0 else 1

        # SSI-DS+ predicted winner: sign of the weighted sum over all
        # discriminating rows (coalitions add, late rows dominate)
        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))
        if E == 0.0:
            continue
        pred_ssi = 0 if E > 0 else 1

        if pred_ssi == pred_mdsc:
            continue  # non-diagnostic trial type

        tot += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == pred_mdsc:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.5925 (var=0.0026)
**Previous candidate values (this loop):**
  - iter 1: 0.6142 (var=0.0034) (Δ vs real +0.0217)
  - iter 2: 0.6271 (var=0.0037) (Δ vs real +0.0346)
  - iter 3: 0.6125 (var=0.0028) (Δ vs real +0.0200)
  - iter 4 (most recent): 0.5946 (var=0.0052) (Δ vs real +0.0021)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6833 (var=0.0038)
- pi_6: 0.2583 (var=0.0034)
- pi_1: 0.1550 (var=0.0102)
- pi_2: 0.4058 (var=0.0050)
- pi_3: 0.1054 (var=0.0016)
- pi_4: 0.3046 (var=0.0036)
- pi_5: 0.2854 (var=0.0033)
- pi_8: 0.6367 (var=0.0043)
- pi_9: 0.6033 (var=0.0045)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)

    GAMMA = 0.90
    LAM = 1.50
    DELTA = 15.0
    LAM_M = 1.725

    hits = 0.0
    tot = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n <= V_DESIGN.size:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])
        v = np.clip(v, 0.5, 0.9995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        if disc.size == 0:
            continue
        signs = np.sign(diff[disc])

        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated
        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        E = float(np.sum(signs * w))
        if abs(E) < 1e-9:
            continue
        ssi_side = 1.0 if E > 0 else -1.0

        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides
        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)
        k = int(np.argmax(sal))
        mdsc_side = float(signs[k])
        if mdsc_side == 0.0:
            continue

        # keep only trials where the two accounts point to OPPOSITE options
        if mdsc_side == ssi_side:
            continue

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        ssi_favours_a = (ssi_side > 0)
        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.2608 (var=0.0079)
**Previous candidate values (this loop):**
  - iter 1: 0.2108 (var=0.0034) (Δ vs real -0.0500)
  - iter 2: 0.2054 (var=0.0027) (Δ vs real -0.0554)
  - iter 3: 0.2317 (var=0.0046) (Δ vs real -0.0292)
  - iter 4 (most recent): 0.2204 (var=0.0043) (Δ vs real -0.0404)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7412 (var=0.0040)
- pi_7: 0.2592 (var=0.0042)
- pi_1: 0.8400 (var=0.0123)
- pi_2: 0.5587 (var=0.0027)
- pi_3: 0.8158 (var=0.0005)
- pi_4: 0.7417 (var=0.0031)
- pi_5: 0.7421 (var=0.0057)
- pi_8: 0.2729 (var=0.0043)
- pi_9: 0.2617 (var=0.0055)

### Experiment 13
**Design**
  A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n < 5:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [int(i) for i in np.nonzero(d != 0.0)[0]]
        # require feature 0 (validity .68) to discriminate
        if 0 not in disc:
            continue
        # require the global doubt anchor, feature 3 (validity .52), to be SILENT
        if 3 in disc:
            continue
        fav_a = [i for i in disc if d[i] > 0.0]
        fav_b = [i for i in disc if d[i] < 0.0]
        if d[0] > 0.0:
            lone, other = fav_a, fav_b
            lone_is_a = True
        else:
            lone, other = fav_b, fav_a
            lone_is_a = False
        # exactly one dissenter (feature 0) against a coalition of >=2 more valid experts
        if len(lone) != 1 or len(other) < 2:
            continue
        try:
            resp = int(r)
        except Exception:
            continue
        chose_lone = (resp == 0) if lone_is_a else (resp == 1)
        hits.append(1.0 if chose_lone else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5460 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1: 0.5060 (var=0.0123) (Δ vs real -0.0400)
  - iter 2: 0.5030 (var=0.0094) (Δ vs real -0.0430)
  - iter 3: 0.5590 (var=0.0150) (Δ vs real +0.0130)
  - iter 4 (most recent): 0.5310 (var=0.0134) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6410 (var=0.0093)
- pi_8: 0.2680 (var=0.0082)
- pi_1: 0.1620 (var=0.0145)
- pi_2: 0.1700 (var=0.0205)
- pi_3: 0.0110 (var=0.0005)
- pi_4: 0.2580 (var=0.0097)
- pi_5: 0.2380 (var=0.0124)
- pi_6: 0.2580 (var=0.0074)
- pi_9: 0.4660 (var=0.0150)

### Experiment 14
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.62, 0.78, 0.86, 0.93, 0.53, 0.97], dtype=float)
    ANCHOR = 4  # the .53 expert (below DTV's doubt trigger)
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].tolist(),
                                  data['option_b_ratings'].tolist(),
                                  data['response'].tolist()):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n != 6:
            continue
        diff = a[:6] - b[:6]
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 3:
            continue
        # the globally most doubtful expert (.53) must be SILENT
        if ANCHOR in set(disc.tolist()):
            continue
        d = diff[disc]
        na = int(np.sum(d > 0.0))
        nb = m - na
        if min(na, nb) != 1 or max(na, nb) < 2:
            continue
        if na == 1:
            lone = int(disc[d > 0.0][0])
            fav = 0  # option A favoured by the lone dissenter
        else:
            lone = int(disc[d < 0.0][0])
            fav = 1
        # the lone dissenter must be the LEAST valid discriminating expert
        if abs(V[lone] - float(np.min(V[disc]))) > 1e-9:
            continue
        tot += 1.0
        if int(resp) == fav:
            hits += 1.0
    if tot < 1.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.4675 (var=0.0144)
**Previous candidate values (this loop):**
  - iter 1: 0.5725 (var=0.0138) (Δ vs real +0.1050)
  - iter 2: 0.5763 (var=0.0118) (Δ vs real +0.1088)
  - iter 3: 0.5813 (var=0.0138) (Δ vs real +0.1138)
  - iter 4 (most recent): 0.5513 (var=0.0220) (Δ vs real +0.0837)
**Other theories' values on this metric (for reference):**
- pi_8: 0.2313 (var=0.0086)
- pi_7: 0.6388 (var=0.0149)
- pi_1: 0.1550 (var=0.0197)
- pi_2: 0.1313 (var=0.0107)
- pi_3: 0.0238 (var=0.0015)
- pi_4: 0.2812 (var=0.0111)
- pi_5: 0.2863 (var=0.0094)
- pi_6: 0.2313 (var=0.0138)
- pi_9: 0.5387 (var=0.0195)

### Experiment 15
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    ANCHOR = 3  # column with the lowest printed validity (0.55)
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in list(a_raw)]
        b = [int(x) for x in list(b_raw)]
        k = min(len(a), len(b))
        if k <= ANCHOR:
            continue
        if a[ANCHOR] == b[ANCHOR]:
            continue  # anchor silent -> not a coalition test
        n_a = 0
        n_b = 0
        for i in range(k):
            if a[i] > b[i]:
                n_a += 1
            elif b[i] > a[i]:
                n_b += 1
        anchor_favors_a = a[ANCHOR] > b[ANCHOR]
        opposing = n_b if anchor_favors_a else n_a
        if opposing < 1:
            continue  # pure dominance for the anchor side -> no coalition
        try:
            resp = int(r)
        except Exception:
            continue
        chose_a = (resp == 0)
        if chose_a == anchor_favors_a:
            hits += 1.0
        tot += 1.0
    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8620 (var=0.0155)
**Previous candidate values (this loop):**
  - iter 1: 0.7773 (var=0.0035) (Δ vs real -0.0847)
  - iter 2: 0.8577 (var=0.0023) (Δ vs real -0.0043)
  - iter 3: 0.7853 (var=0.0034) (Δ vs real -0.0767)
  - iter 4 (most recent): 0.7977 (var=0.0046) (Δ vs real -0.0643)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7610 (var=0.0035)
- pi_9: 0.4777 (var=0.0043)
- pi_1: 0.3003 (var=0.0058)
- pi_2: 0.2920 (var=0.0039)
- pi_3: 0.1703 (var=0.0013)
- pi_4: 0.2870 (var=0.0043)
- pi_5: 0.2827 (var=0.0026)
- pi_6: 0.2450 (var=0.0025)
- pi_8: 0.7143 (var=0.0022)

### Experiment 16
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float).ravel()
        b = np.asarray(list(b_raw), dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k < 6:
            continue
        # the near-chance anchor expert (column 0, validity .52) must discriminate
        if a[0] == b[0]:
            continue
        anchor_side = 0 if a[0] > b[0] else 1  # 0 == option A, 1 == option B
        # count late, mid-doubt rows (columns 3,4,5) that oppose the anchor
        opp = 0
        for j in (3, 4, 5):
            if a[j] == b[j]:
                continue
            fav = 0 if a[j] > b[j] else 1
            if fav != anchor_side:
                opp += 1
        if opp < 2:
            continue
        n += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        if r == anchor_side:
            hits += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)

```

**Observed (real) value:** 0.8260 (var=0.0180)
**Previous candidate values (this loop):**
  - iter 1: 0.8040 (var=0.0053) (Δ vs real -0.0220)
  - iter 2: 0.8770 (var=0.0026) (Δ vs real +0.0510)
  - iter 3: 0.7890 (var=0.0043) (Δ vs real -0.0370)
  - iter 4 (most recent): 0.8180 (var=0.0041) (Δ vs real -0.0080)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4215 (var=0.0085)
- pi_7: 0.7245 (var=0.0054)
- pi_1: 0.4340 (var=0.0037)
- pi_2: 0.2900 (var=0.0065)
- pi_3: 0.1700 (var=0.0026)
- pi_4: 0.3045 (var=0.0052)
- pi_5: 0.3325 (var=0.0058)
- pi_6: 0.3115 (var=0.0039)
- pi_8: 0.6770 (var=0.0061)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
DTG-2R/OC-sharp — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion and the one-reason limb is degraded by OPPOSITION rather than by raw display length. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), and the criterion is now sharp and low: theta ~ .558-.588 with tau ~ .006-.013. This means an expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is treated as ordinary evidence, not as a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows (e.g. three ~.54 columns). (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86), keep_t = (1-eps_t)*rho_t^(m-2)*rho_tc^(k_conf-1) with k_conf the size of the minority side. Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits (2-vs-2, 3-vs-3 tally ties) drift markedly toward chance: the adjudication, not the scan, is what is lost. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E), with its own separable load (rho_m) and opposition (rho_c) lapses, so unanimous 6-row displays stay confident (~.74) while a lone mid-doubt dissenter against a coalition sits near chance (~.46-.55). (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;
    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- printed validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    theta = float(parameters['theta'])
    tau = float(max(float(parameters['tau']), 1e-4))
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_t = float(np.clip(float(parameters['rho_t']), 1e-6, 1.0))
    rho_tc = float(np.clip(float(parameters['rho_tc']), 1e-6, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- lapses ----------
    expo = max(m - 2, 0)
    keep_c = (1.0 - eps0) * (rho_m ** expo) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))
    # one-reason limb: scanning extra rows is nearly free, holding a RIVAL
    # coalition is not -> charge rho_tc per minority row beyond the first
    keep_t = (1.0 - eps_t) * (rho_t ** expo) * (rho_tc ** max(k_conf - 1, 0))
    keep_t = float(min(max(keep_t, 0.0), 1.0))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        p_core = sig(kappa * (n_a - n_b))
        p_a = 0.5 + keep_c * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- absolute doubt trigger over DISCRIMINATING rows ----------
    zz = np.clip((v - theta) / tau, -60.0, 60.0)
    gtrig = 1.0 / (1.0 + np.exp(zz))          # = sigmoid((theta - v)/tau)
    gd = gtrig[disc]
    P_trig = float(np.clip(np.max(gd), 0.0, 1.0))

    # verdict of the consulted (near-chance) expert; recency tilt among them
    logs = np.log(np.clip(gd, 1e-300, None)) + mu * disc.astype(float)
    logs = logs - np.max(logs)
    s = np.exp(np.clip(logs, -700.0, 0.0))
    ssum = float(np.sum(s))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        s = np.full(m, 1.0 / m)
    else:
        s = s / ssum
    p_or_a = float(np.sum(s[signs > 0.0]))
    p_t = 0.5 + keep_t * (p_or_a - 0.5)

    # ---------- compensatory fallback (divisively normalised anti-validity) ----------
    logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
    ld = logw[disc]
    ld = ld - np.max(ld)
    w = np.exp(np.clip(ld, -700.0, 0.0))
    wsum = float(np.sum(w))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        w = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * w)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- doubt-threshold-gated mixture ----------
    p_a = P_trig * p_t + (1.0 - P_trig) * p_c
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- theta: [0.558, 0.588]
- tau: [0.006, 0.013]
- mu: [0.45, 0.95]
- gamma: [1.5, 2.3]
- lam: [0.0, 0.22]
- sigma: [0.40, 0.72]
- beta: [1.8, 2.6]
- eps_t: [0.20, 0.36]
- rho_t: [0.985, 1.00]
- rho_tc: [0.62, 0.86]
- eps0: [0.13, 0.27]
- rho_m: [0.89, 0.97]
- rho_c: [0.74, 0.90]
- tally_kappa: [0.6, 1.3]
- validities: validities

`rationale`: MINIMAL DIFF on the ACCEPTED base (loss .0692): `predict` and `policy` are re-emitted byte-identical; only three parameter ranges change, exactly the two edits the critic prescribed after this base was accepted.

(1) PRIMARY — sharpen and lower the doubt gate: theta [.572,.605] -> [.558,.588], tau [.008,.020] -> [.006,.013]. The critic's decomposition of Exp14 (.581 = P_trig*.85 + (1-P_trig)*.454) shows the compensatory limb is ALREADY at the human value (.468) there and the entire +.114 residual is trigger leakage at v=.62. At the new mid-box (theta=.573, tau=.0095): g(.52)~1.00, g(.538)~1.00, g(.545)~.94, g(.55)~.90, g(.58)~.32, g(.62)~.007, g(.68)~0. So the .62 column in Exp14 is now essentially excluded from the one-reason limb (predicted .58 -> ~.46-.48, the second-largest residual eliminated), while every anchor the corpus depends on stays inside the gate: Exp15's .55, Exp16's .52, Exp8's .535/.540/.545, Exp9's .538/.542/.546, Exp11's .55, Exp12's .51, Exp1's .55, Exp4's .53. Exp13 is untouched (its lone dissenter is v=.68, g~0 before and after). theta is kept >= .558 and tau >= .006 precisely so the .545/.55 anchors keep g >= .9, as the critic required.

(2) SECONDARY — narrow rho_t [.96,1.00] -> [.985,1.00]. rho_t acts only through rho_t^(m-2), so this recovers the lone-anchor cells lost to row-scan decay in long displays: Exp15 (.785, real .862) and Exp16 (.789, real .826) gain ~+.02-.03 and ~+.02, and Exp9 (.791, real .819) gains too, while the m=2-3 displays of Exps 1/3/4 (currently well placed) are unmoved. I did NOT cut eps_t: a blanket eps_t cut was tested in iter 2 and the gate rejected it.

(3) EVERYTHING ELSE FROZEN, as instructed: rho_tc [.62,.86] (the edit the gate rewarded) stays exactly as implemented; eps0/rho_m/rho_c/mu/gamma/lam/sigma/beta/tally_kappa and the .002 degeneracy gate are untouched, protecting the best-in-corpus cells Exp10 (.416 vs .418), Exp7 (.433 vs .438), Exp6 (~.49 vs .507), Exp13 (.559 vs .546). Exp2 is deliberately not chased — it is a documented structural cost of the DTG family (all siblings overshoot it) and the only fixes would undo the accepted rho_tc edit or weaken the near-chance gate that Exps 15/16 depend on. Expected collateral is <= .02 on Exp3/Exp8/Exp11 from the sharper gate (Exp3's .58 column now routes ~2/3 of its mass through the compensatory limb, which is also anti-validity but more lapse-prone on 2-vs-2 ties, so it should move .239 slightly toward the real .286), against a ~.11 gain on Exp14 and ~.03 on Exp15/16/9 combined.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate (DTG-2R) is faithful to the arbiter's prescription: an absolute near-chance validity gate (sigmoid((theta-v)/tau), theta~.59, tau~.014) commanding a load-insensitive one-reason branch, a divisively-normalised anti-validity compensatory fallback when no sub-threshold expert speaks, the label-degeneracy tally collapse, and separate load/conflict lapses. No mechanism drift, and the accept gate confirms it is the best-fitting instantiation so far (loss .0707).

Per-experiment fit is good on 11 of 16 cells (|err| <= .05): Exp1 .253/.282, Exp4 .138/.150, Exp6 .518/.507, Exp7 .452/.438, Exp9 .799/.819, Exp10 .419/.418 (best of all theories), Exp11 .614/.593, Exp12 .211/.261, Exp13 .506/.546, Exp16 .804/.826, Exp8 .810/.770. The degeneracy gate is clearly doing real work (Exp6 at chance, Exp8/9 at ~.80 where every non-DTG theory is either at ceiling or floor), and the compensatory branch is what finally rescues Exp10 and Exp12, which pi_1..pi_6 miss by .3-.5.

The residual errors are systematic and point in one consistent direction rather than being random noise:
(a) The COMPENSATORY branch is a touch too decisive / too anti-validity. Exp2 overshoots badly (+.116: .290 vs .174 — the tally-minus-TTB gap is inflated), Exp3 undershoots (-.069, too anti-validity on tally-tie trials), Exp12 undershoots (-.050, same direction), Exp1 slightly under (-.029), and dominance displays are too confident (Exp5 .801 vs .743; Exp8 .810 vs .770). All six are consistent with beta too high / sigma too small / eps0 too small in the fallback.
(b) The recency+anti-validity tilt inside the fallback over-favours a lone EARLY least-valid dissenter. Exp14 (+.105: .573 vs .468) and Exp13 (-.040: .506 vs .546) both isolate a column-0 lone dissenter against a 2-3 coalition, with human values essentially at chance in both (.47, .55). The model separates them too much: it follows the v=.62 col-0 dissenter at .57 but the v=.68 col-0 dissenter at .51, i.e. the (1-v)^gamma gradient is steeper than humans show in this range.
(c) The TRIGGER branch is slightly too lapse-y: Exp15 .777 vs .862 (-.085) and Exp16 .804 vs .826 (-.022). The arbiter's target for these cells was ~.85-.86 flat; eps_t in [.20,.36] is eating ~5-8 points of anchor-following.
Rationale: Keep DTG-2R exactly as specified by the arbiter — absolute doubt gate + one-reason branch + divisively-normalised anti-validity fallback + degeneracy tally + split load/conflict lapse. Do NOT change the mechanism family; only re-tune three knobs, each of which pushes several cells in the same helpful direction. (1) Soften the compensatory branch: raise sigma to roughly [.60,.95] and/or lower beta to roughly [1.4,2.1], with eps0 nudged up to ~[.17,.30]. This pulls Exp2 down from .29 toward .17-.22, lifts Exp3 (.22->.26) and Exp12 (.21->.25), lifts Exp1 slightly toward .28, and brings the dominance cells Exp5 (.80->.76) and Exp8 (.81->.78) down to their human values. Watch Exp4 (.138) and Exp10 (.419) — both should stay put or improve slightly, so this is a low-risk global move. (2) Flatten the anti-validity gradient in the fallback only: lower gamma to ~[1.1,1.7] and/or trim lam to ~[.00,.12]. Exp13 and Exp14 both isolate a column-0 lone dissenter and both should land near .48-.55; the current gamma=1.9 makes the v=.62 dissenter far too compelling (Exp14 .573 vs .468) while barely helping the v=.68 case (Exp13 .506 vs .546). A shallower exponent compresses the pair toward the .47-.55 band the data demand. Make sure this does NOT change the trigger branch — Exps 15/16 must not be touched by gamma. (3) Make the one-reason branch slightly stiffer: drop eps_t to ~[.12,.24] and keep rho_t high (>=.95). Target Exp15 ~.84-.86 and Exp16 ~.82-.84, exactly the flat-across-coalition-size following the arbiter asked for; this is the single largest remaining residual (-.085 on Exp15). Expect a small cost on Exp1 (first-cue adherence may fall .01-.02 below .282) — that cost is smaller than the Exp15/16 gain, but if it bites, compensate by nudging theta down to the bottom of its range (~.572) so the v=.60 column in Exp2 and the v=.62 column in Exp14 leak less trigger mass into the one-reason branch. Do not widen theta above .605 or shrink tau below .008: the current gate cleanly separates .52-.55 anchors from .62-.68 dissenters and that separation is the theory's main empirical win.

**Outcome of this advice:** iter 1 candidate loss=0.0707 -> iter 2 candidate loss=0.0760 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is a verbatim re-emission of the accepted base's `predict`/`policy` with only parameter-range edits, so it remains fully faithful to the arbiter's DTG-2R family (absolute near-chance gate -> load-free one-reason limb; divisively-normalised anti-validity compensatory fallback; degeneracy tally collapse; split load/conflict lapse). No mechanism drift. But the gate rejected it (.0760 vs .0707), and the per-cell trajectory cleanly separates which of my three previous knobs helped and which hurt.

WHAT HELPED (knob 3, stiffening the one-reason limb: eps_t .20-.36 -> .13-.23, rho_t >= .95): Exp15 .777 -> .858 (real .862; residual -.085 -> -.004, the single biggest win in the whole table) and Exp16 .804 -> .877 (real .826; went from -.022 under to +.051 over). Exp9 .799 -> .845 (real .819) also flipped from under to over. So the direction was right but the step was too large: the trigger limb is now slightly TOO decisive, and it dragged Exp8 from .810 to .860 (real .770; residual +.040 -> +.090), which is now the second-largest error in the set.

WHAT HURT (knobs 1 and 2). The compensatory softening (sigma up, beta down, eps0 up) did NOT move the tally-tie cells toward chance as I predicted; they moved further from the data: Exp3 .217 -> .158 (real .286), Exp1 .253 -> .234 (real .282), Exp4 .138 -> .105 (real .150), Exp12 .211 -> .205 (real .261). Only Exp5 (.801 -> .784, real .743) and Exp2 (.290 -> .274, real .174) improved, and Exp2 is still +.100 off. The gamma/lam flattening was simply inert where it was supposed to act: Exp14 .573 -> .576 (real .468) and Exp13 .506 -> .503 (real .546) are unchanged. So the (1-v)^gamma exponent is not the lever controlling the lone-early-dissenter cells; the sign/magnitude of the lapse on conflicting displays is.

DIAGNOSIS OF THE REMAINING RESIDUAL STRUCTURE: nearly every remaining compensatory-branch error is an error of EXCESS DECISIVENESS ON CONFLICT DISPLAYS, in whichever direction the fallback happens to point. Exp1 (.234 vs .282), Exp3 (.158 vs .286), Exp4 (.105 vs .150), Exp12 (.205 vs .261) are all too far BELOW .5; Exp14 (.576 vs .468) and Exp2 (.274 vs .174) are too far ABOVE. Six of seven compensatory conflict cells would improve simultaneously if choices on displays with a genuine minority coalition were pulled harder toward chance. That is exactly what rho_c (the per-minority-row lapse) is for, and it is currently far too mild (.74-.90). It is also the one knob I have not yet pushed.
Rationale: Stay inside DTG-2R exactly as written (no code changes needed again; ranges only). Three concrete instructions, built on the UNCHANGED accepted base (loss .0707), not on this rejected candidate:

(1) KEEP a reduced-but-smaller eps_t. My iter-1 push here was the one component that clearly worked (Exp15 -.085 -> -.004) but overshot Exp16/Exp8/Exp9. Do not oscillate back to .20-.36 and do not stay at .13-.23; take the midpoint: eps_t in [.17, .27], rho_t in [.95, .99]. Target Exp15 ~.84, Exp16 ~.84, Exp8 ~.80-.82, Exp9 ~.82.

(2) REVERT the compensatory softening, which the gate punished. Put sigma back to ~[.42, .72], beta back to ~[1.8, 2.6], gamma back to ~[1.5, 2.3], lam back to ~[.00, .20], eps0 back to ~[.14, .26]. These edits moved Exp1/3/4/12 the wrong way and left Exp13/14 untouched; there is no evidence they buy anything.

(3) THE NEW KNOB — sharpen the CONFLICT lapse, not the display lapse. Lower rho_c from [.74, .90] to roughly [.60, .78]. rho_c is raised to the power k_conf = size of the minority coalition, so it acts only on displays where both options win rows; it leaves dominance (Exp5, Exp7 6-0 cells) and the trigger limb (Exps 15/16, and Exps 8/9 whose sub-.59 rows fire the gate) untouched. Predicted joint effect, all in the right direction at once: Exp14 .576 -> ~.51 (real .468), Exp2 .274 -> ~.20 (real .174), Exp3 .158 -> ~.22 (real .286), Exp1 .234 -> ~.27 (real .282), Exp4 .105 -> ~.14 (real .150), Exp12 .205 -> ~.24 (real .261). The only cost is Exp13 (.503, real .546) drifting a hair closer to .5, which is <=.01 of loss against six cells each gaining .03-.07.

(4) Leave theta/tau at the accepted base ([.572,.605] / [.008,.020]). The tightening in this iteration produced no isolable gain, and the .52-.55 vs .62-.68 separation is the theory's main empirical win — do not risk it. Also leave the degeneracy gate and tally_kappa alone: Exp10 (.419 vs .418) and Exp6 (~.50) are already the best in the corpus and should not move.

In short: keep one third of the last edit (a milder eps_t cut), discard the other two thirds, and spend the freed budget on a conflict-specific lapse that pushes six compensatory cells toward chance simultaneously.

**Outcome of this advice:** iter 2 candidate loss=0.0760 -> iter 3 candidate loss=0.0692 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's DTG-2R family: absolute near-chance validity gate (sigmoid((theta-v)/tau)) -> one-reason limb; divisively-normalised anti-validity compensatory fallback when no near-chance expert speaks; degeneracy tally collapse; separate load/conflict lapses. The only change from the accepted base is the opposition-scaled lapse on the trigger limb (rho_tc^(k_conf-1)), and it did what the proposer argued: Exp3 .158/.217 -> .239 (real .286), Exp1 -> .248 (real .282), Exp12 -> .232 (real .261), Exp4 .153 (real .150), Exp13 .559 (real .546), with Exp10 (.416 vs .418), Exp7 (.433 vs .438), Exp6 (.485 vs .507) essentially exact. The gate accepted it (.0692, new best). My iter-2 advice mix (keep a mild eps_t, revert the compensatory softening, add a conflict-specific lapse) was vindicated in spirit \u2014 the proposer implemented the conflict lapse on the trigger limb rather than on rho_c and it paid off.

The remaining loss is now extremely concentrated. Two cells carry ~70% of it:
(a) Exp14: .581 vs .468 (+.114). This cell is supposed to be pure compensatory (the .53 anchor is SILENT by construction; the lone dissenter is col0 at v=.62). But with theta up to .605 and tau up to .020, g(v=.62) = sigmoid((.605-.62)/.020) \u2248 .32 \u2014 a third of the trial mass is still being adjudicated by the one-reason limb, which follows the lone dissenter at ~.85. Decomposing the observed .581 = .32*.85 + .68*x gives x \u2248 .454 for the compensatory branch alone \u2014 i.e. the fallback is ALREADY at the human value (.468) and the entire residual is trigger leakage at v=.62. This also explains why my iter-1 gamma/lam flattening was inert on Exp13/14: the trigger mass, not the weight profile, was driving that cell.
(b) Exp2: .362 vs .174 (+.188), and it got WORSE this iteration (.290 -> .362) because rho_tc pulls the 2-vs-2 tally-tie displays (where the .55 column agrees with the TTB winner) down from ~.81 to ~.75, inflating p_tally - p_ttb_tie. Note that the two DTG siblings (pi_7 .230, pi_9 .326) also overshoot here, and the only theories that hit Exp2 (pi_3, pi_4, pi_6) fail catastrophically on Exps 12-16. This looks like a largely structural cost of the family and should NOT be chased hard.
Smaller, coherent residuals: Exp15 .785 vs .862 (-.077) and Exp16 .789 vs .826 (-.037) \u2014 the lone-anchor cells are still slightly too lapse-y in long (m=5,6) displays; Exp5 .799 vs .743 (+.056) and Exp8 .804 vs .770 (+.035) are mildly over-confident; Exp9 -.028, Exp11 +.020 are fine.
Rationale: The candidate was ACCEPTED and should be the base, but there is one quantified, low-risk in-family improvement left that is worth ~.013 of loss on its own, so run one more iteration on THIS candidate (no code changes needed beyond parameter ranges; keep rho_tc exactly as implemented \u2014 that edit was the one the gate rewarded).

(1) PRIMARY: sharpen and slightly lower the doubt gate so that v=.62 no longer leaks into the one-reason limb. Set theta to ~[.555, .585] and tau to ~[.006, .013]. At theta=.575/tau=.010 this gives g(.52)\u22481, g(.545)\u2248.95, g(.55)\u2248.92, g(.58)\u2248.31, g(.60)\u2248.08, g(.62)\u2248.01. Predicted: Exp14 .581 -> ~.46-.48 (real .468) \u2014 the second-largest residual essentially eliminated \u2014 with Exp13 untouched (its lone dissenter is v=.68, already g\u22480) and Exp15/16 anchors (.55/.52) still firmly inside the gate. I flagged in iters 1-2 that theta/tau should be left alone; I am reversing that specifically because the .581 = .32*.85 + .68*.454 decomposition shows the compensatory limb is already correct on Exp14 and only the leak is wrong. Do not push theta below ~.55 or tau below ~.006: the .545/.55 anchors in Exps 8/9/11/15 must keep g>=.9.

(2) SECONDARY, small: recover the lone-anchor cells lost to row-scan decay by narrowing rho_t to [.985, 1.00] (leave eps_t at [.20,.36] \u2014 a blanket eps_t cut was rejected by the gate in iter 2 and I will not repeat it). rho_t only acts through rho_t^(m-2), so this lifts the long-display cells Exp15 (~+.02-.03) and Exp16 (~+.02) while leaving the m=2-3 displays of Exps 1/3/4 \u2014 which are currently well placed \u2014 unmoved.

(3) DO NOT chase Exp2. Its residual is a structural property of any theory that both (i) follows a .55 expert that sides with a 5-row majority and (ii) discounts 2-vs-2 ties via rho_tc; every DTG sibling overshoots it, and the only fixes would either undo the accepted rho_tc edit (which would re-break Exp3 at .286) or weaken the near-chance gate that Exps 15/16 depend on. Leave it as a documented cost of the family.

(4) Freeze everything else: rho_tc [.62,.86], eps0/rho_m/rho_c at current values (Exp10 .416 vs .418, Exp7 .433 vs .438, Exp13 .559 vs .546 are at or near best-in-corpus \u2014 my iter-2 rho_c push was never tested and is now unnecessary given the theta fix does the Exp14 work), tally_kappa and the degeneracy gate untouched (Exp6 at chance is correct). Accept small collateral drift of <=.02 on Exp8/Exp9/Exp11 from the sharper gate; if Exp8 (+.035) worsens, take theta at the top of the new range (.585) rather than re-tuning eps_t.

**Outcome of this advice:** iter 3 candidate loss=0.0692 -> iter 4 candidate loss=0.0549 -> the gate marked it ACCEPTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is fully faithful to the arbiter's DTG-2R prescription — absolute near-chance validity gate g=sigmoid((theta-v)/tau) commanding a one-reason limb, a divisively-normalised anti-validity + recency compensatory fallback when no sub-threshold expert speaks, the .002 label-degeneracy tally collapse, and separate load/opposition lapses. No mechanism drift, and the gate ACCEPTED it as the new running best (.0549, a .014 improvement over the previous base). My iter-3 advice (sharpen/lower theta-tau, narrow rho_t) is vindicated: Exp14 .581 -> .551, Exp15 .785 -> .798, Exp16 .789 -> .818 (real .826, now essentially exact), Exp11 .613 -> .595 (real .593), Exp3 .239 -> .273 (real .286), Exp4 .153 -> .173 (real .150), and the corpus anchors stayed put (Exp10 .404/.418, Exp7 .450/.438, Exp6 .498/.507, Exp13 .531/.546, Exp9 .797/.819).

13 of 16 cells are now within .05 of the human value, and the model remains the only family that simultaneously handles the one-reason cells (Exps 15/16 at .80-.82 where pi_1..pi_6 are all at .17-.43) and the compensatory cells (Exps 10/12 at .40/.22 where pi_1..pi_6 are at .60-.84).

The residual is now concentrated in three places, and the diagnosis has CHANGED in one important respect:
(a) Exp2 (+.147) is still the largest error and remains, as I argued, a structural cost of the family: p_tally (5-vs-1 displays where the .55 anchor sides with the tally) is kept high by the trigger limb while rho_tc^(k_conf-1) discounts the 2-vs-2 tie displays that set p_ttb_tie, inflating the difference. Both DTG siblings (pi_7 .230, pi_9 .326) overshoot too, and the only theories that hit it (pi_3, pi_4, pi_6) are catastrophic on Exps 12-16.
(b) Exp14 (+.084) is now the informative one. With theta~.573 / tau~.0095, g(.62) is ~.007, so the trigger leak I blamed in iter 3 is CLOSED — and the cell still sits at .551. My earlier decomposition (compensatory limb already at .454) was therefore wrong: the compensatory limb itself follows the lone col-0 v=.62 dissenter at ~.55 against a 2-3 row coalition. That is over-decisive in the pro-dissenter direction. Note that Exp13 (lone col-0 dissenter at v=.68, real .546, model .531) is fitted well, so humans show essentially NO anti-validity gradient between v=.62 and v=.68 in these lone-dissenter displays, whereas (1-v)^gamma with gamma~1.9 produces one.
(c) Exp15 (-.064) and the mildly over-confident Exp5 (+.042) / Exp8 (+.025) round out the list; Exp1 (-.039) and Exp12 (-.040) are still slightly too far below chance, i.e. the compensatory limb is a touch too extreme in whichever direction it points.
Rationale: The candidate is the new best base and should NOT be reverted — the next iteration must build on it. But there is one remaining, now properly diagnosed, low-risk in-family knob worth ~.01-.015 of loss, so run one more round on THIS candidate. Code and mechanism stay exactly as written (no family change, no structural edits); ranges only, and ONE knob at a time this round, because the gate punished my iter-2 multi-knob compensatory softening.

(1) PRIMARY, single knob: raise sigma (the divisive-normalisation constant in the COMPENSATORY limb only) from [.40,.72] to ~[.55,.88], leaving beta, gamma, lam, eps0, rho_m, rho_c untouched. This scales the whole compensatory branch toward chance without touching the trigger limb (so Exps 15/16 are unaffected). Six cells move the right way at once: Exp14 .551 -> ~.51-.53 (real .468), Exp1 .244 -> ~.26 (real .282), Exp12 .220 -> ~.235 (real .261), Exp3 .273 -> ~.285 (real .286), Exp5 .785 -> ~.77 (real .743), Exp8 .795 -> ~.785 (real .770). The only costs are small and bounded: Exp4 .173 -> ~.185 (real .150, -.01) and Exp13 .531 -> ~.522 (real .546, -.01). Keep the step modest — do not push sigma past ~.9, or Exp4/Exp13/Exp10 start to flatten out.

(2) DO NOT repeat the two things the gate has already punished. Do not lower beta and do not raise eps0 (iter-2 REJECTED); do not cut eps_t across the board (iter-2 REJECTED). Leave eps_t [.20,.36] and rho_t [.985,1.00] exactly as they are — Exp16 is now nearly exact and Exp8 is already .025 over, so a blanket trigger-limb stiffening would trade Exp15 against Exp8/Exp16 for no net gain.

(3) gamma is the ONLY alternative lever on Exp14, and it is now genuinely operative (unlike in iter 1, when trigger leakage masked it) — but it is second choice, not first. If sigma alone leaves Exp14 above ~.52, a small gamma trim to ~[1.3,1.9] is permissible, but note the human data show NO anti-validity gradient between v=.62 (Exp14, .468) and v=.68 (Exp13, .546), so any gamma cut that improves Exp14 by .05 will cost Exp13 ~.04. Only take it if the net is clearly positive.

(4) FREEZE the rest: theta [.558,.588] and tau [.006,.013] (the edit the gate just rewarded — the .52/.545/.55 anchors must keep g>=.9 and the .62/.68 dissenters g<=.02), rho_tc [.62,.86], tally_kappa and the .002 degeneracy gate (Exp10 .404/.418, Exp6 .498/.507, Exp7 .450/.438 are best-in-corpus).

(5) Continue to NOT chase Exp2. Any fix there would require either undoing rho_tc (re-breaking Exp3/Exp1) or weakening the near-chance gate that Exps 15/16 depend on; it is a documented family-level cost and the arbiter accepted it when prescribing DTG-2R.

**Outcome of this advice:** iter 4 candidate loss=0.0549 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
{
  "theory": "DTG-2R/OC-sharp (sigma-softened compensatory limb) — Doubt-Threshold-Gated one-reason choice with a compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion, the one-reason limb is degraded by OPPOSITION rather than by raw display length, and the compensatory limb carries a LARGER divisive-normalisation constant so that it never commits as hard as the adjudicated limb. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), with theta ~ .558-.588 and tau ~ .006-.013, so an expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is not (g <= .02). Silent near-chance experts are irrelevant: no null-verdict capture and no doubt-gradient leak. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial nearly independently of how many higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows. (3) Because a single row is consulted, scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does: each minority row beyond the first charges rho_tc, keep_t = (1-eps_t)*rho_t^(m-2)*rho_tc^(k_conf-1). Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.80-.86 flat in coalition size, whereas balanced splits drift toward chance. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent with a SUBSTANTIAL normalisation constant sigma (~.55-.88), E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E). The large sigma is the theory's claim that unadjudicated, merely-graded evidence is intrinsically less committal than an adjudicated verdict: a lone mid-doubt dissenter against a coalition sits near chance (~.47-.53) and tally-tie conflict displays stay closer to .5 in whichever direction the anti-validity tilt points. Its own separable load (rho_m) and opposition (rho_c) lapses keep unanimous 6-row displays confident but sub-ceiling (~.74-.78). (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.",
  "predict": "def predict(parameters, state, history):\n    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;\n    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count\n    import numpy as np\n\n    # ---------- parse the (2, n_features) stimulus robustly ----------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()\n            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()\n        else:\n            vals = list(state.values())\n            if len(vals) >= 2:\n                a = np.asarray(vals[0], dtype=float).ravel()\n                b = np.asarray(vals[1], dtype=float).ravel()\n    if a is None or b is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 2 and arr.shape[0] == 2:\n            a, b = arr[0].ravel(), arr[1].ravel()\n        elif arr.ndim == 2 and arr.shape[1] == 2:\n            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()\n        elif arr.ndim == 1:\n            h = arr.shape[0] // 2\n            a, b = arr[:h], arr[h:2 * h]\n        else:\n            f = np.asarray(arr, dtype=float).reshape(2, -1)\n            a, b = f[0].ravel(), f[1].ravel()\n\n    n = int(min(np.size(a), np.size(b)))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = np.asarray(a, dtype=float)[:n]\n    b = np.asarray(b, dtype=float)[:n]\n\n    # ---------- printed validities ----------\n    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n    if v_raw is None:\n        v = np.linspace(0.95, 0.55, n)\n    else:\n        v = np.asarray(v_raw, dtype=float).ravel()\n        if v.size < n:\n            v = np.concatenate([v, np.full(n - v.size, 0.55)])\n        v = v[:n]\n    v = np.clip(v, 0.5, 0.9995)\n\n    # ---------- parameters ----------\n    theta = float(parameters['theta'])\n    tau = float(max(float(parameters['tau']), 1e-4))\n    mu = float(parameters['mu'])\n    gamma = float(parameters['gamma'])\n    lam = float(parameters['lam'])\n    sigma = float(max(float(parameters['sigma']), 1e-6))\n    beta = float(parameters['beta'])\n    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))\n    rho_t = float(np.clip(float(parameters['rho_t']), 1e-6, 1.0))\n    rho_tc = float(np.clip(float(parameters['rho_tc']), 1e-6, 1.0))\n    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))\n    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))\n    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))\n    kappa = float(parameters['tally_kappa'])\n\n    def sig(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # ---------- discriminating rows ----------\n    diff = a - b\n    disc = np.nonzero(diff != 0.0)[0]\n    m = int(disc.size)\n    if m == 0:\n        return np.array([0.5, 0.5])\n    signs = np.sign(diff[disc])\n    n_a = float(np.sum(signs > 0.0))\n    n_b = float(m) - n_a\n    k_conf = int(min(n_a, n_b))\n\n    # ---------- lapses ----------\n    expo = max(m - 2, 0)\n    keep_c = (1.0 - eps0) * (rho_m ** expo) * (rho_c ** k_conf)\n    keep_c = float(min(max(keep_c, 0.0), 1.0))\n    # one-reason limb: scanning extra rows is nearly free, holding a RIVAL\n    # coalition is not -> charge rho_tc per minority row beyond the first\n    keep_t = (1.0 - eps_t) * (rho_t ** expo) * (rho_tc ** max(k_conf - 1, 0))\n    keep_t = float(min(max(keep_t, 0.0), 1.0))\n\n    # ---------- label-degeneracy gate ----------\n    degenerate = False\n    if n > 1:\n        vs = np.sort(v)\n        if bool(np.any(np.diff(vs) < 2.0e-3)):\n            degenerate = True\n\n    if degenerate:\n        p_core = sig(kappa * (n_a - n_b))\n        p_a = 0.5 + keep_c * (p_core - 0.5)\n        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n        p = np.array([p_a, 1.0 - p_a], dtype=float)\n        return p / p.sum()\n\n    # ---------- absolute doubt trigger over DISCRIMINATING rows ----------\n    zz = np.clip((v - theta) / tau, -60.0, 60.0)\n    gtrig = 1.0 / (1.0 + np.exp(zz))          # = sigmoid((theta - v)/tau)\n    gd = gtrig[disc]\n    P_trig = float(np.clip(np.max(gd), 0.0, 1.0))\n\n    # verdict of the consulted (near-chance) expert; recency tilt among them\n    logs = np.log(np.clip(gd, 1e-300, None)) + mu * disc.astype(float)\n    logs = logs - np.max(logs)\n    s = np.exp(np.clip(logs, -700.0, 0.0))\n    ssum = float(np.sum(s))\n    if (not np.isfinite(ssum)) or ssum <= 0.0:\n        s = np.full(m, 1.0 / m)\n    else:\n        s = s / ssum\n    p_or_a = float(np.sum(s[signs > 0.0]))\n    p_t = 0.5 + keep_t * (p_or_a - 0.5)\n\n    # ---------- compensatory fallback (divisively normalised anti-validity) ----------\n    logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)\n    ld = logw[disc]\n    ld = ld - np.max(ld)\n    w = np.exp(np.clip(ld, -700.0, 0.0))\n    wsum = float(np.sum(w))\n    if (not np.isfinite(wsum)) or wsum <= 0.0:\n        w = np.ones(m, dtype=float)\n        wsum = float(m)\n    E = float(np.sum(signs * w)) / (sigma + wsum)\n    p_comp = sig(beta * E)\n    p_c = 0.5 + keep_c * (p_comp - 0.5)\n\n    # ---------- doubt-threshold-gated mixture ----------\n    p_a = P_trig * p_t + (1.0 - P_trig) * p_c\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0.0:\n        return int(np.random.randint(len(probs)))\n    probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "theta": "[0.558, 0.588]",
    "tau": "[0.006, 0.013]",
    "mu": "[0.45, 0.95]",
    "gamma": "[1.5, 2.3]",
    "lam": "[0.0, 0.22]",
    "sigma": "[0.55, 0.88]",
    "beta": "[1.8, 2.6]",
    "eps_t": "[0.20, 0.36]",
    "rho_t": "[0.985, 1.00]",
    "rho_tc": "[0.62, 0.86]",
    "eps0": "[0.13, 0.27]",
    "rho_m": "[0.89, 0.97]",
    "rho_c": "[0.74, 0.90]",
    "tally_kappa": "[0.6, 1.3]",
    "validities": "validities"
  },
  "rationale": "STRICT MINIMAL DIFF on the ACCEPTED base (loss .0549): `predict` and `policy` are re-emitted byte-identical, and exactly ONE parameter range changes — sigma [0.40, 0.72] -> [0.55, 0.88] — which is precisely the single-knob edit the critic prescribed in iteration 4, and nothing else.\n\nWhy this knob and only this knob. sigma enters ONLY the compensatory limb (E = sum(s_j*w_j)/(sigma + sum(w_j))), so raising it scales that branch's evidence toward zero and pulls p_comp toward .5 without touching (i) the trigger limb (Exps 15/16, which are now near-exact at .798/.818 and must not move), (ii) the degeneracy tally (Exps 6/8/9/10 anchors), or (iii) the lapse structure that the gate has twice rewarded (rho_tc, rho_t, eps_t, eps0, rho_m, rho_c). The critic's diagnosis is now properly grounded: with theta~.573/tau~.0095 the v=.62 trigger leak is closed (g~.007), so the residual +.084 on Exp14 is the compensatory limb itself being over-decisive in favour of a lone mid-doubt dissenter. A larger normalisation constant is the theoretically right fix — it encodes the claim that unadjudicated graded evidence is intrinsically less committal than an adjudicated near-chance verdict — and it moves six cells the right way simultaneously: Exp14 .551 -> ~.51-.53 (real .468), Exp1 .244 -> ~.26 (real .282), Exp12 .220 -> ~.235 (real .261), Exp3 .273 -> ~.285 (real .286), Exp5 .785 -> ~.77 (real .743), Exp8 .795 -> ~.785 (real .770). Bounded costs: Exp4 .173 -> ~.185 (real .150) and Exp13 .531 -> ~.522 (real .546), each ~.01.\n\nDeliberate omissions, each grounded in the accept-gate record. I did NOT lower beta or raise eps0 (the iter-2 multi-knob compensatory softening was REJECTED at .0760). I did NOT cut eps_t or stiffen the trigger limb (also REJECTED in iter 2, and Exp16 is now essentially exact while Exp8 is already +.025 over, so any trigger stiffening trades Exp15 against Exp8/16 for no net gain). I did NOT trim gamma: the critic flagged it as second choice only, and the human data show essentially NO anti-validity gradient between v=.62 (Exp14, .468) and v=.68 (Exp13, .546), so a gamma cut large enough to fix Exp14 would cost Exp13 roughly as much. I kept theta/tau and rho_tc frozen — those are the two edits the gate has explicitly rewarded (.0692 then .0549) and the .52/.545/.55 anchors must retain g>=.9. And I continue to NOT chase Exp2: it is a documented family-level cost (both DTG siblings overshoot; the only theories that hit it collapse on Exps 12-16), and any fix would either undo rho_tc or weaken the near-chance gate that Exps 15/16 depend on.\n\nThe step size is kept modest (top of range .88, not past ~.9) precisely so Exp4/Exp13/Exp10 do not flatten out, keeping the expected net movement clearly below the .0549 floor.",
  "rationale_short": "One-knob edit on the accepted base: sigma [0.40,0.72] -> [0.55,0.88], softening only the compensatory limb toward chance, exactly as the critic prescribed; all code and every other range re-emitted verbatim."
}
```

## Usage

```json
{
  "input_tokens": 63045,
  "output_tokens": 5737
}
```
