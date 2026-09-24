# generation_iter_06_attempt_00

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
- THEORY 1 = `pi_10`
- THEORY 2 = `pi_7`
- The recommendation below acts on THEORY 2 (= `pi_7`).

Replace pi_7 entirely. Do NOT keep the sharp silence sigmoid, the categorical unanimity bonus, the saturating majority bonus, or the absolute high-gain read-out \u2014 each of these is directly contradicted by an observed null or reversal (Exp 1, 2, 9, 18) or by systematic over-decisiveness (Exp 4, 13, 14, 17).\n\nProposed new theory ('Ordinal-Tally with Rarity Discount and a Hard Noise Ceiling'):\n1. COMPRESSED / ORDINAL DIAGNOSTICITY. A discriminating expert contributes w_j that depends mainly on its RANK among the validities actually displayed, not on the numeric validity: e.g. w_j = w_lo + (w_hi - w_lo) * (rank_j/(n-1))^gamma with w_hi/w_lo ~ 1.5-2.5 only. This makes a .95 expert worth only ~1.5-2 cheap experts, so that two-or-three-cue coalitions beat one or two heavyweights (Exp 8 = .84, Exp 12 = .85, Exp 4 = .775, Exp 14 = .665) while a single strong cue still ties two weak ones (Exp 5 = .52, Exp 11 = .49). Crucially there is NO oracle premium and NO steep validity gradient.\n2. NUMEROSITY IS THE MAIN DRIVER, BUT GRADED AND NON-CATEGORICAL: add lambda * (k_A - k_B) with no saturation switch and no unanimity step; this reproduces the near-linear increase in majority-following without predicting the jumps that Exp 2 refuted.\n3. RARITY / SENSITIVITY GAP INFERENCE IS THE ONLY CONFIDENCE TERM, and it is attached to the SPEAKING cues, not to the silent fraction: when the discriminating evidence is carried by high-validity (high-sensitivity) experts only, the inferred quality gap is small, so the drive is multiplied by a factor decreasing in the mean validity of the speakers. This is what produces the robust -0.20 in Exp 10 and it should be the ONLY mechanism that touches confidence. There must be NO dependence on the fraction or composition (1,1 vs 0,0) of mute rows, since every silence-composition contrast observed is ~0 (Exp 1, 2, 9, 18).\n4. NO EDGE SALIENCE / NO SERIAL STOP RULE, or at most a very small primacy weight on the top row (|effect| < 3 points), since Exp 5, 6, 11 are all at chance and Exp 15 (.353) shows subjects do NOT follow the first discriminating row.\n5. HARD NOISE CEILING: the final probability is p = 0.5 + (p_raw - 0.5) * (1 - eps) with eps ~ 0.25-0.35 applied UNIFORMLY (not difficulty-scaled), so that even landslide displays top out near .85 \u2014 matching the observed .775-.85 asymptote and killing the .95+ predictions pi_7 kept making.\nParameter ranges should be chosen so that, at central values, the model predicts ~0.00 for the Exp 1 dense-minus-sparse metric, ~0.00 for the Exp 2 composite, ~0.80-0.85 for Exp 8/12, ~0.75 for Exp 4, ~0.50 for Exp 5/6/11, and ~-0.20 for Exp 10. This theory is sharply distinguishable from pi_10 (which retains relative normalisation, an oracle premium, steep edge salience, a smooth silence discount and a top-down commitment stop), so future experiments can contrast 'validity is read ordinally and silence is ignored' against 'validity has a top-end premium and silence dilutes confidence'.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.748)

**Description**
**Dilution-calibrated reason integration with an ORACLE PREMIUM and a SHARPLY-PEAKED edge-salience profile (DCRI-3s): a compressive weighted reason tally whose diagnosticity map is compressive in the mid range but regains a categorical top-end premium near the stated-validity ceiling, read out RELATIVE to the evidence actually on the table, scaled by a smooth, strictly-positive confidence that falls with panel silence and with the coarseness-inference it licenses, plus a two-regime edge-salience tie-breaker whose attention profile is concentrated on the LITERAL first and last rows (steep decay), a slowly-growing numerosity bonus, a difficulty-scaled lapse, and a firm authority-checked top-down stop.**

Mechanism family is unchanged from the accepted base; ONE calibration commitment is sharpened.

1. **Diagnosticity is compressive in the middle but has a TOP-END PREMIUM.** x_j = ((v_j-0.5)/0.5)^rho * (1 + alpha_v*sigmoid(kappa_v*(v_j - v_knee))) with rho ~0.46-0.58, v_knee ~0.91: below ~.88 subjective weight is strongly sub-linear so cheap-cue coalitions routinely outvote a merely-good cue, but a near-certain expert is treated as categorically heavier.

2. **Numerosity is a separate, slowly-growing reason.** N = lambda*sign(k_A-k_B)*sqrt(|k_A-k_B|).

3. **Read-out is RELATIVE:** drive = (V + N + P)/(kappa2 + total speaking diagnosticity).

4. **Confidence = evidence fraction x coarseness inference, never a reversal.** c = c_min + (1-c_min)*frac^psi*max(1 - eta*f_sil^q*vbar, 0), strictly positive and floored, so a lone speaking cue amid a mute panel is still followed above chance and silence only flattens toward chance.

5. **Mutual endorsement is a weaker form of silence than mutual absence** (weight omega ~0.63), giving a small positive processing-load effect.

6. **Edge salience is a two-regime tie-breaker over LITERAL screen rows with a STEEP attention decay.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) with tau ~0.62-0.88 (previously ~1.05). This is the sharpened commitment: the extra attention paid to the top and bottom of the display is *confined to the first and last rows themselves* and does not leak onto rows 1 and n-2. With a broad profile the mid-list heavyweights that sit just inside the panel edges receive almost as much positional boost as the true screen extremes, so the tie-breaker partially cancels itself in exactly the deadlocked extremes-vs-heavyweights displays it is meant to decide; with a steep profile the tie-breaker actually breaks the tie. Full force only on exactly deadlocked reason counts (narrow sigma), a small constant residual otherwise, symmetric so there is no primacy/recency swing.

7. **Choice difficulty scales the lapse**, so near-deadlocked displays draw genuine guessing while landslides stay near ceiling.

8. **Authority-checked top-down stop**: people commit at the first discriminating row only if nothing below it is more diagnostic; the committed choice is still multiplied by the same confidence c.

Individual differences are unimodal jitter in all parameters; no strategy subpopulations, no learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> scaled diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _p(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    rho = _p('rho', 0.52, 0.05, 4.0)
    alpha_v = _p('alpha_v', 0.62, 0.0, 3.0)
    v_knee = _p('v_knee', 0.910, 0.55, 0.999)
    kappa_v = _p('kappa_v', 35.0, 1.0, 200.0)
    lam = _p('lam', 0.56, 0.0, 3.0)
    mu = _p('mu', 2.40, 0.0, 8.0)
    tau = _p('tau', 0.75, 0.2, 8.0)
    sigma = _p('sigma', 0.41, 0.15, 3.0)
    g_res = _p('g_res', 0.10, 0.0, 1.0)
    kappa = _p('kappa', 0.60, 0.05, 5.0)
    kap2 = _p('kap2', 0.42, 0.02, 5.0)
    psi = _p('psi', 0.35, 0.01, 3.0)
    eta = _p('eta', 3.35, 0.0, 10.0)
    q_sil = _p('q_sil', 1.20, 0.3, 8.0)
    omega = _p('omega', 0.63, 0.0, 1.0)
    c_min = _p('c_min', 0.085, 0.0, 0.5)
    beta = _p('beta', 3.10, 0.05, 40.0)
    eps0 = _p('eps0', 0.08, 0.0, 0.5)
    eps1 = _p('eps1', 0.22, 0.0, 0.6)
    q_max = _p('q_max', 0.96, 0.0, 1.0)
    d_com = _p('d_com', 2.25, 0.0, 6.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. compressive reason tally WITH A TOP-END (ORACLE) PREMIUM
    #    mid-range validity is read coarsely (rho < 1), but a near-certain
    #    expert regains a categorical extra weight.
    # ------------------------------------------------------------------
    prem = 1.0 + alpha_v * _sig(kappa_v * (val - v_knee))
    x = np.power(vs, rho) * prem
    idxs = np.flatnonzero(disc)
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))
    Xs = float(np.sum(xs))
    if Xs <= 1e-12:
        return np.array([0.5, 0.5])

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: a separate reason that grows only as sqrt(|dk|)
    # ------------------------------------------------------------------
    N = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * float(np.sqrt(abs(dk)))

    # ------------------------------------------------------------------
    # 6. symmetric edge salience, full force only on deadlocked counts.
    #    STEEP decay (tau < 1): the attention bonus is confined to the
    #    literal first and last rows and does not leak onto rows 1/n-2.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s_sal = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    S = float(np.sum(sgn * s_sal[idxs]))
    g = g_res + (1.0 - g_res) * float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. RELATIVE read-out: evidence judged against evidence available
    # ------------------------------------------------------------------
    drive = (V + N + P) / (kap2 + Xs)

    # ------------------------------------------------------------------
    # 8. confidence = evidence-fraction dilution x coarseness inference
    #    (smooth, strictly positive, never sign-reversing)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    sil_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    sil_mass = float(np.sum(sil_w * x))
    f_sil = float(np.sum(sil_w)) / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))

    frac = Xs / (kappa + Xs + sil_mass)
    frac = float(np.clip(frac, 1e-9, 1.0))
    vbar = float(np.sum(xs * vs[idxs]) / Xs)
    gap = 1.0 - eta * (f_sil ** q_sil) * vbar
    gap = float(max(gap, 0.0))
    c = c_min + (1.0 - c_min) * (frac ** psi) * gap
    c = float(np.clip(c, 0.0, 1.0))

    p_int = float(_sig(beta * c * drive))

    # ------------------------------------------------------------------
    # 9. authority-checked top-down stop (display-contingent, capped)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(vs[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 10. choice-difficulty-scaled lapse
    # ------------------------------------------------------------------
    z = float(np.log(p_a / (1.0 - p_a)))
    eps = eps0 + eps1 * float(np.exp(-min(abs(z), 60.0)))
    eps = float(np.clip(eps, 0.0, 0.9))
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [0.46, 0.58]
- alpha_v: [0.45, 0.85]
- v_knee: [0.895, 0.925]
- kappa_v: [25.0, 45.0]
- lam: [0.48, 0.64]
- mu: [2.15, 2.65]
- tau: [0.62, 0.88]
- sigma: [0.36, 0.46]
- g_res: [0.06, 0.14]
- kappa: [0.45, 0.80]
- kap2: [0.30, 0.55]
- psi: [0.28, 0.45]
- eta: [3.05, 3.65]
- q_sil: [1.05, 1.35]
- omega: [0.55, 0.72]
- c_min: [0.06, 0.11]
- beta: [2.80, 3.50]
- eps0: [0.05, 0.11]
- eps1: [0.16, 0.28]
- q_max: [0.92, 0.99]
- d_com: [2.00, 2.55]
- phi: [0.98, 1.06]
- s_stop: [6.0, 12.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=0.0354 (var=0.0104)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.7435 (var=0.0039)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.4625 (var=0.0178)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.4263 (var=0.0073)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.7650 (var=0.0146)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.5962 (var=0.0062)
- Experiment 7: real=0.0533 (var=0.0494) vs this=0.0433 (var=0.0476)
- Experiment 8: real=-0.2025 (var=0.0220) vs this=-0.1950 (var=0.0223)
- Experiment 9: real=0.4900 (var=0.0105) vs this=0.4683 (var=0.0093)
- Experiment 10: real=0.8500 (var=0.0111) vs this=0.7800 (var=0.0149)
- Experiment 11: real=0.5133 (var=0.0520) vs this=0.8983 (var=0.0498)
- Experiment 12: real=0.6650 (var=0.0058) vs this=0.6758 (var=0.0097)
- Experiment 13: real=0.3533 (var=0.0101) vs this=0.3556 (var=0.0076)
- Experiment 14: real=0.3000 (var=0.0112) vs this=0.2258 (var=0.0051)
- Experiment 15: real=0.8550 (var=0.0485) vs this=0.8383 (var=0.0491)
- Experiment 16: real=0.0967 (var=0.1015) vs this=0.1017 (var=0.0792)
- Experiment 17: real=-0.0014 (var=0.0071) vs this=0.1265 (var=0.0081)
- Experiment 18: real=-0.0333 (var=0.0253) vs this=0.0475 (var=0.0190)


---

### `pi_6` (overall score: 0.411)

**Description**
**Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        else:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-9, 1.0)      # scaled diagnosticity in (0,1]

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 1.15), 0.1, 6.0))
    alpha = float(np.clip(parameters.get('alpha', 0.9), 0.0, 4.0))
    gamma = float(np.clip(parameters.get('gamma', 0.05), 0.0, 3.0))
    tau = float(np.clip(parameters.get('tau', 1.3), 0.2, 8.0))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s = float(np.clip(parameters.get('s', 10.0), 0.5, 80.0))
    beta = float(np.clip(parameters.get('beta', 3.15), 0.05, 40.0))
    kappa = float(np.clip(parameters.get('kappa', 1.6), 0.5, 5.0))
    eta = float(np.clip(parameters.get('eta', 1.8), 0.0, 5.0))
    q_sil = float(np.clip(parameters.get('sil_pow', 2.7), 1.0, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.03), 0.0, 0.5))

    # ------------------------------------------------------------------
    # 4. cue weights: compressed diagnosticity x modest reading-position
    #    attention.  rho ~1.0-1.3 places a .95 cue at roughly the combined
    #    strength of two upper-middling cues, so screen-extreme flankers
    #    neither dominate nor are dominated by a mid-list top expert.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-pos / tau) + gamma * np.exp(-(n - 1.0 - pos) / tau)
    w = np.power(vs, rho) * att
    w = np.clip(w, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. silence calibration: validity = sensitivity.  A difference that
    #    only the most sensitive experts can see, while the rest of the
    #    panel sees none, implies a marginal quality gap -> discounted.
    #    A difference an insensitive expert notices implies a large gap.
    # ------------------------------------------------------------------
    m = int(np.count_nonzero(disc))
    f_sil = float(n - m) / float(n)
    wd = w[disc]
    tot_wd = float(np.sum(wd))
    if tot_wd <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(wd * vs[disc]) / tot_wd)
    c = 1.0 - eta * (f_sil ** q_sil) * vbar
    c = float(np.clip(c, 0.02, 1.0))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # convex (S-shaped) read-out of the relative balance: |R| = 1 is
    # untouched, small residual imbalances are discounted toward chance.
    def _drive(R):
        R = float(np.clip(R, -1.0, 1.0))
        return float(np.sign(R) * (abs(R) ** kappa))

    # ------------------------------------------------------------------
    # 6. serial reading in screen order with graded sufficiency stopping;
    #    evidence is read fully relatively (R in [-1, 1]).
    # ------------------------------------------------------------------
    idxs = np.flatnonzero(disc)
    E = 0.0
    M = 0.0
    remain = 1.0
    p_a = 0.0
    last = int(idxs.shape[0]) - 1
    for t in range(idxs.shape[0]):
        j = int(idxs[t])
        E += w[j] * (1.0 if d[j] > 0 else -1.0)
        M += w[j]
        R = (E / M) if M > 1e-12 else 0.0
        p_here = _sig(beta * c * _drive(R))
        if t >= last:
            q = 1.0
        else:
            tail = w[j + 1:]
            bar = phi * float(np.max(tail)) if tail.size > 0 else 0.0
            if bar <= 1e-12:
                q = 1.0
            else:
                q = _sig(s * (abs(E) / bar - 1.0))
        p_a += remain * q * p_here
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:
        R = (E / M) if M > 1e-12 else 0.0
        p_a += remain * _sig(beta * c * _drive(R))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [1.00, 1.32]
- alpha: [0.7, 1.2]
- gamma: [0.0, 0.15]
- tau: [1.0, 1.6]
- phi: [0.95, 1.20]
- s: [7.0, 14.0]
- beta: [2.80, 3.50]
- kappa: [1.35, 1.85]
- eta: [1.5, 2.1]
- sil_pow: [2.4, 3.0]
- epsilon: [0.0, 0.06]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=0.0110 (var=0.0029)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.7530 (var=0.0025)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.4450 (var=0.0192)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.3663 (var=0.0121)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.7913 (var=0.0231)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.7238 (var=0.0070)
- Experiment 7: real=0.0533 (var=0.0494) vs this=-0.0050 (var=0.0446)
- Experiment 8: real=-0.2025 (var=0.0220) vs this=-0.1838 (var=0.0208)
- Experiment 9: real=0.4900 (var=0.0105) vs this=0.4325 (var=0.0102)
- Experiment 10: real=0.8500 (var=0.0111) vs this=0.2300 (var=0.0174)
- Experiment 11: real=0.5133 (var=0.0520) vs this=0.4233 (var=0.0611)
- Experiment 12: real=0.6650 (var=0.0058) vs this=0.1783 (var=0.0069)
- Experiment 13: real=0.3533 (var=0.0101) vs this=0.7717 (var=0.0096)
- Experiment 14: real=0.3000 (var=0.0112) vs this=0.4683 (var=0.0115)
- Experiment 15: real=0.8550 (var=0.0485) vs this=0.5250 (var=0.0263)
- Experiment 16: real=0.0967 (var=0.1015) vs this=-0.0500 (var=0.0847)
- Experiment 17: real=-0.0014 (var=0.0071) vs this=-0.1332 (var=0.0107)
- Experiment 18: real=-0.0333 (var=0.0253) vs this=0.0283 (var=0.0063)


---

### `pi_7` (overall score: 0.394)

**Description**
**Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity (= sensitivity)
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 0.95), 0.05, 4.0))
    mu = float(np.clip(parameters.get('mu', 1.82), 0.0, 8.0))
    lam = float(np.clip(parameters.get('lam', 0.67), 0.0, 3.0))
    sigma = float(np.clip(parameters.get('sigma', 0.60), 0.15, 3.0))
    g_res = float(np.clip(parameters.get('g_res', 0.25), 0.0, 0.6))
    tau = float(np.clip(parameters.get('tau', 1.0), 0.2, 8.0))
    ups = float(np.clip(parameters.get('ups', 1.50), 0.0, 4.0))
    beta = float(np.clip(parameters.get('beta', 1.20), 0.05, 40.0))
    eta = float(np.clip(parameters.get('eta', 1.55), 0.0, 5.0))
    sil_k = float(np.clip(parameters.get('sil_k', 38.0), 1.0, 80.0))
    sil_f0 = float(np.clip(parameters.get('sil_f0', 0.36), 0.05, 0.9))
    omega = float(np.clip(parameters.get('omega', 0.82), 0.0, 1.0))
    c_min = float(np.clip(parameters.get('c_min', 0.10), 0.0, 0.5))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s_stop = float(np.clip(parameters.get('s_stop', 8.0), 0.5, 60.0))
    d_com = float(np.clip(parameters.get('d_com', 2.50), 0.5, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.035), 0.0, 0.5))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. reason count, graded by stated validity (no position here)
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    idxs = np.flatnonzero(disc)
    m = int(idxs.shape[0])
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))

    # ------------------------------------------------------------------
    # 5. display salience over ABSOLUTE SCREEN POSITION (first & last
    #    rows of the panel), deployed as a TWO-REGIME tie-breaker:
    #    FULL force when the reason count is deadlocked, and a CONSTANT
    #    residual fraction g_res once either side has more reasons --
    #    position stops deciding, but it does not evaporate with the
    #    size of the majority.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    e_sal_full = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    e_sal = e_sal_full[idxs]
    S = float(np.sum(sgn * e_sal))
    dk = float(kA - kB)
    g = float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    g = float(max(g, g_res))

    # ------------------------------------------------------------------
    # 5b. saturating majority bonus (fires only at |dk| >= 2)
    # ------------------------------------------------------------------
    maj = min(max(abs(dk) - 1.0, 0.0), 1.0)
    M = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * maj

    # ------------------------------------------------------------------
    # 6. unanimity among the speakers is categorical
    # ------------------------------------------------------------------
    uni = 0.0
    if kA == 0 or kB == 0:
        uni = ups * (1.0 if kA > 0 else -1.0)

    drive = V + mu * g * S + uni + M

    # ------------------------------------------------------------------
    # 7. gap inference from panel silence (validity = sensitivity).
    #    An expert who endorsed BOTH products has positively asserted the
    #    two are alike on his dimension: he is ALMOST as uninformative
    #    about the quality gap as one who endorsed neither (weight omega).
    #    SHARP collapse, but BOUNDED BELOW at c_min.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    both0 = (~disc) & (~both1)
    n_sil_eff = float(np.count_nonzero(both0)) + omega * float(np.count_nonzero(both1))
    f_sil = n_sil_eff / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))
    tot_x = float(np.sum(xs))
    if tot_x <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(xs * vs[idxs]) / tot_x)
    c = 1.0 - eta * float(_sig(sil_k * (f_sil - sil_f0))) * vbar
    c = float(np.clip(c, c_min, 1.0))

    p_int = _sig(beta * c * drive)

    # ------------------------------------------------------------------
    # 8. authority-capped first-cue commitment (display-contingent)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = 1.0
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = vs[j1] / (phi * v_rest)
        q_com = _sig(s_stop * (ratio - 1.0))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = _sig(beta * c * d_com)
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [0.88, 1.02]
- mu: [1.70, 1.95]
- lam: [0.60, 0.74]
- sigma: [0.55, 0.66]
- g_res: [0.22, 0.28]
- tau: [0.85, 1.25]
- ups: [1.25, 1.80]
- beta: [1.05, 1.40]
- eta: [1.45, 1.65]
- sil_k: [30.0, 45.0]
- sil_f0: [0.345, 0.375]
- omega: [0.72, 0.92]
- c_min: [0.07, 0.13]
- phi: [0.95, 1.12]
- s_stop: [6.0, 10.0]
- d_com: [2.20, 2.90]
- epsilon: [0.01, 0.06]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=0.0921 (var=0.0072)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.8510 (var=0.0036)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.4537 (var=0.0136)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.4238 (var=0.0074)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.8175 (var=0.0092)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.6869 (var=0.0059)
- Experiment 7: real=0.0533 (var=0.0494) vs this=-0.0733 (var=0.0466)
- Experiment 8: real=-0.2025 (var=0.0220) vs this=-0.2100 (var=0.0192)
- Experiment 9: real=0.4900 (var=0.0105) vs this=0.4583 (var=0.0105)
- Experiment 10: real=0.8500 (var=0.0111) vs this=0.8817 (var=0.0095)
- Experiment 11: real=0.5133 (var=0.0520) vs this=1.1533 (var=0.0393)
- Experiment 12: real=0.6650 (var=0.0058) vs this=0.7775 (var=0.0103)
- Experiment 13: real=0.3533 (var=0.0101) vs this=0.2983 (var=0.0068)
- Experiment 14: real=0.3000 (var=0.0112) vs this=0.1350 (var=0.0054)
- Experiment 15: real=0.8550 (var=0.0485) vs this=0.8933 (var=0.0347)
- Experiment 16: real=0.0967 (var=0.1015) vs this=0.2350 (var=0.0752)
- Experiment 17: real=-0.0014 (var=0.0071) vs this=0.3296 (var=0.0047)
- Experiment 18: real=-0.0333 (var=0.0253) vs this=0.3100 (var=0.0160)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1800 -> ACCEPTED
- iter 2: loss=0.1435 -> ACCEPTED
- iter 3: loss=0.1310 -> ACCEPTED
- iter 4: loss=0.1345 -> REJECTED
- iter 5: loss=0.1491 -> REJECTED
- iter 6: loss=0.1410 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.1310 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)
```

**Observed (real) value:** 0.0933 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: -0.4933 (var=0.0083) (Δ vs real -0.5867)
  - iter 2: -0.0635 (var=0.0093) (Δ vs real -0.1569)
  - iter 3: -0.0304 (var=0.0074) (Δ vs real -0.1237)
  - iter 4: -0.0469 (var=0.0075) (Δ vs real -0.1402)
  - iter 5: -0.0350 (var=0.0101) (Δ vs real -0.1283)
  - iter 6 (most recent): -0.0365 (var=0.0070) (Δ vs real -0.1298)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0038 (var=0.0080)
- pi_2: -0.6931 (var=0.0366)
- pi_3: 0.1027 (var=0.0215)
- pi_4: -0.0050 (var=0.0008)
- pi_5: -0.1875 (var=0.0052)
- pi_6: 0.0110 (var=0.0029)
- pi_7: 0.0921 (var=0.0072)
- pi_8: 0.0794 (var=0.0108)
- pi_9: 0.0194 (var=0.0053)
- pi_10: 0.0354 (var=0.0104)

### Experiment 2
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)
```

**Observed (real) value:** 0.7750 (var=0.0072)
**Previous candidate values (this loop):**
  - iter 1: 0.7745 (var=0.0047) (Δ vs real -0.0005)
  - iter 2: 0.7530 (var=0.0055) (Δ vs real -0.0220)
  - iter 3: 0.7095 (var=0.0046) (Δ vs real -0.0655)
  - iter 4: 0.7310 (var=0.0051) (Δ vs real -0.0440)
  - iter 5: 0.7345 (var=0.0047) (Δ vs real -0.0405)
  - iter 6 (most recent): 0.6965 (var=0.0044) (Δ vs real -0.0785)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8660 (var=0.0083)
- pi_1: 0.1530 (var=0.0129)
- pi_3: 0.7780 (var=0.0646)
- pi_4: 0.7860 (var=0.0029)
- pi_5: 0.7755 (var=0.0037)
- pi_6: 0.7530 (var=0.0025)
- pi_7: 0.8510 (var=0.0036)
- pi_8: 0.7965 (var=0.0057)
- pi_9: 0.7880 (var=0.0048)
- pi_10: 0.7435 (var=0.0039)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Rate of choosing the option favoured by the HIGHEST-VALIDITY cue (screen
    # position 3, validity .95) on the 'double-dissociation' pairs, i.e. trials
    # where the two screen-extreme cues (positions 0 and 5) jointly favour one
    # option while the middle top-validity cue favours the other.
    num = 0.0
    den = 0.0
    num_all = 0.0
    den_all = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        d3 = a[3] - b[3]
        if d3 == 0:
            continue
        top_side = 0 if d3 > 0 else 1
        try:
            resp = int(r)
        except Exception:
            continue
        chose_top = 1.0 if resp == top_side else 0.0
        num_all += chose_top
        den_all += 1.0
        d0 = a[0] - b[0]
        d5 = a[5] - b[5]
        if d0 != 0 and d5 != 0:
            s0 = 1.0 if d0 > 0 else -1.0
            s5 = 1.0 if d5 > 0 else -1.0
            s3 = 1.0 if d3 > 0 else -1.0
            if s0 == s5 and s0 == -s3:
                num += chose_top
                den += 1.0
    if den > 0:
        return float(num / den)
    if den_all > 0:
        return float(num_all / den_all)
    return float('nan')
```

**Observed (real) value:** 0.5175 (var=0.0233)
**Previous candidate values (this loop):**
  - iter 1: 0.4500 (var=0.0195) (Δ vs real -0.0675)
  - iter 2: 0.4713 (var=0.0132) (Δ vs real -0.0462)
  - iter 3: 0.4675 (var=0.0154) (Δ vs real -0.0500)
  - iter 4: 0.4550 (var=0.0128) (Δ vs real -0.0625)
  - iter 5: 0.4412 (var=0.0182) (Δ vs real -0.0762)
  - iter 6 (most recent): 0.4213 (var=0.0147) (Δ vs real -0.0962)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8562 (var=0.0143)
- pi_3: 0.0375 (var=0.0022)
- pi_2: 0.1400 (var=0.0151)
- pi_4: 0.5200 (var=0.0187)
- pi_5: 0.6262 (var=0.0216)
- pi_6: 0.4450 (var=0.0192)
- pi_7: 0.4537 (var=0.0136)
- pi_8: 0.4700 (var=0.0191)
- pi_9: 0.3500 (var=0.0111)
- pi_10: 0.4625 (var=0.0178)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    VAL = np.array([0.78, 0.60, 0.95, 0.88, 0.72], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind='stable')
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    def scan_sign(d, lam):
        n = d.shape[0]
        pos = np.arange(n, dtype=float)
        z = -lam * (pos - pos.mean())
        z = z - z.max()
        w = np.exp(z)
        num = float(np.sum(w * d))
        den = float(np.sum(w * np.abs(d)))
        if den <= 1e-12:
            return 0.0
        return num / den

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        val = VAL if n == VAL.shape[0] else np.linspace(0.95, 0.55, n)
        w_ttb = ttb_winner(a, b, val)
        if w_ttb is None:
            continue
        d = a - b
        s_prim = scan_sign(d, 2.0)    # primacy reader (lambda > 0)
        s_rec = scan_sign(d, -2.0)    # recency reader (lambda < 0)
        if abs(s_prim) < 0.15 or abs(s_rec) < 0.15:
            continue
        # scanner choice: S > 0 -> A (0), S < 0 -> B (1)
        c_prim = 0 if s_prim > 0 else 1
        c_rec = 0 if s_rec > 0 else 1
        # keep only cells where BOTH scanning orientations oppose TTB
        if c_prim == c_rec and c_prim != w_ttb:
            try:
                r = int(row['response'])
            except Exception:
                continue
            hits.append(1.0 if r == w_ttb else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.5075 (var=0.0075)
**Previous candidate values (this loop):**
  - iter 1: 0.3906 (var=0.0078) (Δ vs real -0.1169)
  - iter 2: 0.4075 (var=0.0120) (Δ vs real -0.1000)
  - iter 3: 0.4050 (var=0.0119) (Δ vs real -0.1025)
  - iter 4: 0.3713 (var=0.0057) (Δ vs real -0.1362)
  - iter 5: 0.3812 (var=0.0082) (Δ vs real -0.1262)
  - iter 6 (most recent): 0.3994 (var=0.0061) (Δ vs real -0.1081)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0269 (var=0.0009)
- pi_1: 0.8475 (var=0.0132)
- pi_2: 0.1450 (var=0.0133)
- pi_4: 0.4906 (var=0.0065)
- pi_5: 0.4756 (var=0.0085)
- pi_6: 0.3663 (var=0.0121)
- pi_7: 0.4238 (var=0.0074)
- pi_8: 0.3950 (var=0.0070)
- pi_9: 0.3294 (var=0.0078)
- pi_10: 0.4263 (var=0.0073)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != 5 or b.shape[0] != 5:
            continue
        d = a - b
        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)
        pattern_ok = (
            abs(d[4]) < 1e-9
            and abs(abs(d[0]) - 1.0) < 1e-9
            and abs(abs(d[1]) - 1.0) < 1e-9
            and abs(abs(d[2]) - 1.0) < 1e-9
            and abs(abs(d[3]) - 1.0) < 1e-9
            and abs(d[0] - d[3]) < 1e-9
            and abs(d[1] - d[2]) < 1e-9
            and abs(d[0] + d[1]) < 1e-9
        )
        if not pattern_ok:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        # side supported by expert positions 0 and 3 (the first- and last-
        # discriminating cues); opposed by positions 1 and 2
        favored_is_a = d[0] > 0
        chose_favored = (resp == 0) if favored_is_a else (resp == 1)
        hits += 1 if chose_favored else 0
        tot += 1

    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8200 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1: 0.8063 (var=0.0110) (Δ vs real -0.0137)
  - iter 2: 0.8025 (var=0.0124) (Δ vs real -0.0175)
  - iter 3: 0.8400 (var=0.0100) (Δ vs real +0.0200)
  - iter 4: 0.8425 (var=0.0097) (Δ vs real +0.0225)
  - iter 5: 0.8225 (var=0.0085) (Δ vs real +0.0025)
  - iter 6 (most recent): 0.8562 (var=0.0088) (Δ vs real +0.0363)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3500 (var=0.0175)
- pi_3: 0.9513 (var=0.0036)
- pi_1: 0.8250 (var=0.0169)
- pi_2: 0.4850 (var=0.0117)
- pi_5: 0.8325 (var=0.0087)
- pi_6: 0.7913 (var=0.0231)
- pi_7: 0.8175 (var=0.0092)
- pi_8: 0.7863 (var=0.0111)
- pi_9: 0.8000 (var=0.0119)
- pi_10: 0.7650 (var=0.0146)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] < 6:
            continue
        d = a - b
        disc = [int(i) for i in np.nonzero(np.abs(d) > 1e-9)[0]]
        target = None
        # Class (2): d nonzero at {0,2,5}, first and last agree, mid-list top cue opposes
        if disc == [0, 2, 5]:
            if d[0] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        # Class (3): d nonzero at {0,1,2,5}, three cues agree, single top cue opposes
        elif disc == [0, 1, 2, 5]:
            if d[0] == d[1] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        if target is None:
            continue
        n += 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp == target:
            hits += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8413 (var=0.0134)
**Previous candidate values (this loop):**
  - iter 1: 0.5419 (var=0.0087) (Δ vs real -0.2994)
  - iter 2: 0.5294 (var=0.0099) (Δ vs real -0.3119)
  - iter 3: 0.5119 (var=0.0114) (Δ vs real -0.3294)
  - iter 4: 0.5319 (var=0.0073) (Δ vs real -0.3094)
  - iter 5: 0.5194 (var=0.0057) (Δ vs real -0.3219)
  - iter 6 (most recent): 0.4769 (var=0.0093) (Δ vs real -0.3644)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9694 (var=0.0014)
- pi_4: 0.3312 (var=0.0055)
- pi_1: 0.1388 (var=0.0136)
- pi_2: 0.8825 (var=0.0110)
- pi_5: 0.8669 (var=0.0043)
- pi_6: 0.7238 (var=0.0070)
- pi_7: 0.6869 (var=0.0059)
- pi_8: 0.6850 (var=0.0040)
- pi_9: 0.6775 (var=0.0070)
- pi_10: 0.5962 (var=0.0062)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    hi_load_hits, hi_load_n = 0, 0
    lo_load_hits, lo_load_n = 0, 0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b

        pos_idx = np.where(d > 0.5)[0]   # features favouring A
        neg_idx = np.where(d < -0.5)[0]  # features favouring B
        k_a = pos_idx.size
        k_b = neg_idx.size

        # keep only the 3-cue conflict cells: two cues on one side, one on the other
        if not ((k_a == 2 and k_b == 1) or (k_a == 1 and k_b == 2)):
            continue

        # did the subject pick the side backed by the LARGER NUMBER of cues?
        if k_a == 2:
            chose_many = 1 if int(resp) == 0 else 0
        else:
            chose_many = 1 if int(resp) == 1 else 0

        # processing load = shared positive endorsements (non-discriminating 1s)
        L = int(np.count_nonzero((a > 0.5) & (b > 0.5)))

        if L > 0:
            hi_load_hits += chose_many
            hi_load_n += 1
        else:
            lo_load_hits += chose_many
            lo_load_n += 1

    if hi_load_n == 0 or lo_load_n == 0:
        return 0.0

    return float(hi_load_hits / hi_load_n - lo_load_hits / lo_load_n)

```

**Observed (real) value:** 0.0533 (var=0.0494)
**Previous candidate values (this loop):**
  - iter 1: 0.0217 (var=0.0405) (Δ vs real -0.0317)
  - iter 2: 0.0200 (var=0.0446) (Δ vs real -0.0333)
  - iter 3: -0.0367 (var=0.0464) (Δ vs real -0.0900)
  - iter 4: 0.0567 (var=0.0293) (Δ vs real +0.0033)
  - iter 5: -0.0133 (var=0.0323) (Δ vs real -0.0667)
  - iter 6 (most recent): -0.0250 (var=0.0373) (Δ vs real -0.0783)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8050 (var=0.0193)
- pi_3: 0.0033 (var=0.0067)
- pi_1: 0.0017 (var=0.0171)
- pi_2: -0.0050 (var=0.0221)
- pi_4: 0.0283 (var=0.0260)
- pi_6: -0.0050 (var=0.0446)
- pi_7: -0.0733 (var=0.0466)
- pi_8: 0.0183 (var=0.0473)
- pi_9: 0.0133 (var=0.0373)
- pi_10: 0.0433 (var=0.0476)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    weak_idx = {1, 4}      # validities .56 and .62
    strong_idx = {0, 3}    # validities .72 and .95

    weak_hits = []
    strong_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        d = a - b
        nz = np.flatnonzero(np.abs(d) > 1e-9)
        if nz.size != 1:
            continue  # only single-discriminating-cue trials
        j = int(nz[0])
        # 1 if subject chose the option endorsed by the single discriminating expert
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        endorsed_a = d[j] > 0
        hit = 1.0 if (chose_a == endorsed_a) else 0.0
        if j in weak_idx:
            weak_hits.append(hit)
        elif j in strong_idx:
            strong_hits.append(hit)

    if len(weak_hits) == 0 or len(strong_hits) == 0:
        return float('nan')

    return float(np.mean(strong_hits) - np.mean(weak_hits))

```

**Observed (real) value:** -0.2025 (var=0.0220)
**Previous candidate values (this loop):**
  - iter 1: -0.1937 (var=0.0255) (Δ vs real +0.0088)
  - iter 2: -0.1488 (var=0.0201) (Δ vs real +0.0537)
  - iter 3: -0.1837 (var=0.0341) (Δ vs real +0.0188)
  - iter 4: -0.2000 (var=0.0238) (Δ vs real +0.0025)
  - iter 5: -0.1488 (var=0.0215) (Δ vs real +0.0537)
  - iter 6 (most recent): -0.1388 (var=0.0260) (Δ vs real +0.0637)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0062 (var=0.0039)
- pi_5: 0.3350 (var=0.0184)
- pi_1: 0.0063 (var=0.0080)
- pi_2: 0.0037 (var=0.0166)
- pi_4: 0.2738 (var=0.0184)
- pi_6: -0.1838 (var=0.0208)
- pi_7: -0.2100 (var=0.0192)
- pi_8: -0.2113 (var=0.0237)
- pi_9: -0.2037 (var=0.0293)
- pi_10: -0.1950 (var=0.0223)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cells where BOTH scanning orientations (primacy and recency) predict the
    # option favoured by the two screen-extreme cues (pos0 = .72, pos5 = .70),
    # while the advocated model predicts a coin flip (cell 5) or a reversal
    # toward the mid-list crowd (cell 8).
    #   cell 5: d = (+1, 0, -1, 0, 0, +1)  and its mirror
    #   cell 8: d = (+1, -1, -1, -1, 0, +1) and its mirror
    targets = {
        (1, 0, -1, 0, 0, 1),
        (-1, 0, 1, 0, 0, -1),
        (1, -1, -1, -1, 0, 1),
        (-1, 1, 1, 1, 0, -1),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        d = tuple(int(x) - int(y) for x, y in zip(a, b))
        if d not in targets:
            continue
        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        # option favoured by the screen-extreme cues (positions 0 and 5 agree)
        fav_a = (d[0] > 0)
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4900 (var=0.0105)
**Previous candidate values (this loop):**
  - iter 1: 0.3775 (var=0.0088) (Δ vs real -0.1125)
  - iter 2: 0.3508 (var=0.0072) (Δ vs real -0.1392)
  - iter 3: 0.3333 (var=0.0081) (Δ vs real -0.1567)
  - iter 4: 0.3650 (var=0.0104) (Δ vs real -0.1250)
  - iter 5: 0.3592 (var=0.0060) (Δ vs real -0.1308)
  - iter 6 (most recent): 0.3433 (var=0.0082) (Δ vs real -0.1467)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4325 (var=0.0102)
- pi_3: 0.9692 (var=0.0013)
- pi_1: 0.1575 (var=0.0113)
- pi_2: 0.4817 (var=0.0049)
- pi_4: 0.1583 (var=0.0049)
- pi_5: 0.4042 (var=0.0068)
- pi_7: 0.4583 (var=0.0105)
- pi_8: 0.7292 (var=0.0071)
- pi_9: 0.4625 (var=0.0097)
- pi_10: 0.4683 (var=0.0093)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell-5 pairs (trials 9 and 10 of the design), mirrored across sides:
    #   d = a - b = [+1, 0, -1, -1, 0, +1]   -> extremes (pos0=.62, pos5=.58) favour A
    #   d = a - b = [-1, 0, +1, +1, 0, -1]   -> extremes favour B
    # Score = fraction of these trials on which the subject chose the option
    # endorsed by the two SCREEN-EXTREME (weakest) experts, i.e. against the
    # two mid-list heavyweight experts (.95 and .86).
    target_A = (1, 0, -1, -1, 0, 1)
    target_B = (-1, 0, 1, 1, 0, -1)

    hits = 0
    n = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != len(b) or len(a) != 6:
            continue
        d = tuple(int(x - y) for x, y in zip(a, b))
        if d == target_A:
            extreme_side = 0  # option A
        elif d == target_B:
            extreme_side = 1  # option B
        else:
            continue
        n += 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r == extreme_side:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.8500 (var=0.0111)
**Previous candidate values (this loop):**
  - iter 1: 0.6517 (var=0.0260) (Δ vs real -0.1983)
  - iter 2: 0.7517 (var=0.0151) (Δ vs real -0.0983)
  - iter 3: 0.7500 (var=0.0186) (Δ vs real -0.1000)
  - iter 4: 0.7250 (var=0.0215) (Δ vs real -0.1250)
  - iter 5: 0.7367 (var=0.0220) (Δ vs real -0.1133)
  - iter 6 (most recent): 0.7500 (var=0.0150) (Δ vs real -0.1000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9717 (var=0.0024)
- pi_6: 0.2300 (var=0.0174)
- pi_1: 0.1217 (var=0.0142)
- pi_2: 0.5083 (var=0.0237)
- pi_4: 0.0083 (var=0.0006)
- pi_5: 0.1233 (var=0.0134)
- pi_7: 0.8817 (var=0.0095)
- pi_8: 0.8183 (var=0.0177)
- pi_9: 0.8183 (var=0.0155)
- pi_10: 0.7800 (var=0.0149)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def key(row):
        a = ''.join([str(int(round(float(v)))) for v in list(row['option_a_ratings'])])
        b = ''.join([str(int(round(float(v)))) for v in list(row['option_b_ratings'])])
        return a + '|' + b

    # cell -> {pair_key: target_response}  (target = the option predicted pro-A-side)
    cells = {
        'c4': {  # d = [+1,0,-1,-1,+1] : screen extremes vs two mid heavyweights
            '11001|01110': 0,
            '01110|11001': 1,
        },
        'c5': {  # d = [+1,+1,0,-1,0] : early cues favour, later heavyweight opposes
            '11101|00111': 0,
            '00111|11101': 1,
        },
        'c8': {  # d = [+1,+1,0,-1,+1] : three mid experts vs one oracle
            '11101|00110': 0,
            '00110|11101': 1,
        },
    }

    if len(data) == 0:
        return float('nan')

    k = data.apply(key, axis=1)
    resp = pd.to_numeric(data['response'], errors='coerce')

    rates = {}
    for cname, mapping in cells.items():
        hits = []
        for pk, tgt in mapping.items():
            sel = (k == pk)
            if sel.any():
                r = resp[sel]
                hits.append(float(np.mean((r.values == tgt).astype(float))))
        rates[cname] = float(np.mean(hits)) if len(hits) > 0 else 0.5

    # competitor (reason-counting + salience gate) : high on c4, high on c8, lowish on c5
    # advocated (read-until-sufficient, convex read-out) : low on c4, mid on c8, high on c5
    idx = rates['c4'] + rates['c8'] - rates['c5']
    return float(idx)

```

**Observed (real) value:** 0.5133 (var=0.0520)
**Previous candidate values (this loop):**
  - iter 1: 0.9483 (var=0.0489) (Δ vs real +0.4350)
  - iter 2: 1.0017 (var=0.0510) (Δ vs real +0.4883)
  - iter 3: 0.9217 (var=0.0471) (Δ vs real +0.4083)
  - iter 4: 0.9400 (var=0.0422) (Δ vs real +0.4267)
  - iter 5: 0.9783 (var=0.0566) (Δ vs real +0.4650)
  - iter 6 (most recent): 0.9300 (var=0.0482) (Δ vs real +0.4167)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4233 (var=0.0611)
- pi_7: 1.1533 (var=0.0393)
- pi_1: 0.1667 (var=0.0469)
- pi_2: 0.5383 (var=0.0337)
- pi_3: 1.4033 (var=0.2137)
- pi_4: 0.2050 (var=0.0273)
- pi_5: 0.1667 (var=0.0303)
- pi_8: 0.7650 (var=0.0380)
- pi_9: 0.9550 (var=0.0428)
- pi_10: 0.8983 (var=0.0498)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Difference-vector signatures for the two full-direction-disagreement cells
    # (cell 4: deadlock/extremes cell, pairs 4 & 12;  cell 8: four-against-the-oracle, pairs 8 & 16)
    # value = index of the option favoured by the reason-counting (advocated) account
    targets = {
        (1, -1, -1, 0, 0, 0, 1): 0,     # pair 4  -> A
        (-1, 1, 1, 0, 0, 0, -1): 1,     # pair 12 -> B
        (-1, -1, 1, 1, 1, 0, 1): 0,     # pair 8  -> A
        (1, 1, -1, -1, -1, 0, -1): 1,   # pair 16 -> B
    }

    hits = []
    for a_r, b_r, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = [int(v) for v in a_r]
            b = [int(v) for v in b_r]
        except Exception:
            continue
        if len(a) != len(b):
            continue
        d = tuple(int(ai - bi) for ai, bi in zip(a, b))
        if d in targets:
            fav = targets[d]
            try:
                r = int(resp)
            except Exception:
                continue
            hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6650 (var=0.0058)
**Previous candidate values (this loop):**
  - iter 1: 0.7333 (var=0.0078) (Δ vs real +0.0683)
  - iter 2: 0.7150 (var=0.0086) (Δ vs real +0.0500)
  - iter 3: 0.6750 (var=0.0101) (Δ vs real +0.0100)
  - iter 4: 0.6867 (var=0.0104) (Δ vs real +0.0217)
  - iter 5: 0.6908 (var=0.0105) (Δ vs real +0.0258)
  - iter 6 (most recent): 0.6717 (var=0.0047) (Δ vs real +0.0067)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7775 (var=0.0103)
- pi_6: 0.1783 (var=0.0069)
- pi_1: 0.1342 (var=0.0079)
- pi_2: 0.6850 (var=0.0078)
- pi_3: 0.6025 (var=0.0521)
- pi_4: 0.0167 (var=0.0006)
- pi_5: 0.3217 (var=0.0130)
- pi_8: 0.4883 (var=0.0049)
- pi_9: 0.7208 (var=0.0140)
- pi_10: 0.6758 (var=0.0097)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # The six target displays (EXT, AUTH, TAIL cells and their mirrors).
    targets = {
        # EXT
        ("11000101", "01111110"),
        ("01111110", "11000101"),
        # AUTH
        ("10110100", "01111111"),
        ("01111111", "10110100"),
        # TAIL
        ("01110111", "11111000"),
        ("11111000", "01110111"),
    }

    def as_str(x):
        try:
            return "".join([str(int(round(float(v)))) for v in x])
        except Exception:
            return ""

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = [int(round(float(v))) for v in row["option_a_ratings"]]
            b = [int(round(float(v))) for v in row["option_b_ratings"]]
        except Exception:
            continue
        ka = "".join([str(v) for v in a])
        kb = "".join([str(v) for v in b])
        if (ka, kb) not in targets:
            continue
        # first (top-most) screen row on which the two columns disagree
        j = -1
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                j = i
                break
        if j < 0:
            continue
        try:
            resp = int(row["response"])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        favors_a = a[j] > b[j]
        followed = (favors_a and resp == 0) or ((not favors_a) and resp == 1)
        hits += 1 if followed else 0
        tot += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.3533 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1: 0.2567 (var=0.0063) (Δ vs real -0.0967)
  - iter 2: 0.2956 (var=0.0038) (Δ vs real -0.0578)
  - iter 3: 0.3617 (var=0.0051) (Δ vs real +0.0083)
  - iter 4: 0.3511 (var=0.0070) (Δ vs real -0.0022)
  - iter 5: 0.3228 (var=0.0061) (Δ vs real -0.0306)
  - iter 6 (most recent): 0.3778 (var=0.0083) (Δ vs real +0.0244)
**Other theories' values on this metric (for reference):**
- pi_8: 0.9261 (var=0.0024)
- pi_7: 0.2983 (var=0.0068)
- pi_1: 0.6167 (var=0.0031)
- pi_2: 0.1467 (var=0.0094)
- pi_3: 0.7944 (var=0.0830)
- pi_4: 0.4167 (var=0.0016)
- pi_5: 0.3928 (var=0.0039)
- pi_6: 0.7717 (var=0.0096)
- pi_9: 0.3489 (var=0.0052)
- pi_10: 0.3556 (var=0.0076)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Two diagnostic cells (each appears mirrored across sides, 6 reps each).
    # Cell V1  : focal = [1,0,1,0,1,0]  vs  [0,1,1,1,1,1]
    #            (two invisible mid-list heavyweights .95/.86 oppose the focal;
    #             screen extremes .74/.74 split)
    # Cell AUTH: focal = [1,1,0,1,0,0]  vs  [0,1,1,1,1,1]
    #            (pos1 .95 and pos3 .86 are mute mutual endorsements; the top
    #             speaking row .74 backs the focal, three cheaper rows oppose it)
    cells = [
        ((1, 0, 1, 0, 1, 0), (0, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 1, 0, 0), (0, 1, 1, 1, 1, 1)),
    ]

    def _tup(v):
        try:
            return tuple(int(round(float(z))) for z in v)
        except Exception:
            return None

    if len(data) == 0:
        return float('nan')

    av = data['option_a_ratings'].apply(_tup).values
    bv = data['option_b_ratings'].apply(_tup).values
    try:
        rv = data['response'].astype(float).values
    except Exception:
        return float('nan')

    hits = []
    for i in range(len(av)):
        A = av[i]
        B = bv[i]
        if A is None or B is None:
            continue
        r = rv[i]
        if not np.isfinite(r):
            continue
        r = int(round(r))
        for focal, other in cells:
            if (A == focal and B == other) or (A == other and B == focal):
                chosen = A if r == 0 else B
                hits.append(1.0 if chosen == focal else 0.0)
                break

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.3000 (var=0.0112)
**Previous candidate values (this loop):**
  - iter 1: 0.1833 (var=0.0058) (Δ vs real -0.1167)
  - iter 2: 0.2083 (var=0.0074) (Δ vs real -0.0917)
  - iter 3: 0.2017 (var=0.0070) (Δ vs real -0.0983)
  - iter 4: 0.2283 (var=0.0066) (Δ vs real -0.0717)
  - iter 5: 0.2025 (var=0.0075) (Δ vs real -0.0975)
  - iter 6 (most recent): 0.2158 (var=0.0083) (Δ vs real -0.0842)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1350 (var=0.0054)
- pi_8: 0.7658 (var=0.0055)
- pi_1: 0.5075 (var=0.0047)
- pi_2: 0.1275 (var=0.0067)
- pi_3: 0.7725 (var=0.1429)
- pi_4: 0.1692 (var=0.0046)
- pi_5: 0.2717 (var=0.0072)
- pi_6: 0.4683 (var=0.0115)
- pi_9: 0.1950 (var=0.0046)
- pi_10: 0.2258 (var=0.0051)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    oracle = []   # lone .95 expert (row 2) speaks, all other rows mutually ABSENT (0/0)
    weak = []     # lone .58 expert (row 4) speaks, all other rows mutually ABSENT (0/0)
    top = []      # top/bottom reading-swap cells: 6 discriminating rows, row 2 mutual endorsement

    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=int)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=int)
            resp = int(row['response'])
        except Exception:
            continue
        n = min(a.shape[0], b.shape[0])
        if n < 7:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = np.flatnonzero(d != 0)
        chosen = a if resp == 0 else b

        if disc.shape[0] == 1:
            j = int(disc[0])
            others = [k for k in range(n) if k != j]
            silent_zero = all((a[k] == 0 and b[k] == 0) for k in others)
            if silent_zero:
                follow = 1.0 if chosen[j] == 1 else 0.0
                if j == 2:
                    oracle.append(follow)
                elif j == 4:
                    weak.append(follow)
        elif disc.shape[0] == 6 and (2 not in set(disc.tolist())) and a[2] == 1 and b[2] == 1:
            top_is_A = (a[0] == 1 and a[1] == 1)
            top_is_B = (b[0] == 1 and b[1] == 1)
            if top_is_A != top_is_B:
                chose_A = (resp == 0)
                top.append(1.0 if (chose_A == top_is_A) else 0.0)

    m_oracle = float(np.mean(oracle)) if len(oracle) > 0 else 0.5
    m_weak = float(np.mean(weak)) if len(weak) > 0 else 0.5
    m_top = float(np.mean(top)) if len(top) > 0 else 0.5

    return float(m_oracle + m_weak - m_top)
```

**Observed (real) value:** 0.8550 (var=0.0485)
**Previous candidate values (this loop):**
  - iter 1: 0.7892 (var=0.0502) (Δ vs real -0.0658)
  - iter 2: 0.7575 (var=0.0378) (Δ vs real -0.0975)
  - iter 3: 0.7775 (var=0.0517) (Δ vs real -0.0775)
  - iter 4: 0.6358 (var=0.0512) (Δ vs real -0.2192)
  - iter 5: 0.6567 (var=0.0316) (Δ vs real -0.1983)
  - iter 6 (most recent): 0.6817 (var=0.0381) (Δ vs real -0.1733)
**Other theories' values on this metric (for reference):**
- pi_9: 0.2267 (var=0.0482)
- pi_7: 0.8933 (var=0.0347)
- pi_1: 0.8375 (var=0.0390)
- pi_2: 1.1908 (var=0.0665)
- pi_3: 1.3592 (var=0.2426)
- pi_4: 0.4658 (var=0.0219)
- pi_5: 0.6125 (var=0.0290)
- pi_6: 0.5250 (var=0.0263)
- pi_8: 0.5042 (var=0.0413)
- pi_10: 0.8383 (var=0.0491)

### Experiment 16
**Design**
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    sums = {'O0': 0.0, 'O1': 0.0, 'SE': 0.0, 'SA': 0.0}
    cnts = {'O0': 0, 'O1': 0, 'SE': 0, 'SA': 0}

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
            resp = int(row['response'])
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n < 7:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [j for j in range(n) if abs(d[j]) > 1e-9]

        cell = None
        fav_a = None

        if len(disc) == 1 and disc[0] == 1:
            # lone near-oracle (position 1, validity .95) discriminates
            silent = [j for j in range(n) if j != 1]
            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in silent)
            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in silent)
            if all_one:
                cell = 'O1'
            elif all_zero:
                cell = 'O0'
            if cell is not None:
                fav_a = d[1] > 0
        elif set(disc) == set([0, 1, 5, 6]):
            sil = [2, 3, 4]
            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in sil)
            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in sil)
            if all_one:
                cell = 'SE'
            elif all_zero:
                cell = 'SA'
            if cell is not None:
                # majority (3 vs 1) side: position 0 sign marks it
                fav_a = d[0] > 0

        if cell is None or fav_a is None:
            continue

        chose_a = (resp == 0)
        chose_fav = 1.0 if (chose_a == bool(fav_a)) else 0.0
        sums[cell] += chose_fav
        cnts[cell] += 1

    def prop(k):
        if cnts[k] == 0:
            return 0.5
        return sums[k] / float(cnts[k])

    p_SE = prop('SE')
    p_SA = prop('SA')
    p_O1 = prop('O1')
    p_O0 = prop('O0')

    silence_cliff = p_SE - p_SA          # advocated: large (+.25), competitor: ~0
    oracle_reversal = p_O1 - p_O0        # advocated: ~0 (floored c), competitor: large (+.30)

    return float(silence_cliff - oracle_reversal)

```

**Observed (real) value:** 0.0967 (var=0.1015)
**Previous candidate values (this loop):**
  - iter 1: 0.0417 (var=0.0606) (Δ vs real -0.0550)
  - iter 2: 0.1100 (var=0.0487) (Δ vs real +0.0133)
  - iter 3: 0.0683 (var=0.0602) (Δ vs real -0.0283)
  - iter 4: 0.0667 (var=0.0736) (Δ vs real -0.0300)
  - iter 5: -0.0050 (var=0.0596) (Δ vs real -0.1017)
  - iter 6 (most recent): 0.1000 (var=0.0764) (Δ vs real +0.0033)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2350 (var=0.0752)
- pi_9: -0.2500 (var=0.1053)
- pi_1: 0.0467 (var=0.0256)
- pi_2: -0.0183 (var=0.0395)
- pi_3: 0.0133 (var=0.0068)
- pi_4: -0.0167 (var=0.0267)
- pi_5: 0.2833 (var=0.0253)
- pi_6: -0.0500 (var=0.0847)
- pi_8: 0.3117 (var=0.0847)
- pi_10: 0.1017 (var=0.0792)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def key(a, b):
        sa = ''.join([str(int(round(float(v)))) for v in list(a)])
        sb = ''.join([str(int(round(float(v)))) for v in list(b)])
        return sa + '|' + sb

    # group 'D' = dense / high-evidence-mass cells (C, E, H)
    # group 'S' = sparse / silent-panel or cheap-coalition cells (A, B, G1, F)
    # value = (group, favored_response_code)  where 0 = option A, 1 = option B
    table = {
        # cell C : 4-1 dense majority
        '101111|010010': ('D', 0),
        '010010|101111': ('D', 1),
        # cell E : edge-deadlock (screen extremes vs deep heavyweights)
        '110011|011110': ('D', 0),
        '011110|110011': ('D', 1),
        # cell H : cheap coalition vs lone oracle, dk=2
        '110111|001011': ('D', 0),
        '001011|110111': ('D', 1),
        # cell A : lone oracle vs .66 expert, mute rows = mutual endorsements
        '111011|110111': ('S', 0),
        '110111|111011': ('S', 1),
        # cell B : same difference vector, mute rows = mutual abstentions
        '001000|000100': ('S', 0),
        '000100|001000': ('S', 1),
        # cell G1 : two .74 edge experts vs the .62 expert (dk = 1)
        '101111|011110': ('S', 0),
        '011110|101111': ('S', 1),
        # cell F : three cheap/mid reasons vs two better ones (dk = 1)
        '111100|001011': ('S', 0),
        '001011|111100': ('S', 1),
    }

    hitD = []
    hitS = []
    for a, b, r in zip(data['option_a_ratings'].values,
                       data['option_b_ratings'].values,
                       data['response'].values):
        try:
            k = key(a, b)
        except Exception:
            continue
        info = table.get(k)
        if info is None:
            continue
        grp, fav = info
        try:
            resp = int(round(float(r)))
        except Exception:
            continue
        hit = 1.0 if resp == fav else 0.0
        if grp == 'D':
            hitD.append(hit)
        else:
            hitS.append(hit)

    if len(hitD) == 0 or len(hitS) == 0:
        return 0.0
    return float(np.mean(hitD) - np.mean(hitS))
```

**Observed (real) value:** -0.0014 (var=0.0071)
**Previous candidate values (this loop):**
  - iter 1: 0.1529 (var=0.0085) (Δ vs real +0.1543)
  - iter 2: 0.1425 (var=0.0081) (Δ vs real +0.1439)
  - iter 3: 0.0908 (var=0.0074) (Δ vs real +0.0922)
  - iter 4: 0.1069 (var=0.0114) (Δ vs real +0.1083)
  - iter 5: 0.0978 (var=0.0106) (Δ vs real +0.0992)
  - iter 6 (most recent): 0.0953 (var=0.0103) (Δ vs real +0.0967)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1265 (var=0.0081)
- pi_7: 0.3296 (var=0.0047)
- pi_1: -0.2972 (var=0.0124)
- pi_2: 0.0676 (var=0.0083)
- pi_3: 0.2393 (var=0.1177)
- pi_4: -0.4179 (var=0.0059)
- pi_5: 0.0247 (var=0.0061)
- pi_6: -0.1332 (var=0.0107)
- pi_8: 0.2485 (var=0.0077)
- pi_9: 0.1007 (var=0.0068)

### Experiment 18
**Design**
  A=[1, 1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    K2 = "01110101111111"   # cell 2: 3-0 unanimous, 4 mutual-endorsement mute rows
    K3 = "01110101111101"   # cell 3: identical but the weak pos5 expert dissents (3-1)
    K6 = "10101011101011"   # cell 6: oracle+.86 (2 speakers) vs 3 cheap experts
    K7 = "10101001101011"   # cell 7: same, plus one more cheap dissenter at the bottom edge (2-4)

    def canon(a, b):
        sa = ''.join([str(int(v)) for v in a])
        sb = ''.join([str(int(v)) for v in b])
        return min(sa + sb, sb + sa)

    acc = {K2: [0.0, 0.0], K3: [0.0, 0.0], K6: [0.0, 0.0], K7: [0.0, 0.0]}

    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
            r = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        k = canon(a, b)
        if k not in acc:
            continue
        kA = sum(1 for i in range(len(a)) if a[i] > b[i])
        kB = sum(1 for i in range(len(a)) if b[i] > a[i])
        if kA == kB:
            continue
        if k in (K2, K3):
            # did the subject follow the numerically larger camp?
            hit = 1.0 if ((kA > kB and r == 0) or (kB > kA and r == 1)) else 0.0
        else:
            # did the subject follow the smaller, high-validity (oracle-backed) camp?
            hit = 1.0 if ((kA < kB and r == 0) or (kB < kA and r == 1)) else 0.0
        acc[k][0] += hit
        acc[k][1] += 1.0

    def rate(k):
        s, n = acc[k]
        if n <= 0:
            return None
        return s / n

    p2, p3, p6, p7 = rate(K2), rate(K3), rate(K6), rate(K7)

    comps = []
    if (p2 is not None) and (p3 is not None):
        comps.append(p3 - p2)          # dissenter-raises-confidence effect
    if (p6 is not None) and (p7 is not None):
        comps.append(p6 - p7)          # one-extra-cheap-dissenter reversal
    if len(comps) == 0:
        return 0.0
    return float(np.mean(comps))

```

**Observed (real) value:** -0.0333 (var=0.0253)
**Previous candidate values (this loop):**
  - iter 1: 0.1000 (var=0.0094) (Δ vs real +0.1333)
  - iter 2: 0.0983 (var=0.0142) (Δ vs real +0.1317)
  - iter 3: 0.0742 (var=0.0152) (Δ vs real +0.1075)
  - iter 4: 0.0900 (var=0.0133) (Δ vs real +0.1233)
  - iter 5: 0.0642 (var=0.0120) (Δ vs real +0.0975)
  - iter 6 (most recent): 0.0458 (var=0.0084) (Δ vs real +0.0792)
**Other theories' values on this metric (for reference):**
- pi_7: 0.3100 (var=0.0160)
- pi_10: 0.0475 (var=0.0190)
- pi_1: 0.0042 (var=0.0109)
- pi_2: 0.0017 (var=0.0088)
- pi_3: 0.0075 (var=0.0038)
- pi_4: 0.0033 (var=0.0018)
- pi_5: 0.0108 (var=0.0128)
- pi_6: 0.0283 (var=0.0063)
- pi_8: 0.2692 (var=0.0118)
- pi_9: 0.1450 (var=0.0161)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Gap-Inference Tally with an AUTHORITY-COINCIDENCE LEAD and a small GRADED PRIMACY CREDIT (GIT-AL+P).** Same core mechanism family as the accepted base — a graded (non-ordinal) weighted reason tally, a weak linear numerosity reason, speaker-rarity as the ONLY confidence term, a deadlock-licensed first/last-row tie-breaker, relative read-out and a hard uniform noise ceiling — with two calibration moves and one small added term.

1. **Weighted reasons.** Each discriminating expert contributes w_j = ((v_j-0.5)/0.5)^rho, rho ~1.4-1.8 (smooth, no top-end premium, no knee).
2. **NEW — small graded primacy credit on the tally weights.** w_j is multiplied by (1 + pi_prim*(1 - j/(n-1))) with pi_prim ~0.08-0.18. Rows printed early are read slightly more carefully, so a coalition whose members sit near the top of the panel carries a little more weight than the same coalition buried at the bottom. This is an attention multiplier on weights, NOT a serial stop rule and NOT an edge/salience term: it is monotone in screen row, tiny (a .95 expert at row 3 still outweighs a .62 expert at row 0), and it is the missing ingredient behind early-coalition cells (Exp 11 c5, Exp 6 class-2, Exp 13, Exp 9).
3. **Numerosity** is a separate, strictly linear, non-saturating reason lam*(k_A-k_B), now further damped (lam ~0.135), so evidence MASS drives decisiveness less and dense displays are not systematically more decisive than sparse ones (Exp 17, 18) and multi-cue-vs-oracle cells stop saturating (Exp 11 c8, Exp 12).
4. **Speaker-rarity confidence is the only confidence term:** c = max(1 - eta*f_mute^q*s̄_speakers, c_min). A difference only very sensitive experts noticed while the panel is mute implies a marginal true gap. Mute rows are nearly composition-blind (omega ~0.7), so silence-composition contrasts stay near zero.
5. **Position enters the DIRECTION term only as a tie-breaker licensed by an EXACT deadlock in reason counts** (literal first and last screen rows, narrow Gaussian gate, no residual): 2-vs-2 extreme-vs-heavyweight displays run at .85 (Exp 10) and .82 (Exp 5) while structurally identical 2-vs-1 displays sit at chance (Exp 3 = .518, Exp 9 = .490).
6. **Authority-coincidence lead.** When the FIRST discriminating screen row is ALSO more diagnostic than anything printed below it, the panel reads as though the most authoritative expert has already spoken and his verdict dominates the tally (probability q_com, graded by how far his validity exceeds the best remaining row). It fires only where screen order and validity order coincide, so displays with buried heavyweights (Exp 3, 4, 9, 13, 14) are untouched. The committed verdict is still multiplied by the same rarity confidence c, so a lone oracle amid a mute panel stays near chance.
7. **Hard, difficulty-independent noise ceiling** p = 0.5 + (p-0.5)*(1-eps), eps ~0.33, so nothing exceeds ~.83.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> sensitivity / diagnosticity scale
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    s = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _p(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    rho = _p('rho', 1.60, 0.20, 5.0)
    lam = _p('lam', 0.135, 0.0, 2.0)
    pi_prim = _p('pi_prim', 0.13, 0.0, 1.0)
    mu = _p('mu', 1.00, 0.0, 3.0)
    sigma = _p('sigma', 0.45, 0.15, 3.0)
    k2 = _p('k2', 0.48, 0.05, 5.0)
    eta = _p('eta', 4.50, 0.0, 15.0)
    q = _p('q', 2.20, 0.5, 8.0)
    omega = _p('omega', 0.70, 0.0, 1.0)
    c_min = _p('c_min', 0.05, 0.0, 0.5)
    beta = _p('beta', 8.0, 0.1, 40.0)
    eps = _p('eps', 0.32, 0.0, 0.8)
    # authority-coincidence lead
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)
    q_max = _p('q_max', 0.91, 0.0, 1.0)
    d_com = _p('d_com', 0.50, 0.0, 3.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. smooth (no-premium) diagnosticity tally over speaking experts,
    #    with a SMALL GRADED PRIMACY CREDIT on the weights (attention
    #    multiplier, monotone in screen row, never sign-reversing).
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    if n > 1:
        prim = 1.0 + pi_prim * (1.0 - pos / float(n - 1))
    else:
        prim = np.ones(n, dtype=float)
    w = np.power(s, rho) * prim
    idx = np.flatnonzero(disc)
    sgn = np.sign(d[idx])
    ws = w[idx]
    W = float(np.sum(ws))
    if W <= 1e-12:
        return np.array([0.5, 0.5])
    V = float(np.sum(sgn * ws))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: linear, graded, no saturation, no unanimity step
    # ------------------------------------------------------------------
    N = lam * dk

    # ------------------------------------------------------------------
    # 6. anchor-row tie-breaker: literal first and last screen rows,
    #    licensed ONLY by an exact deadlock in the number of reasons
    #    (narrow gate, no residual term).
    # ------------------------------------------------------------------
    edge = np.zeros(n, dtype=float)
    edge[0] = 1.0
    edge[n - 1] = 1.0
    S = float(np.sum(sgn * edge[idx]))
    g = float(np.exp(-min((dk * dk) / (2.0 * sigma * sigma), 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. read-out relative to the evidence actually on the table
    # ------------------------------------------------------------------
    drive = (V + N + P) / (k2 + W)

    # ------------------------------------------------------------------
    # 8. rarity / sensitivity gap-inference: the ONLY confidence term.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    mute_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    f_mute = float(np.sum(mute_w)) / float(n)
    f_mute = float(np.clip(f_mute, 0.0, 1.0))
    sbar = float(np.sum(ws * s[idx]) / W)
    c = 1.0 - eta * (f_mute ** q) * sbar
    c = float(np.clip(c, c_min, 1.0))

    # ------------------------------------------------------------------
    # 9. integrative read-out
    # ------------------------------------------------------------------
    z = float(np.clip(beta * c * drive, -60.0, 60.0))
    p_int = 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # 10. AUTHORITY-COINCIDENCE LEAD: if the topmost discriminating row is
    #     also more diagnostic than anything printed below it, its verdict
    #     dominates.  Still scaled by the same rarity confidence c, so a
    #     lone oracle amid a mute panel stays near chance.
    # ------------------------------------------------------------------
    j1 = int(idx[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(s[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(s[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    # ------------------------------------------------------------------
    # 11. HARD uniform noise ceiling
    # ------------------------------------------------------------------
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [1.40, 1.80]
- lam: [0.11, 0.16]
- pi_prim: [0.08, 0.18]
- mu: [0.85, 1.20]
- sigma: [0.38, 0.52]
- k2: [0.40, 0.58]
- eta: [4.00, 5.20]
- q: [1.90, 2.50]
- omega: [0.62, 0.80]
- c_min: [0.03, 0.08]
- beta: [6.50, 9.50]
- eps: [0.28, 0.38]
- phi: [0.98, 1.08]
- s_stop: [6.0, 12.0]
- q_max: [0.86, 0.96]
- d_com: [0.42, 0.62]
- validities: validities

`rationale`: Minimal diff on the accepted base (loss 0.1435): the equations are re-emitted verbatim except for ONE new line in the weight construction and three parameter-range moves. Everything the critic praised (speaker-rarity confidence as sole confidence term, deadlock-only anchor gate, uniform eps ceiling, no unanimity step, no silence-composition dependence, no serial stop) is untouched.

(1) lam 0.175 -> centre 0.135 (range [0.11,0.16]), exactly the step the critic asked for — a second modest move in the direction that was ACCEPTED last round, not the aggressive cut that the Exp-2 trace argued against. I re-traced the cells this moves: Exp 4's included cell (d=[+1,0,-1,0,+1], extremes .78/.72 against the .95 at row 2) goes from a TTB-follow of ~.39 to ~.42 (real .508); Exp 12 cell 8 (dk=+2) and Exp 11 c8 lose ~0.04/(k2+W) of drive each; Exp 17's D-group (cell C dk=3, cell H dk=2) loses more drive than its S-group (dk=0/1), which is precisely the dense-minus-sparse residual; Exp 18 and Exp 14 move the same way. Exp 2 is protected because its conflict cells (5-1 and 3-1 against the bottom-row .95) carry most of their weight on early rows, where the new primacy credit compensates for the lost numerosity: traced conflict drive changes by < 0.01, so Exp 2 should stay >= .74.

(2) NEW small graded primacy credit w_j *= (1 + pi_prim*(1 - j/(n-1))), pi_prim centre 0.13, capped at 0.18. This is the critic's item 2 and the arbiter's explicitly permitted 'at most a very small primacy weight'. It is the only term that separates 'early two-cue coalition vs a later heavyweight' (Exp 11 c5, currently far too low; Exp 6 class-2; Exp 13 follow-first-row, currently .296 vs .353; Exp 9 cell 1) from the numerically identical late coalitions, and it does so WITHOUT any edge/salience or stop-rule machinery. I kept it at the low end of the critic's range on purpose: traced on Exp 15's top/bottom swap cell (dk=0, so the anchor term is exactly zero there and only the weights matter), pi_prim=0.13 raises m_top by only ~0.03, and the compensating small rise in the two single-cue cells (their drive w/(k2+w) grows slightly) keeps Exp 15 near .75; at pi_prim=0.30 the swap cell would have moved ~0.08 and Exp 15 would have regressed, which is why I capped rather than centred at 0.20. Traced on Exp 4 (the .95 sits at row 2, flanked by rows 0 and 4) the primacy ratio changes V by < 0.005, so Exp 4 is unharmed.

(3) I deliberately do NOT trim mu. I traced Exp 11 c4 — the cell the critic blames on mu — and it is saturated: with mu=1.2 the drive is ~0.39 (p=.96 pre-ceiling) and with mu=1.1 it is ~0.35 (p=.94), i.e. both land on the .83 ceiling, so trimming mu buys essentially nothing on Exp 11 while costing Exp 10 (.752 vs real .850) and Exp 5 (.803 vs .820), which are the two cells that need the anchor term at full strength. Exp 11's residual is instead attacked from the c5/c8 side (primacy up, lam down), which is where the trace says the movable mass is.

(4) Optional low-risk bump the critic sanctioned: q_max centre 0.91 and d_com centre 0.50. Traced on Exp 1 (the one design whose validities descend with screen order, so the authority check fires on every trial): conflict cells rise from ~.762 to ~.79 while agreement cells are already ceiling-bound at ~.82, shrinking the metric residual from -0.157 to roughly -0.03. The rarity floor keeps Exp 15/16's single-cue and silence-composition cells where they are, since the committed verdict is still multiplied by c.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a clear improvement over pi_7 on the cells the arbiter flagged (Exp 2 = .775 vs .775, Exp 8 = -.194 vs -.203, Exp 5 = .806 vs .820, Exp 7 = .022 vs .053, Exp 15 = .789 vs .855, Exp 16 = .042 vs .097, Exp 13 = .257 vs .353), and the hard noise ceiling plus the speaker-sensitivity confidence term are clearly doing useful work: the single-cue reversal (Exp 8) and the silence-composition nulls (Exp 7, 16, 18 sign-wise) are close. But the fit is broken in one huge place and mis-scaled in a cluster of others.

(1) Exp 1 is catastrophic: real +0.093, candidate -0.493 
 this is the largest single residual in the set, and it directly violates the arbiter's own stated target (~0.00 on this metric). The metric is TTB-follow-rate on conflict cells minus on agreement cells. The candidate's conflict cells are being decided almost entirely by the raw count term lam*(kA-kB): with lam=0.23 and five features, a 1-vs-4 or 2-vs-3 display gives N = -0.7 to -0.23, which swamps the weighted-diagnosticity term V even when the lone speaker is the .95 expert (w=.85 vs four cheap cues summing ~.3). So the model follows the tally at the ceiling (~.83) on conflict cells and at the ceiling on agreement cells, producing a difference near -(1-2*eps) instead of ~0. Note that Exp 2 (the metric the lam term was tuned to) does NOT require lam at all: in Exp 2's conflict cells the tally side always carries .9/.8/.7 experts, so the weighted sum alone already prefers the coalition. lam is therefore over-fit to a cell that does not need it and is destroying Exp 1.

(2) A second, separable mis-calibration: the 'two screen-extreme cues beat one mid-list heavyweight' cells are all systematically under-predicted because the edge term is gated to fire only at an EXACT deadlock. Exp 6 (.542 vs .841), Exp 10 (.652 vs .850), Exp 3 (.450 vs .518), Exp 4 (.391 vs .508), Exp 9 (.378 vs .490) are all cells with |kA-kB| = 1 or 2, where sigma=0.45 switches the anchor term essentially off. Every one of these residuals has the same sign, which is a calibration signature, not noise.

(3) A third cluster shows residual over-sensitivity to evidence MASS: Exp 17 (+0.153 vs -0.001) and Exp 18 (+0.100 vs -0.033) both say the candidate is more decisive on dense/high-mass displays than on sparse ones, whereas humans are flat. The k2+W normalisation is not flattening enough, and the linear lam*dk term re-introduces exactly the mass sensitivity the arbiter wanted removed. Exp 11 (0.948 vs 0.513) is the same story in composite form: c8 (three mid cues vs one oracle) is too high, again a numerosity artefact.

Encouragingly, all three diagnoses point the same way on the shared parameters: shrinking lam and moving the coalition-vs-heavyweight work from the raw count into the (widened) extreme-row term would push Exp 1, 3, 4, 6, 9, 10, 11, 12, 17 and 18 all toward the data simultaneously.
Rationale: Stay inside the prescribed family (weighted/compressed diagnosticity tally + linear numerosity + speaker-rarity confidence + uniform noise ceiling) 
 this is a retune, not a redesign. Three concrete, mutually consistent knobs:

1. CUT THE NUMEROSITY TERM HARD. lam is the proximate cause of the Exp 1 blow-up (-0.49 vs +0.09). Drop the range to roughly [0.03, 0.12] (centre ~0.07). Verify that Exp 2 stays at ~.775 
 it should, because in Exp 2 the tally side already wins on weighted diagnosticity alone (.9+.8+.6 vs a lone .95), so nothing is lost. This same change pulls Exp 12 (.733 -> .665), Exp 11 (.948 -> ~.5, via c8), Exp 18 (+.10 -> ~0) and Exp 17 (+.15 -> ~0) toward the data. Keep rho in [1.4, 2.0] so the .95 expert retains enough weight to win Exp 1's small-margin conflict cells once lam is small.

2. WIDEN AND STRENGTHEN THE EXTREME-ROW TERM so it survives |kA-kB| = 1-2 instead of only an exact deadlock: sigma -> [0.85, 1.40], mu -> [0.75, 1.15]. This is the mechanism that has to carry the 'two screen extremes beat one mid-list heavyweight' cells once lam is reduced: Exp 6 (.542 -> ~.84), Exp 10 (.652 -> ~.85), Exp 3 (.450 -> ~.52), Exp 4 (.391 -> ~.51), Exp 9 (.378 -> ~.49). Note that this also resolves the apparent Exp 1 / Exp 6 tension: Exp 1's conflicts are top-row-singleton displays where the extremes split or cancel, while Exp 6's are displays where BOTH extremes back the coalition 
 the position term, not the raw count, is what should distinguish them. Keep it confined to the literal first and last rows with no residual, as now.

3. FLATTEN EVIDENCE-MASS SENSITIVITY a little further for Exp 17/18: either raise k2 to [0.55, 0.95] or normalise the drive by W alone (plus a small floor) rather than k2+W, so dense landslide displays are not systematically more decisive than sparse ones. Leave eps in [0.30, 0.38]; do not raise it further, since Exp 5 (.82), Exp 10 (.85) and Exp 6 (.84) all need the ceiling to stay near .83.

One faithfulness note: the candidate uses a smooth power weighting with an effective .95:.56 weight ratio of ~15-20x, far steeper than the arbiter's prescribed compressed/ordinal ratio of 1.5-2.5. Do NOT compress toward 1.5-2.5 
 that would make Exp 1 worse 
 but do state the weighting explicitly as a graded (non-ordinal) diagnosticity within the family, and rely on the lam cut plus the anchor term (not weight compression) to keep the multi-cue coalition cells where the data put them. Everything else (no unanimity step, no silence-composition dependence, no serial stop rule, speaker-sensitivity as the sole confidence term, uniform noise ceiling) should be kept exactly as it is 
 those parts are already fitting well.

**Outcome of this advice:** iter 1 candidate loss=0.1800 -> iter 2 candidate loss=0.1435 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The retune was clearly productive: loss fell 0.180 -> 0.1435 and the single catastrophic residual (Exp 1, -0.493 -> -0.064 against a real +0.093) was largely repaired by the authority-coincidence lead plus the moderate lam cut, exactly the direction of last round's advice. Exp 10 also improved (.652 -> .752 vs .850) with the higher mu, and Exp 16 is now essentially on target (.110 vs .097). The silence-composition nulls (Exp 7 +.020 vs +.053; Exp 16) and the reversed single-cue ladder (Exp 8 -.149 vs -.203) remain well captured, so the speaker-rarity confidence term and the hard uniform ceiling are doing the work the arbiter asked of them.

What remains is a coherent, signed residual cluster rather than noise:
(1) Exp 11 is now the single largest error (1.002 vs 0.513, +0.49) and it got WORSE than iter 1 (0.948). The composite is c4 + c8 - c5. The candidate is near the .83 ceiling on c4 (the dk=0 extremes-vs-two-heavyweights cell, boosted by the raised mu) and on c8 (three mid cues vs one oracle, boosted by lam*dk), while it is too low on c5 ([+1,+1,0,-1,0]: two EARLY cues against a later heavyweight). So the model over-credits deadlock-edge salience and raw numerosity, and under-credits early two-cue coalitions.
(2) Exp 6 (.529 vs .841, -0.31) is the same under-credit in another guise: its class-2 cells are exactly 'two cues including the top row against a mid-list heavyweight', which neither the deadlock gate (|dk|=1) nor the authority lead (heavyweight below the top row) touches, so the model sits at chance while humans are at ~.84.
(3) A mass-sensitivity residual persists and is almost unchanged: Exp 17 (+0.143 vs -0.001) and Exp 18 (+0.098 vs -0.033) both say the model is still more decisive on dense/high-mass displays than on sparse ones, whereas humans are flat. Exp 12 (+0.05 too high) is the same signature.
(4) A set of cells where the model is uniformly too extreme against the coalition-favoured side: Exp 4 (.408 vs .508), Exp 9 (.351 vs .490), Exp 13 (.296 vs .353), Exp 14 (.208 vs .300), Exp 15 (.758 vs .855). All five have the same sign, and all five move toward the data if numerosity drive is damped further.
Notably (1), (3) and (4) all point at the same knob (lam is still too large at ~0.175), and (1c)+(2) point at the same missing ingredient (a small graded early-row credit for coalitions whose top member is not the most diagnostic).
Rationale: Stay inside the prescribed family (graded weighted-diagnosticity tally + linear numerosity + speaker-rarity confidence + deadlock-gated anchor + uniform noise ceiling). The candidate is faithful to it and should be kept as the base; only three in-family knobs need moving, and they are mutually consistent.

1. SMALLER STEP IN THE SAME (ACCEPTED) DIRECTION ON lam. Last round's cut 0.23 -> 0.175 was accepted; do NOT jump to the aggressive 0.03-0.12 I originally asked for (the proposer's trace argued Exp 2 would fall to ~.60-.68, and that risk is real). Take one more modest step: lam range [0.11, 0.16], centre ~0.135. This is the single lever that simultaneously pulls Exp 11 c8 down, Exp 12 (.715 -> ~.68), Exp 17 (+0.14 -> ~+0.08), Exp 18 (+0.10 -> ~+0.05), Exp 14 (.208 -> ~.25), Exp 9 (.351 -> ~.40) and Exp 4 (.408 -> ~.45) toward the data. Check Exp 2 stays >= .74.

2. ADD A SMALL GRADED PRIMACY CREDIT TO THE TALLY WEIGHTS (not a new family, and explicitly permitted by the arbiter as 'at most a very small primacy weight'): w_j <- w_j * (1 + pi_prim * (1 - j/(n-1))) with pi_prim in [0.10, 0.30]. This is the missing ingredient behind the two largest structural misses: Exp 11 c5 (two EARLY cues vs a later heavyweight, currently too low) and Exp 6 class-2 (.529 vs .841). It also nudges Exp 13 (.296 -> ~.33) and Exp 3 (.471 -> ~.50) the right way. Keep it small enough that Exp 4 (.95 at position 2) and Exp 15's top/bottom swap cell do not degrade by more than a couple of points; if Exp 15 drops below ~.72, cap pi_prim at 0.15.

3. DO NOT RAISE mu FURTHER; if anything trim its upper bound to ~1.10. The mu increase bought Exp 10 (+0.10, good) but simultaneously pushed Exp 11 c4 to the ceiling, which is why Exp 11 regressed from 0.948 to 1.002. Since Exp 5 (.803) and Exp 10 (.752) are still slightly low, prefer buying that back via the primacy term and the lam cut rather than via more edge salience. I am NOT re-asking for a wider sigma / deadlock gate / higher k2 \u2014 the proposer's counter-arguments held up and those cells (Exp 3, Exp 9 2-vs-1) are at chance in the data; leave sigma and k2 as they are.

Optional, low-risk: Exp 1 is still 0.157 below the real value, so a small widening of the authority-lead strength (q_max upper bound to ~0.96, d_com centre ~0.50) is defensible, but only if Exp 15/16's single-cue and silence-composition cells stay put (the rarity floor should protect them). Everything else \u2014 no unanimity step, no silence-composition dependence, no serial stop rule, uniform eps ~0.33 ceiling \u2014 is fitting well and should be left untouched.

**Outcome of this advice:** iter 2 candidate loss=0.1435 -> iter 3 candidate loss=0.1310 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: Iter 3 is the best candidate so far (0.1435 -> 0.1310) and the two moves I asked for both paid off: the second lam step and the small graded primacy credit pulled Exp 13 onto target (.362 vs .353), Exp 12 onto target (.675 vs .665), Exp 17 down (+0.143 -> +0.091), Exp 18 down (+0.098 -> +0.074), Exp 1 further up (-0.064 -> -0.030), Exp 5 now essentially exact (.840 vs .820), and Exp 8 / Exp 16 / Exp 7 are still well captured, so the speaker-rarity confidence term, the uniform eps ceiling and the no-silence-composition constraint continue to earn their place. The residual structure is now quite clean and mostly signed in one direction.

(1) Exp 11 (0.922 vs 0.513) is still by far the largest error, and its composition has not changed: the model sits near the .83 ceiling on c4 (dk=0, deadlock anchor fires at full strength) and on c8 (three mid cues vs the oracle), while it is still too LOW on c5 ([+1,+1,0,-1,0] = two EARLY cues against a later heavyweight). pi_prim=0.13 was too timid to move c5 appreciably.

(2) Exp 6 got WORSE for the third iteration running (.542 -> .529 -> .512 against a real .841) and Exp 9 likewise (.378 -> .351 -> .333 against .490). Both are 'screen extremes agree, one mid-list heavyweight opposes' displays with |dk| = 1, where sigma=0.45 makes the deadlock gate g = exp(-1/(2*0.45^2)) ~ 0.085, i.e. the anchor term is effectively switched off. The models that do well on Exp 6 (pi_2, pi_3, pi_5) are all ones where the extremes/coalition survive a one-reason margin. This is the one cell family where each successive lam cut has made things monotonically worse, and it is now the second-largest residual.

(3) Exp 2 slid a further 0.044 (.753 -> .710 vs real .775), the predicted cost of the second lam cut. It is not catastrophic but it is now a real residual, and importantly it should NOT be repaired by putting lam back (that would immediately re-inflate Exp 17/18/11-c8): Exp 2's conflict coalitions sit on the EARLY rows against a bottom-row .95, so a larger primacy credit and slightly flatter weights recover it for free.

(4) A residual cluster of 'model too extreme against the coalition/weak-cue side' persists and all moves the same way: Exp 4 (.405 vs .508), Exp 14 (.202 vs .300), Exp 15 (.778 vs .855), Exp 3 (.468 vs .518), plus Exp 6/9 above. With rho ~1.6 on ((v-.5)/.5) the effective .95:.56 weight ratio is ~25x, far steeper than anything the data support, and it is what keeps buried heavyweights beating multi-cue coalitions.

(5) Mass sensitivity is reduced but not gone (Exp 17 +0.091, Exp 18 +0.074 against ~0).
Rationale: Accepted candidate, but there is clearly ~0.03-0.05 of loss still on the table in a coherently signed cluster, so keep tuning on THIS base. Stay strictly inside the prescribed family (graded weighted-diagnosticity tally + linear numerosity + speaker-rarity confidence + anchor tie-breaker + uniform eps ceiling) \u2014 the candidate is faithful to it and should not be redesigned. Four in-family knobs, ordered by expected payoff:

1. PUSH THE PRIMACY CREDIT THE REST OF THE WAY. pi_prim centre 0.13 was the low end of the range I sanctioned and it was ACCEPTED, so extend in the same direction rather than reversing: pi_prim -> [0.16, 0.26], centre ~0.21. This is the lever for the three biggest structural misses that all share the signature 'coalition sits on early rows': Exp 11 c5 (raises it, which directly lowers the 0.922 composite), Exp 6 class-2/3, and Exp 2 (whose conflict coalitions are early rows against a bottom-row .95, so this buys back the .710 -> ~.75 without touching lam). Guard-rail: if Exp 15 drops below ~0.74 or Exp 4 moves more than a couple of points, stop at 0.18-0.20.

2. MODEST WIDENING OF THE DEADLOCK GATE \u2014 a value BETWEEN my two previous flanking recommendations. In iter 1 I asked for sigma in [0.85,1.40] (that package was ACCEPTED, though the proposer kept sigma narrow); in iter 2 I said leave sigma alone. Rather than oscillate, split the difference: sigma -> [0.55, 0.80]. At sigma=0.7 the gate at |dk|=1 is g~0.36 instead of 0.085, which is exactly what the |dk|=1 extremes-agree cells need: Exp 9 (.333, real .490), Exp 3 (.468, real .518), Exp 6 (.512, real .841) all rise, while every dk=0 cell (Exp 5, Exp 10, Exp 11 c4, Exp 12 cell 4, Exp 17 cell E) is untouched and every |dk|>=2 cell stays near-zero. Do NOT raise mu at the same time \u2014 mu is already doing all it can and raising it re-saturates Exp 11 c4.

3. SMALL COMPRESSION OF THE DIAGNOSTICITY EXPONENT. rho -> [1.20, 1.50], centre ~1.35. The current ~25x .95:.56 weight ratio is what keeps a single buried heavyweight beating coalitions in Exp 4, 6, 12, 14 and what suppresses m_weak in Exp 15. A mild flattening (ratio ~12-15x) moves Exp 14 (.202 -> ~.24), Exp 15 (.778 -> ~.81), Exp 8 (-0.184 -> slightly more negative, i.e. toward the real -0.203) and Exp 2 the right way. Keep it MILD \u2014 do not go near the arbiter's 1.5-2.5 ordinal ratio, which would break Exp 1. If Exp 1 slips below ~-0.08 as a side effect, compensate only with the authority-lead strength (q_max/d_com), not with rho.

4. OPTIONAL, LAST: one small k2 step ([0.50, 0.70]) to finish flattening the residual mass sensitivity in Exp 17 (+0.091) and Exp 18 (+0.074). Apply this only after 1-3 and check it does not pull Exp 10/Exp 5 off their now-good values.

Explicitly DO NOT touch: lam (leave at ~0.135 \u2014 two cuts in a row were accepted, a third would keep costing Exp 2 and Exp 6 and I am not asking for it), the rarity-confidence form, the silence-composition blindness, the absence of a unanimity step / serial stop rule, or eps (~0.33). Report Exp 11's three sub-cells (c4, c5, c8) separately in the rationale so the composite's movement can be attributed.

**Outcome of this advice:** iter 3 candidate loss=0.1310 -> iter 4 candidate loss=0.1345 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: Iteration 4 was rejected by a hair (0.1345 vs 0.1310), and the per-experiment deltas make the cause unambiguous: two of the three knobs I asked for helped, one backfired badly.

(a) pi_prim 0.13 -> 0.195 was the damaging move. Exp 15 collapsed .778 -> .636 against a real .855 (u22120.14, by far the largest single regression in the round), because the composite is m_oracle + m_weak u2212 m_top and a bigger primacy credit raises m_top on the six-row top/bottom swap cell. The proposer's own trace predicted only ~0.04 of movement there; the realised cost was ~3-4x that. Worse, the move did NOT buy what it was supposed to buy: Exp 11 went the wrong way (.922 -> .940 vs real .513), i.e. c5 did not lift enough to offset the ceiling-bound c4/c8, and Exp 6 gained only +0.02 (.512 -> .532 vs real .841). So the primacy lever is now empirically exhausted: it costs Exp 15 far more than it earns on Exp 11/6. My iteration-3 advice to 'push the primacy credit the rest of the way' was wrong and the gate said so.

(b) The rho compression (1.60 -> 1.40 centre) was genuinely mixed but roughly break-even to mildly positive: gains on Exp 2 (.710 -> .731, real .775), Exp 8 (u22120.184 -> u22120.200, real u22120.203), Exp 14 (.202 -> .228, real .300), Exp 9 (.333 -> .365, real .490), Exp 6 (+0.02); losses on Exp 4 (.405 -> .371, real .508), Exp 1 (u22120.030 -> u22120.047, real +0.093) and Exp 10 (.750 -> .725, real .850). The Exp 10/5 loss is a side-effect the proposer did not anticipate: flattening rho raises total W, which shrinks the anchor term P/(k2+W), so the dk=0 extreme-vs-heavyweight cells lose drive. Exp 17/18 also crept back up (+0.091 -> +0.107, +0.074 -> +0.090) rather than flattening 'for free'.

(c) The sigma widening (0.45 -> 0.50) was near-noise: Exp 9 +0.03 and Exp 6 +0.02 in, Exp 3 u22120.01 and part of Exp 4's u22120.034 out. It does not justify further movement, and I have already oscillated on this knob three times.

Structurally the model remains faithful to the prescribed family (graded weighted tally, linear numerosity, speaker-rarity as the sole confidence term, deadlock-licensed anchor, uniform eps ceiling, no unanimity step, no silence-composition dependence) and continues to nail the parts the arbiter cared most about: Exp 8 (u22120.20 vs u22120.203), Exp 13 (.351 vs .353), Exp 16 (.067 vs .097), Exp 7 (.057 vs .053), Exp 5 (.843 vs .820), Exp 12 (.687 vs .665). The irreducible residuals are Exp 11 (+0.41) and Exp 6 (u22120.31), which are in genuine tension with Exp 3/4/9 (all at chance in the data for structurally similar |dk|=1 coalition-vs-heavyweight displays) and cannot both be fixed by any single weighting within this family.
Rationale: The candidate was REJECTED, so the base stays at iter 3 (loss 0.1310) and the next attempt must be a different, smaller step on that base u2014 still strictly inside the prescribed family (graded weighted-diagnosticity tally + linear numerosity + speaker-rarity confidence + deadlock-gated anchor + uniform eps ceiling). Four instructions, in priority order:

1. REVERT THE PRIMACY PUSH AND FREEZE IT. pi_prim goes back to the iter-3 range (~[0.10, 0.15], centre 0.13) and must not be raised again. My iter-3 advice to extend it was rejected by the gate: the Exp 15 cost (u22120.14) dwarfs the Exp 11/6 gain (+0.02 each). Treat 0.13 as the empirically located optimum of this knob and stop tuning it.

2. TAKE HALF THE rho STEP, NOT THE WHOLE ONE. rho -> [1.45, 1.62], centre ~1.52 (midway between the accepted 1.60 and the rejected 1.40). At half strength the same-signed gains on Exp 2, 8, 9, 14 and 6 should mostly survive while the Exp 4 (u22120.034), Exp 1 (u22120.017) and Exp 10 (u22120.025) costs are roughly halved. Do NOT go below 1.45.

3. COMPENSATE THE ANCHOR DILUTION THAT rho CAUSES, WITH k2 (not mu, not sigma). Because a lower rho inflates W, the dk=0 anchor cells lose drive; offset this by lowering k2 modestly (e.g. [0.32, 0.46], centre ~0.38) so Exp 10 (.750, real .850) and Exp 5 (.842, real .820) keep or regain their anchor strength. Check that Exp 5 does not exceed ~.86 and that Exp 17/18 do not climb above their iter-3 values (+0.091 / +0.074); if they do, back k2 off to ~0.44. Leave mu untouched (raising it re-saturates Exp 11 c4) and leave sigma at the iter-3 value (~0.45) u2014 I have flip-flopped on sigma three times and the two widenings bought essentially nothing, so it is now frozen.

4. SPEND ONE SMALL STEP ON THE AUTHORITY LEAD FOR Exp 1. Exp 1 is still 0.12 below the real +0.093 and is the one large residual nobody has attacked since iter 2, where a lead-strength increase was part of an ACCEPTED package. Nudge d_com to ~0.58-0.65 and q_max to ~0.95-0.96. This raises TTB-following specifically on the conflict cells where the top screen row is also the most diagnostic; the agreement cells are already ceiling-bound, so the difference metric should move up by ~0.04-0.08. Guard-rail: the rarity floor must keep Exp 15's m_oracle and Exp 16's O0/O1 cells unchanged u2014 report those two sub-rates explicitly.

Do not attempt further surgery on Exp 11 or Exp 6. Their requirements (coalition beats a buried heavyweight at |dk|=1 in Exp 6/11-c5, but the same structure is at chance in Exp 3/4/9) are mutually inconsistent under any single graded weighting, and every attempt so far has traded one for the other at zero or negative net gain. Report Exp 11's c4/c5/c8 sub-rates and Exp 15's m_oracle/m_weak/m_top so the next round can attribute movement, and change only the four knobs above so the attribution is clean.

**Outcome of this advice:** iter 4 candidate loss=0.1345 -> iter 5 candidate loss=0.1491 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: Iter 5 was rejected (0.1491 vs the iter-3 base at 0.1310) and it is the worst candidate since iter 1. The per-experiment deltas relative to the iter-3 base give a fairly clean attribution of the three-knob package.

(a) The k2 reduction (0.48 -> 0.40) is the main villain. Lowering k2 raises drive on LOW-mass displays, which (i) pushed both the strong-cue and weak-cue single-discriminator trials of Exp 8 toward the eps ceiling and therefore COMPRESSED the reversal (-0.184 -> -0.149 against a real -0.203), and (ii) inflated the anchor term P/(k2+W) on Exp 15's six-row top/bottom swap cell, raising m_top and collapsing the composite (.778 -> .657 against a real .855, a -0.12 regression, the single largest in the round). The compensation it was supposed to buy did not appear: Exp 10 went DOWN (.750 -> .737, real .850) and Exp 17/18 did not flatten (+0.091 -> +0.098, +0.074 -> +0.064). So the anchor-dilution story I told last round is empirically wrong; k2 should be left alone.

(b) The authority-lead nudge (q_max .91 -> .92, d_com .50 -> .56) also failed on its own terms. Exp 1, the residual it was aimed at, did NOT improve (-0.030 -> -0.035, real +0.093), because the agreement cells are ceiling-bound and rise with the lead just as fast as the conflict cells. Meanwhile it cost Exp 13 (.362 -> .323, real .353, previously ON TARGET) and contributed to Exp 15/16 (.068 -> -0.005). This knob is now exhausted in both directions.

(c) The rho half-step (1.60 -> 1.53) looks mildly positive in isolation and repeats the same-signed gains seen in iter 4: Exp 2 (+.025), Exp 9 (+.026), Exp 6 (+.008), Exp 5 now essentially exact (.8225 vs .820), Exp 18 (+.010); costs on Exp 3 (-.026) and Exp 4 (-.024). Net ~break-even-to-slightly-positive; it was swamped by (a) and (b).

The standing residual structure on the iter-3 base is unchanged and dominated by two cells: Exp 11 (+0.41) and Exp 6 (-0.33), which together are ~40% of the total absolute error, followed by Exp 9 (-0.16), Exp 1 (-0.12), Exp 18/17 (+0.11/+0.09), Exp 4 (-0.10), Exp 10 (-0.10), Exp 14 (-0.10). Everything the arbiter cared most about is still well captured at the base (Exp 8 -0.184 vs -0.203; Exp 13 .362 vs .353; Exp 12 .675 vs .665; Exp 16 .068 vs .097; Exp 5 .840 vs .820; silence-composition contrasts ~0), and the candidate remains fully faithful to the prescribed family (graded weighted diagnosticity tally, linear numerosity, speaker-rarity as the sole confidence term, deadlock-licensed anchor, uniform eps ceiling, no unanimity step, no silence-composition dependence).

A useful decomposition note for Exp 11: its composite c4 + c8 - c5 = .513 is reproducible if humans are at ~.83 on the dk=0 edge cell c4 (consistent with Exp 5/Exp 10, which the model already matches in kind), ~.5 on c8 (three cheap cues vs a lone .95 oracle) and ~.8 on c5 (two early cues vs a later heavyweight). The model's error is therefore concentrated in c8 being far too high \u2014 i.e. raw numerosity is still over-driving the 3-vs-1-against-the-oracle cell \u2014 not in c4. That is the same signature as the residual mass sensitivity in Exp 17/18 and the slight over-shoot in Exp 12.
Rationale: The candidate was rejected by the gate, so the base stays at iter 3 (0.1310). Stay strictly inside the prescribed family \u2014 the candidate is faithful to it and needs no redesign, only a cleaner parameter step. Two hard reversions and one two-knob, mutually-compensating step:

1. REVERT k2 AND THE AUTHORITY LEAD TO THE ITER-3 VALUES AND FREEZE THEM. k2 back to ~0.48 (range [0.44, 0.52]); q_max back to ~0.91, d_com back to ~0.50. The iter-5 evidence against both is direct: k2 down cost Exp 8 (-0.035) and Exp 15 (-0.12) and bought nothing on Exp 10/17/18; the lead increase cost Exp 13 (-0.039) and Exp 15/16 and left Exp 1 flat. My own iter-4 advice on these two knobs was wrong and the gate said so \u2014 do not retry either in either direction.

2. KEEP THE rho HALF-STEP, BUT AS THE ONLY WEIGHT CHANGE: rho [1.50, 1.62], centre ~1.55. Across iter 4 and iter 5 this is the one move with repeatedly same-signed gains (Exp 2 +.021/+.025, Exp 9 +.032/+.026, Exp 6 +.02/+.01, Exp 14 +.026) and the costs are bounded and small (Exp 3, Exp 4 ~-.025 each). With k2 restored, the Exp 8 and Exp 15 regressions seen in iter 5 should not recur.

3. TAKE ONE MORE MODEST lam CUT, PAIRED WITH (2): lam [0.09, 0.13], centre ~0.11. Note the track record: every lam cut so far (0.23 -> 0.175 -> 0.135) has been part of an ACCEPTED package, and the residual structure still points at numerosity. This is the only lever that attacks the dominant Exp 11 error where it actually lives (c8, three cheap cues vs a lone oracle: should be ~.5, model ~.7) while simultaneously pulling Exp 17 (+0.091 -> ~+0.06), Exp 18 (+0.074 -> ~+0.05) and Exp 12 toward zero. Its known cost is Exp 2 (~-0.02), which the rho step in (2) pays for almost exactly \u2014 that is why the two are paired rather than isolated.

Explicitly DO NOT touch: pi_prim (frozen at 0.13 \u2014 the push to 0.195 cost Exp 15 -0.14 for +0.02 on Exp 6/11 and was rejected), mu (raising it re-saturates Exp 11 c4), sigma (frozen at ~0.45 after three flip-flops that bought nothing), eps (~0.33; Exp 5/10/6 need the ceiling where it is), omega/eta/q (the silence-composition nulls and Exp 8's reversal are the best-fitting parts of the model), and the read-out/normalisation FORM.

Guard-rails and reporting: Exp 2 must stay >= 0.72, Exp 5 must stay in [0.80, 0.86], Exp 8 must stay <= -0.18, Exp 13 must stay in [0.34, 0.38], Exp 15 must stay >= 0.76. Report Exp 11's c4/c5/c8 and Exp 15's m_oracle/m_weak/m_top sub-rates explicitly so the next round can attribute movement, and change ONLY rho and lam relative to the iter-3 base \u2014 the last two rounds both bundled three or four simultaneous changes and the attribution had to be reverse-engineered from the residuals, which is exactly how two good knobs got buried under two bad ones. No further surgery on Exp 6 (its |dk|=1 coalition demand is in genuine tension with Exp 3/4/9 being at chance for the same structure).

**Outcome of this advice:** iter 5 candidate loss=0.1491 -> iter 6 candidate loss=0.1410 -> the gate marked it REJECTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: Iter 6 was rejected (0.1410 vs the iter-3 base at 0.1310), and because only two knobs moved the attribution is clean this time. The lam cut (0.135 -> 0.11) is the villain, and it failed for a reason I did not anticipate: lam is not a pure 'numerosity' term. On single-discriminator and 1-vs-0 displays it supplies a constant baseline drive, so cutting it pushed exactly the cells that were fitting best toward 0.5. Exp 8's reversal compressed (-0.184 -> -0.139 against a real -0.203) because both the strong- and weak-cue single-cue trials lost drive, and Exp 15 collapsed (.778 -> .682 against .855) because m_oracle and m_weak (|dk|=1) lost drive while m_top (dk=0, anchor-driven, lam-insensitive) did not \u2014 a -0.096 regression that alone exceeds the whole gate margin. Worse, the cut did NOT buy what it was supposed to buy: Exp 11 moved the WRONG way (.922 -> .930) and Exp 17 was flat (+0.091 -> +0.095). So my iter-5 reading that Exp 11's c8 residual lives in lam*dk is empirically refuted: c8's drive is carried by V (three speaking cues with the primacy multiplier), not by the count term. Gains were real but small and cheap-to-get elsewhere: Exp 18 (+0.028), Exp 16 (+0.032), Exp 14 (+0.014). The rho narrowing to 1.55 also failed to deliver its predicted Exp 2 gain (.7095 -> .6965) \u2014 confounded with the lam cut, but at best neutral, and Exp 3 lost -0.046 and Exp 6 -0.035.

The standing structure on the iter-3 base is unchanged: Exp 11 (+0.41) and Exp 6 (-0.33) together are ~40% of total absolute error, then Exp 9 (-0.16), Exp 1 (-0.12), Exp 18 (+0.11), Exp 4 (-0.10), Exp 10 (-0.10), Exp 14 (-0.10), Exp 17 (+0.09). The signature across Exp 3, 4, 9, 11, 14, 17, 18 is the same: on MULTI-SPEAKER conflict displays the model is confidently decided (p near the .83 ceiling) where humans hover near chance or only mildly favour one side, while on FEW-SPEAKER displays (Exp 8, 13, 15, 5, 10) it is already calibrated. That is a mass-scaling residual, not a weighting residual, and it is the one thing no iteration has actually attacked: every lever tried so far (lam x3, pi_prim, sigma, k2, authority lead, rho) changes the numerator or the ceiling, not how drive scales with the number of speaking experts. The model remains fully faithful to the prescribed family (graded weighted-diagnosticity tally, linear numerosity, speaker-rarity as sole confidence term, deadlock-licensed anchor, uniform eps ceiling, no unanimity step, no silence-composition dependence) and still nails the arbiter's headline targets at the base.
Rationale: Rejected by the gate, so the base stays at iter 3 (0.1310). Stay strictly inside the prescribed family \u2014 no redesign, only a cleaner step. THREE instructions, and change exactly ONE thing relative to iter 3 so attribution is unambiguous (the last three rejected rounds all bundled two-plus knobs).\n\n1. REVERT lam AND FREEZE IT AT THE ITER-3 VALUE (~0.135, range [0.12, 0.15]). My iter-5 advice to take a third lam cut was wrong and the gate said so: lam also supplies baseline drive on |dk|=1 / single-speaker displays, so cutting it costs Exp 8 (-0.045) and Exp 15 (-0.096) while leaving Exp 11 and Exp 17 untouched. Do not retry lam in either direction \u2014 it is now exhausted.\n\n2. REVERT rho TO THE ITER-3 RANGE (~1.60). Its two 'same-signed gains' were measured inside packages that were both rejected, and in the one clean-ish test this round Exp 2 went DOWN rather than up. Park it; the base value is not the bottleneck.\n\n3. THE ONE NEW MOVE \u2014 SUPERLINEAR EVIDENCE-MASS NORMALISATION (an in-family normalisation swap, explicitly permitted; k=1 recovers the current base, so this is a smooth one-parameter extension, not a new mechanism). Replace drive = (V + N + P) / (k2 + W) with drive = (V + N + P) / (k2 + W)^kappa, kappa in [1.08, 1.28], centre ~1.18. Rationale: the residual cluster is uniformly 'too decisive on many-speaker displays, calibrated on few-speaker displays'. With k2 ~0.48, a single-cue display (W ~0.85) is damped only ~5-6%, whereas a 4-5-speaker display (W ~2-2.5) is damped ~18-25%. Expected movement: Exp 11 (.922 -> ~.80), Exp 9 (.333 -> ~.38), Exp 4 (.405 -> ~.44), Exp 3 (.468 -> ~.49), Exp 14 (.202 -> ~.24), Exp 17 (+0.091 -> ~+0.06), Exp 18 (+0.074 -> ~+0.05), Exp 6 (+0.02), with the well-fitting few-speaker cells (Exp 8, 13, 15, 5, 10) essentially untouched. If the small residual damping of single-cue trials visibly shifts Exp 8/13/15, compensate ONLY by raising beta by the matching ~5-6% (e.g. beta centre 8.0 -> 8.5) so that the single-discriminator drive is held at its iter-3 value \u2014 that is a pure re-centring, not a second lever.\n\nGuard-rails (report these explicitly, plus Exp 11's c4/c5/c8 and Exp 15's m_oracle/m_weak/m_top): Exp 8 <= -0.18, Exp 13 in [0.34, 0.38], Exp 15 >= 0.76, Exp 5 in [0.80, 0.86], Exp 10 >= 0.72, Exp 2 >= 0.70, Exp 12 in [0.63, 0.70]. The known risk is Exp 2 and Exp 12, whose conflict cells ARE multi-speaker; if Exp 2 drops below 0.70, back kappa off to ~1.10 rather than reopening lam or rho.\n\nExplicitly DO NOT touch: pi_prim (frozen at 0.13; the push to 0.195 cost Exp 15 -0.14 and was rejected), k2 (lowering it cost Exp 8 and Exp 15 and bought nothing \u2014 rejected), the authority lead q_max/d_com (raising it cost Exp 13 and left Exp 1 flat \u2014 rejected), sigma (three flip-flops, no yield), eps (~0.33), and eta/q/omega and the whole rarity-confidence / silence-composition-blind structure, which produce the best-fitting parts of the model. No further surgery on Exp 6: its |dk|=1 'coalition beats a buried heavyweight' demand is in genuine tension with Exp 3/4/9 sitting at chance for the same structure, and both cannot be satisfied by any single weighting in this family.

**Outcome of this advice:** iter 6 candidate loss=0.1410 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Gap-Inference Tally with an AUTHORITY-COINCIDENCE LEAD, a small GRADED PRIMACY CREDIT, and SUPERLINEAR EVIDENCE-MASS NORMALISATION (GIT-AL+P/k).** Same mechanism family as the accepted base \u2014 a graded (non-ordinal) weighted reason tally, a weak linear numerosity reason, speaker-rarity as the ONLY confidence term, a deadlock-licensed first/last-row tie-breaker, relative read-out and a hard uniform noise ceiling \u2014 with ONE calibration change: the relative read-out divides by (k2 + W)^kappa with kappa slightly greater than 1 rather than by (k2 + W).\n\n1. **Weighted reasons.** Each discriminating expert contributes w_j = ((v_j-0.5)/0.5)^rho, rho ~1.4-1.8 (smooth, no top-end premium, no knee).\n2. **Small graded primacy credit** on the tally weights: w_j is multiplied by (1 + pi_prim*(1 - j/(n-1))) with pi_prim ~0.13. Early screen rows are read slightly more carefully; it is an attention multiplier on weights, not a stop rule and not an edge term.\n3. **Numerosity** is a separate, strictly linear, non-saturating reason lam*(k_A-k_B) with lam ~0.135. It also supplies the baseline drive on 1-vs-0 and |dk|=1 displays, so it is deliberately held fixed.\n4. **Speaker-rarity confidence is the only confidence term:** c = max(1 - eta*f_mute^q*s\u0304_speakers, c_min). Mute rows are near composition-blind (omega ~0.7), so silence-composition contrasts stay ~0.\n5. **Position enters the DIRECTION term only as a tie-breaker licensed by an EXACT deadlock** in reason counts (literal first and last screen rows, narrow Gaussian gate, no residual).\n6. **Authority-coincidence lead.** When the FIRST discriminating screen row is ALSO more diagnostic than anything printed below it, its verdict dominates the tally with probability q_com, still multiplied by the same rarity confidence c.\n7. **NEW \u2014 superlinear evidence-mass normalisation.** drive = (V + N + P) / (k2 + W)^kappa with kappa ~1.10-1.24. The psychological claim is that decisiveness does not merely fail to grow with the amount of evidence on the table; each *additional* speaking expert costs a little more integration capacity than the last, so a panel in which five experts all speak is read LESS decisively per unit of net evidence than a panel in which one speaks. kappa = 1 recovers the previous relative read-out exactly, so this is a one-parameter smooth extension, not a new mechanism. It is the only term in the model whose effect scales with the NUMBER of speakers rather than with their identity or position.\n8. **Hard, difficulty-independent noise ceiling** p = 0.5 + (p-0.5)*(1-eps), eps ~0.33.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ------------------------------------------------------------------\n    # 1. parse the trial stimulus into two rating vectors (screen order)\n    # ------------------------------------------------------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(list(state['option_a_ratings']), dtype=float)\n            b = np.asarray(list(state['option_b_ratings']), dtype=float)\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        arr = arr.reshape(2, -1)\n        a = arr[0].astype(float)\n        b = arr[1].astype(float)\n\n    a = np.ravel(a).astype(float)\n    b = np.ravel(b).astype(float)\n    n = int(min(a.shape[0], b.shape[0]))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = a[:n]\n    b = b[:n]\n    d = a - b\n    disc = np.abs(d) > 1e-12\n    if not np.any(disc):\n        return np.array([0.5, 0.5])\n\n    # ------------------------------------------------------------------\n    # 2. communicated validities -> sensitivity / diagnosticity scale\n    # ------------------------------------------------------------------\n    val = parameters.get('validities', None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = None\n    if val is None or val.shape[0] != n:\n        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])\n    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)\n    s = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)\n\n    # ------------------------------------------------------------------\n    # 3. parameters\n    # ------------------------------------------------------------------\n    def _p(key, dflt, lo, hi):\n        try:\n            v = float(parameters.get(key, dflt))\n        except Exception:\n            v = float(dflt)\n        if not np.isfinite(v):\n            v = float(dflt)\n        return float(np.clip(v, lo, hi))\n\n    rho = _p('rho', 1.60, 0.20, 5.0)\n    lam = _p('lam', 0.135, 0.0, 2.0)\n    pi_prim = _p('pi_prim', 0.13, 0.0, 1.0)\n    mu = _p('mu', 1.00, 0.0, 3.0)\n    sigma = _p('sigma', 0.45, 0.15, 3.0)\n    k2 = _p('k2', 0.48, 0.05, 5.0)\n    kappa = _p('kappa', 1.17, 0.50, 2.50)\n    eta = _p('eta', 4.50, 0.0, 15.0)\n    q = _p('q', 2.20, 0.5, 8.0)\n    omega = _p('omega', 0.70, 0.0, 1.0)\n    c_min = _p('c_min', 0.05, 0.0, 0.5)\n    beta = _p('beta', 8.43, 0.1, 40.0)\n    eps = _p('eps', 0.32, 0.0, 0.8)\n    # authority-coincidence lead\n    phi = _p('phi', 1.02, 0.5, 3.0)\n    s_stop = _p('s_stop', 9.0, 0.5, 60.0)\n    q_max = _p('q_max', 0.91, 0.0, 1.0)\n    d_com = _p('d_com', 0.50, 0.0, 3.0)\n\n    def _sig(z):\n        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))\n\n    # ------------------------------------------------------------------\n    # 4. smooth (no-premium) diagnosticity tally over speaking experts,\n    #    with a SMALL GRADED PRIMACY CREDIT on the weights (attention\n    #    multiplier, monotone in screen row, never sign-reversing).\n    # ------------------------------------------------------------------\n    pos = np.arange(n, dtype=float)\n    if n > 1:\n        prim = 1.0 + pi_prim * (1.0 - pos / float(n - 1))\n    else:\n        prim = np.ones(n, dtype=float)\n    w = np.power(s, rho) * prim\n    idx = np.flatnonzero(disc)\n    sgn = np.sign(d[idx])\n    ws = w[idx]\n    W = float(np.sum(ws))\n    if W <= 1e-12:\n        return np.array([0.5, 0.5])\n    V = float(np.sum(sgn * ws))\n\n    kA = int(np.count_nonzero(sgn > 0))\n    kB = int(np.count_nonzero(sgn < 0))\n    dk = float(kA - kB)\n\n    # ------------------------------------------------------------------\n    # 5. numerosity: linear, graded, no saturation, no unanimity step\n    # ------------------------------------------------------------------\n    N = lam * dk\n\n    # ------------------------------------------------------------------\n    # 6. anchor-row tie-breaker: literal first and last screen rows,\n    #    licensed ONLY by an exact deadlock in the number of reasons\n    #    (narrow gate, no residual term).\n    # ------------------------------------------------------------------\n    edge = np.zeros(n, dtype=float)\n    edge[0] = 1.0\n    edge[n - 1] = 1.0\n    S = float(np.sum(sgn * edge[idx]))\n    g = float(np.exp(-min((dk * dk) / (2.0 * sigma * sigma), 60.0)))\n    P = mu * g * S\n\n    # ------------------------------------------------------------------\n    # 7. read-out relative to the evidence actually on the table, with\n    #    SUPERLINEAR mass normalisation: each extra speaking expert costs\n    #    a little more integration capacity than the last, so many-speaker\n    #    panels are read less decisively per unit of net evidence.\n    #    kappa = 1 recovers the previous linear relative read-out.\n    # ------------------------------------------------------------------\n    den = float(k2 + W)\n    den = max(den, 1e-9)\n    drive = (V + N + P) / float(np.power(den, kappa))\n\n    # ------------------------------------------------------------------\n    # 8. rarity / sensitivity gap-inference: the ONLY confidence term.\n    # ------------------------------------------------------------------\n    both1 = (~disc) & (a > 0.5) & (b > 0.5)\n    mute_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))\n    f_mute = float(np.sum(mute_w)) / float(n)\n    f_mute = float(np.clip(f_mute, 0.0, 1.0))\n    sbar = float(np.sum(ws * s[idx]) / W)\n    c = 1.0 - eta * (f_mute ** q) * sbar\n    c = float(np.clip(c, c_min, 1.0))\n\n    # ------------------------------------------------------------------\n    # 9. integrative read-out\n    # ------------------------------------------------------------------\n    z = float(np.clip(beta * c * drive, -60.0, 60.0))\n    p_int = 1.0 / (1.0 + np.exp(-z))\n\n    # ------------------------------------------------------------------\n    # 10. AUTHORITY-COINCIDENCE LEAD: if the topmost discriminating row is\n    #     also more diagnostic than anything printed below it, its verdict\n    #     dominates.  Still scaled by the same rarity confidence c, so a\n    #     lone oracle amid a mute panel stays near chance.\n    # ------------------------------------------------------------------\n    j1 = int(idx[0])\n    if j1 >= n - 1:\n        q_com = q_max\n    else:\n        v_rest = float(np.max(s[j1 + 1:]))\n        v_rest = max(v_rest, 1e-6)\n        ratio = float(s[j1]) / (phi * v_rest)\n        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))\n    q_com = float(np.clip(q_com, 0.0, 1.0))\n\n    p_dec = float(_sig(beta * c * d_com))\n    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)\n\n    p_a = q_com * p_com_a + (1.0 - q_com) * p_int\n\n    # ------------------------------------------------------------------\n    # 11. HARD uniform noise ceiling\n    # ------------------------------------------------------------------\n    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)\n    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, None)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0:\n        probs = np.ones_like(probs) / len(probs)\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "rho": "[1.40, 1.80]",
    "lam": "[0.11, 0.16]",
    "pi_prim": "[0.08, 0.18]",
    "mu": "[0.85, 1.20]",
    "sigma": "[0.38, 0.52]",
    "k2": "[0.40, 0.58]",
    "kappa": "[1.10, 1.24]",
    "eta": "[4.00, 5.20]",
    "q": "[1.90, 2.50]",
    "omega": "[0.62, 0.80]",
    "c_min": "[0.03, 0.08]",
    "beta": "[6.90, 10.00]",
    "eps": "[0.28, 0.38]",
    "phi": "[0.98, 1.08]",
    "s_stop": "[6.0, 12.0]",
    "q_max": "[0.86, 0.96]",
    "d_com": "[0.42, 0.62]",
    "validities": "validities"
  },
  "rationale": "MINIMAL DIFF on the iter-3 base (loss 0.1310). Relative to that base exactly TWO lines change: (i) the read-out denominator becomes (k2 + W)^kappa instead of (k2 + W), with a new parameter kappa in [1.10, 1.24] (centre 1.17); (ii) beta is re-centred 8.0 -> 8.43 (range [6.90, 10.00]) purely to hold the single-discriminator drive at its iter-3 value. Everything else is re-emitted verbatim: lam is REVERTED/frozen at the iter-3 range [0.11,0.16] and rho at [1.40,1.80] as the critic demanded, and pi_prim, k2, mu, sigma, eps, q_max, d_com, eta/q/omega and the whole rarity-confidence / silence-composition-blind structure are untouched.\n\nWhy this is the right single move. Every lever tried in the three rejected rounds (lam x1, pi_prim, k2, authority lead, rho) changes the NUMERATOR or the ceiling; none changes how drive scales with the NUMBER of speaking experts, yet the standing residual is uniformly 'confidently decided on multi-speaker conflict displays (Exp 3, 4, 9, 11, 14, 17, 18) while already calibrated on few-speaker displays (Exp 5, 8, 10, 13, 15)'. kappa is the one parameter that separates those two families, and kappa = 1 recovers the accepted base exactly, so this is a smooth one-parameter extension inside the prescribed family, not a mechanism swap.\n\nTraced magnitudes at kappa=1.17, k2=0.48, beta=8.43:\n- single strong cue (W~0.90, den=1.38): den^0.17 = 1.056, drive damped 5.3%, exactly offset by the +5.4% beta re-centring -> Exp 13/15 m_oracle and Exp 8 strong arm essentially unchanged.\n- single weak cue (W~0.055, den=0.535): den^0.17 = 0.897, drive AMPLIFIED ~11% (plus beta) -> Exp 8's weak arm rises and the reversal deepens from -0.184 toward the real -0.203; Exp 15's m_weak rises from ~0.80 to ~0.84 (traced: z 1.387 -> 1.626 at c=0.487), so the Exp 15 composite (m_oracle + m_weak - m_top) moves UP from 0.778 toward the real 0.855 rather than down. This is the opposite sign to the k2 cut that was rejected in iter 5, which amplified m_top as well.\n- 4-6 speaker displays (W~2.0-2.6, den~2.5-3.1): den^0.17 ~ 1.17-1.22, net damping ~12-16% after the beta re-centring -> Exp 11 c8 (three cheap cues vs the oracle, currently ~.70 where humans are ~.5) and c4 come off the ceiling while c5 is damped least (only two speakers), so the composite 0.922 should fall toward ~0.80; Exp 9 (.333 -> ~.37), Exp 4 (.405 -> ~.44), Exp 3 (.468 -> ~.49), Exp 14 (.202 -> ~.23), Exp 17 (+0.091 -> ~+0.06) and Exp 18 (+0.074 -> ~+0.05) all move the right way, since all of those residuals are 'too decisive against the coalition/weak side on a crowded panel'.\n- Exp 2 risk (its conflict cells have 4 speakers): traced the 1-vs-3 cell A=[0,0,0,0,0,1] against B=[1,1,1,0,0,0]: drive -0.2855 -> -0.2394, blended p(tally winner) 0.741 -> 0.727, i.e. ~-0.015 on the metric (.7095 -> ~.695). That is why I chose kappa centre 1.17 with a lower bound of 1.10 rather than the critic's 1.18/[1.08,1.28]: the subject-level spread keeps roughly a third of the population at kappa <= 1.14 where Exp 2 is protected, while the Exp 11/9/4/14/17/18 gains (collectively ~0.5 of absolute residual) dwarf the ~0.015 Exp 2 cost.\n- dk=0 anchor cells (Exp 5, Exp 10) are 4-speaker, so they are damped ~12%; Exp 5 is currently 0.840 against a real 0.820 (so a small drop is a GAIN) and Exp 10 is 0.750 against 0.850 (a small loss). Net on those two is roughly neutral, and mu/sigma/k2 are deliberately left frozen so the attribution of this round stays clean: kappa is the only structural change."
}
```

## Usage

```json
{
  "input_tokens": 77162,
  "output_tokens": 10982
}
```
