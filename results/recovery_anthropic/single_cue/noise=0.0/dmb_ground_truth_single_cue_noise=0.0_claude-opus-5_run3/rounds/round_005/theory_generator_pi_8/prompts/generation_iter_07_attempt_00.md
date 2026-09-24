# generation_iter_07_attempt_00

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_7`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 entirely with an ATTENTION-LOTTERY / LIMITED-SAMPLE COMPARISON account that is a real rival to pi_7 rather than another serial-stop integrator.

Core claims to propose:
1. On each trial the subject does NOT integrate all rows. They draw a small, noisy sample of rows to actually compare (effective sample size k ~ 2-3 of n), with sampling probability p_j proportional to a salience term (U-shaped over the literal screen: exp(-j/tau)+exp(-(n-1-j)/tau)) TIMES a mildly compressed diagnosticity term ((v_j-.5)/.5)^rho with rho ~ 0.6-1.0. Position is therefore always active, not gated " but its behavioural *consequence* is largest exactly when the validity-weighted evidence is near-balanced, because that is when the identity of the sampled rows flips the verdict. This reproduces Exp12's .85 (2-vs-2 deadlock resolved by the salient extremes) and Exp11's ~.49 (2-vs-1 with a .95 opponent: the oracle is very likely to be sampled and cancels the extremes) WITHOUT any discrete deadlock gate, and it predicts the Exp11/Exp12 non-monotonicity as a smooth consequence of evidence balance rather than of |k_A-k_B| exactly equalling zero.
2. Choice = sign of the sampled weighted difference, with ties broken by the most salient sampled row. Because the sample is small and random, the mapping from true evidence to P(choice) is COMPRESSIVE with a ceiling near .82-.88 even for landslides: this is the key calibration difference from pi_7, which over-predicts Exp1 (1.153 vs .513), Exp4 (.851 vs .775) and Exp2 (.778 vs .665) whenever its unanimity and majority bonuses stack. No unanimity bonus, no additive majority bonus " numerosity helps only because more rows on one side means the sample is more likely to be dominated by that side (which already yields Exp4 ~.77 and Exp3 ~+.09).
3. Silence-as-sensitivity is retained but as a *sampling* effect plus a gain discount: rows on which both products agree are still sampled (they compete for attention), so when most of the panel is silent the sample often contains no discriminating row and the subject guesses. This reproduces the reversed single-cue validity ladder (Exp10 = -.20: a lone .95 speaker amid silence is discounted more than a lone .60 speaker) and predicts a near-zero density/load effect (Exp9 ~ 0 to +.05), and it should be made indifferent to whether the silent rows are both-1 or both-0 only up to a small weight omega, so the theory is not forced to an exact null on the silence-composition triads.
4. Individual differences are unimodal jitter in k (sample size), tau (salience width), rho, and a lapse rate; no strategy subpopulations, no learning.

Calibration targets the new model must hit simultaneously: Exp1 ~0.51, Exp2 ~0.67, Exp3 ~+0.09, Exp4 ~0.78, Exp5 ~0.52, Exp6 ~0.51, Exp7 ~0.82, Exp8 ~0.84, Exp9 ~+0.05, Exp10 ~-0.20, Exp11 ~0.49, Exp12 ~0.85. In particular it must be MORE decisive than pi_7 on Exp8 (extremes + numerosity vs the mid-list oracle) and LESS decisive than pi_7 on Exp1/Exp4 (stacked unanimity/majority cells) " that pair of constraints is what the compressive small-sample read-out buys and is the sharpest empirical wedge against pi_7.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.648)

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


---

### `pi_6` (overall score: 0.512)

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


---

### `pi_2` (overall score: 0.238)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=-0.6931 (var=0.0366)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.8660 (var=0.0083)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.1400 (var=0.0151)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.1450 (var=0.0133)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.4850 (var=0.0117)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.8825 (var=0.0110)
- Experiment 7: real=0.0533 (var=0.0494) vs this=-0.0050 (var=0.0221)
- Experiment 8: real=-0.2025 (var=0.0220) vs this=0.0037 (var=0.0166)
- Experiment 9: real=0.4900 (var=0.0105) vs this=0.4817 (var=0.0049)
- Experiment 10: real=0.8500 (var=0.0111) vs this=0.5083 (var=0.0237)
- Experiment 11: real=0.5133 (var=0.0520) vs this=0.5383 (var=0.0337)
- Experiment 12: real=0.6650 (var=0.0058) vs this=0.6850 (var=0.0078)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1392 -> ACCEPTED
- iter 2: loss=0.1595 -> REJECTED
- iter 3: loss=0.1285 -> ACCEPTED
- iter 4: loss=0.1310 -> REJECTED
- iter 5: loss=0.1203 -> ACCEPTED
- iter 6: loss=0.1256 -> REJECTED
- iter 7: loss=0.1230 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.1203 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.0946 (var=0.0084) (Δ vs real -0.1879)
  - iter 2: -0.0942 (var=0.0129) (Δ vs real -0.1875)
  - iter 3: 0.0640 (var=0.0112) (Δ vs real -0.0294)
  - iter 4: 0.0310 (var=0.0109) (Δ vs real -0.0623)
  - iter 5: 0.0421 (var=0.0084) (Δ vs real -0.0512)
  - iter 6: 0.0060 (var=0.0097) (Δ vs real -0.0873)
  - iter 7 (most recent): 0.0373 (var=0.0079) (Δ vs real -0.0560)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0038 (var=0.0080)
- pi_2: -0.6931 (var=0.0366)
- pi_3: 0.1027 (var=0.0215)
- pi_4: -0.0050 (var=0.0008)
- pi_5: -0.1875 (var=0.0052)
- pi_6: 0.0110 (var=0.0029)
- pi_7: 0.0921 (var=0.0072)

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
  - iter 1: 0.7215 (var=0.0051) (Δ vs real -0.0535)
  - iter 2: 0.6940 (var=0.0070) (Δ vs real -0.0810)
  - iter 3: 0.7760 (var=0.0062) (Δ vs real +0.0010)
  - iter 4: 0.7515 (var=0.0035) (Δ vs real -0.0235)
  - iter 5: 0.7545 (var=0.0040) (Δ vs real -0.0205)
  - iter 6: 0.7650 (var=0.0035) (Δ vs real -0.0100)
  - iter 7 (most recent): 0.7755 (var=0.0054) (Δ vs real +0.0005)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8660 (var=0.0083)
- pi_1: 0.1530 (var=0.0129)
- pi_3: 0.7780 (var=0.0646)
- pi_4: 0.7860 (var=0.0029)
- pi_5: 0.7755 (var=0.0037)
- pi_6: 0.7530 (var=0.0025)
- pi_7: 0.8510 (var=0.0036)

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
  - iter 1: 0.4512 (var=0.0155) (Δ vs real -0.0662)
  - iter 2: 0.4338 (var=0.0165) (Δ vs real -0.0837)
  - iter 3: 0.4725 (var=0.0203) (Δ vs real -0.0450)
  - iter 4: 0.4163 (var=0.0159) (Δ vs real -0.1012)
  - iter 5: 0.4487 (var=0.0164) (Δ vs real -0.0687)
  - iter 6: 0.4400 (var=0.0141) (Δ vs real -0.0775)
  - iter 7 (most recent): 0.4012 (var=0.0191) (Δ vs real -0.1162)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8562 (var=0.0143)
- pi_3: 0.0375 (var=0.0022)
- pi_2: 0.1400 (var=0.0151)
- pi_4: 0.5200 (var=0.0187)
- pi_5: 0.6262 (var=0.0216)
- pi_6: 0.4450 (var=0.0192)
- pi_7: 0.4537 (var=0.0136)

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
  - iter 1: 0.3362 (var=0.0102) (Δ vs real -0.1712)
  - iter 2: 0.3794 (var=0.0101) (Δ vs real -0.1281)
  - iter 3: 0.3481 (var=0.0086) (Δ vs real -0.1594)
  - iter 4: 0.3412 (var=0.0095) (Δ vs real -0.1662)
  - iter 5: 0.3675 (var=0.0062) (Δ vs real -0.1400)
  - iter 6: 0.3531 (var=0.0071) (Δ vs real -0.1544)
  - iter 7 (most recent): 0.3738 (var=0.0103) (Δ vs real -0.1337)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0269 (var=0.0009)
- pi_1: 0.8475 (var=0.0132)
- pi_2: 0.1450 (var=0.0133)
- pi_4: 0.4906 (var=0.0065)
- pi_5: 0.4756 (var=0.0085)
- pi_6: 0.3663 (var=0.0121)
- pi_7: 0.4238 (var=0.0074)

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
  - iter 1: 0.8087 (var=0.0082) (Δ vs real -0.0112)
  - iter 2: 0.6987 (var=0.0125) (Δ vs real -0.1212)
  - iter 3: 0.8825 (var=0.0049) (Δ vs real +0.0625)
  - iter 4: 0.8850 (var=0.0071) (Δ vs real +0.0650)
  - iter 5: 0.7987 (var=0.0111) (Δ vs real -0.0212)
  - iter 6: 0.8075 (var=0.0104) (Δ vs real -0.0125)
  - iter 7 (most recent): 0.8100 (var=0.0083) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3500 (var=0.0175)
- pi_3: 0.9513 (var=0.0036)
- pi_1: 0.8250 (var=0.0169)
- pi_2: 0.4850 (var=0.0117)
- pi_5: 0.8325 (var=0.0087)
- pi_6: 0.7913 (var=0.0231)
- pi_7: 0.8175 (var=0.0092)

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
  - iter 1: 0.7075 (var=0.0037) (Δ vs real -0.1338)
  - iter 2: 0.7100 (var=0.0058) (Δ vs real -0.1313)
  - iter 3: 0.6981 (var=0.0062) (Δ vs real -0.1431)
  - iter 4: 0.7025 (var=0.0057) (Δ vs real -0.1388)
  - iter 5: 0.7150 (var=0.0077) (Δ vs real -0.1263)
  - iter 6: 0.7188 (var=0.0049) (Δ vs real -0.1225)
  - iter 7 (most recent): 0.7063 (var=0.0063) (Δ vs real -0.1350)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9694 (var=0.0014)
- pi_4: 0.3312 (var=0.0055)
- pi_1: 0.1388 (var=0.0136)
- pi_2: 0.8825 (var=0.0110)
- pi_5: 0.8669 (var=0.0043)
- pi_6: 0.7238 (var=0.0070)
- pi_7: 0.6869 (var=0.0059)

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
  - iter 1: 0.0600 (var=0.0300) (Δ vs real +0.0067)
  - iter 2: 0.1700 (var=0.0386) (Δ vs real +0.1167)
  - iter 3: 0.0933 (var=0.0230) (Δ vs real +0.0400)
  - iter 4: 0.0650 (var=0.0501) (Δ vs real +0.0117)
  - iter 5: 0.0417 (var=0.0412) (Δ vs real -0.0117)
  - iter 6: 0.0433 (var=0.0428) (Δ vs real -0.0100)
  - iter 7 (most recent): 0.0083 (var=0.0551) (Δ vs real -0.0450)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8050 (var=0.0193)
- pi_3: 0.0033 (var=0.0067)
- pi_1: 0.0017 (var=0.0171)
- pi_2: -0.0050 (var=0.0221)
- pi_4: 0.0283 (var=0.0260)
- pi_6: -0.0050 (var=0.0446)
- pi_7: -0.0733 (var=0.0466)

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
  - iter 1: -0.1937 (var=0.0135) (Δ vs real +0.0088)
  - iter 2: -0.1650 (var=0.0250) (Δ vs real +0.0375)
  - iter 3: -0.2137 (var=0.0203) (Δ vs real -0.0112)
  - iter 4: -0.2075 (var=0.0266) (Δ vs real -0.0050)
  - iter 5: -0.2075 (var=0.0290) (Δ vs real -0.0050)
  - iter 6: -0.1925 (var=0.0248) (Δ vs real +0.0100)
  - iter 7 (most recent): -0.2137 (var=0.0319) (Δ vs real -0.0112)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0062 (var=0.0039)
- pi_5: 0.3350 (var=0.0184)
- pi_1: 0.0063 (var=0.0080)
- pi_2: 0.0037 (var=0.0166)
- pi_4: 0.2738 (var=0.0184)
- pi_6: -0.1838 (var=0.0208)
- pi_7: -0.2100 (var=0.0192)

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
  - iter 1: 0.6983 (var=0.0078) (Δ vs real +0.2083)
  - iter 2: 0.6625 (var=0.0123) (Δ vs real +0.1725)
  - iter 3: 0.6992 (var=0.0095) (Δ vs real +0.2092)
  - iter 4: 0.6450 (var=0.0074) (Δ vs real +0.1550)
  - iter 5: 0.6842 (var=0.0079) (Δ vs real +0.1942)
  - iter 6: 0.7075 (var=0.0106) (Δ vs real +0.2175)
  - iter 7 (most recent): 0.7067 (var=0.0074) (Δ vs real +0.2167)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4325 (var=0.0102)
- pi_3: 0.9692 (var=0.0013)
- pi_1: 0.1575 (var=0.0113)
- pi_2: 0.4817 (var=0.0049)
- pi_4: 0.1583 (var=0.0049)
- pi_5: 0.4042 (var=0.0068)
- pi_7: 0.4583 (var=0.0105)

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
  - iter 1: 0.7900 (var=0.0123) (Δ vs real -0.0600)
  - iter 2: 0.7333 (var=0.0117) (Δ vs real -0.1167)
  - iter 3: 0.8233 (var=0.0146) (Δ vs real -0.0267)
  - iter 4: 0.7667 (var=0.0122) (Δ vs real -0.0833)
  - iter 5: 0.8417 (var=0.0109) (Δ vs real -0.0083)
  - iter 6: 0.7767 (var=0.0148) (Δ vs real -0.0733)
  - iter 7 (most recent): 0.7917 (var=0.0112) (Δ vs real -0.0583)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9717 (var=0.0024)
- pi_6: 0.2300 (var=0.0174)
- pi_1: 0.1217 (var=0.0142)
- pi_2: 0.5083 (var=0.0237)
- pi_4: 0.0083 (var=0.0006)
- pi_5: 0.1233 (var=0.0134)
- pi_7: 0.8817 (var=0.0095)

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
  - iter 1: 0.8433 (var=0.0674) (Δ vs real +0.3300)
  - iter 2: 0.7217 (var=0.0463) (Δ vs real +0.2083)
  - iter 3: 0.9150 (var=0.0412) (Δ vs real +0.4017)
  - iter 4: 0.8633 (var=0.0360) (Δ vs real +0.3500)
  - iter 5: 0.8833 (var=0.0422) (Δ vs real +0.3700)
  - iter 6: 0.8417 (var=0.0473) (Δ vs real +0.3283)
  - iter 7 (most recent): 0.7867 (var=0.0375) (Δ vs real +0.2733)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4233 (var=0.0611)
- pi_7: 1.1533 (var=0.0393)
- pi_1: 0.1667 (var=0.0469)
- pi_2: 0.5383 (var=0.0337)
- pi_3: 1.4033 (var=0.2137)
- pi_4: 0.2050 (var=0.0273)
- pi_5: 0.1667 (var=0.0303)

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
  - iter 1: 0.4517 (var=0.0111) (Δ vs real -0.2133)
  - iter 2: 0.4800 (var=0.0100) (Δ vs real -0.1850)
  - iter 3: 0.4350 (var=0.0099) (Δ vs real -0.2300)
  - iter 4: 0.4550 (var=0.0064) (Δ vs real -0.2100)
  - iter 5: 0.4067 (var=0.0092) (Δ vs real -0.2583)
  - iter 6: 0.4417 (var=0.0071) (Δ vs real -0.2233)
  - iter 7 (most recent): 0.4367 (var=0.0129) (Δ vs real -0.2283)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7775 (var=0.0103)
- pi_6: 0.1783 (var=0.0069)
- pi_1: 0.1342 (var=0.0079)
- pi_2: 0.6850 (var=0.0078)
- pi_3: 0.6025 (var=0.0521)
- pi_4: 0.0167 (var=0.0006)
- pi_5: 0.3217 (var=0.0130)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
ATTENTION LOTTERY WITH LIMITED-SAMPLE COMPARISON, AUTHORITY-CONTINGENT CAPTURE, AND PANEL-VOLUME CONFIDENCE. (1) Each display row enters an attention lottery with weight att_j = s_j * x_j * pop_j, where s_j = exp(-j/tau) + gamma*exp(-(n-1-j)/tau) is literal screen salience (asymmetric: top of panel stronger than bottom), x_j = ((v_j-.5)/.5)^rho is mildly compressed stated diagnosticity (rho ~ .55-.85), and pop_j is a pop-out factor: rows on which the two columns are identical do not pop out and keep only a residual share delta, while the FIRST row at which the columns disagree gets a bonus. (2) That first-mismatch bonus is AUTHORITY-CONTINGENT: a row captures attention outright only if it is both the first thing that discriminates AND the most diagnostic discriminating row on the screen (pop = 1 + phi + phi_auth*sigmoid(s_auth*(x_j1/max_other_x - 1))). Position is always active, never gated by a deadlock rule. (3) The subject does not integrate the panel: only k = clip(ceil(k_frac*m), 2, 3) rows are actually compared, drawn without replacement with probability proportional to att. Choice = the attention-weighted sign of the sampled differences plus a small within-sample count term; a 1-1 sample is decided by the more salient sampled row. Numerosity helps only because more rows on one side makes the sample more likely to be dominated by that side; there is no unanimity bonus, no additive majority bonus, no serial stop rule. Because the sample is small and random, the map from true evidence to P(choice) is compressive with a ceiling near .85-.90 even for landslides. (4) Silence is read as sensitivity and discounts read-out gain: c = 1 - eta*sigmoid(kappa*(f_sil - f0))*vbar_speakers, floored at c_min, with mutual endorsements counted at weight omega ~ .88 so density/load is nearly inert; this yields the REVERSED single-cue validity ladder. (5) NEW: CONFIDENCE SCALES WITH THE VOLUME OF THE PANEL, NOT WITH THE AMOUNT ACTUALLY READ. The subject still compares only 2-3 rows, but the felt informativeness of that verdict rises with how many experts were displayed: the read-out gain is beta_eff = beta * (n/6)^zeta with zeta ~ 1-2. This is the classic illusion-of-knowledge dissociation (confidence tracks information volume while the sampling process, and hence accuracy, does not). Its behavioural signature is that SHORT panels (n=5) are systematically less decisive than structurally identical cells on WIDE panels (n=6-7): the same 2-vs-2 screen-extreme deadlock, the same 2-against-the-oracle conflict, and the same three-mid-experts-vs-one-oracle cell all produce choices markedly closer to a coin flip when only five experts are on screen, even though the sampled comparison is the same size. (6) Individual differences are unimodal jitter in k_frac, tau, gamma, rho, phi, phi_auth, s_auth, beta, zeta, eta and a lapse rate; no strategy subpopulations, no learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from itertools import combinations, permutations

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
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _pget(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    tau = _pget('tau', 0.85, 0.2, 8.0)
    gam = _pget('gamma_rec', 0.50, 0.0, 1.5)
    rho = _pget('rho', 0.70, 0.05, 4.0)
    dlt = _pget('delta_sil', 0.07, 0.0, 1.0)
    phi = _pget('phi_first', 0.50, 0.0, 3.0)
    phi_auth = _pget('phi_auth', 6.0, 0.0, 20.0)
    s_auth = _pget('s_auth', 10.0, 0.5, 60.0)
    lam = _pget('lam_w', 0.88, 0.0, 1.0)
    beta = _pget('beta', 2.40, 0.05, 40.0)
    zet = _pget('zeta_n', 1.50, 0.0, 4.0)
    eta = _pget('eta', 1.45, 0.0, 5.0)
    f0 = _pget('f0', 0.365, 0.0, 1.0)
    kap = _pget('kap_s', 25.0, 1.0, 80.0)
    ome = _pget('omega', 0.88, 0.0, 1.0)
    cmin = _pget('c_min', 0.10, 0.0, 0.5)
    kfrac = _pget('k_frac', 0.47, 0.05, 1.0)
    eps = _pget('epsilon', 0.035, 0.0, 0.5)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. attention weights: screen salience x compressed diagnosticity,
    #    matched rows do not pop out (residual share delta), first
    #    mismatching row of the top-down scan gets a pop-out bonus that
    #    is AUTHORITY-CONTINGENT: it captures attention outright only if
    #    no more diagnostic expert further down the panel also speaks.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s = np.exp(-pos / tau) + gam * np.exp(-(n - 1.0 - pos) / tau)
    x = np.power(vs, rho)
    att = s * x
    att = np.where(disc, att, dlt * att)
    idxs = np.flatnonzero(disc)
    att = np.array(att, dtype=float)
    j1 = int(idxs[0])
    if idxs.shape[0] > 1:
        x_rest = float(np.max(x[idxs[1:]]))
        ratio = float(x[j1]) / max(x_rest, 1e-9)
        q_auth = float(sig(s_auth * (ratio - 1.0)))
    else:
        q_auth = 1.0
    att[j1] = att[j1] * (1.0 + phi + phi_auth * q_auth)
    att = np.clip(att, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. silence-as-sensitivity gain discount (sharp threshold, floored)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    n_both0 = float(np.count_nonzero((~disc) & (~both1)))
    n_both1 = float(np.count_nonzero(both1))
    f_sil = float(np.clip((n_both0 + ome * n_both1) / float(n), 0.0, 1.0))
    xw = x[idxs]
    sxw = float(np.sum(xw))
    vbar = float(np.sum(xw * vs[idxs]) / sxw) if sxw > 1e-12 else 0.5
    c = 1.0 - eta * float(sig(kap * (f_sil - f0))) * vbar
    c = float(np.clip(c, cmin, 1.0))

    # ------------------------------------------------------------------
    # 5b. PANEL-VOLUME CONFIDENCE: the sampled comparison stays the same
    #     size, but the felt informativeness of its verdict grows with
    #     how many experts were displayed (illusion of knowledge).
    # ------------------------------------------------------------------
    beta_eff = beta * float(np.power(float(n) / 6.0, zet))

    # ------------------------------------------------------------------
    # 6. the lottery: draw k rows without replacement, prob ~ attention
    # ------------------------------------------------------------------
    m = int(idxs.shape[0])
    k = int(np.clip(int(np.ceil(kfrac * m)), 2, 3))
    k = int(min(k, n))
    if k < 1:
        k = 1

    p = att / float(np.sum(att))
    p = 0.998 * p + 0.002 / float(n)
    p = p / float(np.sum(p))

    tot = 0.0
    pa = 0.0
    for S in combinations(range(n), k):
        ps = 0.0
        for perm in permutations(S):
            q = 1.0
            rem = 1.0
            good = True
            for t in perm:
                if rem <= 1e-9:
                    good = False
                    break
                q *= p[t] / rem
                rem -= p[t]
            if good:
                ps += q
        if ps <= 1e-14:
            continue
        sel = [j for j in S if disc[j]]
        if len(sel) == 0:
            E = 0.0
        else:
            ws = att[sel]
            sg = np.sign(d[sel])
            wsum = float(np.sum(ws))
            Ew = float(np.sum(ws * sg) / wsum) if wsum > 1e-12 else 0.0
            nA = float(np.sum(sg > 0))
            nB = float(np.sum(sg < 0))
            Ec = (nA - nB) / (nA + nB) if (nA + nB) > 0 else 0.0
            E = lam * Ew + (1.0 - lam) * Ec
        pa += ps * float(sig(beta_eff * c * E))
        tot += ps

    p_a = (pa / tot) if tot > 1e-12 else 0.5

    pr = np.array([p_a, 1.0 - p_a], dtype=float)
    pr = (1.0 - eps) * pr + eps * 0.5
    pr = np.clip(pr, 1e-12, None)
    pr = pr / pr.sum()
    return pr

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
- tau: [0.72, 1.00]
- gamma_rec: [0.38, 0.62]
- rho: [0.55, 0.85]
- delta_sil: [0.04, 0.11]
- phi_first: [0.30, 0.70]
- phi_auth: [4.0, 9.0]
- s_auth: [6.0, 16.0]
- lam_w: [0.82, 0.95]
- beta: [2.10, 2.70]
- zeta_n: [1.0, 2.0]
- eta: [1.30, 1.60]
- f0: [0.345, 0.385]
- kap_s: [18.0, 32.0]
- omega: [0.80, 0.94]
- c_min: [0.07, 0.13]
- k_frac: [0.40, 0.55]
- epsilon: [0.02, 0.06]
- validities: validities

`rationale`: MINIMAL DIFF on the accepted iter-3 base (loss 0.1285): every line is re-emitted verbatim except one added read-out-gain line (beta_eff = beta*(n/6)^zeta) and one new parameter (zeta_n). phi_first, phi_auth, s_auth, beta, omega, eta, f0, delta, rho, tau, gamma, lam_w, k_frac and the sampling rule are untouched, exactly as the critic demanded after the iter-4 rejection.

WHY NOT THE PRESCRIBED COUNT-TERM EDIT. I traced the mechanism analytically before spending the edit and it cannot deliver what the critic wants, which is why iter-4 moved Exp6 by +0.005 and Exp12 by +0.020 despite firing. The sample is drawn with probability PROPORTIONAL TO ATTENTION, so the within-sample count is not an independent numerosity signal: it inherits the salience bias of the sampling distribution. Worked example, Exp9 cell-8 (2 screen extremes vs 3 mid rows, m=5, base already uses k=3): the row-0 attention share after the pop-out is ~0.55, so the modal samples are {0,1,5}, {0,2,5}, {0,1,2} whose internal counts are 2-1 or 1-2 with the EXTREME side over-represented relative to the panel's 2-vs-3 — adding nu*Ec therefore pushes the verdict further TOWARD the extremes in exactly the cell that needs the opposite. Widening the trigger or raising lam_wide amplifies this backwards effect while simultaneously pushing Exp2 past the .82 ceiling the arbiter forbade (Exp2's conflict trials are 5-vs-1 and 3-vs-1 count-majorities that all fire the rule). Genuine numerosity would require flattening the SAMPLING distribution, but that immediately destroys Exp10 (2 weak extremes must beat .95/.86 at .85) — Exp10-cell5 and Exp9-cell8 are structurally identical apart from the count, so no lottery-internal reweighting separates them. I am reporting this so the next round does not spend a third edit on that knob.

WHAT I DID INSTEAD, AND WHY IT IS ATTRIBUTABLE. Sorting the 12 residuals by panel width reveals a clean, previously unnoticed regularity: every n=5 experiment whose cell is a conflict is OVER-decisive (Exp5 .883 vs .820, Exp11 index .915 vs .513, Exp4 .348 vs .508 i.e. over-decisive for the anti-TTB side), while every n=6 experiment is at or below human decisiveness (Exp2, Exp3, Exp6, Exp10 all under). The base has a single global gain, so it cannot express this. The edit makes read-out gain grow with the number of experts DISPLAYED while the sampled comparison stays 2-3 rows — the standard illusion-of-knowledge dissociation (confidence tracks information volume, accuracy tracks the sample). It is a read-out-gain modulation of exactly the same kind as the existing silence discount c, so the mechanism family (salience x compressed-diagnosticity lottery, k=2-3 rows, compressive read-out, silence-as-sensitivity; no unanimity bonus, no majority bonus, no deadlock gate, no serial stop) is untouched.

PREDICTED, HAND-SIMULATED EFFECTS (zeta~1.5 gives a gain factor .75 at n=5, 1.0 at n=6, 1.28 at n=7). It is EXACTLY ORTHOGONAL to every cell the critic told me to protect: Exp2, Exp3, Exp6, Exp7, Exp9, Exp10 are all n=6 and move by zero. Gains: Exp5 .883 -> ~.83 (target .820, +.05), Exp4 .348 -> ~.38 (+.03), Exp11 index .915 -> ~.81 via c4 .78->.72, c8 .83->.77, c5 .73->.68 (+.10). Costs, both small and bounded: Exp1 .064 -> ~.050 (-.014) and Exp8 -.214 -> ~-.18 (-.019). Exp12 (n=7) is a near-wash because its pooled value averages a high extremes cell (which rises) and a low four-against-the-oracle cell (which falls): .435 -> ~.43. Net expected improvement ~+0.15 summed absolute deviation, i.e. aggregate loss ~0.116, comfortably below the 0.1285 floor, with the downside capped at ~0.033 because only two of the twelve experiments can lose and each by less than 0.02.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a faithful implementation of the arbiter's attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no discrete deadlock gate, compressive read-out), and it was accepted by the gate at loss 0.1392. It clearly wins on the metrics the arbiter designated as the wedge against pi_7: Exp8 (-0.194 vs -0.203 real; pi_7 -0.210), Exp7 (+0.060 vs +0.053 real; pi_7 -0.073), Exp5 (0.809 vs 0.820), Exp10 (0.790 vs 0.850), Exp3 (0.451 vs 0.518), Exp2 (0.722 vs 0.775 " no longer the .851 overshoot of pi_7), and Exp11 is much closer than pi_7 (0.843 vs real 0.513; pi_7 1.153). So the compressive small-sample read-out plus silence-gain discount is doing the intended work.

However, a coherent set of residuals remains, and they point in a single direction: screen salience is currently over-weighted relative to diagnosticity in the READ-OUT, while numerosity is under-weighted on large panels. (a) Exp9: 0.698 vs 0.490 real " the model follows the two screen-extreme cues (pos0 .72, pos5 .70) far more than humans do, even though humans go strongly with extremes in Exp10 (.62/.58 extremes vs .95/.86 mid). The empirical contrast between Exp10 (0.85) and Exp9 (0.49) is therefore NOT a pure position effect: it tracks how strong the OPPOSING mid-list cues are (heavyweights get discounted, per the reversed ladder of Exp8), which the model currently reproduces mostly via position rather than via the validity-discount. (b) Exp11 index is +0.33 too high, again driven by the c4 extremes-vs-heavyweights cell being too decisive. (c) Exp4: 0.336 vs 0.508 " the model chooses against TTB too often precisely in cells where both scanning orientations oppose TTB, i.e. position beats validity too easily. (d) Exp1: sign flip, -0.095 vs +0.093 (pi_7 got +0.092): on tally-vs-TTB conflict trials the model is slightly LESS TTB-like than on agreement trials, whereas humans are slightly more. The first-mismatch pop-out (phi) is evidently not enough, or is being washed out by the att-weighted read-out. (e) Exp12 (0.452 vs 0.665) and Exp6 (0.708 vs 0.841) are both too indecisive on many-row panels " with k capped at 3 and lam_w near 0.9 (weighted sign dominating the count term), a 4- or 5-row numerical majority barely moves the sampled verdict on the 7-feature panel.
Rationale: Keep the prescribed attention-lottery / limited-sample family exactly as implemented " it is faithful and it is the running best " but tune three in-family knobs on top of this accepted base:

1. Rebalance read-out between salience and diagnosticity. Right now the read-out weight IS the attention weight (att_j), which makes position dominate validity. Introduce a single blend exponent/mixture on the read-out only: w_read_j = att_j^psi * x_j^(1-psi) (or w_read = mu*att + (1-mu)*x, mu ~ 0.55-0.75), leaving the SAMPLING distribution untouched. This should pull Exp9 down from .70 toward ~.50 and Exp4 up from .34 toward ~.50 without hurting Exp10/Exp5, because in Exp10 the opposing mid-list cues are the heavyweights that the silence/validity discount already suppresses. Sanity-check that Exp10 stays >= .80 and Exp5 stays ~.82 as you move psi/mu.

2. Restore numerosity on wide panels. Allow k to grow with m (e.g. k = clip(ceil(k_frac*m), 2, 4) rather than capping at 3) and/or lower lam_w toward ~0.70-0.80 so the within-sample count term has more say. This is the only in-family route to Exp12 (.452 -> ~.66) and Exp6 (.708 -> ~.84) without reintroducing a majority bonus; verify it does not push Exp11's c4/c8 back up (raise the validity-discount rather than position weight if it does).

3. Fix the Exp1 sign. The conflict-vs-agreement contrast is currently -0.09 where humans are +0.09. Strengthen the first-mismatch pop-out (phi_first range upward, e.g. 0.6-1.1) and/or give the pop-out row a read-out bonus as well as a sampling bonus, so that on tally-vs-TTB conflict trials the top discriminating row is slightly more likely to carry the verdict. Watch Exp2 (should stay ~.72-.78, not drift back toward the TTB side) while doing this.

Do NOT add a unanimity bonus, a deadlock gate, or a serial stopping rule " those are outside the prescribed family and are exactly what the arbiter asked to remove. Target residuals: Exp1 ~+0.09, Exp4 ~0.51, Exp6 ~0.84, Exp9 ~0.49, Exp11 ~0.51, Exp12 ~0.67, holding Exp5/7/8/10 at their current (good) values.

**Outcome of this advice:** iter 1 candidate loss=0.1392 -> iter 2 candidate loss=0.1595 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate is still faithful to the arbiter's attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop) " so faithfulness is not the problem. The problem is calibration: the bundle of five simultaneous edits moved the wrong global knob and the gate rejected it (0.1595 vs 0.1392).

Decomposing iter1 -> iter2 by experiment tells a clean story. The edits that HELPED are exactly the ones my previous critique's item (1) predicted: the read-out/sampling split (mu blend of att and raw diagnosticity) pulled Exp9 0.698 -> 0.663 (real .49), Exp11 0.843 -> 0.722 (real .513) and Exp4 0.336 -> 0.379 (real .508) " all three in the right direction, all three still short. The edits that HURT are the ones that lowered global decisiveness or perturbed the silence composition: Exp5 collapsed 0.809 -> 0.699 (real .820, a -0.12 regression on a previously near-perfect cell), Exp10 0.790 -> 0.733 (real .850), Exp2 0.722 -> 0.694 (real .775, now the wrong side by -0.08), Exp3 0.451 -> 0.434 (real .518), Exp8 -0.194 -> -0.165 (real -.203), and Exp7 blew out 0.060 -> 0.170 (real .053) " the load contrast, which was essentially perfect at iter1. The net is negative.

The common cause of the regressions is the beta reduction (2.10-2.70 -> 1.60-2.15): a uniform gain cut deflates EVERY cell, including the ones (Exp5, Exp10, Exp2, Exp6, Exp8) where the model was already at or below the human decisiveness. The arbiter's 'compressive ceiling near .82-.88' is already delivered by the small-sample lottery itself; it does not need to be bought a second time with a lower softmax gain. The omega reduction (.80-.94 -> .55-.75) is the likely cause of the Exp7 load-contrast blow-up (shared 1s became too inert-free, making hi-load panels more decisive than lo-load ones), and it did not buy the promised Exp6/Exp12 gains: Exp6 moved 0.708 -> 0.710 (still -0.13 off) and Exp12 only 0.452 -> 0.480 (still -0.19 off). The k-cap-to-4 + lam_w reduction also failed to deliver numerosity on wide panels while costing the 2-vs-2 deadlock cells (Exp5) that were supposed to be structurally shielded " they were not, because k can now reach 3-4 on those 4-discriminating-row panels too. Exp1 did not move at all (-0.095 -> -0.094 vs real +0.093), so the phi_first / phi_read pop-out strengthening is not touching that contrast.
Rationale: The candidate was REJECTED by the gate, so continue is not available; the next proposal must be built on the iter-1 accepted base (loss 0.1392), inside the same attention-lottery / limited-sample family (no unanimity bonus, no majority bonus, no deadlock gate, no serial stop).

What my own history says: my iter-1 advice bundled THREE knobs and the proposer added TWO more; the result was rejected. Take a SMALLER step, and keep only the part that demonstrably worked.

1. KEEP the read-out/sampling split (this is the one edit whose per-experiment signature was uniformly correct: Exp9, Exp11, Exp4 all moved toward the data). Carry w_read = mu*att_norm + (1-mu)*x_norm over onto the base, with mu in [0.82, 0.92]. Do not go lower than ~0.80 " Exp10/Exp5/Exp12-cell4 need position to win the 2-vs-2 cells.

2. REVERT beta to the accepted base range (~2.10-2.70). This is the explicit correction to my previous advice: the global gain cut is what destroyed Exp5 (-0.11), Exp10 (-0.06), Exp2 (-0.03) and Exp8 (+0.04) and it bought less on the overshoot cells than it cost. The compression the arbiter asks for should come from the sample size and the silence discount, not from a uniform softmax temperature. Target after revert: Exp5 >= .80, Exp10 >= .78, Exp2 ~.72-.78, Exp8 ~-.19.

3. REVERT omega to the base range (~0.80-0.94). Lowering it is what pushed Exp7 from +0.06 (essentially exact) to +0.17 and it did not deliver the promised Exp6/Exp12 gains. Exp7 is one of the few cells the base got exactly right; protect it.

4. For the remaining large residuals (Exp12 .48 vs .665, Exp6 .71 vs .841), do NOT buy numerosity with a global lam_w cut or a blanket k-cap raise " that is what leaked into Exp5. Make the sample size genuinely panel-size-dependent: allow k = 4 ONLY when the number of discriminating rows m >= 4 AND n >= 6 (Exp12's 7-row panel, Exp6's 6-row panel), and leave k <= 3 on 5-feature panels (Exp1/4/5/8/11). Keep lam_w near the base (>= 0.80) so that 2-row samples remain decided by salience/diagnosticity and the 2-vs-2 deadlock cells are untouched. Verify Exp5 and Exp11-c4 do not move.

5. Exp1 (-0.094 vs real +0.093) is now the largest unaddressed residual and the phi_first/phi_read route did not move it at all " stop pushing that knob. The structural reason the sign is wrong is that on the 1-vs-4 conflict trials the within-sample count term drags the verdict away from the top row, while on the agreement (landslide) trials the model is already near its compressive ceiling. An in-family fix is to make the count term scale with the *number of distinct sampled sides actually compared* rather than applying at full strength in every sample " e.g. down-weight the count term when the sample contains the pop-out top row (that row's read-out already carries the verdict), so conflict trials become slightly MORE first-row-driven than agreement trials. Note that this is consistent with Exp2 staying high (~.775), because Exp2's validities are scrambled so the screen-top row is not the .95 oracle there; a position-driven read-out naturally yields TTB-like behaviour in Exp1 and tally-like behaviour in Exp2. Check both simultaneously.

Acceptance targets for the next attempt (must beat 0.1392): hold Exp5 ~.81, Exp7 ~.06, Exp8 ~-.19, Exp10 ~.79, Exp2 ~.72; improve Exp9 toward ~.55-.60, Exp11 toward ~.70, Exp4 toward ~.42, Exp12 toward ~.55, Exp6 toward ~.78, and get Exp1 at least back to ~0.00 rather than -0.09. Change at most two mechanisms at once so the next round's diagnosis is attributable.

**Outcome of this advice:** iter 2 candidate loss=0.1595 -> iter 3 candidate loss=0.1285 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate is faithful to the arbiter's attention-lottery / limited-sample family (salience x compressed-diagnosticity lottery, k=2-3 rows, compressive read-out, silence-as-sensitivity gain discount, no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop) and it is the new running best (0.1285). The authority-contingent pop-out was a genuinely attributable win and my prior advice to stop pushing phi_first blindly plus to protect beta/omega was vindicated: Exp1 flipped sign from -0.094 to +0.064 (real +0.093) and Exp2 landed essentially exactly (0.776 vs 0.775) " the two cells the edit was predicted to move " while the cells I asked to be protected held or improved (Exp5 0.883 vs .820, Exp7 +0.093 vs +0.053, Exp8 -0.214 vs -0.203, Exp10 0.823 vs .850, Exp3 0.473 vs .518). The wedge against pi_7 that the arbiter asked for is intact: less decisive than pi_7 on Exp1/Exp2 stacked cells, essentially tied on Exp8/Exp10, and much better than pi_7 on Exp7.

The residual structure is now narrow and coherent, and it is all on one axis: NUMEROSITY / AGGREGATION ON WIDE PANELS. Exp12 (0.435 vs 0.665, and it moved the wrong way from iter 1), Exp6 (0.698 vs 0.841), Exp9 (0.699 vs 0.490) and Exp4 (0.348 vs 0.508) are all cells in which a numerical crowd of 3-4 mid-list rows should partly or wholly overturn the two screen extremes, and the model instead lets the 2-row sample be decided by salience almost every time. Exp9's pooled value averages a 2-extremes-vs-1-mid cell with a 2-extremes-vs-3-mid cell; humans sit at chance overall, meaning the 3-cue crowd wins its cell, which the model cannot do with k<=3 and lam_w~0.9. The same deficit explains Exp12's 4-vs-3 cell and Exp6's 3-vs-1 classes. Separately, Exp11's index got WORSE with the authority edit (0.843 -> 0.915 vs real 0.513), which says the authority pop-out is firing on some Exp11 cells where it should not " a side effect worth localising rather than a reason to remove the mechanism.
Rationale: The candidate was ACCEPTED and should be the base going forward " do NOT revert the authority-contingent pop-out, and do NOT change mechanism family (stay in the attention-lottery / limited-sample account: no unanimity bonus, no majority bonus, no deadlock gate, no serial stop). But four residuals >= 0.14 remain, all on the same axis, so one more tightly scoped round is warranted. Change at most TWO things.

1) (PRIMARY, never actually implemented) Make aggregation panel-size-dependent instead of globally weaker. Let k = 4 ONLY when the number of discriminating rows m >= 4 AND n >= 6 (Exp12's 7-row and Exp6's 6-row panels, and Exp9's 5-discriminator cell), leaving k <= 3 on 5-feature panels, and let the within-sample count term be stronger only for samples of size >= 3 (e.g. use lam_w as now for 2-row samples but an effective lam_w ~0.70-0.80 for 3-4-row samples). This is the only in-family route to Exp12 (~.435 -> ~.58) , Exp6 (~.70 -> ~.78) and the 3-cue-crowd half of Exp9 (~.70 -> ~.58) that structurally cannot leak into the 2-row-sample cells that currently work (Exp5, Exp10, Exp11-c4). Note the previous attempt at numerosity failed because it was bought with a BLANKET k-cap raise plus a global lam_w cut plus a beta cut " that bundle was rejected by the gate. Do not repeat the global versions: leave beta in [2.10,2.70] and omega in [0.80,0.94] exactly as in the accepted base, since Exp5/Exp7/Exp8/Exp10/Exp2 are all at or near target and a uniform gain change is what destroyed them last time.

2) (SECONDARY, one knob) Localise the authority pop-out so it stops firing in Exp11. It clearly fires where intended (Exp1, Exp2) but Exp11's index rose 0.843 -> 0.915 at exactly the same edit, so some Exp11 cell now gets a spurious capture. Tighten the authority test rather than weakening it globally: require the first discriminating row to beat the best other discriminating row by a MARGIN (e.g. sigmoid(s_auth*(x_j1/max_other_x - 1 - m0)) with m0 ~ 0.05-0.15, or raise the s_auth lower bound to ~12), and/or cap phi_auth's lower bound down to ~3. Verify Exp1 stays >= +0.03 and Exp2 stays in [.74,.80] while Exp11's index falls toward ~.70.

If you prefer a different second knob, a VERY small read-out/sampling split (w_read = mu*att_norm + (1-mu)*x_norm with mu in [0.90,0.95], sampling untouched) is the remaining candidate for Exp4 (.348 -> ~.42) and Exp9 " but note that a strong version of this was part of the rejected iter-2 bundle, so keep mu at the high (near-identity) end and take it only if you are not already spending your second edit on knob 2. There is headroom on Exp5 (0.883 vs .820) to absorb a small loss there.

Acceptance targets to beat 0.1285: hold Exp1 ~+0.05, Exp2 ~.775, Exp5 >= .82, Exp7 ~.06-.09, Exp8 ~-.20, Exp10 >= .80; improve Exp12 toward ~.58, Exp6 toward ~.78, Exp9 toward ~.58, Exp11 toward ~.70. Keep each edit attributable so the next diagnosis is clean.

**Outcome of this advice:** iter 3 candidate loss=0.1285 -> iter 4 candidate loss=0.1310 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The iter-4 candidate remains faithful to the prescribed attention-lottery / limited-sample family (salience x compressed-diagnosticity lottery, k=2-4 sampled rows, compressive read-out, silence-as-sensitivity gain discount; no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop). Faithfulness is not the issue \u2014 calibration is, and the gate rejected it (0.1310 vs base 0.1285).

Decomposing the two edits against the accepted iter-3 base:

(A) The phi_first cut (0.30-0.70 -> 0.05-0.25) did roughly what the proposer predicted in direction but the price exceeded the gain. Gains: Exp11 0.915 -> 0.863 (real .513, still +0.35 off), Exp9 0.699 -> 0.645 (real .490, still +0.155 off), Exp5 unchanged. Costs: Exp1 0.064 -> 0.031 (real +0.093), Exp2 0.776 -> 0.752 (real .775 \u2014 base was essentially exact), Exp3 0.473 -> 0.416 (real .518, a 0.06 regression), Exp10 0.823 -> 0.767 (real .850, a 0.06 regression). Net: the authority term did NOT fully shield Exp1/Exp2 as claimed, and Exp3/Exp10 (which were fine) both degraded. So the generic first-mismatch bonus is load-bearing for the top-row cells and should be restored.

(B) The wide-panel numerosity edit \u2014 the knob I flagged as PRIMARY and which had never been implemented \u2014 was implemented but far too timidly, and its trigger excludes most of the cells it was supposed to fix. With `n>=6 AND m>=5`, it does NOT fire on Exp6's diagnostic classes (disc sets {0,2,5} and {0,1,2,5}, m=3 and m=4) nor on Exp12's cell-4 (d=(+1,-1,-1,0,0,0,+1), m=4). Consequently Exp6 moved 0.698 -> 0.703 (real .841) and Exp12 only 0.435 -> 0.455 (real .665). Where it did fire (Exp12 cell-8, m=6; Exp9's 5-discriminator cell) k=4 with lam_wide=0.70 leaves the weighted-salience term still dominant, so the 4-row crowd still loses to the two screen extremes. The mechanism is right; the dosage and the gate condition are wrong.

Residual structure on the accepted base is therefore unchanged and still concentrated on one axis \u2014 numerosity/aggregation on wide panels (Exp12 -0.23, Exp9 +0.21, Exp6 -0.14, Exp4 -0.16) \u2014 plus the stubborn Exp11 index (+0.40, but note its between-subject variance is the largest in the set, 0.052, so it is the least reliable target).
Rationale: Rejected by the gate, so the next proposal must be built on the UNCHANGED iter-3 base (0.1285), still inside the attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop). Make exactly ONE attributable edit this time.

1) REVERT phi_first to the accepted base range [0.30, 0.70] and leave the authority-contingent capture (phi_auth, s_auth) exactly as in iter 3. Explicitly: my iter-3 suggestion to tighten/weaken the pop-out was tried in its phi_first form and was rejected by the gate \u2014 it bought ~0.05 on Exp11/Exp9 and gave back ~0.06 each on Exp2, Exp3 and Exp10. Do not push that knob again in either direction; leave beta [2.10,2.70], omega [0.80,0.94], eta, f0, delta, rho, tau, gamma verbatim.

2) SPEND THE SINGLE EDIT ON A PROPERLY DOSED, PANEL-SIZE-CONDITIONAL COUNT TERM. The direction is right (it is the only in-family route to Exp6/Exp9/Exp12) but iter 4's version was both mis-triggered and under-powered. Concretely:
   - Widen the trigger from `n>=6 AND m>=5` to `n>=6 AND m>=4`. This is what brings Exp6's four-discriminator class ({0,1,2,5}) and Exp12's cell-4 (m=4) inside the rule. Verify the leak-check cells: Exp3's double-dissociation pairs have m=3 (safe), Exp7's load cells m=3 (safe), all Exp1/4/5/8/11 panels are n=5 (structurally safe), and Exp10's cell-5 / Exp12's cell-4 are 2-vs-2 count TIES so a stronger count term contributes exactly zero there \u2014 it cannot damage them.
   - Strengthen the dose: lam_wide down to ~[0.40, 0.60] (not 0.62-0.78) and allow k up to 5 when m>=6. At lam_wide~0.5 the within-sample count majority genuinely competes with the salience-weighted sign, which is what Exp12's cell-8 (4-against-the-oracle, currently pinned near .1-.2) and Exp9's 3-cue-crowd cell need.
   - Guardrail: Exp2's conflict trials (n=6, m=6) DO fire this rule and a strong count term pushes them toward the tally side. Exp2 must stay in [0.74, 0.82] \u2014 if your simulation pushes it past ~.82 (pi_7's overshoot, the exact thing the arbiter told you to avoid), back lam_wide off to ~0.60 rather than re-tightening the trigger, since the trigger is what buys Exp6/Exp12-cell4.

3) Do NOT spend a second edit on Exp11 this round. It is the noisiest metric in the set (between-subject var 0.052) and the two levers tried so far (authority tightening, phi_first cut) each cost more elsewhere than they recovered. If the numerosity edit incidentally lowers c8 (a 3-vs-1 cell), that is a free gain; do not chase it directly.

Acceptance targets to beat 0.1285: hold Exp1 ~+0.06, Exp2 0.75-0.80, Exp3 >= 0.47, Exp5 >= 0.86, Exp7 ~0.06-0.09, Exp8 ~-0.21, Exp10 >= 0.82; improve Exp12 toward ~0.57, Exp6 toward ~0.78, Exp9 toward ~0.60, Exp4 toward ~0.40. Keep the edit single and attributable so the next diagnosis is clean.

**Outcome of this advice:** iter 4 candidate loss=0.1310 -> iter 5 candidate loss=0.1203 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The iter-5 candidate is faithful to the prescribed attention-lottery / limited-sample family (salience x compressed-diagnosticity lottery, k=2-3 sampled rows, compressive read-out, silence-as-sensitivity gain discount; no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop) and it is the new running best (0.1203). The panel-volume gain modulation (beta_eff = beta*(n/6)^zeta) was a genuinely attributable, well-diagnosed win: on the n=5 panels it pulled Exp5 0.883 -> 0.799 (real .820), Exp4 0.348 -> 0.368, Exp11 0.915 -> 0.883, while the n=6 cells the proposer promised to leave alone did indeed barely move (Exp7 0.042 vs real 0.053, Exp8 -0.2075 vs -0.2025, Exp10 0.842 vs .850 "which is now essentially exact). It also cost a little where predicted (Exp1 0.064 -> 0.042, Exp2 0.776 -> 0.755, Exp3 0.473 -> 0.449) and, contra the hand-simulation, Exp12 got WORSE rather than washing (0.435 -> 0.407 vs real 0.665) because the n=7 gain boost makes the model more decisive in exactly the direction it is already wrong.

I also accept the proposer's analytic argument against the within-sample count term: because rows are sampled proportional to attention, the within-sample count inherits the salience bias and cannot act as an independent numerosity signal. That closes the knob I pushed in iters 3-4 (one attempt rejected), so I will not push it a third time.

What remains is a single, sharply diagnosable residual axis, and it is NOT numerosity per se " it is ORACLE SKEPTICISM. The data say high-validity rows are systematically discounted, not just when they speak alone: Exp8 (-0.20, the reversed single-cue ladder), Exp10 (.85 " two weak screen extremes beat the .95/.86 heavyweights), Exp3 (.52) and Exp4 (.51) (two extremes versus the lone .95 oracle is a coin flip), Exp6 (.84 " the mid-list top cue loses to 2-3 lesser cues), Exp12 (.665 " the four-against-the-oracle cell). The current model implements oracle skepticism ONLY through the global silence gain c (which multiplies by vbar of the speakers); the per-row attention/read-out term x_j = ((v-.5)/.5)^rho is still strictly INCREASING in validity. That is why every cell in which a crowd of mid-list rows must overturn a single heavyweight is under-predicted in the same direction: Exp12 -0.26, Exp6 -0.13, Exp3 -0.07, Exp2 -0.02. Exp9 (+0.19) and Exp11 (+0.37) are the remaining off-axis residuals; Exp11 has by far the largest between-subject variance in the set (0.052) and two direct attempts at it have already been rejected by the gate.
Rationale: The candidate was ACCEPTED and must be the unchanged base going forward " do NOT revert the panel-volume gain (zeta_n), the authority-contingent pop-out, or any of beta/omega/eta/f0/delta/rho/tau/gamma/lam_w/k_frac. Stay strictly inside the prescribed attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop). But three residuals >= 0.19 remain and there is one clean, never-tried, in-family knob, so one more tightly scoped round is warranted. Make exactly ONE edit.

1) SPEND THE EDIT ON MAKING THE DIAGNOSTICITY TERM NON-MONOTONIC (ORACLE SKEPTICISM IN THE ROW WEIGHT, NOT JUST THE GLOBAL GAIN). Replace x_j = ((v_j-.5)/.5)^rho with x_j = ((v_j-.5)/.5)^rho * (1 - psi*sigmoid(s_v*(v_j - v0))), i.e. an inverted-U over validity peaking around .72-.82 and falling for v >~ .88, with psi in [0.20, 0.45], v0 ~ [0.84, 0.90], s_v ~ [15, 40]. This is a re-parameterisation of the SAME 'mildly compressed stated diagnosticity' term the arbiter prescribed " not a new mechanism family " and it is the natural generalisation of the reversed ladder the model already needs for Exp8/Exp10. Apply it to both the sampling weights and the read-out weights (they are the same att in this code), and change nothing else.

Predicted signature, which is what makes it attributable: Exp12 should rise substantially (BOTH its cells help " the four-against-the-oracle cell and the 2-vs-2 extremes deadlock both have a heavyweight on the losing side), target ~.50-.58 from .407; Exp6 should rise (its diagnostic classes are exactly 'mid-list top cue versus 2-3 lesser cues'), target ~.78 from .715; Exp3 ~.48-.52 from .449; Exp2 up slightly toward .775 from .755.

2) GUARDRAILS " check these four before shipping, and back psi down toward 0.20 if any breaks:
   - Exp8 must stay in [-0.24, -0.18]; the new term reinforces the reversed ladder and can easily overshoot past -0.26.
   - Exp4 (.368 vs real .508) moves the WRONG way under this edit, since its TTB winner IS the .95 oracle. It is currently 0.14 too low; do not let it fall below ~0.33. If it does, that is the signal that psi is too large.
   - Exp1 (+0.042 vs +0.093) will soften slightly " keep it >= +0.02; the authority-contingent capture (a sampling bonus) should shield it because the pop-out still fires on the top row.
   - Exp10 must stay >= 0.82 and Exp5 >= 0.78 (both are near-exact now).

3) DO NOT chase Exp11 or Exp9 directly this round. Exp11 is the noisiest metric in the set and both levers tried on it (authority tightening, phi_first cut) were rejected by the gate; note also that oracle skepticism will likely push its c4/c8 UP, so if Exp11 degrades by <= 0.05 while Exp12/Exp6/Exp3 improve by 0.10-0.25, that is still a large net win. Do not add a compensating hack for it.

Acceptance targets to beat 0.1203: hold Exp1 >= +0.02, Exp2 in [.75,.82], Exp5 >= .78, Exp7 ~.04-.09, Exp8 ~-.20, Exp10 >= .82, Exp4 >= .33; improve Exp12 toward ~.55, Exp6 toward ~.78, Exp3 toward ~.50. Keep the edit single so the next diagnosis stays clean.

**Outcome of this advice:** iter 5 candidate loss=0.1203 -> iter 6 candidate loss=0.1256 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter-6 candidate is faithful to the arbiter's attention-lottery / limited-sample family (salience x diagnosticity lottery, k=2-3 rows, compressive read-out, silence-as-sensitivity gain, no unanimity bonus, no majority bonus, no deadlock gate, no serial stop) \u2014 faithfulness is not the problem. The problem is that the ONE edit I prescribed last round (oracle-skeptical, non-monotonic row diagnosticity, x_j scaled by 1-psi*sigmoid(s_v*(v-v0))) failed its own predicted signature and broke a guardrail, so the gate rejected it (0.1256 vs base 0.1203).

Per-experiment decomposition vs the accepted iter-5 base: predicted gains largely did not materialise \u2014 Exp6 0.7150 -> 0.7188 (+0.004 where +0.05 was promised, still -0.12 off), Exp3 0.4487 -> 0.4400 (moved the WRONG way, -0.009), Exp2 0.7545 -> 0.7650 (+0.011, fine), Exp12 0.4067 -> 0.4417 (+0.035, real gain but only 1/7 of the residual). Meanwhile the costs were larger and hit cells that were near-exact: Exp10 0.8417 -> 0.7767 (-0.065, and it broke the explicit '>= 0.82' guardrail), Exp1 0.0421 -> 0.0060 (-0.036, now essentially zero vs real +0.093), Exp8 -0.2075 -> -0.1925, Exp9 0.6842 -> 0.7075 (+0.023, wrong way), Exp4 0.3675 -> 0.3531 (wrong way). Exp11 improved 0.883 -> 0.842 but is still +0.33 off. So per-row oracle skepticism is NOT the missing ingredient: it is already adequately carried by the global silence/vbar gain (which is what produces Exp8's reversed ladder and Exp10's .85), and adding it a second time at the row level mostly destabilises the top-row capture that Exp1/Exp3/Exp10 depend on. I will not push that knob again.

On the unchanged accepted base the residual structure is: Exp11 +0.370, Exp12 -0.258, Exp9 +0.194, Exp4 -0.140, Exp6 -0.126, Exp3 -0.069, Exp1 -0.051, everything else <= 0.02. Crucially, the four biggest residuals share a single signature that is NOT oracle skepticism: the lottery is far too PEAKED. With p proportional to raw att, the pop-out top row plus the two screen extremes absorb nearly all sampling mass, so (a) the low-salience mid-list heavyweight is almost never drawn \u2014 that is exactly why Exp3 (.449 vs .518) and Exp4 (.368 vs .508), both 'does the buried .95 oracle get read at all' cells, are under-predicted; (b) crowds of 3-4 mid rows almost never occupy the sample \u2014 that is why Exp9's 2-extremes-vs-3-crowd cell and Exp12's four-against-the-oracle cell are pinned on the extremes side (Exp9 +0.19, Exp12 -0.26); and (c) Exp11's c4 (extremes vs two mid heavyweights) is over-decisive for the same reason. The arbiter's own text says numerosity should arise ONLY because 'more rows on one side means the sample is more likely to be dominated by that side' \u2014 with sampling this peaked, that channel is effectively switched off.
Rationale: Rejected by the gate, so the next proposal must be built on the UNCHANGED iter-5 base (loss 0.1203), still strictly inside the prescribed attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop). Keep verbatim: the authority-contingent pop-out (phi_first 0.30-0.70, phi_auth, s_auth), the panel-volume gain beta_eff = beta*(n/6)^zeta, beta [2.10,2.70], omega [0.80,0.94], eta, f0, delta_sil, rho, tau, gamma_rec, lam_w, k_frac, epsilon. Explicitly DROP the psi_ora/v0_ora/s_v oracle-skeptical row term \u2014 my own iter-5 advice in that direction was tried and rejected; do not re-attempt it in either direction.

Make exactly ONE new, never-tried, in-family edit: a SAMPLING TEMPERATURE that decouples the sharpness of the lottery from the sharpness of the read-out.

  p_j  proportional to  att_j**alpha_s   (renormalised, then the existing 0.998/0.002 smoothing),
  read-out weights left EXACTLY as att_j (unchanged).

with alpha_s in [0.60, 0.85] (start near the top of that range \u2014 every aggressive global move in this loop, beta cut and blanket k-raise, has been rejected, so take a small step). Nothing else changes; k, beta, c, the pop-out and the silence gain are untouched.

Why this is the right single knob. It is the arbiter's own stated route to numerosity ('numerosity helps only because more rows on one side means the sample is more likely to be dominated by that side'), which is currently inert because sampling mass is monopolised by the pop-out top row and the two screen extremes. Flattening the LOTTERY while keeping the READ-OUT peaked has an asymmetric, attributable signature: in a mixed 2-row sample the weighted sign still follows the salient row (so Exp5/Exp10/Exp12-cell4, which are count ties, are largely shielded), but the buried heavyweight and the mid-list crowds now actually enter the sample.

Predicted signature (use it to verify attribution): Exp3 0.449 -> ~0.49-0.52 and Exp4 0.368 -> ~0.42-0.45 (the buried .95 oracle finally gets drawn); Exp9 0.684 -> ~0.58-0.62 (the 3-cue crowd cell stops being swept by the extremes); Exp11 0.883 -> ~0.75-0.80 (c4 falls, c5 rises); Exp12 0.407 -> ~0.45-0.50 (cell-8, the four-against-the-oracle cell, is the one that moves).

Guardrails \u2014 check all of these and back alpha_s up toward 0.85 if any breaks: Exp10 >= 0.80 (it was 0.842 and is the cell the last edit broke), Exp5 >= 0.76, Exp8 in [-0.24,-0.17], Exp7 in [0.00,0.10], Exp1 >= +0.02, Exp2 in [0.74,0.82], Exp6 >= 0.68 (Exp6's diagnostic classes have m=3-4 so k=2 there; a flatter lottery lets the opposing mid-list top cue in more often and Exp6 is the one cell this edit can plausibly cost \u2014 accept up to -0.04 there against the 0.10-0.30 gains elsewhere).

Do NOT bundle a second edit: no k/lam_w numerosity change (tried, rejected), no beta or omega change (tried, rejected), no phi_first change (tried, rejected), no direct Exp11 hack (Exp11 has the largest between-subject variance in the set, 0.052, and two direct attempts on it have already failed). One knob, one attributable diagnosis.

**Outcome of this advice:** iter 6 candidate loss=0.1256 -> iter 7 candidate loss=0.1230 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-7 candidate stays faithful to the arbiter's attention-lottery / limited-sample family (salience x compressed-diagnosticity lottery, k=2-3 rows, sharp read-out, silence-as-sensitivity gain, no unanimity bonus, no majority bonus, no deadlock gate, no serial stop), so faithfulness is not the issue. But the one edit I prescribed last round \u2014 the sampling temperature p ~ (s*x)^alpha_s \u2014 only half worked, and the gate rejected it (0.1230 vs base 0.1203).

Decomposition vs the accepted iter-5 base. Genuine gains: Exp11 0.883 -> 0.787 (residual +0.370 -> +0.273, the single biggest improvement anyone has produced on that cell), Exp2 0.7545 -> 0.7755 (now essentially exact), Exp12 0.407 -> 0.437, Exp5 0.799 -> 0.810, Exp4 0.368 -> 0.374. Costs: Exp3 0.449 -> 0.401 (moved the WRONG way, opposite to my prediction of 0.49-0.52), Exp10 0.842 -> 0.792 (broke the >=0.80 guardrail), Exp9 0.684 -> 0.707 (wrong way), Exp7 0.042 -> 0.008, Exp6 0.715 -> 0.706. Net slightly negative.

The reason Exp3/Exp4/Exp9 did not move as predicted is now diagnosable and it is my error, not the proposer's: the exponent was applied to the PRODUCT s*x, so it flattened the diagnosticity factor too, which is precisely the factor that gives the buried .95 oracle its only chance of being sampled. Flattening x therefore cancels the intended 'buried heavyweight finally gets drawn' effect and leaves only the 'extremes lose share' effect \u2014 which is exactly what costs Exp10 and Exp3.

More importantly, an arithmetic check on the remaining big residuals shows the Exp3/Exp4 deficit is NOT a sampling problem at all, it is a read-out-gain FLOOR problem. In Exp3 the diagnostic pairs have 3 of 6 rows both-0 silent, so f_sil = 0.50 >> f0 = 0.365, the sigmoid saturates, and c = 1 - eta*vbar goes negative and is clipped at c_min = 0.10. With beta_eff = 2.4 (n=6) the model is emitting sigmoid(0.24*E); with E ~ -0.8 that is 0.45 \u2014 exactly the observed base value. Humans are at 0.518, i.e. essentially at chance. The model is stuck 0.05-0.07 away from chance purely because the floor c_min is too high. The same floor/near-floor regime governs the Exp4 cells (2-3 silent rows out of 5, c ~ 0.13-0.28), which is the -0.14 residual there. Exp10's cell-5 by contrast has f_sil = omega*2/6 = 0.29 < f0, so it is NOT discounted at all and is structurally immune to a floor change; Exp5 (one silent row) and Exp12's cell-8 (one silent row) are likewise immune. That makes the floor a rare thing in this loop: a knob with a large, targeted effect on the two cells that need it and provable near-zero leakage into the cells that are already near-exact.
Rationale: Rejected by the gate, so the next proposal must be built on the UNCHANGED iter-5 base (loss 0.1203), still strictly inside the prescribed attention-lottery / limited-sample family (no unanimity bonus, no additive majority bonus, no deadlock gate, no serial stop). Keep verbatim: the authority-contingent pop-out (phi_first 0.30-0.70, phi_auth, s_auth), the panel-volume gain beta_eff = beta*(n/6)^zeta, beta [2.10,2.70], omega [0.80,0.94], eta, f0, kap_s, delta_sil, rho, tau, gamma_rec, lam_w, k_frac, epsilon. Do NOT re-add the alpha_s sampling temperature this round (tried, rejected) and do NOT re-attempt the psi_ora oracle-skeptical row term (tried, rejected).

MAKE EXACTLY ONE NEW EDIT: LOWER THE SILENCE-GAIN FLOOR. Change c_min from [0.07, 0.13] to [0.015, 0.055] (centre ~0.03). Nothing else changes. Rationale and predicted, attributable signature: the heavily-silent panels are already saturating the sigmoid and sitting ON the floor, so the floor \u2014 not eta, not f0, not beta \u2014 is what pins them away from chance. Expected: Exp3 0.449 -> ~0.48-0.50 (real 0.518), Exp4 0.368 -> ~0.40-0.44 (real 0.508, its cells are at/near the floor too). If after the change Exp3 still sits below ~0.47, the permitted follow-up (next round, not bundled now) is a small eta raise to [1.40,1.65]; do not raise eta in the same edit.

GUARDRAILS \u2014 verify all of these and pull c_min back toward 0.05 if any breaks:
  * Exp8 in [-0.24, -0.17]. The .95 single-cue trials are floored and the .56/.62 ones are not, so the reversed ladder will get slightly MORE negative (expect ~-0.215 to -0.225). This is the one cell the edit can plausibly cost.
  * Exp7 in [0.00, 0.10]. Both its load cells are floored, so lowering the floor compresses both toward .5 and shrinks the contrast; it is currently +0.042 on the base, so expect ~+0.02-0.04. Acceptable.
  * Exp10 >= 0.82, Exp5 >= 0.78, Exp12 >= 0.40, Exp1 >= +0.02, Exp2 in [0.74, 0.82]. These are structurally immune (their diagnostic cells have f_sil below f0 or only one silent row) \u2014 if any of them moves by more than ~0.02 that is a sign the floor is being hit somewhere unexpected and should be reported, not patched.

WHY NOT THE OTHER KNOBS. (a) alpha_s: its per-cell signature was genuinely mixed, and the failure mode is now understood \u2014 flattening the PRODUCT s*x also flattens diagnosticity and so destroys the buried-oracle sampling advantage that motivated it. If a later round revisits it, the correct form is p ~ (s^alpha_s)*x with x left sharp; but that variant threatens Exp10 (weak extremes vs .95/.86 heavyweights) and must not be bundled with anything else. (b) Numerosity via k / lam_w (tried, rejected twice), beta or omega cuts (tried, rejected), phi_first changes (tried, rejected), per-row oracle skepticism (tried, rejected) \u2014 all closed. (c) Do NOT chase Exp11 directly: it has the largest between-subject variance in the set (0.052) and every direct attempt has failed; the iter-7 run showed it does move as a side effect of global sampling changes, which is the only route worth taking.

Acceptance targets to beat 0.1203: hold Exp1 >= +0.02, Exp2 in [0.74,0.82], Exp5 >= 0.78, Exp6 ~0.71, Exp7 in [0.00,0.10], Exp8 in [-0.24,-0.17], Exp10 >= 0.82, Exp12 >= 0.40; improve Exp3 toward ~0.49 and Exp4 toward ~0.42. One knob, one attributable diagnosis.

**Outcome of this advice:** iter 7 candidate loss=0.1230 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "ATTENTION LOTTERY WITH LIMITED-SAMPLE COMPARISON, AUTHORITY-CONTINGENT CAPTURE, PANEL-VOLUME CONFIDENCE, AND A NEAR-ZERO SILENCE FLOOR. (1) Each display row enters an attention lottery with weight att_j = s_j * x_j * pop_j, where s_j = exp(-j/tau) + gamma*exp(-(n-1-j)/tau) is literal screen salience (asymmetric: top of panel stronger than bottom), x_j = ((v_j-.5)/.5)^rho is mildly compressed stated diagnosticity (rho ~ .55-.85), and pop_j is a pop-out factor: rows on which the two columns are identical do not pop out and keep only a residual share delta, while the FIRST row at which the columns disagree gets a bonus. (2) That first-mismatch bonus is AUTHORITY-CONTINGENT: a row captures attention outright only if it is both the first thing that discriminates AND the most diagnostic discriminating row on the screen. Position is always active, never gated by a deadlock rule. (3) The subject does not integrate the panel: only k = clip(ceil(k_frac*m), 2, 3) rows are actually compared, drawn without replacement with probability proportional to att. Choice = the attention-weighted sign of the sampled differences plus a small within-sample count term; a 1-1 sample is decided by the more salient sampled row. Numerosity helps only because more rows on one side makes the sample more likely to be dominated by that side; there is no unanimity bonus, no additive majority bonus, no serial stop rule. Because the sample is small and random, the map from true evidence to P(choice) is compressive with a ceiling near .85-.90 even for landslides. (4) Silence is read as sensitivity and discounts read-out gain: c = 1 - eta*sigmoid(kappa*(f_sil - f0))*vbar_speakers, with mutual endorsements counted at weight omega ~ .88 so density/load is nearly inert; this yields the REVERSED single-cue validity ladder. (5) NEW COMMITMENT: THE COLLAPSE OF CONFIDENCE UNDER PANEL SILENCE IS ESSENTIALLY COMPLETE, NOT PARTIAL. When most of the panel says nothing, the subject does not retain a small residual tilt toward the speaking expert \u2014 the verdict of the sampled comparison becomes almost literally uninformative and the choice approaches a coin flip (c is floored at c_min ~ .03, an order of magnitude below a 'modest residual'). This is the sharpest quantitative signature of silence-as-sensitivity: a display in which three of six experts are mute and the rest disagree is a genuine 50/50 for people, not a .45/.55 lean, whereas a display in which only one expert is mute is followed decisively. Because f_sil must exceed the sharp threshold f0 for the discount to fire at all, this near-total collapse is confined to heavily-silent panels and leaves dense panels (one or two mute rows) completely untouched. (6) Confidence also scales with the VOLUME of the panel, not with the amount actually read: beta_eff = beta*(n/6)^zeta (illusion of knowledge \u2014 short panels are systematically less decisive than structurally identical cells on wide panels). (7) Individual differences are unimodal jitter in k_frac, tau, gamma, rho, phi, phi_auth, s_auth, beta, zeta, eta, c_min and a lapse rate; no strategy subpopulations, no learning.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    from itertools import combinations, permutations\n\n    # ------------------------------------------------------------------\n    # 1. parse the trial stimulus into two rating vectors (screen order)\n    # ------------------------------------------------------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(list(state['option_a_ratings']), dtype=float)\n            b = np.asarray(list(state['option_b_ratings']), dtype=float)\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        arr = arr.reshape(2, -1)\n        a = arr[0].astype(float)\n        b = arr[1].astype(float)\n    a = np.ravel(a).astype(float)\n    b = np.ravel(b).astype(float)\n    n = int(min(a.shape[0], b.shape[0]))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = a[:n]\n    b = b[:n]\n    d = a - b\n    disc = np.abs(d) > 1e-12\n    if not np.any(disc):\n        return np.array([0.5, 0.5])\n\n    # ------------------------------------------------------------------\n    # 2. communicated validities -> diagnosticity\n    # ------------------------------------------------------------------\n    val = parameters.get('validities', None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = None\n    if val is None or val.shape[0] != n:\n        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])\n    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)\n    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)\n\n    # ------------------------------------------------------------------\n    # 3. parameters\n    # ------------------------------------------------------------------\n    def _pget(key, dflt, lo, hi):\n        try:\n            v = float(parameters.get(key, dflt))\n        except Exception:\n            v = float(dflt)\n        if not np.isfinite(v):\n            v = float(dflt)\n        return float(np.clip(v, lo, hi))\n\n    tau = _pget('tau', 0.85, 0.2, 8.0)\n    gam = _pget('gamma_rec', 0.50, 0.0, 1.5)\n    rho = _pget('rho', 0.70, 0.05, 4.0)\n    dlt = _pget('delta_sil', 0.07, 0.0, 1.0)\n    phi = _pget('phi_first', 0.50, 0.0, 3.0)\n    phi_auth = _pget('phi_auth', 6.0, 0.0, 20.0)\n    s_auth = _pget('s_auth', 10.0, 0.5, 60.0)\n    lam = _pget('lam_w', 0.88, 0.0, 1.0)\n    beta = _pget('beta', 2.40, 0.05, 40.0)\n    zet = _pget('zeta_n', 1.50, 0.0, 4.0)\n    eta = _pget('eta', 1.45, 0.0, 5.0)\n    f0 = _pget('f0', 0.365, 0.0, 1.0)\n    kap = _pget('kap_s', 25.0, 1.0, 80.0)\n    ome = _pget('omega', 0.88, 0.0, 1.0)\n    cmin = _pget('c_min', 0.03, 0.0, 0.5)\n    kfrac = _pget('k_frac', 0.47, 0.05, 1.0)\n    eps = _pget('epsilon', 0.035, 0.0, 0.5)\n\n    def sig(z):\n        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))\n\n    # ------------------------------------------------------------------\n    # 4. attention weights: screen salience x compressed diagnosticity,\n    #    matched rows do not pop out (residual share delta), first\n    #    mismatching row of the top-down scan gets a pop-out bonus that\n    #    is AUTHORITY-CONTINGENT: it captures attention outright only if\n    #    no more diagnostic expert further down the panel also speaks.\n    # ------------------------------------------------------------------\n    pos = np.arange(n, dtype=float)\n    s = np.exp(-pos / tau) + gam * np.exp(-(n - 1.0 - pos) / tau)\n    x = np.power(vs, rho)\n    att = s * x\n    att = np.where(disc, att, dlt * att)\n    idxs = np.flatnonzero(disc)\n    att = np.array(att, dtype=float)\n    j1 = int(idxs[0])\n    if idxs.shape[0] > 1:\n        x_rest = float(np.max(x[idxs[1:]]))\n        ratio = float(x[j1]) / max(x_rest, 1e-9)\n        q_auth = float(sig(s_auth * (ratio - 1.0)))\n    else:\n        q_auth = 1.0\n    att[j1] = att[j1] * (1.0 + phi + phi_auth * q_auth)\n    att = np.clip(att, 1e-12, None)\n\n    # ------------------------------------------------------------------\n    # 5. silence-as-sensitivity gain discount (sharp threshold, floored\n    #    at a NEAR-ZERO residual: a heavily-mute panel leaves essentially\n    #    no tilt at all, not a modest one)\n    # ------------------------------------------------------------------\n    both1 = (~disc) & (a > 0.5) & (b > 0.5)\n    n_both0 = float(np.count_nonzero((~disc) & (~both1)))\n    n_both1 = float(np.count_nonzero(both1))\n    f_sil = float(np.clip((n_both0 + ome * n_both1) / float(n), 0.0, 1.0))\n    xw = x[idxs]\n    sxw = float(np.sum(xw))\n    vbar = float(np.sum(xw * vs[idxs]) / sxw) if sxw > 1e-12 else 0.5\n    c = 1.0 - eta * float(sig(kap * (f_sil - f0))) * vbar\n    c = float(np.clip(c, cmin, 1.0))\n\n    # ------------------------------------------------------------------\n    # 5b. PANEL-VOLUME CONFIDENCE: the sampled comparison stays the same\n    #     size, but the felt informativeness of its verdict grows with\n    #     how many experts were displayed (illusion of knowledge).\n    # ------------------------------------------------------------------\n    beta_eff = beta * float(np.power(float(n) / 6.0, zet))\n\n    # ------------------------------------------------------------------\n    # 6. the lottery: draw k rows without replacement, prob ~ attention\n    # ------------------------------------------------------------------\n    m = int(idxs.shape[0])\n    k = int(np.clip(int(np.ceil(kfrac * m)), 2, 3))\n    k = int(min(k, n))\n    if k < 1:\n        k = 1\n\n    p = att / float(np.sum(att))\n    p = 0.998 * p + 0.002 / float(n)\n    p = p / float(np.sum(p))\n\n    tot = 0.0\n    pa = 0.0\n    for S in combinations(range(n), k):\n        ps = 0.0\n        for perm in permutations(S):\n            q = 1.0\n            rem = 1.0\n            good = True\n            for t in perm:\n                if rem <= 1e-9:\n                    good = False\n                    break\n                q *= p[t] / rem\n                rem -= p[t]\n            if good:\n                ps += q\n        if ps <= 1e-14:\n            continue\n        sel = [j for j in S if disc[j]]\n        if len(sel) == 0:\n            E = 0.0\n        else:\n            ws = att[sel]\n            sg = np.sign(d[sel])\n            wsum = float(np.sum(ws))\n            Ew = float(np.sum(ws * sg) / wsum) if wsum > 1e-12 else 0.0\n            nA = float(np.sum(sg > 0))\n            nB = float(np.sum(sg < 0))\n            Ec = (nA - nB) / (nA + nB) if (nA + nB) > 0 else 0.0\n            E = lam * Ew + (1.0 - lam) * Ec\n        pa += ps * float(sig(beta_eff * c * E))\n        tot += ps\n\n    p_a = (pa / tot) if tot > 1e-12 else 0.5\n\n    pr = np.array([p_a, 1.0 - p_a], dtype=float)\n    pr = (1.0 - eps) * pr + eps * 0.5\n    pr = np.clip(pr, 1e-12, None)\n    pr = pr / pr.sum()\n    return pr",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0:\n        probs = np.ones_like(probs) / len(probs)\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "tau": "[0.72, 1.00]",
    "gamma_rec": "[0.38, 0.62]",
    "rho": "[0.55, 0.85]",
    "delta_sil": "[0.04, 0.11]",
    "phi_first": "[0.30, 0.70]",
    "phi_auth": "[4.0, 9.0]",
    "s_auth": "[6.0, 16.0]",
    "lam_w": "[0.82, 0.95]",
    "beta": "[2.10, 2.70]",
    "zeta_n": "[1.0, 2.0]",
    "eta": "[1.30, 1.60]",
    "f0": "[0.345, 0.385]",
    "kap_s": "[18.0, 32.0]",
    "omega": "[0.80, 0.94]",
    "c_min": "[0.015, 0.055]",
    "k_frac": "[0.40, 0.55]",
    "epsilon": "[0.02, 0.06]",
    "validities": "validities"
  },
  "rationale": "MINIMAL DIFF on the accepted iter-5 base (loss 0.1203). Exactly ONE line changes: the silence-gain floor range c_min goes from [0.07, 0.13] to [0.015, 0.055] (centre ~0.035), plus the matching default in the `_pget` call for cosmetic consistency. Every other line \u2014 the authority-contingent pop-out (phi_first, phi_auth, s_auth), the panel-volume gain beta_eff = beta*(n/6)^zeta, beta, omega, eta, f0, kap_s, delta_sil, rho, tau, gamma_rec, lam_w, k_frac, epsilon, the lottery and read-out code \u2014 is re-emitted verbatim. I did NOT re-add the alpha_s sampling temperature (tried iter 7, rejected) nor the psi_ora oracle-skeptical row term (tried iter 6, rejected), exactly as instructed.\n\nWHY THIS KNOB. I verified the critic's arithmetic on the base. Exp3's diagnostic pairs have 3 of 6 rows both-0 mute, so f_sil = 0.50 vs f0 = 0.365 and kap_s = 25 \u2014 the sigmoid is fully saturated, 1 - eta*vbar is negative, and c is pinned exactly ON the floor. With beta_eff = 2.4 (n = 6) the model emits sigmoid(0.24*E); with |E| ~ 0.8 that is 0.45, which is the observed base value (0.449) to two decimals. Humans are at 0.518, i.e. essentially at chance. Nothing but the floor is holding it there: eta, f0 and kap_s are all downstream of a saturated sigmoid and have zero gradient in that regime. Dropping the floor to 0.035 gives sigmoid(0.084*0.8) = 0.517, and after the lapse mixture ~0.515 \u2014 essentially the human value. The Exp4 conflict cells are in the same regime (2-3 mute of 5, c at or just above the floor, beta_eff = 1.82 at n = 5), so they move from 0.368 toward ~0.42-0.45 against a target of 0.508. Together those are the two largest addressable residuals after Exp11/Exp12.\n\nWHY THE LEAKAGE IS PROVABLY SMALL. The discount only fires when f_sil exceeds the sharp f0 threshold, so every near-exact cell with 0-2 mute rows is structurally immune, and I checked them individually: Exp10 cell-5 has 2 mute of 6 (f_sil = 0.88*2/6 = 0.29 < 0.365, no discount at all \u2014 stays 0.842); Exp5 has 1 mute of 5 (f_sil = 0.20, immune \u2014 stays 0.799); Exp12 cell-8 has 1 mute of 7 (f_sil = 0.126, immune) and cell-4 has 3 both-1 mute of 7 (f_sil = 0.377, only marginally over threshold so c ~ 0.5, far above either floor \u2014 immune); Exp1's conflict trials are mostly fully discriminating; Exp2's conflict trials are 5-vs-1 and 3-vs-1 with few mute rows. The two cells that can cost are exactly the ones the critic named: Exp8's reversed ladder gets slightly steeper (the .95/.72 single-cue trials are floored, the .56/.62 ones are not, so expect ~-0.22 against real -0.2025, still inside the [-0.24,-0.17] guardrail) and Exp7's load contrast compresses a little (both its load cells are floored, so expect ~+0.02 against real +0.053, still inside [0.00,0.10]).\n\nNET. Expected gains of ~+0.05 on Exp3 and ~+0.06 on Exp4 against expected costs of ~-0.015 on Exp8 and ~-0.02 on Exp7, with ten of twelve experiments moving by under 0.02 \u2014 an asymmetric, attributable edit that should land below the 0.1203 floor. Theoretically it is not a free parameter tweak but a genuine sharpening of the silence-as-sensitivity claim the arbiter prescribed: when most of the panel is mute, the subject's verdict is not merely discounted, it is essentially voided, which is what the near-chance human data on heavily-silent displays actually show."
}
```

## Usage

```json
{
  "input_tokens": 61312,
  "output_tokens": 8231
}
```
