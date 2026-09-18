# generation_iter_00_attempt_00

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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace pi_5 (WC-ST) with a new theory that keeps the domain-invariant polarity claim but abandons the Weber-threshold and strict validity-blindness. Sketch fff 'Sign-Dominant Flag Difference with a Weighter Minority (SD-FD/M)':

1. NEGATIVE POLARITY (retained, uncontroversial): a rating of '1' is a warning flag; the option with the smaller badness is chosen. This is what makes the population look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs (Exps 3fff7).

2. SIGN-DOMINANT DIFFERENCE DETECTION (the key departure from both incumbents): the decision variable is the flag-count difference, and the psychometric function is *saturating in |fff|, essentially magnitude-invariant in the base count*. A one-flag advantage is nearly as diagnostic at 4-vs-5 as at 0-vs-1: predict ~0.85fff0.88 for fff=1 at any base count (Exp 1 = 0.872, Exp 9 = 0.773) rising to ~0.90+ for ffffff2. Base count enters only through a small additive attention/interference cost (of order 0.02fff0.04 per extra shared flag, saturating), which reproduces Exp 10's small positive easy-minus-hard gap (~0.17) without any collapse toward chance. Explicitly no divisive compression by total mass and no ratio/Weber scaling.

3. WEAK, MINORITY-CARRIED VALIDITY MODULATION: instead of a population-wide strong severity term (PW-FMS) or none at all (WC-ST), posit a *mixture*: a majority (~70fff80%) of subjects are essentially validity-blind counters whose composition contrast is ~0, and a minority (~20fff30%) apply a validity-weighted mass / worst-flag reading strongly enough to reverse on composition pairs. The mixture yields a group-mean composition contrast of ~0.10fff0.20 with LARGE between-subject variance (Exp 2: 0.156, var 0.129) fff matching both the mean and the dispersion, which neither incumbent does (pi_5 has zero mean and zero spread; pi_6 has too large a mean). The same mixture makes Exp 8's differential small (~0.05fff0.10) rather than ~0.15.

4. EQUAL-COUNT TIEBREAK: on exact count ties the top discriminating expert's flag is mildly repellent, with moderate strength (p(choose flagged option) ~0.25fff0.40) and moderate spread fff intermediate between PW-FMS's extreme 0.07 and WC-ST's broad prior.

5. HETEROGENEITY: dispersion arises jointly from the counter/weighter mixture (bimodal on composition pairs) and from a modest lapse/sensitivity spread (unimodal on count pairs), matching the observation that per-subject variance is small on the magnitude ladder (0.009fff0.033) but large on composition contrasts (0.06fff0.13).

This theory competes sharply with PW-FMS: it predicts a FLAT magnitude ladder where PW-FMS predicts a monotone decline, and a SMALL, minority-driven composition effect where PW-FMS predicts a large population-wide severity reversal. Future experiments should target (i) fff=1 accuracy at base counts 0,2,4,6 with cue identity held fixed, and (ii) the per-subject correlation between composition reversal and count-ladder accuracy, which PW-FMS says is positive (high-gamma = high-beta subjects) and SD-FD/M says is near zero or negative.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.937)

**Description**
**Partially Weighted Flag Mass with Worst-Flag Severity (PW-FMS).**

Subjects read a binary expert rating of '1' as a *warning flag* rather than an endorsement, and choose the option carrying the smaller *badness*. Five claims:

1. **Negative polarity.** Badness grows with the flags an option carries; the option with the smaller badness is preferred. This single mechanism makes the population look 'anti-TTB' on dominance/agreement pairs and 'pro-TTB' on conflict pairs.

2. **Partial validity weighting on a subject-level continuum.** Each cue gets w_j = 1 + gamma*(r_j - mean(r)) with r_j = (v_j-0.5)/0.5 and gamma a subject-level tilt running from 0 (pure equal-weight counter) to strongly validity-tilted. Because the tilt is *centred*, its behavioural consequence scales with the design's validity spread, which is why the same population looks like a counter in one design and like a weigher in another.

3. **Mild divisive compression, not a threshold.** Evidence s = |D| / (1 + c*(mass_a + mass_b)) fed through a plain logistic plus lapse, so discriminability declines *shallowly and monotonically* with base flag count rather than collapsing to chance. The compression denominator is the plain flag mass — the perceived *quantity* of complaints — because that is what sets the scale against which a difference is judged.

4. **Worst-flag severity, and it is strong.** An option is judged not only by *how much* flag mass it carries but also by *how bad its single worst flag is*. The comparative badness signal is D = (mass_a - mass_b) + lam*(sev_a - sev_b), where sev_i = max_j w_j x_ij is the weight of the most reliable expert who flagged option i (0 if unflagged) and lam is a subject-level severity weight, typically above the point of mere cancellation. Severity is a salience term, not a re-weighting of the tally — it enters the numerator of the evidence only, never the compression scale, and it is never gated on the count difference.

5. **A graded-evidence cascade at near-ties, terminating in strong worst-flag repulsion.** When |D| falls inside a subject-specific resolution band tau the weighted comparison is untrustworthy and the subject falls back lexicographically: (i) if the raw flag *counts* differ, take the option with fewer flags (same compression/logistic applied to counts); (ii) if the counts are also equal, ask which single flag is worst and be *strongly* repelled by the option carrying the top discriminating expert's flag (tie_bias ~ 0.07).

6. **Heterogeneous evidence sensitivity is the primary source of population dispersion.** The subject-level inverse temperature beta is broadly distributed, not narrowly clustered: some subjects convert badness differences into choices almost deterministically while others remain shallow even on large gaps. This spread — not lapse spread — is what generates the large between-subject variance in per-subject accuracy observed on the count-driven designs, and it simultaneously softens near-ceiling agreement/dominance contrasts (low-beta tail) while lifting count-driven accuracy (high-beta tail).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n = a.size

    # ---------------- 2. design validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.90, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    c = float(parameters["comp_c"])
    tau = float(parameters["tau"])
    lam = float(parameters["severity"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    gamma = max(gamma, 0.0)
    beta = max(beta, 0.0)
    c = max(c, 0.0)
    tau = max(tau, 0.0)
    lam = max(lam, 0.0)

    # ---------------- 4. partially weighted flag mass ----------------
    r = np.clip((val - 0.5) / 0.5, 0.0, 1.0)     # normalised validity strength
    rbar = float(np.mean(r))
    w = 1.0 + gamma * (r - rbar)                 # centred tilt: mean weight == 1
    w = np.clip(w, 0.05, None)                   # weights stay positive

    mass_a = float(np.dot(w, a))
    mass_b = float(np.dot(w, b))

    # ---------- 4b. worst-flag severity ----------
    fa = w * a
    fb = w * b
    sev_a = float(np.max(fa)) if np.any(a > 0.5) else 0.0
    sev_b = float(np.max(fb)) if np.any(b > 0.5) else 0.0

    # signed badness difference (A - B); severity enters the NUMERATOR only
    D = (mass_a - mass_b) + lam * (sev_a - sev_b)
    scale = mass_a + mass_b                      # compression scale = plain flag mass

    def _logistic_diff(diff, sc):
        s = abs(diff) / (1.0 + c * max(sc, 0.0))
        z = float(np.clip(beta * s, -50.0, 50.0))
        core = 1.0 / (1.0 + np.exp(-z))          # P(choose the LESS bad option)
        core = float(np.clip(core, 0.0, 1.0))
        win = 0 if diff < 0 else 1               # less badness is chosen
        out = np.empty(2, dtype=float)
        out[win] = core
        out[1 - win] = 1.0 - core
        return out

    if abs(D) >= tau and abs(D) > 1e-9:
        # ------------- 5. divisive compression + logistic on badness -------
        p_core = _logistic_diff(D, scale)
    else:
        # ------------- 6. near-tie cascade -------------------------------
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if abs(ca - cb) > 1e-9:
            # (i) badness unresolvable -> fall back on raw flag counting
            p_core = _logistic_diff(ca - cb, ca + cb)
        else:
            # (ii) counts tie too -> worst discriminating flag, strongly repellent
            owner = None
            order = np.argsort(-val, kind="stable")
            for j in order:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
            if owner is None:
                p_core = np.ones(2) / 2.0
            else:
                p_core = np.empty(2, dtype=float)
                p_core[owner] = tie_bias         # option carrying the worst flag
                p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- gamma: [0.0, 1.25]
- severity: [0.55, 1.65]
- beta: [1.8, 5.4]
- comp_c: [0.12, 0.28]
- tau: [0.10, 0.55]
- epsilon: [0.18, 0.40]
- tie_bias: [0.0, 0.14]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.7733 (var=0.0059)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.6247 (var=0.0113)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=-0.6000 (var=0.0208)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.6867 (var=0.0243)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.5167 (var=0.0572)
- Experiment 6: real=0.0667 (var=0.0622) vs this=0.1583 (var=0.1034)
- Experiment 7: real=0.7733 (var=0.0327) vs this=0.7042 (var=0.0140)
- Experiment 8: real=0.1660 (var=0.0789) vs this=0.1430 (var=0.0116)
- Experiment 9: real=0.8720 (var=0.0086) vs this=0.7127 (var=0.0095)
- Experiment 10: real=0.1560 (var=0.1289) vs this=0.2620 (var=0.0844)


---

### `pi_5` (overall score: 0.779)

**Description**
**Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:

1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.

2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.

3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.

4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.

WC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n = a.size

    # ---------------- 2. validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.9, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    theta = float(parameters["theta"])
    kappa = float(parameters["kappa"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    ip = float(np.clip(float(parameters["ignore_prone"]), 0.0, 1.0))
    theta = max(theta, 1e-3)
    kappa = max(kappa, 0.5)

    # ---------------- 4. discrete expert triage (minority of subjects) ----
    # most subjects count every expert; a minority discard the 1 (or 2)
    # least-valid experts outright before counting.
    if ip < 0.84:
        n_drop = 0
    elif ip < 0.96:
        n_drop = 1
    else:
        n_drop = 2
    mask = np.ones(n, dtype=bool)
    if n_drop > 0 and n > 2:
        order_asc = np.argsort(val, kind="stable")   # ascending validity
        dropped = 0
        for idx in order_asc:
            if dropped >= n_drop or (n - dropped) <= 2:
                break
            if val[idx] < 0.65:                       # only near-chance experts
                mask[idx] = False
                dropped += 1
    af = a[mask]
    bf = b[mask]
    if af.size == 0:
        af, bf, mask = a, b, np.ones(n, dtype=bool)

    # ---------------- 5. Weber-noisy numerosity comparison --------------
    ca = float(np.sum(af))
    cb = float(np.sum(bf))

    if abs(ca - cb) > 1e-9:
        c_hi = max(ca, cb)
        c_lo = min(ca, cb)
        denom = np.sqrt(c_hi * c_hi + c_lo * c_lo + 1.0)
        s = (c_hi - c_lo) / denom               # ratio-scaled evidence
        x = (s / theta) ** kappa
        x = float(np.clip(x, 0.0, 50.0))
        core = 1.0 - 0.5 * np.exp(-x)           # Weibull psychometric fn
        core = float(np.clip(core, 0.5, 1.0))
        winner = 0 if ca < cb else 1            # fewer flags is chosen
        p_core = np.empty(2, dtype=float)
        p_core[winner] = core
        p_core[1 - winner] = 1.0 - core
    else:
        # ------------- 6. severity tiebreak on exact count ties ----------
        idxs = np.where(mask)[0]
        owner = None
        order = idxs[np.argsort(-val[idxs], kind="stable")]
        for j in order:
            if a[j] > b[j]:
                owner = 0
                break
            if b[j] > a[j]:
                owner = 1
                break
        if owner is None:                        # fall back to full cue set
            order_all = np.argsort(-val, kind="stable")
            for j in order_all:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
        if owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.empty(2, dtype=float)
            p_core[owner] = tie_bias             # option carrying the worst flag
            p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- theta: [0.34, 0.46]
- kappa: [4.5, 7.5]
- epsilon: [0.12, 0.48]
- tie_bias: [0.05, 0.45]
- ignore_prone: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.7900 (var=0.0075)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.6361 (var=0.0184)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=-0.5296 (var=0.0160)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.6633 (var=0.0347)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.5350 (var=0.0250)
- Experiment 6: real=0.0667 (var=0.0622) vs this=0.1433 (var=0.1053)
- Experiment 7: real=0.7733 (var=0.0327) vs this=0.5408 (var=0.0203)
- Experiment 8: real=0.1660 (var=0.0789) vs this=0.2835 (var=0.0155)
- Experiment 9: real=0.8720 (var=0.0086) vs this=0.5227 (var=0.0061)
- Experiment 10: real=0.1560 (var=0.1289) vs this=0.0080 (var=0.0347)


---

### `pi_4` (overall score: 0.662)

**Description**
**Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [1.3, 2.7]
- epsilon: [0.05, 0.25]
- tie_bias: [0.10, 0.70]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.8083 (var=0.0076)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.6325 (var=0.0187)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=-0.5608 (var=0.0168)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.6133 (var=0.0274)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.8367 (var=0.0105)
- Experiment 6: real=0.0667 (var=0.0622) vs this=-0.0117 (var=0.0319)
- Experiment 7: real=0.7733 (var=0.0327) vs this=0.8217 (var=0.0065)
- Experiment 8: real=0.1660 (var=0.0789) vs this=0.0065 (var=0.0132)
- Experiment 9: real=0.8720 (var=0.0086) vs this=0.8093 (var=0.0072)
- Experiment 10: real=0.1560 (var=0.1289) vs this=0.0120 (var=0.0219)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def ttb_winner(a, b):
        # features are ordered by descending validity (index 0 = 0.9, ... index 4 = 0.55)
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    def tally_winner(a, b):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            return 0
        if b_wins > a_wins:
            return 1
        return -1

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row["option_a_ratings"]]
            b = [int(v) for v in row["option_b_ratings"]]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        w_ttb = ttb_winner(a, b)
        w_tal = tally_winner(a, b)
        # keep only trials where the two heuristics make opposite, well-defined predictions
        if w_ttb < 0 or w_tal < 0 or w_ttb == w_tal:
            continue
        resp = row["response"]
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        n += 1
        if int(resp) == w_ttb:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7744 (var=0.0262)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8172 (var=0.0109)
- pi_2: 0.1522 (var=0.0104)
- pi_3: 0.7611 (var=0.0116)
- pi_4: 0.8083 (var=0.0076)
- pi_5: 0.7900 (var=0.0075)
- pi_6: 0.7733 (var=0.0059)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agree_hits = []
    conflict_hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        diff = a - b
        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            continue
        # features are listed in descending validity order in this design
        ttb_choice = 0 if diff[nz[0]] > 0 else 1
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin == 0:
            continue
        tally_choice = 0 if margin > 0 else 1
        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        chose_ttb = 1.0 if resp == ttb_choice else 0.0
        if tally_choice == ttb_choice and abs(margin) >= 3:
            agree_hits.append(chose_ttb)
        elif tally_choice != ttb_choice and abs(margin) >= 2:
            conflict_hits.append(chose_ttb)
    if len(agree_hits) == 0 or len(conflict_hits) == 0:
        return 0.0
    return float(np.mean(agree_hits) - np.mean(conflict_hits))
```

**Observed (real) value:** -0.6339 (var=0.0508)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7178 (var=0.0313)
- pi_1: -0.0008 (var=0.0076)
- pi_3: -0.6858 (var=0.0254)
- pi_4: -0.6325 (var=0.0187)
- pi_5: -0.6361 (var=0.0184)
- pi_6: -0.6247 (var=0.0113)

### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def key(x):
        return ''.join([str(int(v)) for v in list(x)])

    # Class I / II 'polarity-reversal' pairs: the option favoured by the
    # top-validity expert ALSO carries the larger weighted mass of 1s.
    rev = {
        ('11111', '00000'), ('00000', '11111'),
        ('11100', '00011'), ('00011', '11100'),
        ('10000', '00000'), ('00000', '10000'),
        ('10000', '00010'), ('00010', '10000'),
    }
    # Class IV 'agreement' pairs: top-cue option carries the SMALLER mass of 1s.
    agr = {
        ('10000', '01111'), ('01111', '10000'),
        ('10000', '01100'), ('01100', '10000'),
    }

    hits_rev = []
    hits_agr = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            ka = key(a)
            kb = key(b)
        except Exception:
            continue
        pair = (ka, kb)
        in_rev = pair in rev
        in_agr = pair in agr
        if not (in_rev or in_agr):
            continue
        aa = int(list(a)[0])
        bb = int(list(b)[0])
        if aa == bb:
            continue
        winner = 0 if aa == 1 else 1
        hit = 1.0 if int(r) == winner else 0.0
        if in_rev:
            hits_rev.append(hit)
        else:
            hits_agr.append(hit)

    if len(hits_rev) == 0 or len(hits_agr) == 0:
        return 0.0
    return float(np.mean(hits_rev) - np.mean(hits_agr))
```

**Observed (real) value:** -0.5767 (var=0.0708)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0012 (var=0.0066)
- pi_3: -0.6121 (var=0.0311)
- pi_2: 0.5846 (var=0.0241)
- pi_4: -0.5608 (var=0.0168)
- pi_5: -0.5296 (var=0.0160)
- pi_6: -0.6000 (var=0.0208)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def as_str(x):
        return ''.join([str(int(v)) for v in list(x)])

    dom_hits = []
    conf_hits = []

    for _, row in data.iterrows():
        a = [int(v) for v in list(row['option_a_ratings'])]
        b = [int(v) for v in list(row['option_b_ratings'])]
        if len(a) != len(b) or len(a) == 0:
            continue
        # Cue order = feature index order (validities are strictly descending by design)
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        sa = as_str(a)
        sb = as_str(b)
        key = tuple(sorted([sa, sb]))

        # DOMINANCE base pair: [1,0,0,0,0] vs [0,0,0,0,0]  (trials 1 and 9)
        if key == ('00000', '10000'):
            dom_hits.append(hit)
        # CONFLICT base pair: [1,0,0,0,0] vs [0,1,1,1,1]  (trials 4 and 12)
        elif key == ('01111', '10000'):
            conf_hits.append(hit)

    if len(dom_hits) == 0 or len(conf_hits) == 0:
        return 0.0

    return float(np.mean(conf_hits) - np.mean(dom_hits))
```

**Observed (real) value:** 0.6600 (var=0.0633)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6550 (var=0.0311)
- pi_1: 0.0183 (var=0.0212)
- pi_2: -0.6800 (var=0.0529)
- pi_4: 0.6133 (var=0.0274)
- pi_5: 0.6633 (var=0.0347)
- pi_6: 0.6867 (var=0.0243)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # design-time validities (fixed by the experiment)
    val_default = np.array([0.97, 0.66, 0.60, 0.56, 0.52], dtype=float)

    def weights(n):
        if n == len(val_default):
            v = val_default.copy()
        elif n == 1:
            v = np.array([0.9])
        else:
            v = np.linspace(0.9, 0.55, n)
        r = np.clip((v - 0.5) / 0.5, 1e-6, 1.0)
        return r / float(np.mean(r))

    hits_strong = []
    hits_any = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(row["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(row["option_b_ratings"], dtype=float).ravel()
            resp = int(row["response"])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        w = weights(a.size)
        ca = float(a.sum())
        cb = float(b.sum())
        if abs(ca - cb) < 1e-9:
            continue  # equal-count pairs are not informative here
        ma = float(np.dot(w, a))
        mb = float(np.dot(w, b))
        fewer = 0 if ca < cb else 1
        # does the option with FEWER flags carry the LARGER validity-weighted mass?
        conflict = (ma > mb) if fewer == 0 else (mb > ma)
        if not conflict:
            continue
        hit = 1.0 if resp == fewer else 0.0
        hits_any.append(hit)
        if abs(ma - mb) > 1.5:
            hits_strong.append(hit)

    if len(hits_strong) > 0:
        return float(np.mean(hits_strong))
    if len(hits_any) > 0:
        return float(np.mean(hits_any))
    return float("nan")

```

**Observed (real) value:** 0.5167 (var=0.1333)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8367 (var=0.0105)
- pi_3: 0.2383 (var=0.0433)
- pi_1: 0.8500 (var=0.0208)
- pi_2: 0.1383 (var=0.0121)
- pi_5: 0.5350 (var=0.0250)
- pi_6: 0.5167 (var=0.0572)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    tgt1 = tuple(sorted([(1, 0, 0, 0, 1), (0, 1, 1, 0, 0)]))
    tgt2 = tuple(sorted([(1, 0, 0, 0, 0), (0, 0, 0, 0, 1)]))

    x1 = []
    x2 = []

    for _, row in data.iterrows():
        try:
            a = tuple(int(v) for v in row["option_a_ratings"])
            b = tuple(int(v) for v in row["option_b_ratings"])
            r = int(row["response"])
        except Exception:
            continue
        if len(a) != 5 or len(b) != 5:
            continue
        key = tuple(sorted([a, b]))
        if key != tgt1 and key != tgt2:
            continue
        chosen = a if r == 0 else b
        other = b if r == 0 else a
        # feature 0 (validity 0.90) must discriminate on these pairs
        if chosen[0] == other[0]:
            continue
        hit = 1.0 if chosen[0] == 1 else 0.0
        if key == tgt1:
            x1.append(hit)
        else:
            x2.append(hit)

    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    return float(np.mean(x1) - np.mean(x2))
```

**Observed (real) value:** 0.0667 (var=0.0622)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5383 (var=0.0564)
- pi_4: -0.0117 (var=0.0319)
- pi_1: -0.0100 (var=0.0127)
- pi_2: -0.0150 (var=0.0488)
- pi_5: 0.1433 (var=0.1053)
- pi_6: 0.1583 (var=0.1034)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        n = a.size
        ca = float(a.sum())
        cb = float(b.sum())
        d = ca - cb
        # keep only constant-difference-of-one comparisons with a LARGE base count
        if abs(abs(d) - 1.0) > 1e-9:
            continue
        if min(ca, cb) < 2.0 - 1e-9:
            continue
        # drop triage-sensitive pairs: the sign of the count difference must be
        # preserved when the two least-valid experts are discarded
        k = n - 2
        if k < 1:
            continue
        dh = float(a[:k].sum()) - float(b[:k].sum())
        if abs(dh) < 1e-9:
            continue
        if np.sign(dh) != np.sign(d):
            continue
        winner = 0 if d < 0 else 1  # fewer flags wins
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        tot += 1
        if resp == winner:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.7733 (var=0.0327)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8217 (var=0.0065)
- pi_5: 0.5408 (var=0.0203)
- pi_1: 0.5058 (var=0.0051)
- pi_2: 0.1475 (var=0.0108)
- pi_3: 0.7983 (var=0.0115)
- pi_6: 0.7042 (var=0.0140)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    if data is None or len(data) == 0:
        return 0.0

    easy_pairs = {(0, 1), (1, 2), (2, 4), (3, 6)}
    hard_pairs = {(2, 3), (3, 4)}

    easy_hits = []
    hard_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row["option_a_ratings"]), dtype=float).ravel()
            b = np.asarray(list(row["option_b_ratings"]), dtype=float).ravel()
            resp = int(row["response"])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        ca = int(round(float(np.sum(a))))
        cb = int(round(float(np.sum(b))))
        if ca == cb:
            continue
        cp = (min(ca, cb), max(ca, cb))
        # winner = option with FEWER flags (both theories agree on this direction)
        winner = 0 if ca < cb else 1
        hit = 1.0 if resp == winner else 0.0

        n = a.size
        if n >= 3:
            low_flag = bool(np.any(a[-2:] > 0.5) or np.any(b[-2:] > 0.5))
        else:
            low_flag = False

        if cp in easy_pairs:
            easy_hits.append(hit)
        elif cp in hard_pairs and not low_flag:
            hard_hits.append(hit)

    if len(easy_hits) == 0 or len(hard_hits) == 0:
        return 0.0

    return float(np.mean(easy_hits) - np.mean(hard_hits))
```

**Observed (real) value:** 0.1660 (var=0.0789)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2835 (var=0.0155)
- pi_4: 0.0065 (var=0.0132)
- pi_1: -0.3640 (var=0.0215)
- pi_2: 0.0000 (var=0.0073)
- pi_3: 0.1210 (var=0.0253)
- pi_6: 0.1430 (var=0.0116)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    if data is None or len(data) == 0:
        return 0.5

    hits = 0
    total = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if abs(ca - cb) != 1.0:
            continue
        # strictly nested pairs only (the difference is a single shared cue)
        nested = bool(np.all(a <= b + 1e-9)) or bool(np.all(b <= a + 1e-9))
        if not nested:
            continue
        # high base-count region only: the smaller option already carries >= 2 flags
        if min(ca, cb) < 2.0:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        better = 0 if ca < cb else 1  # fewer warning flags is the 'correct' option
        total += 1
        if resp == better:
            hits += 1

    if total == 0:
        return 0.5
    return float(hits) / float(total)
```

**Observed (real) value:** 0.8720 (var=0.0086)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7127 (var=0.0095)
- pi_5: 0.5227 (var=0.0061)
- pi_1: 0.1333 (var=0.0092)
- pi_2: 0.1760 (var=0.0144)
- pi_3: 0.7907 (var=0.0112)
- pi_4: 0.8093 (var=0.0072)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    hits_no_top = []   # fewer-flag option does NOT carry the top expert's flag (P7-type)
    hits_top = []      # fewer-flag option DOES carry the top expert's flag (P6-type)

    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) < 1:
            continue
        ca = sum(a)
        cb = sum(b)
        # matched-composition contrast: 2-flag vs 3-flag pairs
        if sorted([ca, cb]) != [2, 3]:
            continue
        # top-validity expert (index 0) must discriminate the two options
        if a[0] == b[0]:
            continue
        # which side has fewer flags?
        if ca < cb:
            fewer_side = 0
            fewer_vec = a
        else:
            fewer_side = 1
            fewer_vec = b
        chose_fewer = 1.0 if resp == fewer_side else 0.0
        if fewer_vec[0] == 1:
            hits_top.append(chose_fewer)
        else:
            hits_no_top.append(chose_fewer)

    if len(hits_top) == 0 or len(hits_no_top) == 0:
        return 0.0

    return float(np.mean(hits_no_top) - np.mean(hits_top))

```

**Observed (real) value:** 0.1560 (var=0.1289)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0080 (var=0.0347)
- pi_6: 0.2620 (var=0.0844)
- pi_1: -0.7480 (var=0.0373)
- pi_2: -0.0380 (var=0.0144)
- pi_3: 0.3800 (var=0.1248)
- pi_4: 0.0120 (var=0.0219)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "**Sign-Dominant Flag Difference with Configural Dominance and a Weighting Sub-population (SD-FD/CD-M).**\n\nA binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the flags it carries and subjects pick the option that seems *less* flagged. Six claims:\n\n1. **Negative polarity (domain-invariant).** Badness is monotone in flags; the less-flagged option is chosen. This single fact makes the population look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs (Exps 2–4, 7).\n\n2. **Sign-dominant, magnitude-invariant difference detection.** The primary decision variable is the *flag-count difference*, and the psychometric function is almost completely saturating in |Δ|: a one-flag advantage is nearly as diagnostic as a four-flag advantage (t_c = sign(Δ)·|Δ|^rho with rho≈0.08). This predicts the empirically FLAT ladder — Exp 1 (Δ=1,2,3 mixed) = Exp 7 (Δ=1) = Exp 2-conflict (Δ=2,3,4) ≈ 0.78 — and explicitly denies both a divisive/Weber compression by total mass and any steep monotone decline.\n\n3. **Configural dominance detection is the only large discriminability boost.** When one option's flag *set* is a subset of the other's (componentwise dominance, i.e. the options differ on a single side only), the comparison is solved perceptually rather than by tallying, and accuracy jumps to ~0.88–0.91 irrespective of Δ. This is why nested Δ=1 pairs (Exp 9, 0.872) are far more accurate than non-nested Δ=1 pairs (Exp 7, 0.773) even at base counts of 2–4 — a dissociation no pure magnitude model can produce.\n\n4. **A small additive base-count interference cost.** Discriminability falls by a few points per shared flag already on the table (gain × (1 − kappa·min(base,6)), kappa ≈ 0.05), saturating and never collapsing to chance. This yields Exp 10's and part of Exp 8's small positive easy-minus-hard gap without a Weber threshold.\n\n5. **A bimodal counter/weigher mixture whose expression is design-scaled.** About half the population are validity-blind counters (phi = 0). The rest blend in a validity-weighted flag mass (weights u_j = r_j/mean(r), r_j = 2v_j − 1) with weight phi ∈ [0.5, 0.95]. Because the mass term is proportional (not centred), its ability to *override* the count depends on the design's validity spread: it reverses decisively when one expert dominates (Exp 5's 0.97/0.6x design → group mean 0.52 with LARGE between-subject variance), only mildly perturbs graded-validity designs (Exps 8, 10 composition contrasts of ~0.1–0.2), and leaves moderate-spread agreement pairs intact (Exps 2, 3). Dispersion is therefore bimodal on composition contrasts and unimodal (lapse/sensitivity driven) on the magnitude ladder.\n\n6. **Count-ties are solved by everyone the same way.** When the raw flag counts are exactly equal the count read is silent for counters *and* weighers alike; all subjects fall back on 'who complained loudest', being mildly repelled by the option carrying the top discriminating expert's flag (p ≈ 0.25–0.40). Because this rule is identical across equal-count pairs regardless of how many flags are on the table, it predicts a near-zero differential among equal-count contrasts (Exp 6 ≈ 0), unlike graded weighted-mass accounts.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------------- 1. parse stimulus ----------------\n    a = b = None\n    if isinstance(state, dict):\n        if \"option_a_ratings\" in state and \"option_b_ratings\" in state:\n            a = np.asarray(state[\"option_a_ratings\"], dtype=float).ravel()\n            b = np.asarray(state[\"option_b_ratings\"], dtype=float).ravel()\n    if a is None:\n        stim = np.asarray(state, dtype=float)\n        if stim.ndim == 1:\n            if stim.size % 2 != 0:\n                return np.ones(2) / 2.0\n            half = stim.size // 2\n            a, b = stim[:half], stim[half:]\n        else:\n            stim = stim.reshape(stim.shape[0], -1)\n            if stim.shape[0] < 2:\n                return np.ones(2) / 2.0\n            a, b = stim[0], stim[1]\n    a = np.asarray(a, dtype=float).ravel()\n    b = np.asarray(b, dtype=float).ravel()\n    if a.size == 0 or a.size != b.size:\n        return np.ones(2) / 2.0\n    n = a.size\n\n    # ---------------- 2. design validities ----------------\n    val = parameters.get(\"validities\", None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = np.array([])\n    if val.size != n:\n        val = np.array([0.9]) if n == 1 else np.linspace(0.90, 0.55, n)\n    val = np.clip(val, 0.5, 1.0)\n\n    # ---------------- 3. subject-level parameters ----------------\n    beta = max(float(parameters[\"beta\"]), 0.0)\n    rho = float(np.clip(float(parameters[\"rho\"]), 0.0, 1.0))\n    nu = max(float(parameters[\"nu\"]), 0.0)\n    kappa = max(float(parameters[\"kappa\"]), 0.0)\n    mu = max(float(parameters[\"mu\"]), 0.0)\n    eps = float(np.clip(float(parameters[\"epsilon\"]), 0.0, 1.0))\n    tie_bias = float(np.clip(float(parameters[\"tie_bias\"]), 0.0, 1.0))\n    wp = float(np.clip(float(parameters[\"w_prone\"]), 0.0, 1.0))\n\n    # bimodal mixture: ~52% pure counters (phi = 0), rest weighers (phi in [0.5,0.95])\n    W_THRESH = 0.52\n    if wp < W_THRESH:\n        phi = 0.0\n    else:\n        phi = 0.50 + 0.45 * (wp - W_THRESH) / (1.0 - W_THRESH)\n    phi = float(np.clip(phi, 0.0, 1.0))\n\n    # ---------------- 4. flag counts and weighted flag mass ------------\n    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)\n    rbar = float(np.mean(r))\n    if rbar <= 1e-9:\n        u = np.ones(n)\n    else:\n        u = r / rbar                      # mean-1 proportional validity weights\n\n    ca = float(np.sum(a))\n    cb = float(np.sum(b))\n    d = ca - cb                            # >0 : A carries more warning flags\n    ma = float(np.dot(u, a))\n    mb = float(np.dot(u, b))\n    base = min(ca, cb)                     # shared flag load already on the table\n\n    # configural dominance: one flag set is a subset of the other\n    nested = bool(np.all(a <= b + 1e-9) or bool(np.all(b <= a + 1e-9)))\n\n    # ---------------- 5. decision ----------------\n    if abs(d) > 1e-9:\n        # sign-dominant (saturating) count evidence, positive => choose A\n        t_c = -np.sign(d) * (abs(d) ** rho)\n        # validity-weighted flag-mass evidence, positive => choose A\n        t_m = mu * (mb - ma)\n        V = (1.0 - phi) * t_c + phi * t_m\n\n        gain = (1.0 + (nu if nested else 0.0)) * max(0.25, 1.0 - kappa * min(base, 6.0))\n        z = float(np.clip(beta * gain * V, -50.0, 50.0))\n        pA = 1.0 / (1.0 + np.exp(-z))\n        p_core = np.array([pA, 1.0 - pA], dtype=float)\n    else:\n        # ---------- 6. exact count tie: worst-flag mild repulsion ----------\n        owner = None\n        order = np.argsort(-val, kind=\"stable\")\n        for j in order:\n            if a[j] > b[j] + 1e-9:\n                owner = 0\n                break\n            if b[j] > a[j] + 1e-9:\n                owner = 1\n                break\n        if owner is None:\n            p_core = np.ones(2) / 2.0\n        else:\n            p_core = np.empty(2, dtype=float)\n            p_core[owner] = tie_bias       # option carrying the top discriminating flag\n            p_core[1 - owner] = 1.0 - tie_bias\n\n    if not np.all(np.isfinite(p_core)):\n        p_core = np.ones(2) / 2.0\n\n    # ---------------- 7. attention lapse ----------------\n    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)\n    p = np.clip(p, 1e-12, 1.0)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        probs = np.ones(len(probs)) / len(probs)\n    else:\n        probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "beta": "[1.5, 2.7]",
    "rho": "[0.0, 0.16]",
    "nu": "[0.85, 1.5]",
    "kappa": "[0.02, 0.09]",
    "mu": "[0.40, 0.65]",
    "epsilon": "[0.08, 0.24]",
    "tie_bias": "[0.18, 0.45]",
    "w_prone": "[0, 1]",
    "validities": "validities"
  },
  "rationale": "The arbiter's diagnosis is right on two counts and I implement both: (i) the magnitude ladder is FLAT, not compressed — Exp 1 (Δ=1,2,3 mixed) = 0.774, Exp 7 (Δ=1, base≥2) = 0.773 and Exp 2's conflict cell (Δ=2,3,4) all sit at ~0.78, so pi_6's divisive compression by total flag mass and pi_5's Weber threshold both mis-shape the data; (ii) composition effects must be small in the mean but bimodally dispersed, which needs a mixture rather than a population-wide severity term.\n\nWhere I go beyond the arbiter is the source of the *one* genuinely high-accuracy pocket. Exp 9 (0.872) uses strictly NESTED Δ=1 pairs at base counts 2–4, while Exp 7 (0.773) uses NON-nested Δ=1 pairs at the same base counts. A pure magnitude/base account cannot separate them — that is exactly why pi_6 bottomed out at 0.713 on Exp 9 and pi_5 at 0.523. I therefore add a *configural dominance detector*: when one option's flag set is a subset of the other's, the option differs on a single side and the judgement is made perceptually, multiplying discriminability by (1+nu). This simultaneously (a) lifts Exp 9 to ~0.88, (b) leaves Exp 1/7/2-conflict flat at ~0.77, (c) keeps Exp 3's reversal pairs ('11111' vs '00000', '10000' vs '00000' — both nested) very accurate, reproducing the −0.58 contrast, (d) supplies the (0,1) nested trial that is a large part of Exp 8's easy-minus-hard gap, and (e) makes the dominance baseline in Exp 4 near-perfect, giving the +0.66 conflict-minus-dominance contrast.\n\nThe severity/weighting machinery is replaced by a spike-and-slab mixture: ~52% of subjects are strict validity-blind counters, the remainder blend in a *proportional* (not centred) validity-weighted flag mass. Because the weights are proportional to (2v−1)/mean, the weigher's override strength scales with the design's validity spread: it flips decisively in Exp 5's 0.97/0.6x design (group mean ≈0.53 with between-subject variance ≈0.10–0.12, matching the observed 0.5167/0.133 bimodality that a graded gamma-continuum cannot produce), only mildly perturbs graded designs (Exp 10 ≈0.10–0.20, Exp 8 ≈0.10–0.13 versus pi_6's inflated 0.262), and barely touches Exps 2/3 agreement pairs, where mass still tracks the count.\n\nFinally, count ties route ALL subjects — counters and weighers alike — to the same mild worst-flag repulsion. Since both Exp 6 target pairs are exact count ties broken by the same top expert, this predicts a differential of ~0 (real 0.0667, versus pi_5's 0.143 and pi_6's 0.158) with the right binomial-sized dispersion, while the base-count interference term (a few points per shared flag, saturating) supplies the residual easy-minus-hard gaps without any collapse toward chance."
}
```

## Usage

```json
{
  "input_tokens": 29812,
  "output_tokens": 55778
}
```
