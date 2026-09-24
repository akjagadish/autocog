# generation_iter_02_attempt_00

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
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_11`).

Replace SSG-6 entirely. Its two defining commitments 
s (a) pi_one = 0 whenever the discriminating rows are unanimous, and (b) the relative salience-share gate share = sum g_j o_j / sum o_j 
s are both directly falsified: observed anchor-follow on unanimous anchor-carried displays is .86-.88 (SSG predicted .67-.72), and it does not decline as mid-doubt opponents accumulate. Re-parameterising cannot save it without erasing the description.

Propose a NEW theory that is a strong competitor to DTG-2R/OC-sharp rather than a clone. Suggested sketch 
s call it 'AUTHORITY-INVERSION with a graded cost-of-integration' (AIG):

1. Keep what is overwhelmingly supported: an ABSOLUTE, sharp near-chance criterion on PRINTED validity. When some discriminating expert prints v near .50-.55, that expert's verdict is adopted essentially wholesale. Crucially, make the adoption flat in BOTH display length AND opposition: the retained fidelity should be ~.86-.90 with at most a very mild opposition charge (rho_opp in .90-1.0), because the observed anchor-follow is .862 (Exp 1), .875 (Exp 2), .862 (Exp 17, all coalition trials), .826 (Exp 18, >=2 late mid-doubt opponents) 
s all AT or ABOVE DTG's predictions, which currently under-shoot Exps 17 and 1/2. This is the one place where DTG is systematically biased and where a new theory can win.

2. Differ from DTG in the FALLBACK. Instead of DTG's divisively-normalised anti-validity sum with a load leak, make the no-anchor case a SINGLE-CUE lottery over rows sampled with probability proportional to salience s_j = (1-v_j)^gamma * exp(lam*j) (a probabilistic one-reason / 'sampling the oddest late row' rule) rather than a compensatory sum. A sampling rule naturally reproduces (i) the ~.70-.75 ceiling on long unanimous non-anchor displays (Exp 7 obs .743) without needing a large explicit eps0, because samples from a unanimous display all agree but an occasional lapse/guess remains; (ii) the strongly below-chance validity-weighted metrics of Exps 3, 5, 6, 12, 14 (obs .28, .29, .15, .42, .26) because the sampled row is almost always the LEAST valid / latest one; (iii) crucially it predicts a SMALLER tally-vs-TTB gap than a compensatory sum on Exp 4 (obs .174, DTG over-predicts at .318), because a sampler does not add coalition members the way a weighted sum does. That Exp-4 discrepancy is the clearest remaining crack in DTG and should be the new theory's target.

3. Make the near-chance trigger and the sampler ONE mechanism rather than two limbs: p(consult row j) = softmax over delta*log(1-v_j) + lam*j with a LARGE contrast exponent delta (so a .52 row wins the lottery with probability ~1 whenever it speaks, reproducing the .86 anchor-follow automatically), and a moderate delta-driven spread among mid rows when no anchor speaks. Add a single lapse/guess rate eps (.12-.20) and, optionally, a mild length-dependent guess increment to match the .74 dominance ceiling. This gives a parsimonious, genuinely different architecture (one graded sampling rule, no mode-switch gate, no compensatory integration) that makes distinguishable predictions from DTG on: coalition-size effects with the anchor silent (Exps 15, 16, 19, where observed values hug .47-.55 
s a pure sampler predicts near-chance there naturally), the Exp-4 style tally-vs-TTB difference, and the exact ceiling of long unanimous displays.

4. Retain the degeneracy clause (validities within .002 -> unweighted graded row tally) and stationarity, since nothing in the data contradicts them.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.926)

**Description**
DTG-2R/OC-sharp (load-graded compensatory limb) — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion, the one-reason limb is degraded by OPPOSITION rather than by raw display length, and the compensatory limb is degraded by DISPLAY LOAD more steeply than previously assumed. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), with theta ~ .558-.588 and tau ~ .006-.013. An expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is ordinary evidence, not a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows. (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86). Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits drift toward chance. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E). NEW, SHARPENED CLAIM: the compensatory rule is a genuinely effortful integration whose thread is lost quickly as the display lengthens — every discriminating row beyond the second multiplies the retained evidence by rho_m ~ .84-.93, appreciably steeper than the one-reason limb's near-free scan. This asymmetry is the theory's signature: one-reason choice is cheap to run on long displays and expensive only under opposition, whereas compensatory integration is expensive in sheer list length. Consequently long unanimous displays stay confident but clearly below ceiling (~.74-.76), and long conflict displays settle appreciably closer to chance than short ones, while short displays (m=2-3) are essentially unaffected. (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;
    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count,
    # compensatory limb degraded by DISPLAY LOAD (rho_m)
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
- rho_m: [0.84, 0.93]
- rho_c: [0.74, 0.90]
- tally_kappa: [0.6, 1.3]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2820 (var=0.0035) vs this=0.2522 (var=0.0026)
- Experiment 2: real=0.1744 (var=0.0127) vs this=0.3175 (var=0.0088)
- Experiment 3: real=0.2856 (var=0.0160) vs this=0.2650 (var=0.0064)
- Experiment 4: real=0.1500 (var=0.0194) vs this=0.1600 (var=0.0080)
- Experiment 5: real=0.7427 (var=0.0173) vs this=0.7707 (var=0.0058)
- Experiment 6: real=0.5067 (var=0.0112) vs this=0.5017 (var=0.0113)
- Experiment 7: real=0.4383 (var=0.0023) vs this=0.4500 (var=0.0038)
- Experiment 8: real=0.7697 (var=0.0057) vs this=0.7960 (var=0.0025)
- Experiment 9: real=0.8189 (var=0.0074) vs this=0.7953 (var=0.0023)
- Experiment 10: real=0.4178 (var=0.0051) vs this=0.4236 (var=0.0018)
- Experiment 11: real=0.5925 (var=0.0026) vs this=0.5946 (var=0.0036)
- Experiment 12: real=0.2608 (var=0.0079) vs this=0.2013 (var=0.0047)
- Experiment 13: real=0.5460 (var=0.0126) vs this=0.5080 (var=0.0118)
- Experiment 14: real=0.4675 (var=0.0144) vs this=0.5413 (var=0.0121)
- Experiment 15: real=0.8620 (var=0.0155) vs this=0.8087 (var=0.0029)
- Experiment 16: real=0.8260 (var=0.0180) vs this=0.8075 (var=0.0040)
- Experiment 17: real=0.4925 (var=0.0074) vs this=0.4906 (var=0.0058)
- Experiment 18: real=0.3183 (var=0.0405) vs this=0.3550 (var=0.0214)
- Experiment 19: real=0.8617 (var=0.0175) vs this=0.8450 (var=0.0051)
- Experiment 20: real=0.8750 (var=0.0100) vs this=0.8570 (var=0.0034)


---

### `pi_11` (overall score: 0.697)

**Description**
**SSG-6 — Salience-Share Gating with an opposition-graded adjudication limb and a leakier integration limb.**

The architecture is exactly SSG-5's; one quantitative commitment is revised, none is reversed.

**(1) Two modes: adjudication vs. integration.** On every trial the subject either lets ONE expert adjudicate the choice or integrates all disagreeing rows. Mode is decided by a RELATIVE quantity — the share of the display's total *oddness* owned by genuinely near-chance experts — not by display length nor by an absolute knife edge. Each discriminating row has oddness o_j = (1-v_j)^gamma_s (gamma_s ≈ 6.4-8.6, a contrast-enhanced, nearly ordinal 'who is the strangest expert here' read-out) and a scandal value g_j = sigmoid((v_crit - v_j)/w_h) with a crisp boundary (w_h ≈ .008-.011, v_crit ≈ .576-.586). The gate is share = Σ_j g_j·o_j / Σ_j o_j and pi_one = (max_j g_j)·[f + (1-f)·sigmoid((share - c)/w)]: adjudication requires BOTH that some expert be near-chance AND that near-chance experts own most of the display's oddness.

**(2) Adjudication is faithful but NOT flat in opposition; it is flat only in display length.** When a near-chance expert commands the trial, subjects follow its verdict with high fidelity (eps_t ≈ .19-.29) and scanning extra, silent or agreeing rows costs nothing at all — one-reason choice is free in sheer list length, which is why anchor-follow rates stay high on six-row displays. What it is NOT free in is genuine CONTRADICTION: each additional row that argues the other way beyond the first charges rho_opp ≈ .82-.90. A scandalous dissenter standing alone against one opposing voice is followed almost as often as a scandalous dissenter in a unanimous display, but once two or three experts are lined up the other way the adjudication itself becomes contested and the verdict is dragged appreciably toward chance. Among co-scandalous rows the consulted verdict is drawn from softmax(log g_j + mu·j) with a WEAK positional gain (mu ≈ .35-.62); reading order can never lift an ordinary expert over a near-chance one.

**(3) REVISED — integration is an intrinsically effortful, frequently-abandoned operation, and it is fragile under DISAGREEMENT.** When the gate is shut, evidence is integrated with w_j = (1-v_j)^gamma_w·exp(lam·j) (gamma_w ≈ 1.32-1.58, lam ≈ .10-.17, so doubt, not position, buys weight), divisively normalised to the loudest dissent, E = Σ sign_j·ŵ_j/(sigma + Σ ŵ_j), p = sigmoid(beta·E). The sharpened claim is that the BASELINE abandonment rate of integration is appreciably higher than previously assumed (eps0 ≈ .26-.34, not ≈ .22-.30): weighing several probabilistically-labelled experts against one another is an operation subjects decline to complete on roughly a third of trials even before any conflict or load penalty, so integrated verdicts are systematically less decided than the weights alone imply. On top of this baseline the retained evidence leaks further with display load (rho_m) and much more steeply with every additional row that argues the other way (rho_c ≈ .62-.75 per minority row). Because the baseline lapse applies to EVERY integration trial — including the unanimous/dominance displays where there is nothing to adjudicate — it predicts that even fully consistent multi-row displays stay clearly below ceiling (~.74-.75), while conflict displays are pulled nearer chance still and never reach a near-deterministic anti-validity verdict.

The theory's signature is therefore that BOTH modes are degraded by contradiction and NEITHER is degraded by list length — they differ in how steeply contradiction bites (mild for adjudication, severe for integration) and in their baseline completion rates (adjudication is cheap and usually completed; integration is expensive and often abandoned). When all discriminating experts agree there is nothing to adjudicate (pi_one = 0) and only the load/baseline leak applies. If two printed validities are indistinguishable (gap < .002) oddness has no referent and choice collapses to a graded unweighted row tally. No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # SSG: Salience-Share Gating
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
    gamma_s = float(parameters['gamma_s'])
    v_crit = float(parameters['v_crit'])
    w_h = float(max(float(parameters['w_h']), 1e-4))
    c_share = float(parameters['c_share'])
    w_share = float(max(float(parameters['w_share']), 1e-4))
    floor = float(np.clip(float(parameters['gate_floor']), 0.0, 1.0))
    mu = float(parameters['mu'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))
    gamma_w = float(parameters['gamma_w'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
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
    k_conf = int(min(n_a, n_b))

    # ---------- leak on the integration limb ----------
    keep_c = (1.0 - eps0) * (rho_m ** max(m - 2, 0)) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))

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

    vd = v[disc]
    jd = disc.astype(float)

    # ---------- scandal values and sharpened oddness ----------
    zz = np.clip((v_crit - vd) / w_h, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(-zz))                       # near-chance "scandal" weight
    log_g = -np.logaddexp(0.0, -zz)                     # stable log sigmoid
    log_o = gamma_s * np.log(np.clip(1.0 - vd, 1e-12, None))
    log_o_shift = log_o - np.max(log_o)
    o = np.exp(np.clip(log_o_shift, -700.0, 0.0))
    o_sum = float(np.sum(o))
    if (not np.isfinite(o_sum)) or o_sum <= 0.0:
        o = np.ones(m, dtype=float)
        o_sum = float(m)

    share = float(np.sum(g * o) / o_sum)
    g_share = sig((share - c_share) / w_share)
    P_trig = float(np.clip(np.max(g), 0.0, 1.0))

    # unanimous displays need no adjudication
    if k_conf == 0:
        pi_one = 0.0
    else:
        pi_one = P_trig * (floor + (1.0 - floor) * g_share)
    pi_one = float(np.clip(pi_one, 0.0, 1.0))

    # ---------- which row adjudicates (recency only breaks scandal ties) ----------
    sc = log_g + mu * jd
    sc = sc - np.max(sc)
    ww = np.exp(np.clip(sc, -700.0, 0.0))
    ws = float(np.sum(ww))
    if (not np.isfinite(ws)) or ws <= 0.0:
        ww = np.full(m, 1.0 / m)
    else:
        ww = ww / ws
    p_or_a = float(np.sum(ww[signs > 0.0]))
    fid = (1.0 - eps_t) * (rho_opp ** max(k_conf - 1, 0))
    fid = float(min(max(fid, 0.0), 1.0))
    p_one_a = 0.5 + fid * (p_or_a - 0.5)

    # ---------- integration limb ----------
    logw = gamma_w * np.log(np.clip(1.0 - vd, 1e-12, None)) + lam * jd
    logw = logw - np.max(logw)
    wi = np.exp(np.clip(logw, -700.0, 0.0))
    wsum = float(np.sum(wi))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        wi = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * wi)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c_a = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- mixture ----------
    p_a = pi_one * p_one_a + (1.0 - pi_one) * p_c_a
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
- gamma_s: [6.4, 8.6]
- v_crit: [0.576, 0.586]
- w_h: [0.008, 0.011]
- c_share: [0.69, 0.75]
- w_share: [0.040, 0.065]
- gate_floor: [0.33, 0.47]
- mu: [0.35, 0.62]
- eps_t: [0.19, 0.29]
- rho_opp: [0.82, 0.90]
- gamma_w: [1.32, 1.58]
- lam: [0.10, 0.17]
- sigma: [0.58, 0.78]
- beta: [2.15, 2.65]
- eps0: [0.26, 0.34]
- rho_m: [0.87, 0.93]
- rho_c: [0.62, 0.75]
- tally_kappa: [0.7, 1.3]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2820 (var=0.0035) vs this=0.2610 (var=0.0020)
- Experiment 2: real=0.1744 (var=0.0127) vs this=0.2156 (var=0.0155)
- Experiment 3: real=0.2856 (var=0.0160) vs this=0.2578 (var=0.0035)
- Experiment 4: real=0.1500 (var=0.0194) vs this=0.1267 (var=0.0084)
- Experiment 5: real=0.7427 (var=0.0173) vs this=0.7587 (var=0.0089)
- Experiment 6: real=0.5067 (var=0.0112) vs this=0.4925 (var=0.0142)
- Experiment 7: real=0.4383 (var=0.0023) vs this=0.4725 (var=0.0025)
- Experiment 8: real=0.7697 (var=0.0057) vs this=0.8106 (var=0.0027)
- Experiment 9: real=0.8189 (var=0.0074) vs this=0.8128 (var=0.0023)
- Experiment 10: real=0.4178 (var=0.0051) vs this=0.4203 (var=0.0023)
- Experiment 11: real=0.5925 (var=0.0026) vs this=0.6012 (var=0.0039)
- Experiment 12: real=0.2608 (var=0.0079) vs this=0.2096 (var=0.0037)
- Experiment 13: real=0.5460 (var=0.0126) vs this=0.4950 (var=0.0133)
- Experiment 14: real=0.4675 (var=0.0144) vs this=0.4950 (var=0.0100)
- Experiment 15: real=0.8620 (var=0.0155) vs this=0.8033 (var=0.0034)
- Experiment 16: real=0.8260 (var=0.0180) vs this=0.8455 (var=0.0028)
- Experiment 17: real=0.4925 (var=0.0074) vs this=0.5025 (var=0.0063)
- Experiment 18: real=0.3183 (var=0.0405) vs this=0.3325 (var=0.0167)
- Experiment 19: real=0.8617 (var=0.0175) vs this=0.6958 (var=0.0069)
- Experiment 20: real=0.8750 (var=0.0100) vs this=0.7045 (var=0.0056)


---

### `pi_7` (overall score: 0.685)

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
- Experiment 17: real=0.4925 (var=0.0074) vs this=0.6956 (var=0.0072)
- Experiment 18: real=0.3183 (var=0.0405) vs this=0.0125 (var=0.0220)
- Experiment 19: real=0.8617 (var=0.0175) vs this=0.7700 (var=0.0082)
- Experiment 20: real=0.8750 (var=0.0100) vs this=0.7355 (var=0.0059)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0575 -> ACCEPTED
- iter 2: loss=0.0457 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0457 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.2512 (var=0.0026) (Δ vs real -0.0307)
  - iter 2 (most recent): 0.2505 (var=0.0023) (Δ vs real -0.0315)
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
- pi_10: 0.2522 (var=0.0026)
- pi_11: 0.2610 (var=0.0020)

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
  - iter 1: 0.2431 (var=0.0117) (Δ vs real +0.0686)
  - iter 2 (most recent): 0.2231 (var=0.0119) (Δ vs real +0.0486)
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
- pi_10: 0.3175 (var=0.0088)
- pi_11: 0.2156 (var=0.0155)

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
  - iter 1: 0.2411 (var=0.0037) (Δ vs real -0.0444)
  - iter 2 (most recent): 0.2461 (var=0.0040) (Δ vs real -0.0394)
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
- pi_10: 0.2650 (var=0.0064)
- pi_11: 0.2578 (var=0.0035)

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
  - iter 1: 0.1283 (var=0.0089) (Δ vs real -0.0217)
  - iter 2 (most recent): 0.1250 (var=0.0103) (Δ vs real -0.0250)
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
- pi_10: 0.1600 (var=0.0080)
- pi_11: 0.1267 (var=0.0084)

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
  - iter 1: 0.8147 (var=0.0051) (Δ vs real +0.0720)
  - iter 2 (most recent): 0.7820 (var=0.0057) (Δ vs real +0.0393)
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
- pi_10: 0.7707 (var=0.0058)
- pi_11: 0.7587 (var=0.0089)

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
  - iter 1: 0.4975 (var=0.0112) (Δ vs real -0.0092)
  - iter 2 (most recent): 0.5017 (var=0.0124) (Δ vs real -0.0050)
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
- pi_10: 0.5017 (var=0.0113)
- pi_11: 0.4925 (var=0.0142)

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
  - iter 1: 0.4308 (var=0.0037) (Δ vs real -0.0075)
  - iter 2 (most recent): 0.4314 (var=0.0037) (Δ vs real -0.0069)
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
- pi_10: 0.4500 (var=0.0038)
- pi_11: 0.4725 (var=0.0025)

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
  - iter 1: 0.7877 (var=0.0018) (Δ vs real +0.0180)
  - iter 2 (most recent): 0.7860 (var=0.0022) (Δ vs real +0.0163)
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
- pi_10: 0.7960 (var=0.0025)
- pi_11: 0.8106 (var=0.0027)

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
  - iter 1: 0.8125 (var=0.0022) (Δ vs real -0.0064)
  - iter 2 (most recent): 0.8028 (var=0.0022) (Δ vs real -0.0161)
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
- pi_10: 0.7953 (var=0.0023)
- pi_11: 0.8128 (var=0.0023)

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
  - iter 1: 0.4342 (var=0.0024) (Δ vs real +0.0164)
  - iter 2 (most recent): 0.4336 (var=0.0034) (Δ vs real +0.0158)
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
- pi_10: 0.4236 (var=0.0018)
- pi_11: 0.4203 (var=0.0023)

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
  - iter 1: 0.5746 (var=0.0035) (Δ vs real -0.0179)
  - iter 2 (most recent): 0.5729 (var=0.0037) (Δ vs real -0.0196)
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
- pi_10: 0.5946 (var=0.0036)
- pi_11: 0.6012 (var=0.0039)

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
  - iter 1: 0.2329 (var=0.0035) (Δ vs real -0.0279)
  - iter 2 (most recent): 0.2179 (var=0.0040) (Δ vs real -0.0429)
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
- pi_10: 0.2013 (var=0.0047)
- pi_11: 0.2096 (var=0.0037)

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
  - iter 1: 0.4330 (var=0.0131) (Δ vs real -0.1130)
  - iter 2 (most recent): 0.4060 (var=0.0160) (Δ vs real -0.1400)
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
- pi_10: 0.5080 (var=0.0118)
- pi_11: 0.4950 (var=0.0133)

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
  - iter 1: 0.4250 (var=0.0114) (Δ vs real -0.0425)
  - iter 2 (most recent): 0.4300 (var=0.0179) (Δ vs real -0.0375)
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
- pi_10: 0.5413 (var=0.0121)
- pi_11: 0.4950 (var=0.0100)

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
  - iter 1: 0.8583 (var=0.0014) (Δ vs real -0.0037)
  - iter 2 (most recent): 0.8690 (var=0.0022) (Δ vs real +0.0070)
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
- pi_10: 0.8087 (var=0.0029)
- pi_11: 0.8033 (var=0.0034)

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
  - iter 1: 0.8490 (var=0.0027) (Δ vs real +0.0230)
  - iter 2 (most recent): 0.8365 (var=0.0044) (Δ vs real +0.0105)
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
- pi_10: 0.8075 (var=0.0040)
- pi_11: 0.8455 (var=0.0028)

### Experiment 17
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = [int(x) for x in a_raw]
            b = [int(x) for x in b_raw]
        except Exception:
            continue
        if len(a) < 6 or len(b) < 6:
            continue
        # expert 4 (validity .52, the near-chance 'anchor') must be SILENT
        if a[4] != b[4]:
            continue
        # expert 5 (validity .66, mid-doubt) must discriminate
        d5 = a[5] - b[5]
        if d5 == 0:
            continue
        # there must be a genuine opposing coalition among the more valid experts 0..3
        opposed = False
        for j in range(4):
            dj = a[j] - b[j]
            if dj != 0 and (dj * d5) < 0:
                opposed = True
                break
        if not opposed:
            continue
        fav = 0 if d5 > 0 else 1  # option favoured by the .66 expert
        tot += 1.0
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == fav:
            hits += 1.0
    if tot <= 0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.4925 (var=0.0074)
**Previous candidate values (this loop):**
  - iter 1: 0.5031 (var=0.0055) (Δ vs real +0.0106)
  - iter 2 (most recent): 0.4906 (var=0.0058) (Δ vs real -0.0019)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6956 (var=0.0072)
- pi_10: 0.4906 (var=0.0058)
- pi_1: 0.4988 (var=0.0030)
- pi_2: 0.3175 (var=0.0067)
- pi_3: 0.3875 (var=0.0033)
- pi_4: 0.7712 (var=0.0079)
- pi_5: 0.7512 (var=0.0059)
- pi_6: 0.7525 (var=0.0064)
- pi_8: 0.3119 (var=0.0063)
- pi_9: 0.5931 (var=0.0080)
- pi_11: 0.5025 (var=0.0063)

### Experiment 18
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    ANCHOR = 4   # printed validity .52  (the only sub-.558 expert)
    SECOND = 5   # printed validity .68, last column (secondary doubt row)

    t_num = 0.0; t_den = 0.0   # anchor-follow rate on LONG, low-opposition displays
    c_num = 0.0; c_den = 0.0   # c5-follow rate when the .52 anchor is SILENT

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_, b_, r_ in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_, dtype=float).ravel()
            b = np.asarray(b_, dtype=float).ravel()
            r = int(r_)
        except Exception:
            continue
        if a.size < 6 or b.size < 6:
            continue
        d = a[:6] - b[:6]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m == 0:
            continue
        nA = int(np.sum(d[disc] > 0.0))
        nB = m - nA
        kconf = min(nA, nB)
        chose_a = (r == 0)

        anchor_speaks = bool(np.any(disc == ANCHOR))
        second_speaks = bool(np.any(disc == SECOND))

        if anchor_speaks and m >= 4 and kconf <= 1:
            # long display, anchor either alone against a coalition or with everyone
            fav_a = (d[ANCHOR] > 0.0)
            t_den += 1.0
            if chose_a == fav_a:
                t_num += 1.0
        elif (not anchor_speaks) and second_speaks and kconf == 1:
            # anchor silent; is the .68 row the LONE dissenter on its side?
            fav_a_5 = (d[SECOND] > 0.0)
            side_count = nA if fav_a_5 else nB
            if side_count == 1:
                c_den += 1.0
                if chose_a == fav_a_5:
                    c_num += 1.0

    t_rate = (t_num / t_den) if t_den > 0 else 0.5
    c_rate = (c_num / c_den) if c_den > 0 else 0.5
    return float(t_rate - c_rate)
```

**Observed (real) value:** 0.3183 (var=0.0405)
**Previous candidate values (this loop):**
  - iter 1: 0.3725 (var=0.0150) (Δ vs real +0.0542)
  - iter 2 (most recent): 0.3558 (var=0.0213) (Δ vs real +0.0375)
**Other theories' values on this metric (for reference):**
- pi_10: 0.3550 (var=0.0214)
- pi_7: 0.0125 (var=0.0220)
- pi_1: -0.0267 (var=0.0062)
- pi_2: 0.1300 (var=0.0079)
- pi_3: 0.1117 (var=0.0039)
- pi_4: -0.2992 (var=0.0200)
- pi_5: -0.3408 (var=0.0161)
- pi_6: -0.3108 (var=0.0127)
- pi_8: 0.3800 (var=0.0208)
- pi_9: -0.0500 (var=0.0184)
- pi_11: 0.3325 (var=0.0167)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    base_vals = [0.66, 0.62, 0.64, 0.90, 0.94, 0.52]
    hits = 0
    tot = 0
    for a_, b_, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = np.asarray(list(a_), dtype=float).ravel()
            b = np.asarray(list(b_), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n <= len(base_vals):
            v = np.asarray(base_vals[:n], dtype=float)
        else:
            v = np.asarray(base_vals + [0.55] * (n - len(base_vals)), dtype=float)
        anchor = int(np.argmin(v))
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue
        if anchor not in set(disc.tolist()):
            continue
        s = np.sign(d[disc])
        # keep only UNANIMOUS displays (every discriminating expert agrees)
        if not (bool(np.all(s > 0.0)) or bool(np.all(s < 0.0))):
            continue
        fav_a = bool(d[anchor] > 0.0)
        try:
            chose_a = (int(r) == 0)
        except Exception:
            continue
        tot += 1
        if chose_a == fav_a:
            hits += 1
    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.8617 (var=0.0175)
**Previous candidate values (this loop):**
  - iter 1: 0.8783 (var=0.0047) (Δ vs real +0.0167)
  - iter 2 (most recent): 0.8608 (var=0.0057) (Δ vs real -0.0008)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6958 (var=0.0069)
- pi_10: 0.8450 (var=0.0051)
- pi_1: 0.8608 (var=0.0098)
- pi_2: 0.8458 (var=0.0088)
- pi_3: 0.9317 (var=0.0023)
- pi_4: 0.7633 (var=0.0090)
- pi_5: 0.7458 (var=0.0049)
- pi_6: 0.7750 (var=0.0095)
- pi_7: 0.7700 (var=0.0082)
- pi_8: 0.7592 (var=0.0081)
- pi_9: 0.7983 (var=0.0067)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # printed validities of this design (fixed at design time)
    V = np.array([0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625])
    anchor = int(np.argmin(V))  # the lone near-chance expert, v = .52 (column 5)

    follows = []
    for a_r, b_r, resp in zip(data['option_a_ratings'],
                              data['option_b_ratings'],
                              data['response']):
        try:
            a = np.asarray(list(a_r), dtype=float)
            b = np.asarray(list(b_r), dtype=float)
        except Exception:
            continue
        n = int(min(a.size, b.size, V.size))
        if n <= anchor:
            continue
        d = a[:n] - b[:n]
        if d[anchor] == 0.0:
            continue  # anchor silent -> not a test trial
        s = np.sign(d[anchor])
        opp = [j for j in range(n) if j != anchor and d[j] != 0.0 and np.sign(d[j]) != s]
        # exclude displays where a high-validity expert opposes the anchor
        if any(V[j] >= 0.9 for j in opp):
            continue
        # keep only unanimous-with-anchor displays (0 opponents) and displays with
        # a COALITION of >=2 mid-doubt opponents; drop the single-mid-opponent cell
        # where the two theories coincide
        if len(opp) == 1:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        follows.append(1.0 if ((s > 0) == chose_a) else 0.0)

    if len(follows) == 0:
        return 0.5
    return float(np.mean(follows))
```

**Observed (real) value:** 0.8750 (var=0.0100)
**Previous candidate values (this loop):**
  - iter 1: 0.8680 (var=0.0043) (Δ vs real -0.0070)
  - iter 2 (most recent): 0.8740 (var=0.0023) (Δ vs real -0.0010)
**Other theories' values on this metric (for reference):**
- pi_10: 0.8570 (var=0.0034)
- pi_11: 0.7045 (var=0.0056)
- pi_1: 0.5870 (var=0.0029)
- pi_2: 0.5850 (var=0.0031)
- pi_3: 0.5750 (var=0.0011)
- pi_4: 0.6200 (var=0.0047)
- pi_5: 0.6060 (var=0.0054)
- pi_6: 0.5860 (var=0.0046)
- pi_7: 0.7355 (var=0.0059)
- pi_8: 0.7665 (var=0.0047)
- pi_9: 0.6670 (var=0.0050)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**AIG-2 — Authority-Inversion with Null-Verdict Capture and a load-leaky single-cue fallback lottery.** (Same architecture as AIG; one quantitative commitment sharpened.)

One graded sampling rule governs every trial; there is no mode switch and no compensatory integration anywhere in the architecture.

**(1) Salience = an absolute near-chance bonus plus a weak doubt/recency tilt.** Each printed expert j carries a *scandal* value g_j = sigmoid((theta - v_j)/tau) with an ABSOLUTE, SHARP criterion. SHARPENED: the criterion is placed and steepened so that a printed .55 is UNAMBIGUOUSLY a coin flip (theta ~ .562-.572, tau ~ .003-.006 gives g(.55) >= .92, g(.58) <= .12, g(.60) ~ 0). Salience is log u_j = g_j*(kappa + mu*j) + gamma*log(1-v_j) + lam*j. The scandal term is an enormous additive log-bonus (kappa ~ 5-7.5), so whenever a genuinely near-chance expert is present it owns essentially all salience; the ordinary doubt tilt gamma and the positional tilt lam are weak and only matter among ordinary experts. The recency gain mu is carried by the scandal term, i.e. reading order arbitrates only BETWEEN co-scandalous experts; it can never lift an ordinary expert over a near-chance one.

**(2) A single row is consulted — never a sum.** The trial is settled by the verdict of ONE row drawn from softmax(log u) — a probabilistic one-reason rule. Coalitions buy almost nothing: adding agreeing rows does not add evidence, it only adds lottery tickets. This is why tally-vs-TTB gaps stay small and why validity-weighted metrics sit far below chance.

**(3) Authority inversion is adopted wholesale, flat in display length.** A discriminating near-chance expert wins the lottery with probability ~1 no matter how many rows precede or oppose it. Its verdict is retained with fidelity (1-eps) ~ .70-.77 and is charged only a very mild opposition tax rho_opp per minority row beyond the first. Scanning extra rows costs the authority limb NOTHING: anchor-follow is ~.86 alike on one-row, three-row and six-row displays and with 1-5 opponents.

**(4) NULL-VERDICT CAPTURE unifies the ceiling effects.** Scandal is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the least credible expert in the WHOLE display before the subject checks whether it discriminates. With probability q the subject simply goes with whoever captured attention; if that expert rated both products alike it returns no verdict and the subject guesses.

**(5) SHARPENED — the fallback lottery is genuinely EXPENSIVE in sheer list length, far more than previously assumed.** When no scandal is present the subject must hold the whole list in view to run the lottery over ordinary rows, and the thread is lost quickly: retained evidence is multiplied by rho_len ~ .87-.95 per discriminating row beyond the second — appreciably steeper than the earlier .94-1.0 estimate — a penalty the authority limb never pays. This one asymmetry now carries four otherwise separate facts: (i) long unanimous NON-anchor displays top out near .74 rather than at the authority limb's .87; (ii) silent-anchor conflict cells, where a mid-doubt (v ~ .62-.68) lone row faces a more valid coalition, sit at .47-.55 rather than being pushed decisively toward the coalition; (iii) tally-vs-TTB and validity-weighted metrics on long six-row designs are pulled a few points back toward chance relative to the short-display versions; (iv) the anchor-speaking vs anchor-silent contrast is compressed. Meanwhile the anchor-follow rate on long or heavily opposed anchor displays remains flat at ~.86, which is the theory's signature dissociation.

**(6) Degeneracy and stationarity.** If two printed validities are within .002 the experts cannot be individuated, salience has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # AIG: Authority-Inversion with null-verdict capture and a single-cue fallback lottery
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
    kappa = float(parameters['kappa'])
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    q = float(np.clip(float(parameters['q_capture']), 0.0, 1.0))
    eps = float(np.clip(float(parameters['eps']), 0.0, 1.0))
    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))
    rho_len = float(np.clip(float(parameters['rho_len']), 1e-6, 1.0))
    tkap = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(logv):
        logv = np.asarray(logv, dtype=float)
        if logv.size == 0:
            return logv
        z = logv - np.max(logv)
        e = np.exp(np.clip(z, -700.0, 0.0))
        s = float(np.sum(e))
        if (not np.isfinite(s)) or s <= 0.0:
            return np.full(logv.size, 1.0 / logv.size)
        return e / s

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    fav_a = signs > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        keep = (1.0 - eps) * (rho_len ** max(m - 2, 0))
        keep = float(min(max(keep, 0.0), 1.0))
        p_core = sig(tkap * (n_a - n_b))
        p_a = 0.5 + keep * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- salience: absolute scandal bonus + weak doubt/recency tilt ----------
    j_all = np.arange(n, dtype=float)
    zz = np.clip((theta - v) / tau, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(-zz))                      # scandal (near-chance) value
    logu = g * (kappa + mu * j_all) \
        + gamma * np.log(np.clip(1.0 - v, 1e-12, None)) \
        + lam * j_all

    # global salience over ALL rows (capture is pre-attentive)
    s_all = softmax(logu)
    # restricted lottery over rows that actually speak
    t_disc = softmax(logu[disc])

    # (i) capture branch: go with whoever grabbed attention; guess if it is silent
    mask = np.zeros(n, dtype=bool)
    mask[disc] = True
    s_silent = float(np.sum(s_all[~mask]))
    s_d = s_all[disc]
    p_glob_a = float(np.sum(s_d[fav_a])) + 0.5 * s_silent

    # (ii) re-targeted branch: lottery restricted to discriminating rows
    p_rest_a = float(np.sum(t_disc[fav_a]))

    core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- fidelity: authority limb is free in length, fallback is not ----------
    A_speak = float(np.clip(np.max(g[disc]), 0.0, 1.0))
    len_pen = rho_len ** ((1.0 - A_speak) * float(max(m - 2, 0)))
    opp_pen = rho_opp ** float(max(k_conf - 1, 0))
    fid = (1.0 - eps) * opp_pen * len_pen
    fid = float(min(max(fid, 0.0), 1.0))

    p_a = 0.5 + fid * (core_a - 0.5)
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
- theta: [0.562, 0.572]
- tau: [0.003, 0.006]
- kappa: [5.0, 7.5]
- mu: [0.60, 1.00]
- gamma: [0.36, 0.58]
- lam: [0.05, 0.20]
- q_capture: [0.40, 0.56]
- eps: [0.23, 0.30]
- rho_opp: [0.90, 0.99]
- rho_len: [0.87, 0.95]
- tally_kappa: [0.7, 1.3]
- validities: validities

`rationale`: MINIMAL DIFF: `predict` and `policy` are re-emitted verbatim; only four parameter RANGES change. No architectural edit, no compensatory limb, no new terms.

(1) **rho_len [0.94,1.00] -> [0.87,0.95]** — the single biggest lever, and it is the critic's item (2) implemented through the knob that already exists. Because the length penalty is gated by (1 - A_speak), it bites ONLY when no printed near-chance expert speaks, so the anchor limb stays exactly flat in length (Exps 15/16/19/20 untouched, still ~.86-.87), while every no-anchor display is pulled toward chance in proportion to load. I checked the sign of this edit against all thirteen residuals: it moves Exp 5 (.815->~.75 vs .743), Exp 13 (.433->~.48 vs .546), Exp 14 (.425->~.45 vs .468), Exp 1 (.251->~.26 vs .282), Exp 3 (.241->~.25 vs .286), Exp 4 (.128->~.14 vs .150), Exp 12 (.233->~.24 vs .261), Exp 18 (.373->~.34 vs .318) and Exp 2 (.243->~.22 vs .174) ALL in the correct direction, and costs only ~.01 on Exps 10 and 11. That is an unusually one-sided trade because almost every residual of the base model is 'model too extreme relative to humans' in precisely the cells where the scandal expert is mute.

(2) **theta [0.556,0.570] -> [0.562,0.572] and tau [0.004,0.009] -> [0.003,0.006]** — the critic's item (3). In the base, a draw of theta=.556/tau=.009 gave g(.55) only ~.66, making the Exp-2 and Exp-15 anchor rows unstable across parameter draws; now g(.55) >= .92 on every draw while g(.58) <= .12 and g(.60) ~ 0, so mediocre experts remain non-scandalous. This both removes the draw noise that inflated the Exp-2 tie cell and, as a side benefit, protects Exp 15 from the new stronger rho_len (its .55 anchor now reliably sets A_speak ~ 1, exponent ~ 0).

(3) **gamma [0.28,0.50] -> [0.36,0.58]** — the critic's item (1), but adopted only in the half that has the right sign. I deliberately did NOT lower q_capture: in the Exp-13/14 cell the restricted lottery still favours the (more numerous) valid coalition, so shifting mass out of the capture branch would push those cells DOWN, away from the observed .55/.47, the opposite of what the critic intended. Raising the ordinary doubt tilt instead makes the lone least-valid speaker win the restricted lottery more often, which raises Exps 13/14/11 with only a small cost on Exps 3/4/12 that the rho_len edit more than repays. The raise is kept modest so that the anti-validity metrics (Exps 1, 3, 4, 12), which are already slightly too extreme, are not driven further from the data.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: AIG fits the corpus remarkably evenly. Of the 20 experiments, 13 are within ~.03 of the observed value, including the three metrics that previously separated the theory families most sharply: Exp 15 (.858 vs .862), Exp 17 (.503 vs .493), Exp 20 (.868 vs .875), plus Exp 19 (.878 vs .862) and Exp 16 (.849 vs .826). The prescribed mechanism family is clearly honoured: an absolute, sharp printed-validity criterion with an additive log-bonus inside a SINGLE softmax lottery over rows, no compensatory sum, no mode gate, a null-verdict capture branch, degeneracy clause and stationarity all present. The known weakness of the DTG family 
s under-shooting anchor-follow on long/opposed anchor displays 
s is fixed (Exps 15, 19, 20 all now at or slightly above observation rather than 5-6 points below), and the below-chance validity-weighted metrics (Exps 3, 4, 10, 12) come out naturally from the sampler rather than from tuned leak terms.

The residual errors are small but systematic and in identifiable directions:
(1) Silent-anchor lone-dissenter cells are under-predicted. Exp 13 .433 vs .546 (the single largest error, -.113) and Exp 14 .425 vs .468 (-.043). Both are the same design logic: the near-chance scandal expert is mute, and a mid-doubt (v ~ .62-.68) lone row faces a more valid coalition. The capture branch is dumping too much of that mass into a 50/50 guess, so the model sits below the .47-.55 the humans show. The sampler's ordinary doubt tilt (gamma) plus positional tilt (lam) are too weak relative to q_capture in exactly this cell.
(2) The unanimous-dominance ceiling is over-predicted. Exp 5 .815 vs .743 (+.072). Long unanimous non-anchor displays should top out near .74; here the capture branch has no silent rows to leak into on full-dominance trials, so fidelity stays too high.
(3) The tally-vs-TTB gap is over-predicted on Exp 2 (.243 vs .174) and the anchor-vs-silent-anchor contrast on Exp 18 (.373 vs .318). In Exp 2 the .55 row sits right at the theta boundary (theta .556-.570, tau .004-.009 makes g for v=.55 only ~.7-.9 depending on draw), which makes the capture of that row unstable and pushes the tie-cell TTB-follow too far below chance.
None of these require abandoning the architecture; they are calibration of q_capture/gamma/lam, of the unanimous-display lapse, and of the theta/tau placement.
Rationale: The candidate was ACCEPTED (loss 0.0575, new running best) and is faithful to the arbiter's AIG prescription, so I am asking only for in-family calibration on top of THIS base 
s do not change the architecture, do not reintroduce compensatory integration, and do not touch what already works (Exps 6-11, 15-17, 19, 20 are at or near the observed values and must not be disturbed). Three targeted knobs, in priority order:

(1) Exp 13/14 (model .433/.425 vs observed .546/.468) 
s raise the mid-doubt sampler's pull when the scandal expert is SILENT. Concretely, lower q_capture slightly (e.g. .32-.45 instead of .40-.56) and/or raise the ordinary doubt tilt gamma (e.g. .45-.75 instead of .28-.50) so that on silent-anchor conflict displays the least-valid speaking row wins the restricted lottery more often. This is the biggest single residual and moves in a direction (more anti-validity among ordinary rows) that the corpus supports everywhere else.

(2) Exp 5 (model .815 vs observed .743) 
s the unanimous full-discrimination ceiling is too high. Add the arbiter's optional mild length-dependent guess increment that applies to unanimous displays specifically (e.g. let rho_len also bite when the display is unanimous and the anchor is silent, or add a small eps_len * (m-2) term), rather than simply raising the global eps 
s a global eps rise would drag Exps 15/19/20 off their currently excellent .86-.88 and must be avoided.

(3) Exp 2/18 (gap over-predicted by ~.06) 
s stabilise the scandal criterion so a printed .55 is unambiguously scandalous: nudge theta up a touch (e.g. .565-.578) or shrink tau (e.g. .003-.006) so g(.55)->1 while g(.60) stays ~0. That removes the parameter-draw noise in Exp 2's tie cell and should shrink the t_rate-c_rate contrast in Exp 18 toward .32.

Keep the degeneracy clause, stationarity, the flat-in-length authority limb, and the single-softmax formulation exactly as they are. If the retuned version does not beat 0.0575 the gate will discard it and the base is preserved, so the expected cost of trying is zero.

**Outcome of this advice:** iter 1 candidate loss=0.0575 -> iter 2 candidate loss=0.0457 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: Iteration 2 is a clear net win (loss .0575 -> .0457, ACCEPTED) and it is fully faithful to the arbiter's AIG prescription: one absolute, sharp printed-validity criterion feeding an additive log-bonus inside a single softmax lottery over rows, a null-verdict capture branch, a load-leaky fallback, the degeneracy clause and stationarity. No compensatory limb has crept in.

Where it now stands: the signature authority cells are essentially exact 
s Exp 15 .869 vs .862, Exp 16 .837 vs .826, Exp 17 .491 vs .493, Exp 19 .861 vs .862, Exp 20 .874 vs .875. This is the family's decisive advantage over every DTG-style competitor (pi_1-pi_9 sit 20-50 points off on 15/16/20) and it must not be disturbed. The steeper rho_len I asked for last round paid off almost exactly as predicted: Exp 5 .815 -> .782 (obs .743), Exp 2 .243 -> .223 (obs .174), Exp 18 .373 -> .356 (obs .318), Exp 16 .849 -> .837, Exp 20 .868 -> .874, all in the right direction with negligible collateral.

The one place my last critique was wrong is now visible. I asked for the Exp 13/14 gap to be closed by raising the ordinary doubt tilt gamma (and/or lowering q_capture). The proposer took the gamma half and declined the q_capture half; the result is that Exp 13 moved the WRONG way, .433 -> .406 against an observed .546 (residual now -.140, by far the largest in the corpus), and Exp 14 barely moved (.430 vs .468). The arithmetic is instructive: in these silent-anchor cells the lone least-valid dissenter sits EARLY (Exp 13 feature 0, v=.68) while the more-valid coalition sits LATE and owns 2+ lottery tickets, so lam*j plus ticket-count beat any plausible gamma. Pushing gamma harder cannot win this cell without simultaneously driving Exps 1/3/4/12 (already 3-4 points too anti-valid: .251/.246/.125/.218 vs .282/.286/.150/.261) even further below the humans.

The residual sign pattern is now remarkably one-directional: in EVERY cell where the near-chance scandal expert is MUTE, the model is more extreme than the humans 
s Exp 13 -.140, Exp 12 -.043, Exp 3 -.039, Exp 14 -.038, Exp 1 -.032, Exp 4 -.025 (all too far below chance) and Exp 5 +.039, Exp 2 +.049, Exp 18 +.038 (all too far above chance / too large a contrast). Everything with the anchor speaking is already spot on. That is a single missing quantity: too little guess mass on anchor-silent displays.
Rationale: Stay inside AIG exactly as prescribed 
s do not touch the architecture, the flat-in-length authority limb, the degeneracy clause, or the theta/tau placement (the sharpened .562-.572 / .003-.006 criterion is working: Exps 2, 15, 19, 20 all stabilised).

One primary knob this round, and it is the OPPOSITE of what I wrongly pushed last time (my 'lower q_capture / raise gamma' advice for Exp 13 was declined on the q half and the gamma half made Exp 13 worse 
s do not repeat it):

(1) RAISE q_capture, from [0.40,0.56] to roughly [0.58,0.72]. Because the capture branch guesses whenever the pre-attentively captured (most scandalous) expert returns no verdict, q is the only parameter in the model whose effect is confined to exactly the anchor-silent cells where all remaining residuals live. Check the signs: Exp 13 .406 -> ~.46 (obs .546), Exp 14 .430 -> ~.46 (obs .468), Exp 5 .782 -> ~.76 (obs .743), Exp 2 .223 -> ~.20 (obs .174), Exp 18 .356 -> ~.33 (obs .318), Exp 1 .251 -> ~.26, Exp 3 .246 -> ~.26, Exp 4 .125 -> ~.14, Exp 12 .218 -> ~.23, Exp 10 .434 -> ~.43 
s all toward the data. Anchor-speaking cells (15/16/17/19/20) are untouched because there is no silent-scandal mass to leak. The only costs are Exp 9 (-.016 -> maybe -.03) and Exp 11 (-.020 -> maybe -.03) and a point on Exp 7; those are small and partially offset by Exp 8 (currently +.016) improving. Keep the raise moderate 
s q above ~.75 will start flattening Exps 8/9/11 visibly.

(2) Partially back off gamma, from [0.36,0.58] to about [0.30,0.46]. The gamma raise was the one component of last round's edit whose sign is wrong: it deepens the already-too-extreme anti-validity metrics (Exps 1/3/4/12) while failing to rescue Exp 13. Splitting the difference between iter-1 (.28-.50) and iter-2 (.36-.58) is the non-oscillating choice.

(3) Optional, second-order: a small further step on rho_len (e.g. [0.85,0.93]) IF (1)+(2) leave Exp 5 and Exp 2 still over-shooting. That direction was ACCEPTED last round, so a further small step is low-risk, but take it only as a secondary adjustment 
s q_capture is the better-targeted lever for the dominant Exp-13 residual and a big rho_len step risks Exps 9/11.

Accept that Exp 13 will land near .46-.48 rather than .546: a pure capture-guess caps that cell at .50, and exceeding .50 would require a gamma/lam regime that the rest of the corpus rules out. Reducing that residual from -.14 to ~-.07 while simultaneously improving eight other experiments is the right trade. If the retune does not beat .0457 the gate discards it and the current base is preserved, so the expected cost is zero.

**Outcome of this advice:** iter 2 candidate loss=0.0457 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**AIG-3 — Authority-Inversion with a stronger pre-attentive capture branch and a load-leaky single-cue fallback lottery.** (Same architecture as AIG-2; one quantitative commitment sharpened, one partially reversed.)\n\nOne graded sampling rule governs every trial; there is no mode switch and no compensatory integration anywhere in the architecture.\n\n**(1) Salience = an absolute near-chance bonus plus a weak doubt/recency tilt.** Each printed expert j carries a *scandal* value g_j = sigmoid((theta - v_j)/tau) with an ABSOLUTE, SHARP criterion placed so that a printed .55 is unambiguously a coin flip (theta ~ .562-.572, tau ~ .003-.006 gives g(.55) >= .92, g(.58) <= .12, g(.60) ~ 0). Salience is log u_j = g_j*(kappa + mu*j) + gamma*log(1-v_j) + lam*j. The scandal term is an enormous additive log-bonus (kappa ~ 5-7.5), so whenever a genuinely near-chance expert is present it owns essentially all salience; the ordinary doubt tilt gamma and the positional tilt lam are weak and only arbitrate among ordinary experts. The recency gain mu rides on the scandal term, so reading order arbitrates only BETWEEN co-scandalous experts and can never lift an ordinary expert over a near-chance one.\n\n**(2) A single row is consulted — never a sum.** The trial is settled by the verdict of ONE row drawn from softmax(log u) — a probabilistic one-reason rule. Coalitions buy almost nothing: adding agreeing rows does not add evidence, it only adds lottery tickets. Hence tally-vs-TTB gaps stay small and validity-weighted metrics sit far below chance.\n\n**(3) Authority inversion is adopted wholesale, flat in display length.** A discriminating near-chance expert wins the lottery with probability ~1 no matter how many rows precede or oppose it. Its verdict is retained with fidelity (1-eps) ~ .70-.77 and is charged only a very mild opposition tax rho_opp per minority row beyond the first. Scanning extra rows costs the authority limb NOTHING: anchor-follow is ~.86 alike on one-row, three-row and six-row displays and with 1-5 opponents.\n\n**(4) SHARPENED — pre-attentive NULL-VERDICT CAPTURE is the DOMINANT mode of responding, not a minority one.** Scandal is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the least credible expert in the WHOLE display before the subject checks whether it actually discriminates. The revised claim is that subjects commit to whoever captured attention on a clear MAJORITY of trials (q ~ .58-.72, not ~.40-.56): the decision to re-scan the display for a speaking cue is effortful and usually declined. The consequence is a signature dissociation with no free parameter to tune it: when the scandalous expert SPEAKS, capture and re-targeting agree, so anchor-follow stays high and flat; when the scandalous expert is MUTE, most of the response mass is delivered to a verdictless expert and the subject simply guesses, so every anchor-silent cell is pulled systematically toward chance — lone mid-doubt dissenters facing a valid coalition land near .46-.50 rather than being swept to the coalition, long unanimous non-anchor displays top out near .74-.76, and the tally-vs-TTB and anchor-vs-silent contrasts are compressed. This single quantity, not extra leak terms, carries the whole anchor-silent residual family.\n\n**(5) The fallback lottery is expensive in sheer list length.** When no scandal is present the subject must hold the whole list in view to run the lottery over ordinary rows, and the thread is lost quickly: retained evidence is multiplied by rho_len ~ .87-.95 per discriminating row beyond the second — a penalty the authority limb never pays, because the length exponent is gated by (1 - A_speak).\n\n**(6) The ordinary doubt tilt is WEAK.** gamma is returned to ~.30-.46: among non-scandalous experts, doubt buys only a modest advantage in the lottery, and it cannot overturn ticket-count plus position. Over-weighting it deepens the already too-extreme anti-validity metrics without rescuing silent-anchor conflict cells.\n\n**(7) Degeneracy and stationarity.** If two printed validities are within .002 the experts cannot be individuated, salience has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.",
  "predict": "def predict(parameters, state, history):\n    # AIG: Authority-Inversion with null-verdict capture and a single-cue fallback lottery\n    import numpy as np\n\n    # ---------- parse the (2, n_features) stimulus robustly ----------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()\n            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()\n        else:\n            vals = list(state.values())\n            if len(vals) >= 2:\n                a = np.asarray(vals[0], dtype=float).ravel()\n                b = np.asarray(vals[1], dtype=float).ravel()\n    if a is None or b is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 2 and arr.shape[0] == 2:\n            a, b = arr[0].ravel(), arr[1].ravel()\n        elif arr.ndim == 2 and arr.shape[1] == 2:\n            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()\n        elif arr.ndim == 1:\n            h = arr.shape[0] // 2\n            a, b = arr[:h], arr[h:2 * h]\n        else:\n            f = np.asarray(arr, dtype=float).reshape(2, -1)\n            a, b = f[0].ravel(), f[1].ravel()\n\n    n = int(min(np.size(a), np.size(b)))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = np.asarray(a, dtype=float)[:n]\n    b = np.asarray(b, dtype=float)[:n]\n\n    # ---------- printed validities ----------\n    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n    if v_raw is None:\n        v = np.linspace(0.95, 0.55, n)\n    else:\n        v = np.asarray(v_raw, dtype=float).ravel()\n        if v.size < n:\n            v = np.concatenate([v, np.full(n - v.size, 0.55)])\n        v = v[:n]\n    v = np.clip(v, 0.5, 0.9995)\n\n    # ---------- parameters ----------\n    theta = float(parameters['theta'])\n    tau = float(max(float(parameters['tau']), 1e-4))\n    kappa = float(parameters['kappa'])\n    mu = float(parameters['mu'])\n    gamma = float(parameters['gamma'])\n    lam = float(parameters['lam'])\n    q = float(np.clip(float(parameters['q_capture']), 0.0, 1.0))\n    eps = float(np.clip(float(parameters['eps']), 0.0, 1.0))\n    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))\n    rho_len = float(np.clip(float(parameters['rho_len']), 1e-6, 1.0))\n    tkap = float(parameters['tally_kappa'])\n\n    def sig(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    def softmax(logv):\n        logv = np.asarray(logv, dtype=float)\n        if logv.size == 0:\n            return logv\n        z = logv - np.max(logv)\n        e = np.exp(np.clip(z, -700.0, 0.0))\n        s = float(np.sum(e))\n        if (not np.isfinite(s)) or s <= 0.0:\n            return np.full(logv.size, 1.0 / logv.size)\n        return e / s\n\n    # ---------- discriminating rows ----------\n    diff = a - b\n    disc = np.nonzero(diff != 0.0)[0]\n    m = int(disc.size)\n    if m == 0:\n        return np.array([0.5, 0.5])\n    signs = np.sign(diff[disc])\n    fav_a = signs > 0.0\n    n_a = float(np.sum(fav_a))\n    n_b = float(m) - n_a\n    k_conf = int(min(n_a, n_b))\n\n    # ---------- label-degeneracy gate ----------\n    degenerate = False\n    if n > 1:\n        vs = np.sort(v)\n        if bool(np.any(np.diff(vs) < 2.0e-3)):\n            degenerate = True\n\n    if degenerate:\n        keep = (1.0 - eps) * (rho_len ** max(m - 2, 0))\n        keep = float(min(max(keep, 0.0), 1.0))\n        p_core = sig(tkap * (n_a - n_b))\n        p_a = 0.5 + keep * (p_core - 0.5)\n        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n        p = np.array([p_a, 1.0 - p_a], dtype=float)\n        return p / p.sum()\n\n    # ---------- salience: absolute scandal bonus + weak doubt/recency tilt ----------\n    j_all = np.arange(n, dtype=float)\n    zz = np.clip((theta - v) / tau, -60.0, 60.0)\n    g = 1.0 / (1.0 + np.exp(-zz))                      # scandal (near-chance) value\n    logu = g * (kappa + mu * j_all) \\\n        + gamma * np.log(np.clip(1.0 - v, 1e-12, None)) \\\n        + lam * j_all\n\n    # global salience over ALL rows (capture is pre-attentive)\n    s_all = softmax(logu)\n    # restricted lottery over rows that actually speak\n    t_disc = softmax(logu[disc])\n\n    # (i) capture branch: go with whoever grabbed attention; guess if it is silent\n    mask = np.zeros(n, dtype=bool)\n    mask[disc] = True\n    s_silent = float(np.sum(s_all[~mask]))\n    s_d = s_all[disc]\n    p_glob_a = float(np.sum(s_d[fav_a])) + 0.5 * s_silent\n\n    # (ii) re-targeted branch: lottery restricted to discriminating rows\n    p_rest_a = float(np.sum(t_disc[fav_a]))\n\n    core_a = q * p_glob_a + (1.0 - q) * p_rest_a\n\n    # ---------- fidelity: authority limb is free in length, fallback is not ----------\n    A_speak = float(np.clip(np.max(g[disc]), 0.0, 1.0))\n    len_pen = rho_len ** ((1.0 - A_speak) * float(max(m - 2, 0)))\n    opp_pen = rho_opp ** float(max(k_conf - 1, 0))\n    fid = (1.0 - eps) * opp_pen * len_pen\n    fid = float(min(max(fid, 0.0), 1.0))\n\n    p_a = 0.5 + fid * (core_a - 0.5)\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0.0:\n        return int(np.random.randint(len(probs)))\n    probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "theta": "[0.562, 0.572]",
    "tau": "[0.003, 0.006]",
    "kappa": "[5.0, 7.5]",
    "mu": "[0.60, 1.00]",
    "gamma": "[0.30, 0.46]",
    "lam": "[0.05, 0.20]",
    "q_capture": "[0.58, 0.72]",
    "eps": "[0.23, 0.30]",
    "rho_opp": "[0.90, 0.99]",
    "rho_len": "[0.87, 0.95]",
    "tally_kappa": "[0.7, 1.3]",
    "validities": "validities"
  },
  "rationale": "MINIMAL DIFF: `predict` and `policy` are re-emitted VERBATIM from the running-best iter-2 base. Only TWO parameter ranges change — exactly the critic's two prioritised knobs, in the priority order given.\n\n(1) **q_capture [0.40,0.56] -> [0.58,0.72]** — the primary edit. This is the only parameter in the architecture whose effect is structurally confined to anchor-SILENT displays: the capture branch contributes 0.5*s_silent to p_glob_a, and s_silent is essentially zero whenever a printed near-chance expert speaks (the scandal bonus kappa makes that expert own ~all salience). So raising q leaves Exps 15/16/17/19/20 — the family's decisive wins, currently exact to within .01 — literally untouched, while injecting guess mass into precisely the cells where every remaining residual lives. Sign check on all nine anchor-silent residuals: Exp 13 .406 -> ~.46 (obs .546, residual -.140 -> ~-.09), Exp 14 .430 -> ~.45 (obs .468), Exp 5 .782 -> ~.76 (obs .743), Exp 2 .223 -> ~.20 (obs .174), Exp 18 .356 -> ~.33 (obs .318), Exp 1 .251 -> ~.26 (obs .282), Exp 3 .246 -> ~.26 (obs .286), Exp 4 .125 -> ~.14 (obs .150), Exp 12 .218 -> ~.23 (obs .261) — ALL toward the data. The residual sign pattern the critic identified is strictly one-directional (model too extreme on every mute-anchor cell), which is exactly the signature of one missing scalar, and q is that scalar. The raise is kept moderate (upper bound .72, not >.75) so that Exps 8/9/11 — which depend on the restricted lottery resolving late low-validity rows — are not flattened; expected cost there is ~.01-.02 each, more than repaid by the nine gains.\n\n(2) **gamma [0.36,0.58] -> [0.30,0.46]** — a deliberate partial reversal of the one component of last round's edit whose sign was wrong. Raising gamma deepened the already-too-anti-valid Exps 1/3/4/12 and moved Exp 13 the WRONG way (.433 -> .406), because in those silent-anchor cells the lone least-valid dissenter sits EARLY while the more-valid coalition sits LATE and holds 2+ lottery tickets, so lam*j plus ticket count dominate any plausible gamma. Splitting the difference between iter-1 (.28-.50) and iter-2 (.36-.58) is the non-oscillating choice and it pushes Exps 1/3/4/12 back up toward the humans in concert with the q raise.\n\nI deliberately did NOT take the critic's optional third step (a further rho_len cut to [0.85,0.93]). q_capture already moves Exp 5 and Exp 2 in the same direction, and stacking two length/guess-mass edits in one iteration risks over-shooting Exps 9/11 and tripping the gate; rho_len stays at the ACCEPTED [0.87,0.95]. Architecture, theta/tau placement, the flat-in-length authority limb, the degeneracy clause and stationarity are all unchanged."
}
```

## Usage

```json
{
  "input_tokens": 68135,
  "output_tokens": 6722
}
```
