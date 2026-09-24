# generation_iter_03_attempt_00

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
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with a CHORUS-FREE, PURELY INVERSE-VALIDITY (contrarian) weighting theory \u2014 call it e.g. 'Flat Maverick / Inverse-Seniority Normalised Integration (FM-ISN)'. Core commitments:

(1) Keep the parameter-free count gate: if the two options win on equally many features (including wholly non-discriminating displays), P = exactly 0.5, no validity tie-break, no take-the-best fallback. This is required by Exp3 (.474) and is common ground.

(2) Weight each discriminating cue by a STEEPLY DECREASING function of panel-relative seniority s_j = (v_j \u2212 v_min)/(v_max \u2212 v_min): w_j = eta + A\u00b7(1 \u2212 s_j)^kappa, with kappa \u2248 4\u201312 and A/eta large (A \u2248 1\u20133, eta \u2248 0.02\u20130.15). The floor voice carries almost all the weight; mid and star voices carry only a thin, weakly graded residual.

(3) THE DECISIVE NEW CLAIM \u2014 NO CHORUS AMPLIFICATION OF ANY KIND. The bottom voice's weight is fixed by seniority alone and is completely independent of (a) how many voices oppose it, (b) how senior those opposing voices are, and (c) whether it sits inside a unanimous own-side bloc. This is exactly what Exp1 (\u22120.017) and Exp2 (+0.015) demand: opposed-floor cells and unanimous-bloc cells must come out at the SAME rate, and the seniority titration across an opposing pair must be FLAT. Both pi_10 (seniority-unit gate) and pi_9 (raw-count knee) are falsified here; the new theory's signature is a null on both contrasts.

(4) Normalised / saturating integration so that upper-only displays land at exact chance at every margin: z = beta\u00b7(\u03a3 dir_j w_j)/((c_n + \u03a3_j w_j)\u00b7D^gamma) with gamma \u2248 0.8\u20131.2 and c_n chosen so that a 3v1 or 2v0 display made entirely of upper-band voices yields .48\u2013.55 (Exp15 .489, Exp16 .489, Exp10 .485, Exp17 \u2248 0, Exp18 \u2248 +.05), while a floor-backed majority yields \u2248 .80 and a 3-cue upper majority opposed by the lone floor voice reverses to \u2248 .13\u2013.17 (Exp14 .145).

(5) Only a MILD capacity effect: at n = 8 the model must still deliver \u2248 .75 tally adherence (Exp6) while star-backed 8-expert majorities stay at chance (Exp10). Do NOT include the strong noise inflation of pi_9, which over-flattened, nor pi_10's amplification collapse, which under-predicted Exp6 (.699 vs .755). Capacity should be at most a small slope reduction, or absent entirely.

(6) A lapse epsilon (\u2248 .05\u2013.20) mixing with a coin flip; parameter ranges should be set so that the model reproduces Exp4 \u2248 .65, Exp5 \u2248 .79, Exp7 \u2248 \u2212.76, Exp8 \u2248 \u2212.9, Exp9 \u2248 .66, Exp11 \u2248 .41, Exp13 \u2248 .70.

The new theory is thus a strictly simpler, strictly more task-invariant competitor to pi_10: identical on the count gate and on anti-validity dominance, but committed to a context-free maverick weight, which is the one place where pi_10 demonstrably over-predicts (Exp1 +.20 vs observed \u2212.02; Exp2 +.16 vs observed +.02; Exp15 .571 vs observed .489).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.795)

**Description**
**Mute-Band Maverick Arbitration with a HARD (near-step) Band and a Stronger Capacity Noise Term (MBM-PG7).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD, NEAR-STEP MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information, and *within* that band the listening function is now strongly convex: m_j = max(0, 1 − s_j/theta)^rho with rho ≈ 1.7-2.5. The maverick construal is therefore close to categorical — only the panel's genuinely idiosyncratic BOTTOM voice speaks with full force; a voice that merely sits low-ish (a fifth to a half of the way into the band) is already nearly as mute as a star. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE, STEEP KNEE — and the post-knee amplification is DEEP.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. For cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 205) and a LARGE post-knee asymptote (alpha ≈ 52). Because (2) is now near-categorical, this large amplification is applied essentially only to true floor voices: a partial, mid-band voice can no longer borrow a 20-30x chorus gain and masquerade as a maverick.

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Displays in which no voice is inside the mute band hover near chance at any count margin, while the preserved slope still generates the anti-validity contrasts on conflict displays.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and mixed with an attention lapse epsilon.

**(6) CAPACITY IS BOTH A FLATTENING AND A SUBSTANTIAL NOISE SOURCE.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 AND integration becomes markedly noisier (divisor × (1 + chi·(n−7)) with chi now appreciably larger), so eight-expert star-backed majorities sit at chance and eight-expert plain tally adherence is systematically LOWER than the six-expert equivalent even though the tally still governs direction.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; only the panel's bottom voice (not the second- or third-from-bottom) can reverse a majority; a lone floor voice opposed by a mere PAIR lands at chance while the same voice opposed by a three-cue bloc reverses it to ≈.15; near-ceiling following of dense unanimous displays and of majorities containing the floor voice; and visibly attenuated, noisy integration on eight-expert panels.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: parameter-free count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                      # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    theta = _g('theta', 0.16, 0.02, 0.90)
    rho = _g('rho', 2.1, 0.20, 4.00)
    eta = _g('eta', 0.23, 0.00, 1.00)
    c_res = _g('c_res', 2.0, 0.00, 6.00)
    beta = _g('beta', 0.80, 0.00, 5.00)
    gamma = _g('gamma', 0.95, 0.00, 2.50)
    alpha = _g('alpha', 52.0, 0.00, 400.0)
    kappa = _g('kappa', 205.0, 0.05, 2000.0)
    p_ch = _g('p_ch', 6.4, 0.50, 12.0)
    lam_own = _g('lam_own', 0.36, 0.00, 1.00)
    psi = _g('psi', 0.22, 0.00, 1.00)
    chi = _g('chi', 0.39, 0.00, 3.00)
    eps = _g('epsilon', 0.09, 0.00, 0.60)

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    # ---------- Commitment 2: hard, near-step, chorus-independent mute band ----------
    band = np.clip(1.0 - s / max(theta, 1e-6), 0.0, 1.0)
    mav = np.power(band, rho)                      # maverick strength, 0 outside the band

    # ---------- Commitment 3: GRADED chorus amplification (cue-specific) ----------
    def _G(base):
        base = float(max(0.0, base))
        try:
            bp = base ** p_ch
        except Exception:
            bp = base
        if (not np.isfinite(bp)) or bp < 0.0:
            bp = 0.0
        g = 1.0 + alpha * bp / (bp + kappa)
        if (not np.isfinite(g)) or g < 1.0:
            g = 1.0
        return g

    # the bloc a cue OPPOSES counts fully; the bloc it sits INSIDE counts at
    # weight lam_own; an unopposed cue hears only the consensus it belongs to.
    if mB > 0:
        base_A = (float(mB) - 1.0) + lam_own * (float(mA) - 1.0)
    else:
        base_A = (float(mA) - 1.0)
    if mA > 0:
        base_B = (float(mA) - 1.0) + lam_own * (float(mB) - 1.0)
    else:
        base_B = (float(mB) - 1.0)

    G_A = _G(base_A)
    G_B = _G(base_B)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    if over > 0.0:
        shrink = max(0.0, 1.0 - psi * over)
        G_A = 1.0 + (G_A - 1.0) * shrink
        G_B = 1.0 + (G_B - 1.0) * shrink
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # ---------- Commitment 4: the panel gradient (quiet, not empty) ----------
    resid = eta + c_res * (1.0 - s)

    w = np.array(resid, dtype=float, copy=True)
    if iA.size > 0:
        w[iA] = G_A * mav[iA] + resid[iA]
    if iB.size > 0:
        w[iB] = G_B * mav[iB] + resid[iB]

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: dilution by cue count, then lapse ----------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * noise

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- theta: [0.12, 0.20]
- rho: [1.7, 2.5]
- eta: [0.18, 0.28]
- c_res: [1.7, 2.3]
- beta: [0.70, 0.90]
- gamma: [0.88, 1.02]
- alpha: [46, 58]
- kappa: [180, 230]
- p_ch: [6.1, 6.7]
- lam_own: [0.28, 0.45]
- psi: [0.10, 0.35]
- chi: [0.26, 0.52]
- epsilon: [0.04, 0.14]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5081 (var=0.0086)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.7087 (var=0.0069)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.7742 (var=0.0020)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.7492 (var=0.0014)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.6542 (var=0.0050)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.9213 (var=0.0220)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.6750 (var=0.0035)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.4931 (var=0.0066)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.3756 (var=0.0029)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=0.0663 (var=0.0326)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.6533 (var=0.0064)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.0788 (var=0.0054)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.5542 (var=0.0056)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.4929 (var=0.0046)
- Experiment 15: real=0.0067 (var=0.0123) vs this=-0.0583 (var=0.0163)
- Experiment 16: real=0.0517 (var=0.0159) vs this=-0.0381 (var=0.0193)
- Experiment 17: real=-0.0167 (var=0.0075) vs this=-0.2711 (var=0.0109)
- Experiment 18: real=0.0150 (var=0.0121) vs this=-0.3142 (var=0.0128)


---

### `pi_10` (overall score: 0.795)

**Description**
**Chorus-Threshold Novelty Weighting, Balanced-Gradient variant (CTN-BG).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 14–16.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it speaks into, and a chorus is measured in *mid-expert equivalents*.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i — voices it contradicts counted in full, voices it merely stands beside echoed at a discount. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar: two mid voices are not yet a chorus, two stars just barely are, three upper voices fully are.

**(4) The ungated panel gradient is THIN and only MILDLY convex — a bare ordinal pull that does not, by itself, give the panel's bottom voice an edge over a pair of mid voices.** Every discriminating cue contributes r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.7–2.05. The scale is small enough that all-upper displays drift near chance at any count margin, and the exponent is flat enough that a single floor cue's residual is worth about the same as two mid cues' residuals — so a floor voice opposed by a mid PAIR is a dead heat (it needs the chorus gate, not the gradient, to win), while a floor voice opposed by a three-cue upper bloc is carried past chance by the gate.

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight of evidence on the table plus a constant of indecision, and is diluted by the number of mutually discriminating attributes with gamma ≈ 1, so a lone cue and an unopposed pair of the same tier feel about equally compelling. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices, with only mild extra noise.** For n > 7 the amplification essentially vanishes (Gmax × (1 − psi·(n−7))) while integration noise grows only slightly, so eight-expert star-backed majorities sit near chance rather than being reversed, and eight-expert plain tally adherence survives on the residual gradient.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; moderate (not ceiling) adherence to majorities containing no bottom voice; a bottom voice facing two MID voices lands at chance, facing two STARS reverses substantially, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of majorities that contain the bottom voice; attenuated integration on eight-expert panels.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: parameter-free count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                      # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    eta = _g('eta', 0.040, 0.0, 0.5)
    c_res = _g('c_res', 0.12, 0.0, 1.5)
    nu = _g('nu', 1.88, 0.5, 5.0)
    kappa = _g('kappa', 15.0, 1.0, 30.0)
    tau = _g('tau', 1.2, 0.0, 4.0)
    lam_own = _g('lam_own', 0.60, 0.0, 1.0)
    gmax = _g('gmax', 1.90, 0.0, 5.0)
    q0 = _g('q0', 2.45, 0.20, 8.0)
    wid = _g('wid', 0.14, 0.02, 1.0)
    c_n = _g('c_n', 2.9, 0.10, 12.0)
    beta = _g('beta', 41.0, 0.0, 150.0)
    gamma = _g('gamma', 0.96, 0.0, 2.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)
    psi = _g('psi', 0.88, 0.0, 1.0)
    chi = _g('chi', 0.05, 0.0, 3.0)

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    one_minus = np.clip(1.0 - s, 0.0, 1.0)

    # ---------- Commitment 2: smooth convex novelty, steep descent ----------
    nov = np.power(one_minus, kappa)
    nov = np.where(np.isfinite(nov), nov, 0.0)

    # ---------- Commitment 4: THIN, mildly convex panel gradient ----------
    resid = eta + c_res * np.power(one_minus, nu)
    resid = np.where(np.isfinite(resid), resid, eta)

    # ---------- Commitment 3: chorus measured in MID-EXPERT EQUIVALENTS ----------
    denom_u = 1.0 + 0.5 * tau
    if (not np.isfinite(denom_u)) or denom_u <= 0.0:
        denom_u = 1.0
    u = (1.0 + tau * s) / denom_u        # mid voice == 1 unit, star ~1.4, floor ~0.6
    u = np.where(np.isfinite(u), u, 1.0)

    U_A = float(np.sum(u[iA])) if mA > 0 else 0.0
    U_B = float(np.sum(u[iB])) if mB > 0 else 0.0

    Q = np.zeros(n, dtype=float)
    if mA > 0:
        Q[iA] = U_B + lam_own * (U_A - u[iA])
    if mB > 0:
        Q[iB] = U_A + lam_own * (U_B - u[iB])
    Q = np.clip(Q, 0.0, None)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    gmax_eff = gmax * max(0.0, 1.0 - psi * over)
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # smooth logistic threshold in the chorus size
    xg = (Q - q0) / max(wid, 1e-6)
    xg = np.clip(xg, -60.0, 60.0)
    gate = gmax_eff / (1.0 + np.exp(-xg))
    gate = np.where(np.isfinite(gate), gate, 0.0)

    # ---------- weights ----------
    w = resid + nov * gate
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)

    E = 0.0
    if mA > 0:
        E += float(np.sum(w[iA]))
    if mB > 0:
        E -= float(np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    W_tot = 0.0
    if mA > 0:
        W_tot += float(np.sum(w[iA]))
    if mB > 0:
        W_tot += float(np.sum(w[iB]))
    if (not np.isfinite(W_tot)) or W_tot < 0.0:
        W_tot = 0.0

    # ---------- Commitment 5: Luce-like saturating integration ----------
    den = (c_n + W_tot) * (float(D) ** gamma) * noise
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- eta: [0.030, 0.048]
- c_res: [0.10, 0.14]
- nu: [1.70, 2.05]
- kappa: [14.0, 16.5]
- tau: [1.0, 1.5]
- lam_own: [0.50, 0.70]
- gmax: [1.70, 2.10]
- q0: [2.38, 2.52]
- wid: [0.11, 0.17]
- c_n: [2.70, 3.05]
- beta: [34, 48]
- gamma: [0.88, 1.05]
- epsilon: [0.09, 0.15]
- psi: [0.80, 0.95]
- chi: [0.02, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5112 (var=0.0071)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.7531 (var=0.0055)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.8108 (var=0.0026)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.6992 (var=0.0030)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.7558 (var=0.0048)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.7575 (var=0.0424)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.6647 (var=0.0029)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.4850 (var=0.0089)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.3747 (var=0.0028)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=0.0387 (var=0.0312)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.6479 (var=0.0051)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.1225 (var=0.0080)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.5713 (var=0.0051)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.5083 (var=0.0056)
- Experiment 15: real=0.0067 (var=0.0123) vs this=-0.0167 (var=0.0167)
- Experiment 16: real=0.0517 (var=0.0159) vs this=0.0369 (var=0.0127)
- Experiment 17: real=-0.0167 (var=0.0075) vs this=0.1994 (var=0.0261)
- Experiment 18: real=0.0150 (var=0.0121) vs this=0.1575 (var=0.0164)


---

### `pi_6` (overall score: 0.590)

**Description**
**Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    a = None
    b = None
    try:
        if isinstance(state, dict):
            if 'option_a_ratings' in state and 'option_b_ratings' in state:
                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()
                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()
    except Exception:
        a = None
        b = None

    if a is None or b is None:
        try:
            stim = np.asarray(state, dtype=float)
            if stim.ndim == 1:
                stim = stim.reshape(2, -1)
            elif stim.ndim > 2:
                stim = stim.reshape(2, -1)
            if stim.shape[0] != 2:
                stim = stim.reshape(2, -1)
            a = np.asarray(stim[0], dtype=float).ravel()
            b = np.asarray(stim[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- feature-wise comparison ----------
    d = np.sign(a - b)                       # +1 A wins cue, -1 B wins cue, 0 tie
    nA = int(np.sum(d > 0))
    nB = int(np.sum(d < 0))
    D = nA + nB

    # ---- Commitment 1: the count comparator is a gate. No majority -> guess.
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    lam = _get('lam', 1.45)
    beta = _get('beta', 6.7)
    gamma = _get('gamma', 0.90)
    kappa_d = _get('kappa_d', 0.10)
    psi = _get('psi', 0.40)
    chi = _get('chi', 0.95)
    eps = _get('epsilon', 0.17)
    omega = _get('omega', 0.34)              # saturating-redundancy exponent

    lam = float(min(max(lam, 0.0), 2.5))
    beta = float(min(max(beta, 0.0), 30.0))
    gamma = float(min(max(gamma, 0.0), 2.5))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))
    psi = float(min(max(psi, 0.0), 1.0))
    chi = float(min(max(chi, 0.0), 5.0))
    eps = float(min(max(eps, 0.0), 0.6))
    omega = float(min(max(omega, 0.2), 1.5))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    rng = v_max - v_min
    if not np.isfinite(rng) or rng <= 1e-9:
        s = np.zeros(n, dtype=float)          # no validity contrast in the panel
    else:
        s = (v - v_min) / rng                 # panel-relative informativeness, in [0,1]

    # ---- Commitment 3: the redundancy construal SATURATES above the floor ----
    # strongly concave transform: being appreciably above the panel's least-accurate
    # voice is already enough to be heard as "one of the accurate ones".
    s = np.power(np.clip(s, 0.0, 1.0), omega)

    # ---- Commitment 5: capacity overload of the panel (~7 experts) --------
    N_CAP = 7.5
    W_CAP = 0.20
    arg_o = (float(n) - N_CAP) / W_CAP
    arg_o = float(min(max(arg_o, -60.0), 60.0))
    o = 1.0 / (1.0 + np.exp(-arg_o))

    lam_n = lam * (1.0 - psi * o)
    if not np.isfinite(lam_n) or lam_n < 0.0:
        lam_n = 0.0

    # ---- Commitment 4: redundancy is comparative (needs a chorus) -------
    lam_eff = lam_n * (float(D) - 1.0) / ((float(D) - 1.0) + kappa_d)
    if not np.isfinite(lam_eff) or lam_eff < 0.0:
        lam_eff = 0.0

    # ---- Commitment 2: inverse-informativeness weights ------------------
    w = 1.0 - lam_eff * s
    E = float(np.sum(w * d))

    # ---- Commitment 4b: compression by the number of contradicting cues --
    denom = float(D) ** gamma
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    overload_noise = 1.0 + chi * o
    if not np.isfinite(overload_noise) or overload_noise <= 0.0:
        overload_noise = 1.0
    denom = denom * overload_noise

    z = beta * E / denom
    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---- attention lapse ------------------------------------------------
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- lam: [1.10, 1.80]
- beta: [5.8, 7.6]
- gamma: [0.74, 1.06]
- kappa_d: [0.01, 0.45]
- psi: [0.32, 0.48]
- chi: [0.75, 1.15]
- epsilon: [0.07, 0.27]
- omega: [0.26, 0.42]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5106 (var=0.0056)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.7025 (var=0.0100)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.7831 (var=0.0039)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.7175 (var=0.0042)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.7525 (var=0.0071)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.9162 (var=0.0494)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.6775 (var=0.0023)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.4919 (var=0.0121)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.3791 (var=0.0043)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=0.3338 (var=0.0282)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.5292 (var=0.0060)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.2562 (var=0.0218)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.3654 (var=0.0148)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.2792 (var=0.0159)
- Experiment 15: real=0.0067 (var=0.0123) vs this=0.3375 (var=0.0364)
- Experiment 16: real=0.0517 (var=0.0159) vs this=0.4219 (var=0.0362)
- Experiment 17: real=-0.0167 (var=0.0075) vs this=0.1228 (var=0.0386)
- Experiment 18: real=0.0150 (var=0.0121) vs this=0.1600 (var=0.0457)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1001 -> ACCEPTED
- iter 2: loss=0.0650 -> ACCEPTED
- iter 3: loss=0.0577 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0577 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4738 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1: 0.4288 (var=0.0115) (Δ vs real -0.0450)
  - iter 2: 0.4706 (var=0.0090) (Δ vs real -0.0031)
  - iter 3 (most recent): 0.4569 (var=0.0065) (Δ vs real -0.0169)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)
- pi_5: 0.5169 (var=0.0089)
- pi_6: 0.5106 (var=0.0056)
- pi_7: 0.4819 (var=0.0066)
- pi_8: 0.5075 (var=0.0083)
- pi_9: 0.5081 (var=0.0086)
- pi_10: 0.5112 (var=0.0071)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
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
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6538 (var=0.0092)
**Previous candidate values (this loop):**
  - iter 1: 0.8337 (var=0.0043) (Δ vs real +0.1800)
  - iter 2: 0.7562 (var=0.0056) (Δ vs real +0.1025)
  - iter 3 (most recent): 0.7694 (var=0.0054) (Δ vs real +0.1156)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)
- pi_5: 0.9281 (var=0.0024)
- pi_6: 0.7025 (var=0.0100)
- pi_7: 0.7050 (var=0.0185)
- pi_8: 0.7425 (var=0.0058)
- pi_9: 0.7087 (var=0.0069)
- pi_10: 0.7531 (var=0.0055)

### Experiment 3
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a, b, r in zip(a_col, b_col, r_col):
        av = np.asarray(a, dtype=float).ravel()
        bv = np.asarray(b, dtype=float).ravel()
        if av.size == 0 or av.size != bv.size:
            continue
        nA = float(np.sum(av > bv))
        nB = float(np.sum(bv > av))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

```

**Observed (real) value:** 0.7937 (var=0.0071)
**Previous candidate values (this loop):**
  - iter 1: 0.8067 (var=0.0015) (Δ vs real +0.0129)
  - iter 2: 0.7921 (var=0.0021) (Δ vs real -0.0017)
  - iter 3 (most recent): 0.8271 (var=0.0014) (Δ vs real +0.0333)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)
- pi_5: 0.8167 (var=0.0020)
- pi_6: 0.7831 (var=0.0039)
- pi_7: 0.7769 (var=0.0024)
- pi_8: 0.8994 (var=0.0010)
- pi_9: 0.7742 (var=0.0020)
- pi_10: 0.8108 (var=0.0026)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Adherence to the full-profile tally winner.

    For every trial we recompute the global feature-wise win counts
    nA = #(A > B) and nB = #(B > A) over ALL features, identify the
    option favoured by the complete tally, and score whether the subject
    chose it. The returned scalar is the proportion of such trials on
    which the subject followed the complete-tally winner.

    Works identically on the pooled DataFrame and on a single subject's
    slice.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        nA = float(np.sum(a > b))
        nB = float(np.sum(b > a))
        if nA == nB:
            continue  # undiagnostic (no complete-tally winner)
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except (TypeError, ValueError):
            continue
        if resp not in (0, 1):
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7546 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: 0.7971 (var=0.0022) (Δ vs real +0.0425)
  - iter 2: 0.7381 (var=0.0027) (Δ vs real -0.0165)
  - iter 3 (most recent): 0.7583 (var=0.0023) (Δ vs real +0.0037)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)
- pi_5: 0.7783 (var=0.0043)
- pi_6: 0.7175 (var=0.0042)
- pi_7: 0.7642 (var=0.0035)
- pi_8: 0.8350 (var=0.0023)
- pi_9: 0.7492 (var=0.0014)
- pi_10: 0.6992 (var=0.0030)

### Experiment 5
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.94, 0.57, 0.86, 0.62, 0.70, 0.78], dtype=float)

    coh_hit = 0.0
    coh_n = 0.0
    con_hit = 0.0
    con_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if n == 0:
            continue
        if validities.shape[0] == n:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        a_win = a > b
        b_win = b > a
        nA = int(a_win.sum())
        nB = int(b_win.sum())
        if nA == nB:
            continue  # no count winner; excluded
        winner = 0 if nA > nB else 1

        disc = a_win | b_win
        if not disc.any():
            continue
        # highest-validity discriminating cue
        idx = int(np.argmax(np.where(disc, v, -np.inf)))
        top_owner = 0 if a_win[idx] else 1
        coherent = (top_owner == winner)

        try:
            r = int(resp)
        except Exception:
            continue
        hit = 1.0 if r == winner else 0.0

        if coherent:
            coh_hit += hit
            coh_n += 1.0
        else:
            con_hit += hit
            con_n += 1.0

    if coh_n == 0.0 or con_n == 0.0:
        return 0.0
    return float(coh_hit / coh_n - con_hit / con_n)

```

**Observed (real) value:** -0.7633 (var=0.0200)
**Previous candidate values (this loop):**
  - iter 1: -0.7046 (var=0.0040) (Δ vs real +0.0587)
  - iter 2: -0.6608 (var=0.0068) (Δ vs real +0.1025)
  - iter 3 (most recent): -0.7283 (var=0.0062) (Δ vs real +0.0350)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)
- pi_5: -0.7671 (var=0.0062)
- pi_6: -0.7525 (var=0.0071)
- pi_7: -0.7733 (var=0.0093)
- pi_8: -0.7063 (var=0.0049)
- pi_9: -0.6542 (var=0.0050)
- pi_10: -0.7558 (var=0.0048)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55], dtype=float)

    coh_hit = 0.0; coh_n = 0
    con_hit = 0.0; con_n = 0
    us_hit = 0.0; us_n = 0
    ud_hit = 0.0; ud_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(a.size)
        if n == 0 or int(b.size) != n:
            continue
        if n == int(V.size):
            w = V - 0.5
        elif n > 1:
            w = np.linspace(0.95, 0.55, n) - 0.5
        else:
            w = np.array([0.25])

        aw = a > b
        bw = b > a
        nA = int(np.sum(aw))
        nB = int(np.sum(bw))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        hit = 1.0 if r == winner else 0.0
        D = nA + nB
        if winner == 0:
            Ww = float(np.sum(w[aw])); Wl = float(np.sum(w[bw])); n_lose = nB
        else:
            Ww = float(np.sum(w[bw])); Wl = float(np.sum(w[aw])); n_lose = nA

        if n_lose == 0:
            # unanimous profiles: sparse (few discriminating cues) vs dense
            if D <= 2:
                us_hit += hit; us_n += 1
            else:
                ud_hit += hit; ud_n += 1
        else:
            if Ww > Wl:
                coh_hit += hit; coh_n += 1
            elif Ww < Wl:
                con_hit += hit; con_n += 1

    if coh_n > 0 and con_n > 0:
        d1 = coh_hit / coh_n - con_hit / con_n
    else:
        d1 = 0.0
    if us_n > 0 and ud_n > 0:
        d2 = us_hit / us_n - ud_hit / ud_n
    else:
        d2 = 0.0

    return float(d1 + d2)

```

**Observed (real) value:** -0.9375 (var=0.0693)
**Previous candidate values (this loop):**
  - iter 1: -0.9638 (var=0.0281) (Δ vs real -0.0263)
  - iter 2: -0.9100 (var=0.0385) (Δ vs real +0.0275)
  - iter 3 (most recent): -1.0006 (var=0.0470) (Δ vs real -0.0631)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)
- pi_5: -0.8519 (var=0.0288)
- pi_6: -0.9162 (var=0.0494)
- pi_7: -0.8237 (var=0.0280)
- pi_8: -1.0012 (var=0.0482)
- pi_9: -0.9213 (var=0.0220)
- pi_10: -0.7575 (var=0.0424)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [0.95, 0.57, 0.88, 0.60, 0.66, 0.83]
    # Median split -> STAR tier (high validity)  = features 0, 2, 5
    #              -> MARGINAL tier (low validity) = features 1, 3, 4
    validities = np.array([0.95, 0.57, 0.88, 0.60, 0.66, 0.83], dtype=float)

    hits = 0.0
    n = 0.0

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        m = a.size
        if validities.size == m:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, m)
        order = np.argsort(-v, kind='stable')
        k = m // 2
        star_idx = order[:k]
        marg_idx = order[k:]

        d = np.sign(a - b)  # +1 A wins the cue, -1 B wins the cue, 0 tie
        star_score = float(np.sum(d[star_idx]))
        marg_score = float(np.sum(d[marg_idx]))

        # keep only trials where BOTH tiers discriminate and they DISAGREE
        if star_score == 0.0 or marg_score == 0.0:
            continue
        if np.sign(star_score) == np.sign(marg_score):
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        marginal_preferred = 0 if marg_score > 0 else 1  # 0 = A, 1 = B
        hits += 1.0 if resp == marginal_preferred else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.6606 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.7706 (var=0.0035) (Δ vs real +0.1100)
  - iter 2: 0.7656 (var=0.0043) (Δ vs real +0.1050)
  - iter 3 (most recent): 0.7778 (var=0.0036) (Δ vs real +0.1172)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9059 (var=0.0023)
- pi_4: 0.4641 (var=0.0042)
- pi_1: 0.1278 (var=0.0098)
- pi_2: 0.4997 (var=0.0029)
- pi_3: 0.4981 (var=0.0037)
- pi_6: 0.6775 (var=0.0023)
- pi_7: 0.7197 (var=0.0017)
- pi_8: 0.6572 (var=0.0028)
- pi_9: 0.6750 (var=0.0035)
- pi_10: 0.6647 (var=0.0029)

### Experiment 8
**Design**
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-time validities (fixed by the experiment), used only to define the
    # median-split 'star' tier structurally.
    v = np.array([0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80], dtype=float)
    n = v.shape[0]
    order = np.argsort(-v, kind='stable')
    star = set(int(i) for i in order[: n // 2])

    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        d = np.sign(a - b)
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        D = nA + nB
        # dense / mid displays only (D >= 4), with a strict count winner
        if D < 4 or nA == nB:
            continue
        winner = 0 if nA > nB else 1
        if winner == 0:
            win_idx = set(int(i) for i in np.where(d > 0)[0])
            lose_idx = set(int(i) for i in np.where(d < 0)[0])
        else:
            win_idx = set(int(i) for i in np.where(d < 0)[0])
            lose_idx = set(int(i) for i in np.where(d > 0)[0])
        # keep only STAR-BACKED majorities: the count winner's wins are all
        # high-validity (star) experts, the count loser's wins are all
        # low-validity (marginal) experts.
        if not win_idx.issubset(star):
            continue
        if len(lose_idx & star) > 0:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4850 (var=0.0085)
**Previous candidate values (this loop):**
  - iter 1: 0.4419 (var=0.0075) (Δ vs real -0.0431)
  - iter 2: 0.4619 (var=0.0079) (Δ vs real -0.0231)
  - iter 3 (most recent): 0.4988 (var=0.0073) (Δ vs real +0.0138)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7144 (var=0.0105)
- pi_5: 0.1506 (var=0.0044)
- pi_1: 0.8488 (var=0.0147)
- pi_2: 0.8550 (var=0.0140)
- pi_3: 0.5913 (var=0.0076)
- pi_6: 0.4919 (var=0.0121)
- pi_7: 0.4775 (var=0.0150)
- pi_8: 0.4850 (var=0.0181)
- pi_9: 0.4931 (var=0.0066)
- pi_10: 0.4850 (var=0.0089)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [f0..f5]
    val = np.array([0.95, 0.78, 0.82, 0.55, 0.76, 0.80], dtype=float)
    n_feat = val.shape[0]
    order = np.argsort(-val, kind='stable')
    marginal_idx = [int(j) for j in order[n_feat // 2:]]  # bottom-half tier: f1, f4, f3

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n_feat or b.shape[0] != n_feat:
            continue
        d = np.sign(a - b)
        D = int(np.sum(d != 0))
        if D < 2:
            continue  # drop the solo-cue cells (K5/K6), where the two theories nearly agree
        net_marg = float(np.sum(d[marginal_idx]))
        if net_marg == 0.0:
            continue
        pred = 0 if net_marg > 0 else 1  # 0 = chose A, 1 = chose B
        try:
            rr = int(r)
        except Exception:
            continue
        hits += 1.0 if rr == pred else 0.0
        tot += 1.0

    if tot <= 0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.4050 (var=0.0022)
**Previous candidate values (this loop):**
  - iter 1: 0.3884 (var=0.0032) (Δ vs real -0.0166)
  - iter 2: 0.3903 (var=0.0028) (Δ vs real -0.0147)
  - iter 3 (most recent): 0.3981 (var=0.0023) (Δ vs real -0.0069)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8438 (var=0.0038)
- pi_6: 0.3791 (var=0.0043)
- pi_1: 0.3253 (var=0.0042)
- pi_2: 0.4994 (var=0.0020)
- pi_3: 0.4828 (var=0.0041)
- pi_4: 0.5244 (var=0.0029)
- pi_7: 0.3969 (var=0.0046)
- pi_8: 0.3950 (var=0.0035)
- pi_9: 0.3756 (var=0.0029)
- pi_10: 0.3747 (var=0.0028)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell C3 (tally-tied 2v2, tier-mismatched): discriminating cues {f1,f2,f5,f6};
    #   one option is backed by the two BOTTOM experts {f2(.55), f5(.52)},
    #   the other by the two STARS {f1(.95), f6(.90)}.
    # Cell C4 (mid-pair vs floor): discriminating cues {f0,f4,f5};
    #   one option is backed by the mid pair {f0(.74), f4(.70)},
    #   the other only by the floor expert {f5(.52)}.
    C3_ALL = frozenset([1, 2, 5, 6])
    C3_BOTTOM = frozenset([2, 5])
    C4_ALL = frozenset([0, 4, 5])
    C4_FLOOR = frozenset([5])

    c3_hits = 0
    c3_n = 0
    c4_hits = 0
    c4_n = 0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
            resp = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) < 7:
            continue
        pos = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        neg = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        allset = frozenset(pos | neg)

        if allset == C3_ALL:
            # which side is the bottom-expert-backed one?
            if pos == C3_BOTTOM:
                target = 0  # A
            elif neg == C3_BOTTOM:
                target = 1  # B
            else:
                target = None
            if target is not None:
                c3_n += 1
                if resp == target:
                    c3_hits += 1

        elif allset == C4_ALL:
            # which side is backed only by the floor expert f5?
            if pos == C4_FLOOR:
                target = 0
            elif neg == C4_FLOOR:
                target = 1
            else:
                target = None
            if target is not None:
                c4_n += 1
                if resp == target:
                    c4_hits += 1

    if c3_n == 0 or c4_n == 0:
        return float('nan')

    p_c4_floor = c4_hits / float(c4_n)
    p_c3_bottom = c3_hits / float(c3_n)
    return float(p_c4_floor - p_c3_bottom)

```

**Observed (real) value:** -0.0225 (var=0.0100)
**Previous candidate values (this loop):**
  - iter 1: -0.0050 (var=0.0175) (Δ vs real +0.0175)
  - iter 2: 0.0188 (var=0.0232) (Δ vs real +0.0413)
  - iter 3 (most recent): 0.0425 (var=0.0132) (Δ vs real +0.0650)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3338 (var=0.0282)
- pi_5: -0.7275 (var=0.0246)
- pi_1: -0.0050 (var=0.0123)
- pi_2: -0.3575 (var=0.0255)
- pi_3: -0.1062 (var=0.0243)
- pi_4: -0.2200 (var=0.0268)
- pi_7: -0.0475 (var=0.0412)
- pi_8: -0.0162 (var=0.0372)
- pi_9: 0.0663 (var=0.0326)
- pi_10: 0.0387 (var=0.0312)

### Experiment 11
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    cells = {}
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ai = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        bi = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(ai) == len(bi):
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if len(ai) > len(bi):
            maj, mino = ai, bi
            follow = 1.0 if resp == 0 else 0.0
        else:
            maj, mino = bi, ai
            follow = 1.0 if resp == 1 else 0.0
        key = (maj, mino)
        if key not in cells:
            cells[key] = []
        cells[key].append(follow)

    def rate(maj, mino):
        k = (frozenset(maj), frozenset(mino))
        v = cells.get(k)
        if v is None or len(v) == 0:
            return None
        return float(np.mean(v))

    # cells identified by which expert indices each side wins
    # validities = [0.78, 0.95, 0.68, 0.84, 0.90, 0.72]
    d3 = rate({0, 3}, {1})          # mid-pair majority vs LONE STAR (.95)
    c2 = rate({0, 3}, {2})          # same majority vs LONE FLOOR (.68)
    e4 = rate({1, 2, 4}, {0, 5})    # star+star+floor majority vs mid-pair
    a1 = rate({0, 3, 5}, {1, 2})    # mid-trio majority vs star+floor

    terms = []
    if (d3 is not None) and (c2 is not None):
        terms.append(d3 - c2)
    if e4 is not None:
        terms.append(e4)
    if a1 is not None:
        terms.append(1.0 - a1)

    if len(terms) == 0:
        return float('nan')
    return float(np.mean(terms))
```

**Observed (real) value:** 0.6983 (var=0.0122)
**Previous candidate values (this loop):**
  - iter 1: 0.5950 (var=0.0078) (Δ vs real -0.1033)
  - iter 2: 0.6071 (var=0.0074) (Δ vs real -0.0912)
  - iter 3 (most recent): 0.6583 (var=0.0049) (Δ vs real -0.0400)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1512 (var=0.0062)
- pi_6: 0.5292 (var=0.0060)
- pi_1: 0.3317 (var=0.0037)
- pi_2: 0.3525 (var=0.0038)
- pi_3: 0.3254 (var=0.0065)
- pi_4: 0.3283 (var=0.0079)
- pi_5: 0.2325 (var=0.0054)
- pi_8: 0.6917 (var=0.0064)
- pi_9: 0.6533 (var=0.0064)
- pi_10: 0.6479 (var=0.0051)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in list(a_raw)])
            b = np.asarray([float(x) for x in list(b_raw)])
        except Exception:
            continue
        if a.size != b.size or a.size < 5:
            continue
        d = np.sign(a - b)
        iA = np.where(d > 0)[0]
        iB = np.where(d < 0)[0]
        try:
            resp = int(r)
        except Exception:
            continue
        # K4-type displays: a 3-cue majority opposed by exactly ONE cue,
        # and that lone opposing cue is the panel's floor expert (index 4, v=0.52).
        if iA.size == 3 and iB.size == 1 and int(iB[0]) == 4:
            tot += 1
            if resp == 0:
                hits += 1
        elif iB.size == 3 and iA.size == 1 and int(iA[0]) == 4:
            tot += 1
            if resp == 1:
                hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.1450 (var=0.0099)
**Previous candidate values (this loop):**
  - iter 1: 0.3625 (var=0.0136) (Δ vs real +0.2175)
  - iter 2: 0.2913 (var=0.0134) (Δ vs real +0.1463)
  - iter 3 (most recent): 0.2225 (var=0.0106) (Δ vs real +0.0775)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2562 (var=0.0218)
- pi_7: 0.9187 (var=0.0063)
- pi_1: 0.8500 (var=0.0170)
- pi_2: 0.8438 (var=0.0149)
- pi_3: 0.5925 (var=0.0166)
- pi_4: 0.7612 (var=0.0210)
- pi_5: 0.8588 (var=0.0103)
- pi_8: 0.1375 (var=0.0089)
- pi_9: 0.0788 (var=0.0054)
- pi_10: 0.1225 (var=0.0080)

### Experiment 13
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # stated validities of the six experts in screen order (design-time constant)
    V = np.array([0.88, 0.55, 0.72, 0.95, 0.63, 0.80], dtype=float)
    THR = 0.70  # 'upper panel' voices: 0.72, 0.80, 0.88, 0.95

    hits = 0
    tot = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in a_raw], dtype=float)
            b = np.asarray([float(x) for x in b_raw], dtype=float)
            r = int(r_raw)
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        if a.size == V.size:
            v = V
        else:
            v = np.full(a.size, 1.0, dtype=float)

        d = np.sign(a - b)
        idx = np.nonzero(d)[0]
        if idx.size == 0:
            continue
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        if nA == nB:
            continue  # no strict count winner: both theories predict chance
        # keep only displays where EVERY discriminating voice is an upper-panel expert
        if np.any(v[idx] < THR):
            continue

        winner = 0 if nA > nB else 1  # 0 == chose A, 1 == chose B
        tot += 1
        if r == winner:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.4892 (var=0.0049)
**Previous candidate values (this loop):**
  - iter 1: 0.5729 (var=0.0057) (Δ vs real +0.0837)
  - iter 2: 0.5267 (var=0.0040) (Δ vs real +0.0375)
  - iter 3 (most recent): 0.5246 (var=0.0048) (Δ vs real +0.0354)
**Other theories' values on this metric (for reference):**
- pi_8: 0.6446 (var=0.0042)
- pi_6: 0.3654 (var=0.0148)
- pi_1: 0.6813 (var=0.0037)
- pi_2: 0.8667 (var=0.0111)
- pi_3: 0.5975 (var=0.0056)
- pi_4: 0.7529 (var=0.0048)
- pi_5: 0.4121 (var=0.0048)
- pi_7: 0.0892 (var=0.0030)
- pi_9: 0.5542 (var=0.0056)
- pi_10: 0.5713 (var=0.0051)

### Experiment 14
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Target cells: displays where the TALLY-MAJORITY side is carried by the two
    # most valid experts (features 0 = .92 and 2 = .84).
    # C1: majority wins {0,2}, minority wins {}        (unopposed star pair, 2v0)
    # C3: majority wins {0,2}, minority wins {5}       (star pair vs lone mid, 2v1)
    # C4: majority wins {0,2}, minority wins {1}       (star pair vs lone low, 2v1)
    # C7: majority wins {0,2,4}, minority wins {5}     (all-upper 3v1)
    targets = {
        (frozenset([0, 2]), frozenset()),
        (frozenset([0, 2]), frozenset([5])),
        (frozenset([0, 2]), frozenset([1])),
        (frozenset([0, 2, 4]), frozenset([5])),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        idxA = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        idxB = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(idxA) == len(idxB):
            continue
        if len(idxA) > len(idxB):
            maj_set, min_set, maj_resp = idxA, idxB, 0  # A is count winner
        else:
            maj_set, min_set, maj_resp = idxB, idxA, 1  # B is count winner
        if (maj_set, min_set) not in targets:
            continue
        try:
            r = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if r == maj_resp else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.4892 (var=0.0049)
**Previous candidate values (this loop):**
  - iter 1: 0.4963 (var=0.0040) (Δ vs real +0.0071)
  - iter 2: 0.5042 (var=0.0053) (Δ vs real +0.0150)
  - iter 3 (most recent): 0.5192 (var=0.0055) (Δ vs real +0.0300)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2792 (var=0.0159)
- pi_8: 0.5913 (var=0.0058)
- pi_1: 0.8583 (var=0.0102)
- pi_2: 0.8646 (var=0.0075)
- pi_3: 0.6125 (var=0.0071)
- pi_4: 0.7679 (var=0.0054)
- pi_5: 0.2533 (var=0.0084)
- pi_7: 0.0717 (var=0.0021)
- pi_9: 0.4929 (var=0.0046)
- pi_10: 0.5083 (var=0.0056)

### Experiment 15
**Design**
  A=[1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Canonical (order-free) keys for the four diagnostic cells.
    # SOLO cells: exactly one discriminating expert, and it is NOT the panel floor
    #   C1 = solo .95 star, C2 = solo .75 mid
    SOLO = {
        ("1001010", "1101010"),   # C1 (trials 1/2)
        ("0010101", "1010101"),   # C2 (trials 3/4)
    }
    # FLAT cells: CH >= 2 displays whose every discriminating voice is upper-band
    #   C4 = unopposed .79/.75 pair (2 vs 0), C5 = low trio vs lone .95 (3 vs 1)
    FLAT = {
        ("0011001", "1011011"),   # C4 (trials 7/8)
        ("0101000", "1011010"),   # C5 (trials 9/10)
    }

    solo_hits = 0.0
    solo_n = 0.0
    flat_hits = 0.0
    flat_n = 0.0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ka = ''.join([str(x) for x in a])
        kb = ''.join([str(x) for x in b])
        key = tuple(sorted((ka, kb)))

        if key in SOLO:
            grp = 0
        elif key in FLAT:
            grp = 1
        else:
            continue

        mA = sum(1 for i in range(len(a)) if a[i] > b[i])
        mB = sum(1 for i in range(len(a)) if b[i] > a[i])
        if mA == mB:
            continue
        winner = 0 if mA > mB else 1

        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        if grp == 0:
            solo_hits += hit
            solo_n += 1.0
        else:
            flat_hits += hit
            flat_n += 1.0

    if solo_n == 0.0 or flat_n == 0.0:
        return float('nan')

    return float(solo_hits / solo_n - flat_hits / flat_n)
```

**Observed (real) value:** 0.0067 (var=0.0123)
**Previous candidate values (this loop):**
  - iter 1: -0.0517 (var=0.0173) (Δ vs real -0.0583)
  - iter 2: -0.0158 (var=0.0226) (Δ vs real -0.0225)
  - iter 3 (most recent): 0.0058 (var=0.0233) (Δ vs real -0.0008)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4275 (var=0.0123)
- pi_9: -0.0583 (var=0.0163)
- pi_1: 0.3392 (var=0.0260)
- pi_2: -0.0092 (var=0.0081)
- pi_3: -0.0492 (var=0.0130)
- pi_4: 0.1650 (var=0.0092)
- pi_5: -0.1375 (var=0.0105)
- pi_6: 0.3375 (var=0.0364)
- pi_7: 0.2758 (var=0.0115)
- pi_10: -0.0167 (var=0.0167)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # cell accumulators: hits / total
    hits = {}
    tot = {}

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        winsA = [i for i in range(len(a)) if a[i] > b[i]]
        winsB = [i for i in range(len(a)) if b[i] > a[i]]
        mA, mB = len(winsA), len(winsB)
        if mA == mB:
            continue
        if mA > mB:
            win_idx, lose_idx, winner_is_A = set(winsA), set(winsB), True
        else:
            win_idx, lose_idx, winner_is_A = set(winsB), set(winsA), False
        D = mA + mB

        cell = None
        if D == 1:
            j = list(win_idx)[0]
            if j == 2:
                cell = 'C1'      # solo star (v=.95)
            elif j == 3:
                cell = 'C2'      # solo low-upper (v=.78)
        elif D == 2 and len(lose_idx) == 0:
            if win_idx == {0, 3}:
                cell = 'C4'      # unopposed low pair
            elif win_idx == {2, 4}:
                cell = 'C5'      # unopposed senior pair
        elif D == 4 and win_idx == {0, 3, 5} and lose_idx == {2}:
            cell = 'C6'          # all-upper 3v1

        if cell is None:
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_A = (resp == 0)
        follow = 1.0 if (chose_A == winner_is_A) else 0.0

        hits[cell] = hits.get(cell, 0.0) + follow
        tot[cell] = tot.get(cell, 0.0) + 1.0

    def rate(c):
        if tot.get(c, 0.0) <= 0:
            return None
        return hits[c] / tot[c]

    g1 = [r for r in (rate('C1'), rate('C2')) if r is not None]
    g2 = [r for r in (rate('C4'), rate('C5'), rate('C6')) if r is not None]

    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    return float(np.mean(g1) - np.mean(g2))
```

**Observed (real) value:** 0.0517 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1: -0.0489 (var=0.0149) (Δ vs real -0.1006)
  - iter 2: -0.0097 (var=0.0137) (Δ vs real -0.0614)
  - iter 3 (most recent): -0.0022 (var=0.0167) (Δ vs real -0.0539)
**Other theories' values on this metric (for reference):**
- pi_9: -0.0381 (var=0.0193)
- pi_8: 0.4408 (var=0.0102)
- pi_1: 0.2372 (var=0.0099)
- pi_2: -0.0078 (var=0.0092)
- pi_3: -0.0428 (var=0.0142)
- pi_4: 0.1047 (var=0.0067)
- pi_5: -0.0139 (var=0.0137)
- pi_6: 0.4219 (var=0.0362)
- pi_7: 0.5256 (var=0.0124)
- pi_10: 0.0369 (var=0.0127)

### Experiment 17
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    FLOOR = 2          # validity 0.54, unique panel floor
    # cells in which the floor voice sides with a set F against opposition O
    POS = {
        (frozenset({2}), frozenset({1, 4})),        # C2: floor vs STAR-PAIR
        (frozenset({1, 2, 4}), frozenset({0})),     # C4: floor+2 stars vs lone mid
        (frozenset({2, 3, 5}), frozenset({1})),     # C5: floor+2 mids vs lone star
    }
    NEG = {
        (frozenset({1, 2, 4}), frozenset()),        # C7: unanimous floor+star+star
    }

    pos_hits = 0
    pos_n = 0
    neg_hits = 0
    neg_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        winA = frozenset(int(i) for i in np.where(a > b)[0])
        winB = frozenset(int(i) for i in np.where(b > a)[0])
        if FLOOR in winA:
            F, O = winA, winB
            follow = 1 if int(resp) == 0 else 0
        elif FLOOR in winB:
            F, O = winB, winA
            follow = 1 if int(resp) == 1 else 0
        else:
            continue
        key = (F, O)
        if key in POS:
            pos_n += 1
            pos_hits += follow
        elif key in NEG:
            neg_n += 1
            neg_hits += follow

    if pos_n == 0 or neg_n == 0:
        return float('nan')

    return float(pos_hits) / pos_n - float(neg_hits) / neg_n

```

**Observed (real) value:** -0.0167 (var=0.0075)
**Previous candidate values (this loop):**
  - iter 1: -0.0189 (var=0.0106) (Δ vs real -0.0022)
  - iter 2: -0.0011 (var=0.0119) (Δ vs real +0.0156)
  - iter 3 (most recent): 0.0000 (var=0.0123) (Δ vs real +0.0167)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1994 (var=0.0261)
- pi_9: -0.2711 (var=0.0109)
- pi_1: -0.4772 (var=0.0289)
- pi_2: -0.2306 (var=0.0167)
- pi_3: -0.1383 (var=0.0308)
- pi_4: -0.3794 (var=0.0138)
- pi_5: -0.1172 (var=0.0108)
- pi_6: 0.1228 (var=0.0386)
- pi_7: -0.2450 (var=0.0180)
- pi_8: -0.0839 (var=0.0108)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    titration = []   # cells C3 ({1} vs {0,4}) and C4 ({1} vs {2,4})
    unanimous = []   # cell C6 ({1,2,4} unanimous bloc)

    target_others = (frozenset({0, 4}), frozenset({2, 4}))
    floor_only = frozenset({1})
    bloc = frozenset({1, 2, 4})

    for a, b, r in zip(a_list, b_list, resp):
        try:
            av = [int(x) for x in a]
            bv = [int(x) for x in b]
        except Exception:
            continue
        n = len(av)
        if n != len(bv) or n != 7:
            continue
        setA = frozenset([i for i in range(n) if av[i] > bv[i]])
        setB = frozenset([i for i in range(n) if bv[i] > av[i]])
        if len(setA) == len(setB):
            continue
        if 1 in setA:
            focal, other, focal_is_a = setA, setB, True
        elif 1 in setB:
            focal, other, focal_is_a = setB, setA, False
        else:
            continue
        try:
            chose_a = (int(r) == 0)
        except Exception:
            continue
        chose_focal = 1.0 if (chose_a == focal_is_a) else 0.0

        if focal == floor_only and other in target_others:
            titration.append(chose_focal)
        elif focal == bloc and len(other) == 0:
            unanimous.append(chose_focal)

    if len(titration) == 0 or len(unanimous) == 0:
        return float('nan')

    return float(np.mean(titration) - np.mean(unanimous))
```

**Observed (real) value:** 0.0150 (var=0.0121)
**Previous candidate values (this loop):**
  - iter 1: -0.0400 (var=0.0151) (Δ vs real -0.0550)
  - iter 2: -0.0217 (var=0.0140) (Δ vs real -0.0367)
  - iter 3 (most recent): -0.0133 (var=0.0118) (Δ vs real -0.0283)
**Other theories' values on this metric (for reference):**
- pi_9: -0.3142 (var=0.0128)
- pi_10: 0.1575 (var=0.0164)
- pi_1: -0.6817 (var=0.0373)
- pi_2: -0.7100 (var=0.0442)
- pi_3: -0.2942 (var=0.0319)
- pi_4: -0.5683 (var=0.0297)
- pi_5: 0.0417 (var=0.0215)
- pi_6: 0.1600 (var=0.0457)
- pi_7: 0.0750 (var=0.0179)
- pi_8: -0.2258 (var=0.0197)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC3).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), now VERY steep with a near-vanishing residual.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 10.5–14 and a very thin flat floor eta ≈ 0.008–0.030. The panel's bottom voice speaks with essentially all the weight; the second-from-bottom keeps only a small share; mid and star voices collapse onto the flat residual and are therefore effectively interchangeable and nearly mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, and further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.30–0.50. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair — spreading evidence across more attributes never buys confidence when the attributes are all senior. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a reduced gain rho_tie ≈ 0.55–0.75, because the subject construes the display as only weakly diagnostic. Tie displays therefore hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) MILD CAPACITY.** Holding a whole validity ordering in mind gets harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi small. Eight-expert panels are slightly flatter — star-backed dense majorities stay at chance — but plain tally adherence survives essentially intact.

**(5) LAPSE.** An attention lapse epsilon mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    eta = _g('eta', 0.018, 0.0, 0.50)        # thin flat listening floor
    kappa = _g('kappa', 12.0, 1.0, 24.0)     # steepness of inverse-seniority descent
    beta = _g('beta', 9.5, 0.0, 40.0)        # integration slope
    c_n = _g('c_n', 1.35, 0.05, 20.0)        # constant of indecision
    gamma = _g('gamma', 0.40, 0.0, 2.0)      # mild count dilution
    rho_tie = _g('rho_tie', 0.65, 0.0, 1.0)  # confidence gain when counts are tied
    eps = _g('epsilon', 0.11, 0.0, 0.60)     # attention lapse
    chi = _g('chi', 0.14, 0.0, 2.0)          # mild capacity flattening past 7 voices

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    # ---------- Commitment 1: context-free inverse-seniority weights ----------
    base = np.clip(1.0 - s, 0.0, 1.0)
    try:
        w = eta + np.power(base, kappa)
    except Exception:
        w = eta + base
    w = np.where(np.isfinite(w), w, eta)
    w = np.clip(w, 0.0, None)

    # ---------------- feature-wise comparison ----------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    D = mA + mB
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    sumA = float(np.sum(w[iA])) if mA > 0 else 0.0
    sumB = float(np.sum(w[iB])) if mB > 0 else 0.0
    E = sumA - sumB
    W = sumA + sumB
    if not np.isfinite(E):
        E = 0.0
    if (not np.isfinite(W)) or W < 0.0:
        W = 0.0

    # ---------- Commitment 3: count comparator gates CONFIDENCE only ----------
    gain = rho_tie if mA == mB else 1.0

    # ---------- Commitment 4: mild capacity flattening ----------
    cap = 1.0 + chi * max(0.0, float(n) - 7.0)
    if (not np.isfinite(cap)) or cap <= 0.0:
        cap = 1.0

    # ---------- Commitment 2: Luce-like saturating integration + mild dilution ----------
    dil = float(D) ** gamma
    if (not np.isfinite(dil)) or dil <= 0.0:
        dil = 1.0

    den = (c_n + W) * dil * cap
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E * gain / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 5: lapse ----------
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- eta: [0.008, 0.030]
- kappa: [10.5, 14.0]
- beta: [8.5, 10.5]
- c_n: [1.1, 1.6]
- gamma: [0.30, 0.50]
- rho_tie: [0.55, 0.75]
- epsilon: [0.08, 0.15]
- chi: [0.05, 0.25]
- validities: validities

`rationale`: Minimal-diff edit on the accepted base (loss 0.0650): no mechanism, no equation and no parameter name changed. Only four sampling ranges (and the matching internal clamp/defaults) were re-tuned exactly as the critic prescribed in iteration 2:

(1) kappa [8,11] -> [10.5,14] and eta [0.015,0.045] -> [0.008,0.030]. Steepening the context-free inverse-seniority descent and thinning the flat residual is the single knob pair that moves BOTH remaining residual directions at once. It collapses the second-from-bottom / mid-band voices onto the near-zero residual, pulling the over-shooting cells down (Exp7 ~.77 -> ~.70, Exp2 ~.76 -> ~.70, Exp13 ~.53 -> ~.50, Exp4 ~.74 stays near .75), and it simultaneously removes the dilution that upper cues currently exert on floor-driven displays, sharpening Exp12 (~.29 -> ~.18) and Exp11 (~.61 -> ~.65). kappa is capped at 14 as instructed so the second-from-bottom voice keeps a small but non-zero share and Exp10's tied bottom-pair cell and Exp6's dense cells do not break.

(2) beta [7,9] -> [8.5,10.5] with c_n [1.3,1.9] -> [1.1,1.6]. Because eta is now very thin, upper-only displays still put almost no subjective weight on the table, so the Luce normalisation keeps Exp8/Exp13/Exp14/Exp15/Exp16 near chance; but floor-containing displays now reach the near-ceiling/near-floor rates Exp11 (~.70) and Exp12 (~.15-.20) demand, and the anti-star contrast in Exp5 should regain ~0.05-0.08 of magnitude (-.66 -> ~-.72).

(3) gamma nudged to [0.30,0.50] (the critic's 'leave near current band'); it is NOT raised further, since more dilution acts hardest on D=4 displays and would push Exp12 back toward chance to chase Exp16's remaining 0.06, which lies inside its between-subject noise.

(4) rho_tie and epsilon left untouched, as instructed: rho_tie is doing exactly the right job on Exp1 (.471 vs .474), and raising epsilon would flatten Exp5/Exp6/Exp11/Exp12, all of which need more extremity.

All four changes are context-free, so the family-diagnostic nulls that are this candidate's principal asset (Exp17 ~-0.00 vs -0.017; Exp18 ~-0.02 vs +0.015) are structurally preserved: an opposed floor voice and a floor voice inside a unanimous bloc still receive identical loudness, there is no chorus term, no validity-rank tie-break, and the count comparator still never supplies a direction.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is the best entry so far (loss 0.1001, ACCEPTED) and it nails the hardest, most family-diagnostic contrasts: Exp17 (-0.019 vs -0.017) and Exp18 (-0.040 vs +0.015) are essentially null, exactly as the arbiter's no-chorus commitment requires, and Exp10 (-0.005 vs -0.023), Exp14 (0.496 vs 0.489), Exp3 (0.807 vs 0.794), Exp6 (-0.964 vs -0.938), Exp8 (0.442 vs 0.485), Exp9 (0.388 vs 0.405) and Exp5 (-0.705 vs -0.763) are all within noise. The theory is faithful to the prescribed family (context-free inverse-seniority weights, count gate on confidence only, Luce-style normalisation, mild capacity, lapse); the only deviation 
is the soft tie gate (rho_tie) instead of an exact 0.5, and that deviation is empirically defensible because it is what keeps Exp10 near zero.

The residuals are, however, systematically structured and clearly parameter-driven rather than mechanism-driven. (i) Exp12 is the single largest miss: 0.363 vs 0.145 
s the lone floor voice opposed by a three-cue upper majority does not reverse hard enough (pi_8/pi_9/pi_10 all reach .08
s.14 here). (ii) Exp11 under-shoots (0.595 vs 0.698) 
s floor-backed cells are again not extreme enough. (iii) In the opposite direction, Exp2 (0.834 vs 0.654), Exp7 (0.771 vs 0.661), Exp13 (0.573 vs 0.489) and Exp4 (0.797 vs 0.755) all over-shoot 
s mid-band voices (second- and third-from-bottom) are still carrying too much weight, so displays whose 'low' side is merely mid-validity are read too decisively. Both directions point to the SAME fix: the seniority descent is not steep enough and the flat residual eta is too fat, so upper/mid voices dilute floor-driven displays (hurting Exp11/Exp12) while themselves driving mid-vs-star displays away from chance (hurting Exp2/Exp7/Exp13). (iv) Exp15 (-0.052 vs +0.007) and Exp16 (-0.049 vs +0.052) have the wrong sign: adding a second thin upper voice currently raises confidence above the solo cell, whereas humans show solo >= multi-cue-upper. The arbiter's spec included a D^gamma count-dilution term that the candidate dropped; a mild version of it would flip these two signs at no cost elsewhere.
Rationale: Stay inside the prescribed FM-ISN family 
s do not change mechanisms, only re-tune three knobs and restore one term the arbiter specified.

1) Steepen the inverse-seniority descent and thin the residual: kappa range ~[8, 12] (currently 5
s7.5) and eta range ~[0.015, 0.045] (currently 0.045
s0.085). This simultaneously (a) removes the dilution that is blunting Exp12 (target ~.15, now .36) and Exp11 (target ~.70, now .60), and (b) collapses the second/third-from-bottom voices toward the residual, which is what pulls Exp7 (.77 -> ~.68), Exp2 (.83 -> ~.75) and Exp13 (.57 -> ~.52) back down. It is the one change that moves both over- and under-shooting experiments in the right direction at once.

2) Sharpen the floor-driven gain slightly: beta toward the top of its range (~[6.5, 8.5]) and c_n a little lower (~[1.3, 1.9]). With eta small, upper-only displays still put almost no weight on the table so Exp8/Exp13/Exp14 stay at chance, but floor-containing displays reach the near-ceiling rates Exp11/Exp12 demand.

3) Restore the arbiter's count-dilution term, mildly: den = (c_n + W) * D^gamma * cap with gamma ~[0.25, 0.6]. This is explicitly part of the prescribed integration rule (point 4) and is the cheapest way to flip the sign of Exp15 (-0.05 -> ~+0.01) and Exp16 (-0.05 -> ~+0.04), where a solo thin voice must be at least as decisive as an unopposed thin pair. Compensate the dilution of Exp12 through the beta/c_n change in (2), not by removing the term.

4) Do NOT touch the tie gate. Lowering rho_tie would improve Exp1 by ~0.03 but would push Exp10's tied cell toward .5 and blow that metric from -0.005 to ~+0.10; keep rho_tie in [0.5, 0.7]. Likewise leave epsilon near its current band ([0.08, 0.15]); raising it to tame Exp2/Exp7 would cost more on Exp5, Exp6 and Exp12 than it buys. Exp2 will remain the residual worst case for any anti-validity account 
s accept ~.75 there rather than distorting the weight function to chase it.

Keep every commitment intact: no chorus amplification, no validity-rank tie-break, direction never set by the count comparator. The Exp17/Exp18 nulls are this candidate's main asset and must survive the re-tune (they will, since all four changes are context-free).

**Outcome of this advice:** iter 1 candidate loss=0.1001 -> iter 2 candidate loss=0.0650 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate (loss 0.0650, ACCEPTED) is now the best entry in this loop and remains faithful to the prescribed FM-ISN family: context-free inverse-seniority weights, count comparator gating confidence only, Luce-style normalisation with mild D^gamma dilution, mild capacity, lapse. My previous prescription (steepen kappa, thin eta, raise beta/lower c_n, restore the arbiter's D^gamma term) was ACCEPTED and moved essentially every flagged residual in the right direction: Exp1 0.429->0.471 (obs 0.474), Exp2 0.834->0.756 (obs 0.654), Exp3 0.807->0.792 (obs 0.794), Exp4 0.797->0.738 (obs 0.755), Exp12 0.363->0.291 (obs 0.145), Exp13 0.573->0.527 (obs 0.489), Exp15 -0.052->-0.016 (obs +0.007), Exp16 -0.049->-0.010 (obs +0.052), and the family-diagnostic nulls survived (Exp17 -0.001 vs -0.017; Exp18 -0.022 vs +0.015). Exp5/Exp6/Exp8/Exp9/Exp14 are all within noise.

The remaining residuals are still structured in exactly the same two directions as before, only smaller. (a) Floor-driven displays are still not extreme enough: Exp12 0.291 vs 0.145 (the lone floor voice must reverse a three-cue upper majority much harder; pi_8/pi_9/pi_10 reach .08-.14 there), Exp11 0.607 vs 0.698, and Exp5 -0.661 vs -0.763 (anti-star effect a bit too weak). (b) Non-floor low/mid voices still carry too much weight, so displays whose 'weak' side is merely second-from-bottom are read too decisively: Exp7 0.766 vs 0.661, Exp2 0.756 vs 0.654, Exp13 0.527 vs 0.489. Both directions are again diagnosed by the same single knob pair: the descent is still not steep enough to isolate the true floor voice, while the overall floor gain is slightly too low. Two minor caveats: Exp10 drifted the wrong way (-0.005 -> +0.019 vs obs -0.023) and Exp16 is still -0.06 off, but both have large between-subject variance and should be treated as low-priority constraints, not as targets to chase.
Rationale: Keep the FM-ISN mechanism exactly as is u2014 no family change, no chorus term, no validity-rank tie-break, direction still never supplied by the count comparator. Take one further, SMALLER step in the same direction that the gate just rewarded:

1) Steepen the seniority descent again, modestly: kappa [8,11] -> ~[10.5,14], eta [0.015,0.045] -> ~[0.008,0.030]. This is the single change that moves both residual directions at once: it collapses the second-from-bottom voice (f3/f4 in Exp7, f4 in Exp2, the mid band in Exp13) onto the thin residual, pulling Exp7 ~.77->~.70, Exp2 ~.76->~.70 and Exp13 ~.53->~.50, while removing the dilution that is currently blunting the floor in Exp12/Exp11. Do not go past kappa~14: beyond that the second-from-bottom voice is effectively mute and Exp10's tied bottom-pair cell and Exp6's dense cells start to break.

2) Raise the floor-driven gain a touch: beta [7,9] -> ~[8.5,10.5] with c_n [1.3,1.9] -> ~[1.1,1.6]. Because eta is now very thin, upper-only displays still put almost no subjective weight on the table (Exp8, Exp13, Exp14, Exp15, Exp16 stay near chance), but floor-containing displays get the near-ceiling / near-floor rates Exp11 (~.70) and Exp12 (~.15-.20) demand, and it should also restore ~0.05-0.08 of magnitude on Exp5.

3) Leave gamma near its current band ([0.30,0.50]) u2014 do not raise it to chase the last 0.06 of Exp16. Exp16's target (+0.05) is within its between-subject noise (var .016), and more dilution acts on D=4 displays, which would immediately push Exp12 back toward chance, i.e. cost more than it buys.

4) Do NOT touch rho_tie or epsilon. rho_tie is currently doing exactly the right job on Exp1 (0.471 vs 0.474); lowering it would improve nothing and raising it would regress Exp1. Raising epsilon would shrink Exp7/Exp2 but would simultaneously flatten Exp5, Exp6, Exp11 and Exp12, all of which need MORE extremity.

5) Accept Exp2 (~.70) and Exp10 (+0.02) as the residual worst cases; they are the two cells where any purely context-free inverse-seniority account must pay a price, and distorting the weight function to chase them would destroy the Exp17/Exp18 nulls, which remain this candidate's principal asset and must be preserved verbatim (they will be, since all the proposed changes are context-free).

**Outcome of this advice:** iter 2 candidate loss=0.0650 -> iter 3 candidate loss=0.0577 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate (loss 0.0577, ACCEPTED) is the best entry in this loop and stays fully faithful to the prescribed FM-ISN family: context-free inverse-seniority weights (no chorus term, no validity-rank tie-break), a count comparator that only gates confidence, Luce-style normalisation with mild D^gamma dilution, mild capacity, lapse. The family-diagnostic nulls that motivated the arbiter's recommendation are intact and are this candidate's principal asset: Exp17 0.000 vs -0.017, Exp18 -0.013 vs +0.015, Exp15 +0.006 vs +0.007, Exp14 0.519 vs 0.489, Exp8 0.499 vs 0.485. My iteration-2 prescription (steepen kappa, thin eta, raise beta / lower c_n) was rewarded again and moved the floor-driven block substantially in the right direction: Exp12 0.291 -> 0.2225 (obs 0.145), Exp11 0.607 -> 0.658 (obs 0.698), Exp5 -0.661 -> -0.728 (obs -0.763), Exp9 0.398 vs 0.405, Exp4 0.758 vs 0.755.

What did NOT move is the second half of the same prescription. I predicted that steepening kappa would pull the mid-band-driven cells DOWN; instead Exp7 went 0.766 -> 0.778 (obs 0.661) and Exp2 went 0.756 -> 0.769 (obs 0.654), and Exp3 went 0.792 -> 0.827 (obs 0.794). The reason is diagnosable: the kappa step and the beta step were taken simultaneously, and the gain increase more than cancelled the weight reduction for the second-from-bottom voice. These two cells are now by far the dominant residuals (+0.12 each, roughly twice any other), and they are of one kind: displays whose 'weak' side is carried by the SECOND-from-bottom voice (Exp7 f3 = .60 against a .57 floor; Exp2 f4 = .62 against a .57 floor) are still read far too decisively. At kappa = 12 with a panel-relative s of only 0.08, that voice still retains ~35-40% of the floor's loudness, which is what drives Exp7/Exp2/Exp3/Exp13 above the human rate. Minor, low-priority residuals: Exp10 drifted further the wrong way (+0.043 vs -0.023) and Exp16 is still ~0.05 short, but both have large between-subject variance (0.013, 0.017) and their fixes (rho_tie up, gamma up) each cost more on Exp1 and Exp12 respectively than they buy.
Rationale: Keep the FM-ISN mechanism exactly as prescribed \u2014 no family change, no chorus amplification, no validity-rank tie-break, direction still never supplied by the count comparator, all weights context-free so the Exp17/Exp18 nulls survive verbatim. Take one final, CLEANLY SEPARATED step in the direction the gate has now rewarded twice, but this time decouple the two knobs that cancelled each other last round:\n\n1) Steepen the descent once more, and this time ALONE: kappa [10.5,14] -> ~[14,18], eta [0.008,0.030] -> ~[0.005,0.022]. This is the one change that isolates the true panel floor from the second-from-bottom voice (in Exp7 w(f3) falls ~0.37 -> ~0.27 relative to the floor; in Exp2 w(f4) falls similarly), which is exactly where the two big residuals live. Expected: Exp7 ~0.78 -> ~0.73, Exp2 ~0.77 -> ~0.72, Exp13 ~0.52 -> ~0.51, Exp6 -1.00 -> ~-0.96 (obs -0.94, currently slightly over-extreme), while floor-driven cells actually SHARPEN because the floor's relative loudness rises: Exp12 ~0.22 -> ~0.19, Exp11 ~0.66 -> ~0.68.\n\n2) Do NOT raise beta again \u2014 shave its top slightly instead: beta [8.5,10.5] -> ~[8.0,9.5], c_n unchanged at [1.1,1.6]. Last round's beta increase is what masked the kappa effect and pushed Exp3 (0.792 -> 0.827), Exp7 and Exp2 up. The kappa step in (1) compensates the floor-driven cells, so trimming gain should cost Exp11/Exp12/Exp5 almost nothing while pulling Exp3 back to ~0.80 and helping Exp7/Exp2/Exp14.\n\n3) Leave gamma, rho_tie, epsilon, chi untouched. rho_tie is doing the right job on Exp1 (0.457 vs 0.474); raising it to chase Exp10's tied bottom-pair cell would regress Exp1, which has half the between-subject variance. Raising gamma to chase Exp16's remaining 0.05 acts hardest on D=4 displays and would push Exp12 back toward chance. Raising epsilon would shrink Exp7/Exp2 but flatten Exp5, Exp6, Exp11, Exp12, which all still need more extremity.\n\n4) Treat this as the terminal step of the kappa programme. If at kappa~16-18 Exp7 and Exp2 still sit above ~0.72, that residual is intrinsic to a purely context-free inverse-seniority account (humans are near-ceiling anti-validity only when the true floor voice speaks, and merely moderate when the second-from-bottom carries), and the proposer should stop pushing kappa rather than distort the weight function or add any context-sensitive term \u2014 doing so would destroy the Exp17/Exp18/Exp15 nulls, which are the reason this family was prescribed.

**Outcome of this advice:** iter 3 candidate loss=0.0577 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC4).**\n\nWhen people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.\n\n**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), now NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j \u2212 v_min)/(v_max \u2212 v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 \u2212 s_j)^kappa with kappa \u2248 14\u201318 and a very thin flat floor eta \u2248 0.005\u20130.022. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness \u2014 their observed rates differ only through the ordinary arithmetic of the weighted balance.\n\n**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = \u03a3_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, and further diluted by the number of mutually discriminating attributes: z = beta\u00b7E/((c_n + \u03a3_j w_j)\u00b7D^gamma) with gamma \u2248 0.30\u20130.50. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.\n\n**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a reduced gain rho_tie \u2248 0.55\u20130.75, so tie displays hover close to chance \u2014 slightly *anti*-validity, never pro-validity.\n\n**(4) MILD CAPACITY.** Holding a whole validity ordering in mind gets harder past about seven voices; the integration slope is divided by 1 + chi\u00b7(n \u2212 7) with chi small. Eight-expert panels are slightly flatter \u2014 star-backed dense majorities stay at chance \u2014 but plain tally adherence survives essentially intact.\n\n**(5) LAPSE.** An attention lapse epsilon mixes the result with a coin flip.\n\n**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo \u2265 multi-cue upper; a null on the two contrasts that falsify chorus theories \u2014 an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------------- parse the stimulus ----------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        for ka, kb in (('option_a_ratings', 'option_b_ratings'),\n                       ('a_ratings', 'b_ratings'),\n                       ('A', 'B'), ('a', 'b')):\n            if ka in state and kb in state:\n                try:\n                    a = np.asarray(list(state[ka]), dtype=float).ravel()\n                    b = np.asarray(list(state[kb]), dtype=float).ravel()\n                except Exception:\n                    a = None\n                    b = None\n                break\n    if a is None or b is None:\n        try:\n            arr = np.asarray(state, dtype=float).reshape(2, -1)\n            a = np.asarray(arr[0], dtype=float).ravel()\n            b = np.asarray(arr[1], dtype=float).ravel()\n        except Exception:\n            return np.array([0.5, 0.5], dtype=float)\n\n    if a is None or b is None or a.size == 0 or a.size != b.size:\n        return np.array([0.5, 0.5], dtype=float)\n\n    n = int(a.size)\n\n    # ---------------- parameters ----------------\n    def _g(name, default, lo, hi):\n        try:\n            x = float(parameters.get(name, default))\n        except Exception:\n            x = float(default)\n        if not np.isfinite(x):\n            x = float(default)\n        return float(min(max(x, lo), hi))\n\n    eta = _g('eta', 0.013, 0.0, 0.50)        # thin flat listening floor\n    kappa = _g('kappa', 16.0, 1.0, 30.0)     # steepness of inverse-seniority descent\n    beta = _g('beta', 8.75, 0.0, 40.0)       # integration slope\n    c_n = _g('c_n', 1.35, 0.05, 20.0)        # constant of indecision\n    gamma = _g('gamma', 0.40, 0.0, 2.0)      # mild count dilution\n    rho_tie = _g('rho_tie', 0.65, 0.0, 1.0)  # confidence gain when counts are tied\n    eps = _g('epsilon', 0.11, 0.0, 0.60)     # attention lapse\n    chi = _g('chi', 0.14, 0.0, 2.0)          # mild capacity flattening past 7 voices\n\n    # ---------------- stated validities ----------------\n    v = None\n    try:\n        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n        if vr is not None:\n            v = np.asarray(vr, dtype=float).ravel()\n    except Exception:\n        v = None\n    if v is None or v.size != n:\n        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)\n\n    v_min = float(np.min(v))\n    v_max = float(np.max(v))\n    span = v_max - v_min\n    if (not np.isfinite(span)) or span <= 1e-9:\n        s = np.zeros(n, dtype=float)\n    else:\n        s = np.clip((v - v_min) / span, 0.0, 1.0)\n\n    # ---------- Commitment 1: context-free inverse-seniority weights ----------\n    base = np.clip(1.0 - s, 0.0, 1.0)\n    try:\n        w = eta + np.power(base, kappa)\n    except Exception:\n        w = eta + base\n    w = np.where(np.isfinite(w), w, eta)\n    w = np.clip(w, 0.0, None)\n\n    # ---------------- feature-wise comparison ----------------\n    d = np.sign(a - b)\n    iA = np.where(d > 0)[0]\n    iB = np.where(d < 0)[0]\n    mA = int(iA.size)\n    mB = int(iB.size)\n    D = mA + mB\n    if D == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    sumA = float(np.sum(w[iA])) if mA > 0 else 0.0\n    sumB = float(np.sum(w[iB])) if mB > 0 else 0.0\n    E = sumA - sumB\n    W = sumA + sumB\n    if not np.isfinite(E):\n        E = 0.0\n    if (not np.isfinite(W)) or W < 0.0:\n        W = 0.0\n\n    # ---------- Commitment 3: count comparator gates CONFIDENCE only ----------\n    gain = rho_tie if mA == mB else 1.0\n\n    # ---------- Commitment 4: mild capacity flattening ----------\n    cap = 1.0 + chi * max(0.0, float(n) - 7.0)\n    if (not np.isfinite(cap)) or cap <= 0.0:\n        cap = 1.0\n\n    # ---------- Commitment 2: Luce-like saturating integration + mild dilution ----------\n    dil = float(D) ** gamma\n    if (not np.isfinite(dil)) or dil <= 0.0:\n        dil = 1.0\n\n    den = (c_n + W) * dil * cap\n    if (not np.isfinite(den)) or den <= 0.0:\n        den = 1.0\n\n    z = beta * E * gain / den\n    if not np.isfinite(z):\n        z = 0.0\n    z = float(min(max(z, -30.0), 30.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-z))\n\n    # ---------- Commitment 5: lapse ----------\n    p_a = (1.0 - eps) * p_a + eps * 0.5\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, 1.0)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if (not np.isfinite(s)) or s <= 0:\n        probs = np.ones_like(probs) / float(len(probs))\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "eta": "[0.005, 0.022]",
    "kappa": "[14.0, 18.0]",
    "beta": "[8.0, 9.5]",
    "c_n": "[1.1, 1.6]",
    "gamma": "[0.30, 0.50]",
    "rho_tie": "[0.55, 0.75]",
    "epsilon": "[0.08, 0.15]",
    "chi": "[0.05, 0.25]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff edit on the accepted base (loss 0.0577). No mechanism, no equation, no parameter name changed; only three sampling ranges (plus the matching internal defaults and one clamp upper bound) were re-tuned exactly as the critic prescribed in iteration 3, and this time the two knobs that cancelled each other last round are DECOUPLED.\n\n(1) kappa [10.5,14] -> [14,18], eta [0.008,0.030] -> [0.005,0.022]. This is the one change that isolates the true panel floor from the SECOND-from-bottom voice, which is precisely where the two dominant residuals live. In Exp7 the second-from-bottom f3 (v=.60 against a .57 floor, s\u22480.08) currently keeps ~35-40% of the floor's loudness; at kappa\u224816 that falls to ~25%, so Exp7 (.778 -> ~.72) and Exp2 (.769 -> ~.72) come down, Exp13 (.525 -> ~.51) comes down, and Exp6 (-1.00 -> ~-0.96, obs -0.94) becomes slightly less over-extreme. At the same time the floor's RELATIVE loudness rises, so the floor-driven block should sharpen rather than regress: Exp12 (.223 -> ~.19, obs .145), Exp11 (.658 -> ~.68, obs .698), Exp5 (~-0.73 -> ~-0.75, obs -0.763). The internal kappa clamp ceiling was raised from 24 to 30 purely so the new sampling range is never silently truncated; the range itself stops at 18 as instructed, so the second-from-bottom voice keeps a small but non-zero share and Exp10's tied bottom-pair cell and Exp6's dense cells do not break.\n\n(2) beta [8.5,10.5] -> [8.0,9.5], c_n unchanged. Last round's simultaneous gain increase is what masked the kappa effect and pushed Exp3 (.792 -> .827), Exp7 and Exp2 upward. Trimming the top of the gain range while the kappa step compensates the floor-driven cells should pull Exp3 back to ~.80 (obs .794) and help Exp7/Exp2/Exp14 at almost no cost to Exp11/Exp12/Exp5.\n\n(3) gamma, rho_tie, epsilon and chi are left untouched, as instructed: rho_tie is doing the right job on Exp1 (.457 vs .474) and raising it to chase Exp10's tied bottom-pair cell would regress Exp1 (half the between-subject variance); raising gamma to chase Exp16's remaining ~.05 acts hardest on D=4 displays and would push Exp12 back toward chance; raising epsilon would flatten Exp5, Exp6, Exp11 and Exp12, all of which still need more extremity.\n\nEvery change is context-free, so the family-diagnostic nulls that are this candidate's principal asset are structurally preserved verbatim: no chorus term, no validity-rank tie-break, direction never supplied by the count comparator, so an opposed floor voice and a floor voice inside a unanimous bloc still receive identical loudness (Exp17 \u2248 0.00 vs -0.017; Exp18 \u2248 -0.01 vs +0.015; Exp15 \u2248 +0.01 vs +0.007). Per the critic's point (4) this is the terminal step of the kappa programme: if Exp7/Exp2 remain above ~.72 the residual is intrinsic to a purely context-free inverse-seniority account and should not be chased with a context-sensitive term."
}
```

## Usage

```json
{
  "input_tokens": 62870,
  "output_tokens": 5438
}
```
