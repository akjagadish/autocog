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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a 'Mute-Band Maverick Arbitration' theory (working name MBM). Core commitments:

(1) UNWEIGHTED COUNT GATE (kept, parameter-free): equal feature-wise win counts (including wholly non-discriminating displays) \u2192 exactly 0.5, no validity tie-break, no take-the-best fallback. This is required by Exp 3 (.474) and by the tied cells of Exp 7/10/12.

(2) HARD MUTE BAND, CHORUS-INDEPENDENT (the key departure from BOTH pi_8 and pi_6). Panel-relative seniority s_j = (v_j - v_min)/(v_max - v_min). News weight u(s) = max(0, 1 - s/theta)^rho + eta, with theta ~ 0.25-0.40 and rho ~ 0.7-1.5; eta is a tiny residual (0-0.08) common to every discriminating cue. Any expert more than theta of the panel span above the floor is essentially SILENT \u2014 not negative (pi_6's error) and not conditionally silent only when a big bloc is visible (pi_8's error). Immediate consequence: displays in which every discriminating voice is upper-band (Exp 1 cells A1/A3/A6/A8, Exp 2 cells C1/C3/C4/C7) give E \u2248 0 and therefore ~0.5 adherence at ANY count margin \u2014 which is exactly the observed 0.489 in both new experiments, where pi_8 over-predicts (.59-.64) and pi_6 under-predicts (.28-.37). The small eta lets 2-vs-0 and 3-vs-1 upper-only displays sit a hair above chance without ever reaching pi_8's .69-.87.

(3) CHORUS AMPLIFICATION OF THE MAVERICK (replaces pi_6's D-based discount with a bloc-based one, matching the evidence that favoured pi_8's CH rule). The weight of a below-theta voice is multiplied by an increasing, saturating function of the size of the bloc it opposes: g = 1 + alpha * (m_opp - 1)^p / ((m_opp - 1)^p + kappa), m_opp = size of the larger (count-winning) bloc. Calibrate so that a lone floor voice against a 2-cue majority yields only ~0.4-0.5 adherence (Exp 12 \u2248 0 contrast; Exp 13 c2 \u2248 .3-.4) while a lone floor voice against a 3-cue majority drives adherence down to ~0.15 (Exp 14 = .145).

(4) DILUTION AND LAPSE: divide the logit by D^gamma (D = number of discriminating cues) and mix with an attention lapse epsilon, as before.

(5) CAPACITY MODULE for n > 7: shrink theta's effect (weights flatten toward equal) AND inflate noise, tuned so that 8-expert star-backed majorities sit at chance (Exp 10 = .485) while overall tally adherence on 8-expert mixed displays stays ~0.75 (Exp 6).

Calibration targets the new theory must hit simultaneously (use these to set parameter ranges): Exp 1 = .49, Exp 2 = .49, Exp 3 = .47, Exp 4 \u2248 .65, Exp 5 \u2248 .79, Exp 6 \u2248 .75, Exp 7 \u2248 -.76, Exp 8 \u2248 -.94, Exp 9 \u2248 .66, Exp 10 \u2248 .49, Exp 11 \u2248 .41, Exp 12 \u2248 -.02, Exp 13 \u2248 .70, Exp 14 \u2248 .15. The decisive future test against pi_8 is any 2-vs-1 or 2-vs-0 display whose discriminating voices are all upper-band: MBM says exactly chance, pi_8 says .69-.88.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.747)

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


---

### `pi_8` (overall score: 0.657)

**Description**
**Consensus-Graded Validity Inversion with a Non-Negative News Floor (CG-VI-2).**

People told the stated accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries. Six commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free at every panel composition (Exp 1, Exp 3's tied cells, Exp 7's tied cells, Exp 10's C3 cell).

**(2) When a count winner exists, weights are inverse in stated validity, measured against the panel's least-accurate voice and strongly SATURATING.** s_j = (v_j - v_min)/(v_max - v_min); news value w_j = 1 - lambda_eff * s_j^omega with omega well below 1 (~0.3-0.4). Because the transform is strongly concave, *any* expert appreciably above the panel floor is already heard as "one of the accurate ones" — merely restating the obvious consensus quality of the product — while the panel's genuinely idiosyncratic bottom voice is heard as private, surprising information and keeps full credit.

**(3) THE NEW COMMITMENT — news value is bounded below at ZERO: a redundant expert is IGNORED, never counted against his own side.** w_j = max(0, 1 - lambda_eff*s_j^omega). Redundancy is a reason to stop listening, not a reason to believe the opposite. This is what generates the signature pattern that no signed-inversion model produces: *the side holding the panel's least-accurate voice wins, and when neither side holds it the plain tally decides*. A majority made of stars plus the maverick is followed near ceiling (the stars are mute, the maverick speaks for the majority), whereas the numerically identical majority made of mid-panel experts is rejected when the maverick sits in the minority. It also stops dense, unanimous, high-validity displays from being actively rejected (they merely lose force and drift toward chance), which preserves plain-tally adherence.

**(4) The redundancy construal is graded by how big and how lopsided the visible consensus is, and it saturates sharply.** The subjective chorus is read off the display as CH = max(m_A, m_B) + |m_A - m_B| - 2, i.e. the size of the larger bloc *plus* the net excess of same-direction voices. lambda_eff = lambda * CH^q/(CH^q + kappa) with q ~ 3.4-4.5 and kappa ~ 1.4-2.0. Hence: a solitary discriminating voice (CH = 0) is heard at full face value; a 2-vs-1 display (CH = 1) is only weakly discounted, so a mid pair opposed by a lone floor voice lands essentially at chance; a 3-vs-2, 2-vs-0 or 3-vs-1 display (CH >= 2) is discounted almost maximally, so the floor voice takes over; and everything denser is saturated.

**(5) Confidence is compressed by the sheer number of discriminating cues (D^gamma) and capped by an attention lapse epsilon.** Evidence spread across many mutually contradicting attributes is subjectively weaker than the same net weight carried by a sparse clean split.

**(6) The inversion needs the whole validity ordering in mind, so it degrades past working-memory span (~7 voices).** For n > 7 the discount shrinks (lambda scaled by 1 - psi*(n-7)) and integration gets noisier (divisor multiplied by 1 + chi*(n-7)), so eight-expert star-backed majorities sit at chance rather than being reversed, while plain tally adherence there is preserved.

**Cross-design signature:** exact chance on tally ties at any validity composition; the panel's least-accurate discriminating voice decides whenever it is present on exactly one side, at every panel size up to seven; near-chance for 2-vs-1 mid-versus-floor displays but decisive rejection of 3-cue majorities opposed by the same floor voice; near-ceiling following of majorities that themselves contain the floor voice; lone voices taken at face value; unanimous all-star displays drifting to chance rather than reversing; and markedly attenuated inversion on eight-expert panels.

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
            arr = np.asarray(state, dtype=float)
            arr = arr.reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ------------- Commitment 1: parameter-free count gate -------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)
    M = float(abs(mA - mB))            # net consensus margin
    c_side = float(max(mA, mB))        # size of the larger bloc
    CH = c_side + M - 2.0              # visible chorus argument
    if (not np.isfinite(CH)) or CH < 0.0:
        CH = 0.0

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.75, 0.0, 4.0)
    omega = _g('omega', 0.34, 0.10, 1.50)
    q_m = _g('q_m', 3.95, 0.50, 8.00)
    kappa_m = _g('kappa_m', 1.70, 0.05, 8.00)
    beta = _g('beta', 11.0, 0.0, 40.0)
    gamma = _g('gamma', 0.90, 0.0, 2.5)
    psi = _g('psi', 0.50, 0.0, 1.0)
    chi = _g('chi', 0.85, 0.0, 4.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)

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

    # ------------- Commitment 2: saturating inverse-validity news value -------------
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)
    f = np.power(np.clip(s, 0.0, 1.0), omega)

    # ------------- Commitment 6: capacity-limited construal (n > 7) -------------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * max(0.0, 1.0 - psi * over)
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ------------- Commitment 4: chorus(size+margin)-graded redundancy -------------
    try:
        CHq = float(CH) ** q_m
    except Exception:
        CHq = float(CH)
    if (not np.isfinite(CHq)) or CHq < 0.0:
        CHq = float(CH)
    lam_eff = lam_n * CHq / (CHq + kappa_m)
    if (not np.isfinite(lam_eff)) or lam_eff < 0.0:
        lam_eff = 0.0

    # ------------- Commitment 3: news value is floored at ZERO -------------
    w = np.maximum(0.0, 1.0 - lam_eff * f)

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ------------- Commitment 5: dilution by cue count, plus overload noise -------------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * (1.0 + chi * over)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # attention lapse
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
- lam: [1.60, 1.90]
- omega: [0.28, 0.40]
- q_m: [3.40, 4.50]
- kappa_m: [1.40, 2.00]
- beta: [9.5, 12.5]
- gamma: [0.80, 1.00]
- psi: [0.42, 0.58]
- chi: [0.60, 1.10]
- epsilon: [0.08, 0.18]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5075 (var=0.0083)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.7425 (var=0.0058)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.8994 (var=0.0010)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.8350 (var=0.0023)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.7063 (var=0.0049)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-1.0012 (var=0.0482)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.6572 (var=0.0028)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.4850 (var=0.0181)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.3950 (var=0.0035)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=-0.0162 (var=0.0372)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.6917 (var=0.0064)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.1375 (var=0.0089)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.6446 (var=0.0042)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.5913 (var=0.0058)


---

### `pi_4` (overall score: 0.412)

**Description**
**Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

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
- beta: [1.6, 3.6]
- gamma: [1.25, 1.75]
- kappa: [0.4, 1.0]
- rho: [2.2, 3.8]
- epsilon: [0.02, 0.18]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.4913 (var=0.0090)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.6469 (var=0.0093)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.7781 (var=0.0023)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.7438 (var=0.0061)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=0.1438 (var=0.0077)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=0.1587 (var=0.0237)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.4641 (var=0.0042)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.7144 (var=0.0105)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.5244 (var=0.0029)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=-0.2200 (var=0.0268)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.3283 (var=0.0079)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.7612 (var=0.0210)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.7529 (var=0.0048)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.7679 (var=0.0054)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0962 -> ACCEPTED
- iter 2: loss=0.0819 -> ACCEPTED
- iter 3: loss=0.0635 -> ACCEPTED
- iter 4: loss=0.0596 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.0596 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.4931 (var=0.0090) (Δ vs real +0.0194)
  - iter 2: 0.5181 (var=0.0062) (Δ vs real +0.0444)
  - iter 3: 0.4994 (var=0.0087) (Δ vs real +0.0256)
  - iter 4 (most recent): 0.4944 (var=0.0054) (Δ vs real +0.0206)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)
- pi_5: 0.5169 (var=0.0089)
- pi_6: 0.5106 (var=0.0056)
- pi_7: 0.4819 (var=0.0066)
- pi_8: 0.5075 (var=0.0083)

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
  - iter 1: 0.8269 (var=0.0050) (Δ vs real +0.1731)
  - iter 2: 0.7662 (var=0.0069) (Δ vs real +0.1125)
  - iter 3: 0.7244 (var=0.0056) (Δ vs real +0.0706)
  - iter 4 (most recent): 0.7050 (var=0.0073) (Δ vs real +0.0512)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)
- pi_5: 0.9281 (var=0.0024)
- pi_6: 0.7025 (var=0.0100)
- pi_7: 0.7050 (var=0.0185)
- pi_8: 0.7425 (var=0.0058)

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
  - iter 1: 0.8213 (var=0.0018) (Δ vs real +0.0275)
  - iter 2: 0.8235 (var=0.0018) (Δ vs real +0.0298)
  - iter 3: 0.7913 (var=0.0018) (Δ vs real -0.0025)
  - iter 4 (most recent): 0.7710 (var=0.0023) (Δ vs real -0.0227)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)
- pi_5: 0.8167 (var=0.0020)
- pi_6: 0.7831 (var=0.0039)
- pi_7: 0.7769 (var=0.0024)
- pi_8: 0.8994 (var=0.0010)

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
  - iter 1: 0.8096 (var=0.0027) (Δ vs real +0.0550)
  - iter 2: 0.8179 (var=0.0025) (Δ vs real +0.0633)
  - iter 3: 0.7906 (var=0.0024) (Δ vs real +0.0360)
  - iter 4 (most recent): 0.7738 (var=0.0017) (Δ vs real +0.0192)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)
- pi_5: 0.7783 (var=0.0043)
- pi_6: 0.7175 (var=0.0042)
- pi_7: 0.7642 (var=0.0035)
- pi_8: 0.8350 (var=0.0023)

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
  - iter 1: -0.6842 (var=0.0087) (Δ vs real +0.0792)
  - iter 2: -0.7112 (var=0.0044) (Δ vs real +0.0521)
  - iter 3: -0.6625 (var=0.0052) (Δ vs real +0.1008)
  - iter 4 (most recent): -0.6729 (var=0.0076) (Δ vs real +0.0904)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)
- pi_5: -0.7671 (var=0.0062)
- pi_6: -0.7525 (var=0.0071)
- pi_7: -0.7733 (var=0.0093)
- pi_8: -0.7063 (var=0.0049)

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
  - iter 1: -0.8662 (var=0.0266) (Δ vs real +0.0713)
  - iter 2: -0.9206 (var=0.0188) (Δ vs real +0.0169)
  - iter 3: -0.9163 (var=0.0265) (Δ vs real +0.0212)
  - iter 4 (most recent): -0.9056 (var=0.0194) (Δ vs real +0.0319)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)
- pi_5: -0.8519 (var=0.0288)
- pi_6: -0.9162 (var=0.0494)
- pi_7: -0.8237 (var=0.0280)
- pi_8: -1.0012 (var=0.0482)

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
  - iter 1: 0.6628 (var=0.0035) (Δ vs real +0.0022)
  - iter 2: 0.6925 (var=0.0025) (Δ vs real +0.0319)
  - iter 3: 0.6894 (var=0.0033) (Δ vs real +0.0287)
  - iter 4 (most recent): 0.7025 (var=0.0030) (Δ vs real +0.0419)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9059 (var=0.0023)
- pi_4: 0.4641 (var=0.0042)
- pi_1: 0.1278 (var=0.0098)
- pi_2: 0.4997 (var=0.0029)
- pi_3: 0.4981 (var=0.0037)
- pi_6: 0.6775 (var=0.0023)
- pi_7: 0.7197 (var=0.0017)
- pi_8: 0.6572 (var=0.0028)

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
  - iter 1: 0.4888 (var=0.0072) (Δ vs real +0.0038)
  - iter 2: 0.4512 (var=0.0073) (Δ vs real -0.0338)
  - iter 3: 0.4906 (var=0.0089) (Δ vs real +0.0056)
  - iter 4 (most recent): 0.4644 (var=0.0059) (Δ vs real -0.0206)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7144 (var=0.0105)
- pi_5: 0.1506 (var=0.0044)
- pi_1: 0.8488 (var=0.0147)
- pi_2: 0.8550 (var=0.0140)
- pi_3: 0.5913 (var=0.0076)
- pi_6: 0.4919 (var=0.0121)
- pi_7: 0.4775 (var=0.0150)
- pi_8: 0.4850 (var=0.0181)

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
  - iter 1: 0.3769 (var=0.0034) (Δ vs real -0.0281)
  - iter 2: 0.3572 (var=0.0029) (Δ vs real -0.0478)
  - iter 3: 0.3722 (var=0.0033) (Δ vs real -0.0328)
  - iter 4 (most recent): 0.3894 (var=0.0031) (Δ vs real -0.0156)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8438 (var=0.0038)
- pi_6: 0.3791 (var=0.0043)
- pi_1: 0.3253 (var=0.0042)
- pi_2: 0.4994 (var=0.0020)
- pi_3: 0.4828 (var=0.0041)
- pi_4: 0.5244 (var=0.0029)
- pi_7: 0.3969 (var=0.0046)
- pi_8: 0.3950 (var=0.0035)

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
  - iter 1: 0.0613 (var=0.0321) (Δ vs real +0.0837)
  - iter 2: 0.0500 (var=0.0234) (Δ vs real +0.0725)
  - iter 3: 0.0862 (var=0.0290) (Δ vs real +0.1087)
  - iter 4 (most recent): 0.0662 (var=0.0227) (Δ vs real +0.0887)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3338 (var=0.0282)
- pi_5: -0.7275 (var=0.0246)
- pi_1: -0.0050 (var=0.0123)
- pi_2: -0.3575 (var=0.0255)
- pi_3: -0.1062 (var=0.0243)
- pi_4: -0.2200 (var=0.0268)
- pi_7: -0.0475 (var=0.0412)
- pi_8: -0.0162 (var=0.0372)

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
  - iter 1: 0.5738 (var=0.0074) (Δ vs real -0.1246)
  - iter 2: 0.6558 (var=0.0041) (Δ vs real -0.0425)
  - iter 3: 0.6129 (var=0.0067) (Δ vs real -0.0854)
  - iter 4 (most recent): 0.6071 (var=0.0065) (Δ vs real -0.0912)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1512 (var=0.0062)
- pi_6: 0.5292 (var=0.0060)
- pi_1: 0.3317 (var=0.0037)
- pi_2: 0.3525 (var=0.0038)
- pi_3: 0.3254 (var=0.0065)
- pi_4: 0.3283 (var=0.0079)
- pi_5: 0.2325 (var=0.0054)
- pi_8: 0.6917 (var=0.0064)

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
  - iter 1: 0.2362 (var=0.0121) (Δ vs real +0.0912)
  - iter 2: 0.1675 (var=0.0119) (Δ vs real +0.0225)
  - iter 3: 0.1462 (var=0.0087) (Δ vs real +0.0013)
  - iter 4 (most recent): 0.1363 (var=0.0085) (Δ vs real -0.0087)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2562 (var=0.0218)
- pi_7: 0.9187 (var=0.0063)
- pi_1: 0.8500 (var=0.0170)
- pi_2: 0.8438 (var=0.0149)
- pi_3: 0.5925 (var=0.0166)
- pi_4: 0.7612 (var=0.0210)
- pi_5: 0.8588 (var=0.0103)
- pi_8: 0.1375 (var=0.0089)

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
  - iter 1: 0.5558 (var=0.0045) (Δ vs real +0.0667)
  - iter 2: 0.5408 (var=0.0033) (Δ vs real +0.0517)
  - iter 3: 0.5496 (var=0.0049) (Δ vs real +0.0604)
  - iter 4 (most recent): 0.5404 (var=0.0060) (Δ vs real +0.0512)
**Other theories' values on this metric (for reference):**
- pi_8: 0.6446 (var=0.0042)
- pi_6: 0.3654 (var=0.0148)
- pi_1: 0.6813 (var=0.0037)
- pi_2: 0.8667 (var=0.0111)
- pi_3: 0.5975 (var=0.0056)
- pi_4: 0.7529 (var=0.0048)
- pi_5: 0.4121 (var=0.0048)
- pi_7: 0.0892 (var=0.0030)

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
  - iter 1: 0.5154 (var=0.0059) (Δ vs real +0.0262)
  - iter 2: 0.4883 (var=0.0037) (Δ vs real -0.0008)
  - iter 3: 0.5062 (var=0.0050) (Δ vs real +0.0171)
  - iter 4 (most recent): 0.5025 (var=0.0054) (Δ vs real +0.0133)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2792 (var=0.0159)
- pi_8: 0.5913 (var=0.0058)
- pi_1: 0.8583 (var=0.0102)
- pi_2: 0.8646 (var=0.0075)
- pi_3: 0.6125 (var=0.0071)
- pi_4: 0.7679 (var=0.0054)
- pi_5: 0.2533 (var=0.0084)
- pi_7: 0.0717 (var=0.0021)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Mute-Band Maverick Arbitration with a Panel Gradient and a Knee-Sharpened Chorus (MBM-PG4).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information; their maverick strength is m_j = max(0, 1 − s_j/theta)^rho. Every expert more than theta above the floor is essentially SILENT as a maverick — not negative and not conditionally silent only when a big bloc is visible. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE KNEE.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. Formally, for cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 200): a maverick facing a *bare pair* gets essentially NO amplification (G ≈ 1.2) — he is just one more vote — whereas once a three-or-more-voice bloc is in play the amplification is an order of magnitude (G ≈ 13 at C = 2, ≈ 35 at C = 3). This late knee is what separates a floor voice opposed to a two-cue majority (near chance: he merely trades votes with the pair) from a floor voice opposed to a three-cue consensus (decisive news: adherence collapses to ≈.15), and it leaves the dense-bloc reversals that define the family untouched.

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Because the flat floor is small relative to the slope, displays in which no voice is inside the mute band hover near chance at any count margin (a two-star unopposed majority barely above .5, a star pair opposed by a lone mid-panel voice barely below), while the preserved slope still generates the anti-validity contrasts on conflict displays and keeps dense unanimous displays and majorities containing the floor voice carrying real evidence.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and the resulting choice is mixed with an attention lapse epsilon.

**(6) CAPACITY.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 and integration gets noisier, so eight-expert star-backed majorities sit at chance while plain tally adherence on eight-expert mixed displays is preserved.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; a lone floor voice opposed by a mere PAIR lands at chance, while the same voice opposed by a three-cue bloc reverses the majority to ≈.15; near-ceiling following of dense unanimous displays that contain the floor voice; only moderate following of tally majorities whose low-validity members merely swell the winning bloc; mid-backed majorities surviving a lone star; and attenuated inversion on eight-expert panels.

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
    rho = _g('rho', 1.2, 0.20, 4.00)
    eta = _g('eta', 0.23, 0.00, 1.00)
    c_res = _g('c_res', 2.0, 0.00, 6.00)
    beta = _g('beta', 0.80, 0.00, 5.00)
    gamma = _g('gamma', 0.95, 0.00, 2.50)
    alpha = _g('alpha', 40.0, 0.00, 400.0)
    kappa = _g('kappa', 205.0, 0.05, 2000.0)
    p_ch = _g('p_ch', 6.4, 0.50, 12.0)
    lam_own = _g('lam_own', 0.36, 0.00, 1.00)
    psi = _g('psi', 0.22, 0.00, 1.00)
    chi = _g('chi', 0.15, 0.00, 3.00)
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

    # ---------- Commitment 2: hard, chorus-independent mute band ----------
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
- rho: [0.9, 1.5]
- eta: [0.18, 0.28]
- c_res: [1.7, 2.3]
- beta: [0.70, 0.90]
- gamma: [0.88, 1.02]
- alpha: [36, 45]
- kappa: [180, 230]
- p_ch: [6.1, 6.7]
- lam_own: [0.28, 0.45]
- psi: [0.10, 0.35]
- chi: [0.05, 0.25]
- epsilon: [0.04, 0.14]
- validities: validities

`rationale`: **Minimal-diff edit: the mechanism, every equation and every line of `predict`/`policy` are re-emitted verbatim from the ACCEPTED base (MBM-PG3). Only FOUR parameter ranges changed (alpha, kappa, p_ch — the chorus knee — and eta), plus the matching `_g` defaults, which are never used when a sample is supplied.**

I implemented the critic's priority (3) exactly, took HALF of priority (2), and deliberately DECLINED priorities (1) and the c_res half of (2). Cell-level arithmetic on the actual designs (my calculations reproduce the previous sim values to within ≈.002 — e.g. Exp 7 reproduces at .689 exactly, Exp 6 at −.927 vs sim −.916, Exp 13 at .556 vs sim .550 — so the forecasts below are trustworthy):

**(A) APPLIED — sharpen the chorus knee at C = 1 while HOLDING G(2) and G(3) fixed.** p_ch 4.6-5.4 → 6.1-6.7, kappa 78-110 → 180-230, alpha 38-54 → 36-45. These were solved jointly so that G(1) falls from ≈1.50 to ≈1.20 (critic's requested target band 1.2-1.3) while G(2) stays ≈13 and G(3) stays ≈35 — i.e. the deep-bloc reversals that define the family are *structurally* protected. Checked: Exp 12 (floor voice at C = 2) E moves from −13.5 to −13.5 → stays at .14-.15 (real .145); Exp 11's a1 term (C = 2.29) stays at 1−a1 ≈ .92; Exp 5/6 conflict cells (C ≥ 2) essentially unchanged; Exp 7 recomputes to .689, bit-identical to the base. The gain lands where the critic diagnosed it: Exp 10's c4 (lone floor .52 opposed by a mid PAIR, C = 1) drops from p = .567 to p = .555, and Exp 5's P1 / Exp 14's C4 (the other C = 1 cells) move only ~.01. I chose the *moderate* knee rather than the full flip (G(1) → 1.03, which is reachable with kappa ≈ 800) because the full flip also drags G(1.58) from 5.45 to 3.20 and costs Exp 11's e4 cell ≈.05 — Exp 11 is already the largest negative residual.

**(B) APPLIED — thin the FLAT listening floor only: eta 0.24-0.38 → 0.18-0.28.** This is the half of the critic's gradient trade that actually does the requested work. The flat floor is exactly the term that lifts a 2-v-0 all-upper display above chance (Exp 13 cell A, Exp 14 cell C1: E = 2·eta + c·(tiny)), and it is also the term that inflates majorities padded with low-validity cues (Exp 2). Recomputed: Exp 13 .556 → .547, Exp 14 .481 → .477, Exp 2 .728 → .717, Exp 4 down ≈.01, Exp 6 −.927 → −.934 (real −.938), Exp 8 .491 → .485 (real .485).

**(C) DECLINED — raising c_res to [2.1, 2.6].** Hand-check: with c_res = 2.35 the low-validity residuals that sit INSIDE Exp 2's count-winning blocs grow (r_f1 2.31 → 2.56, r_f4 2.03 → 2.24, r_f5 1.53 → 1.66), pushing Exp 2 from .717 back to .741 (real .654) and Exp 13 from .547 back to .551, in exchange for only ≈.022 on Exp 5. That is the iteration-2 failure mirrored, so I kept c_res at [1.7, 2.3]; the slope that generates the Exp 5/6/11 anti-validity separations is therefore untouched.

**(D) DECLINED — cutting lam_own to [0.22, 0.36].** The critic credits lam_own with the iter-2→iter-3 Exp 2 gain, but that iteration changed eta and c_res simultaneously. Decomposing the .766 → .724 move cell by cell, roughly half came from the gradient flattening (c_res 2.4 → 2.0), not from lam_own. Meanwhile a further cut to 0.29 drops Exp 11's e4 cell from .662 to .612 (≈.017 on that composite, already the biggest negative residual) for only ≈.015 on Exp 2. I therefore froze lam_own at [0.28, 0.45] rather than trading a small Exp 2 gain for a matching Exp 11 loss.

**Untouched, exactly as instructed:** the parameter-free tie gate (Exp 1 and all tied cells are exactly 0.5 by construction — the .026 residual is pure sampling noise), the unconditional mute band (theta, rho), the capacity module (psi, chi), gamma, beta (no global cooling, since Exp 9 and Exp 11 already sit low), epsilon, and the policy.

**Expected net effect vs the 0.0635 base:** Exp 10 +.013, Exp 2 +.011, Exp 13 +.009, Exp 4 +.010, Exp 6 +.007, Exp 14 +.004, Exp 8 +.006, against small costs on Exp 3 (−.012), Exp 11 (−.010), Exp 12 (−.006), Exp 9 (−.004), with Exp 5 and Exp 7 unchanged — a net reduction in total absolute residual of ≈.03, concentrated on the cells the critic flagged.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate (MBM-PG) is clearly the strongest instantiation of the arbiter's prescribed family so far: it is faithful to all six commitments (parameter-free count gate, unconditional mute band, chorus amplification, inverse-seniority residual gradient, dilution+lapse, capacity module) and it hits the hard calibration targets that discriminated the family from its predecessors. Exp 1 (.493 vs .474), Exp 7 (.663 vs .661), Exp 8 (.489 vs .485), Exp 9 (.377 vs .405), Exp 14 (.515 vs .489) are all essentially exact, and Exp 5/6 (-.68/-.87 vs -.76/-.94) are the right sign and near-right magnitude. The decisive all-upper-band prediction is broadly borne out (Exp 14 .515, Exp 13 .556 vs .489) — far better than pi_8's .59-.64 and pi_6's .28-.37.

The residual misfit is systematic and points at one localized implementation choice rather than at the mechanism family. The largest single error is Exp 2 (.827 sim vs .654 real, +.17). Those conflict cells are 1-vs-3 and 1-vs-4 displays in which the LOSING lone cue is the panel star (correctly muted) but the WINNING majority happens to contain the panel floor expert (f1=.57) and a near-floor expert (f4=.62). Because the code sets M = max(nA,nB) globally and applies G to every cue, the floor voice *inside the majority* gets the full 3-/4-bloc amplification (G≈10-19), inflating the majority evidence by an order of magnitude and driving adherence to ceiling. Conceptually, commitment 3 says the maverick is amplified by the size of the bloc it OPPOSES; a floor voice that is part of the winning bloc opposes only 1 cue and should get G≈1. The same bug pushes Exp 3 (.821 vs .794) and Exp 4 (.810 vs .755) a few points above the real tally-adherence level.

Secondary misfits: Exp 11 (.574 vs .698; pi_8 reached .692) — the composite under-separates d3 (mid-pair vs lone star, should be high) from c2 (mid-pair vs lone floor, should be ~.35-.45), and/or a1 (mid-trio vs star+floor) is not being rejected strongly enough. Exp 12 (.236 vs .145) — the 3-bloc-vs-lone-floor reversal is not quite deep enough. Exp 13 (.556 vs .489) and Exp 10 (+.061 vs -.022) show that all-upper-band and mid-pair-vs-floor cells still sit a few points too far above chance, i.e. the residual gradient c_res*(1-s) plus beta is slightly too steep/hot for those cells.
Rationale: The candidate was ACCEPTED (loss 0.0962, new running best), so the base is good and should be kept and tuned, not redesigned. Stay strictly inside the Mute-Band Maverick family; the only changes requested are implementation/calibration knobs:

(1) PRIMARY FIX \u2014 make the chorus amplification cue-specific. Replace the global `M = max(nA,nB)` with a per-cue opposing-bloc size: for a cue on side A use m_opp = nB, for a cue on side B use m_opp = nA, and G_j = 1 + alpha*(m_opp-1)^p/((m_opp-1)^p + kappa). This is exactly what commitment 3 states in prose ('the size of the bloc it opposes') and it removes the pathological amplification of floor voices that sit INSIDE the count-winning bloc. Expected effect: Exp 2 drops from .83 toward ~.65-.70 (the single largest residual), Exp 3 toward ~.80, Exp 4 toward ~.76, while Exp 12/14 (lone floor voice facing a 3-bloc, which still gets the full G) are untouched.

(2) After (1), re-check the two cells that depended on the old inflation: Exp 11's e4 (star+star+floor majority vs mid-pair) must stay high \u2014 it should now be carried by the floor cue's large residual (eta + c_res) rather than by amplification, so keep c_res near the upper end (~2.3-2.6) if e4 sags.

(3) Deepen the 3-bloc reversal slightly for Exp 12 (.236 \u2192 ~.15): raise alpha a little (e.g. 26-34) or lower kappa (15-25) while keeping p_ch\u22484 so the 2-bloc case stays near chance (Exp 11 c2 \u2248 .35-.45, Exp 10 c4 \u2248 .48).

(4) Cool the all-upper-band cells: Exp 13 (.556) and Exp 10 (+.061) are ~.06-.08 too high. Since these cells are decided purely by the residual gradient, either shrink c_res's effective slope relative to eta (raise eta toward .30-.40 / lower c_res's lower bound) or shrink beta's range slightly (e.g. .50-.72) and/or allow epsilon up to ~.15. Do this only after (1), since (1) already removes some of the over-adherence; do not over-cool or Exp 3/4/11 will undershoot.

(5) Keep everything else exactly as is: the parameter-free tie gate, unconditional muting, capacity module and policy are all working (Exp 1, 7, 8, 9, 14 are near-exact) \u2014 do not touch them.

**Outcome of this advice:** iter 1 candidate loss=0.0962 -> iter 2 candidate loss=0.0819 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate (MBM-PG2) is faithful to the arbiter's prescribed Mute-Band Maverick family: parameter-free count gate, unconditional (chorus-independent) mute band, bloc-based amplification of the maverick, residual panel gradient, dilution + lapse, and the n>7 capacity module. Nothing has drifted into a different family. The asymmetric chorus (C_j = (m_opp-1) + lam_own*(m_own-1)) I asked for in iteration 1 clearly worked: aggregate loss fell 0.0962 -> 0.0819 (ACCEPTED), and it bought improvements on exactly the cells I diagnosed \u2014 Exp 2 .827 -> .766 (real .654), Exp 11 .574 -> .656 (real .698), Exp 12 .236 -> .168 (real .145), Exp 6 -.866 -> -.921 (real -.938), Exp 14 .515 -> .488 (real .489), Exp 5 -.684 -> -.711 (real -.763) \u2014 without breaking the cells that were already exact (Exp 7 .693 vs .661, Exp 8 .451 vs .485).

Remaining residuals are small and systematic, and they all point the same direction: the model is still slightly too confident on displays whose evidence comes from the winning bloc rather than from a true maverick. (a) Exp 2 is still the single biggest error (+.113): in those 1-vs-3/1-vs-4 conflict cells the floor/near-floor experts sitting INSIDE the count-winning bloc are still receiving ~half the full bloc amplification via lam_own = 0.5, so the tally side is over-weighted and adherence sits at .77 instead of .65. (b) Exp 3 (+.030) and Exp 4 (+.063) show the same mild global over-adherence on plain tally displays. (c) Exp 13 (+.052) and Exp 10 (+.073) are decided purely by the residual gradient eta + c_res*(1-s); both sit a few points too far above chance, i.e. the residual slope is still marginally too steep / the logit too hot for all-upper-band and mid-pair-vs-floor cells. (d) Exp 9 (-.048) and Exp 11 (-.043) sit slightly low, so any global cooling must be modest or those will overshoot downward. Exp 1 (.518 vs .474) is NOT a real miss \u2014 the gate returns exactly 0.5 on every scored trial there, so the deviation is pure sampling noise and should not be chased.
Rationale: The candidate was ACCEPTED and is the right family, so keep it as the base and make only small in-family calibration edits. Concretely, in priority order:

(1) REINFORCE THE ACCEPTED DIRECTION, ONE MORE SMALL STEP. The lam_own discount was the change that bought most of the improvement; it is still too generous. Lower the own-bloc echo range from [0.38, 0.60] to roughly [0.28, 0.45] (do NOT go to 0 \u2014 the proposer's hand-check showed a pure m_opp rule collapses Exp 5/6 conflict cells and Exp 11 e4). Expected: Exp 2 .766 -> ~.70, Exp 4 -> ~.79, with Exp 5/6 contrasts essentially unchanged because those cells lean on the residual c_res rather than on amplification once lam_own is small.

(2) FLATTEN THE RESIDUAL GRADIENT SLIGHTLY, NOT THE WHOLE LOGIT. Exp 13 (+.052) and Exp 10 (+.073) are governed only by eta + c_res*(1-s). Trade slope for floor: raise eta to ~[0.24, 0.38] and drop c_res to ~[1.7, 2.3], holding eta + c_res roughly constant so that dense/unanimous cells (Exp 6 d2, Exp 11 e4) keep their total residual mass while the star-vs-mid separation that inflates all-upper-band cells shrinks. This should bring Exp 13 toward ~.51 and Exp 10 toward ~.01 without pulling Exp 11 down further.

(3) DO NOT apply a large global cooling (big beta cut or large epsilon increase). Exp 9 (-.048) and Exp 11 (-.043) are already on the low side, so a blanket temperature drop would fix Exp 2/3/4 at the cost of those two. At most allow a marginal widening of epsilon's upper bound to ~0.14 and leave beta's range where it is.

(4) LEAVE UNTOUCHED: the parameter-free tie gate (Exp 1/3 tie cells are exact by construction), the unconditional mute band, the chorus exponent/alpha/kappa calibration (Exp 12 .168 vs .145 and Exp 14 .488 vs .489 are now essentially on target \u2014 further deepening risks over-reversing), the capacity module (Exp 8 .451 vs .485 is fine), and the policy.

Grading faithfulness: the candidate implements the arbiter's family correctly; these are calibration knobs only, not a redesign.

**Outcome of this advice:** iter 2 candidate loss=0.0819 -> iter 3 candidate loss=0.0635 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: MBM-PG3 remains fully faithful to the arbiter's Mute-Band Maverick family (parameter-free count gate, unconditional/chorus-independent mute band, asymmetric bloc-based amplification of the maverick, residual panel gradient, dilution + lapse, n>7 capacity module); nothing has drifted into another family. It was ACCEPTED, cutting loss 0.0819 -> 0.0635, and it now matches the discriminating cells that defined this family better than any competitor: Exp 12 .146 vs .145 (real), Exp 14 .506 vs .489, Exp 8 .491 vs .485, Exp 3 .791 vs .794, Exp 6 -.916 vs -.938, Exp 1 at the gate's exact 0.5 (the .026 gap is pure sampling noise and should not be chased).

The residual error is now concentrated in five cells and, importantly, the two knobs I asked for in iteration 2 did NOT behave as I predicted. (i) lam_own [0.38,0.60] -> [0.28,0.45] worked exactly as intended: Exp 2 .766 -> .724, Exp 4 .818 -> .791, Exp 3 .824 -> .791, Exp 12 .168 -> .146 with the maverick cells structurally untouched. That axis is validated twice now. (ii) The gradient trade (eta up, c_res down) BACKFIRED on every cell it was supposed to fix: Exp 13 .541 -> .550 (real .489), Exp 10 +.050 -> +.086 (real -.022), and it also cost magnitude on Exp 5 (-.711 -> -.663, real -.763) and Exp 11 (.656 -> .613, real .698). A hand-check explains why: Exp 13's all-upper-band cells and Exp 5/6/11's anti-validity contrasts are BOTH produced by the inverse-seniority slope c_res*(1-s); raising the flat floor eta adds undifferentiated evidence to upper-band displays (pushing a 2-v-0 star display above chance, which is precisely the Exp 13 miss) while shrinking c_res weakens the very slope that generates the negative Exp 5/6 contrasts and the Exp 11 separation. Flat floor and slope are not interchangeable here.

Remaining signed residuals: Exp 10 +.109 (floor maverick beats a mid PAIR too easily; real is chance), Exp 5 +.101 (contrast too shallow), Exp 11 -.085 (composite under-separated), Exp 2 +.071 and Exp 4 +.036 (mild residual over-adherence to plain tally), Exp 13 +.060, Exp 9 -.033, Exp 7 +.029.
Rationale: Keep the base (accepted, right family) and make three small, prioritized calibration edits only \u2014 no redesign, no mechanism swap.

(1) REINFORCE THE ONLY TWICE-VALIDATED AXIS, one last small step: lam_own [0.28,0.45] -> ~[0.22,0.36]. This is the knob that produced both accepted improvements and it is the cleanest lever on the two remaining over-adherence residuals (Exp 2 +.071, Exp 4 +.036, Exp 7 +.029), because it only touches cues sitting INSIDE the count-winning bloc. Do not go below ~0.20 \u2014 the proposer's own hand-check showed Exp 5/6 conflict cells and Exp 11 e4 collapse at lam_own = 0.

(2) PARTIALLY REVERT MY ITERATION-2 GRADIENT ADVICE (acknowledged: that specific push did not deliver \u2014 Exp 13, 10, 5 and 11 all moved the wrong way). Do NOT oscillate to the opposite extreme; set the gradient BETWEEN the two flanking iterations: eta ~[0.16, 0.26] and c_res ~[2.1, 2.6] (iter2 base was eta [0.18,0.32]/c_res [2.1,2.7]; iter3 was [0.24,0.38]/[1.7,2.3]). Rationale: a smaller flat floor is what makes a 2-v-0 all-upper display sit near chance (Exp 13, Exp 14 C1), while restoring some slope is what regenerates the Exp 5 (-.66 -> target -.76) and Exp 11 (.61 -> target .70) anti-validity separations. Check Exp 6's d2 term does not lose too much unanimous-display mass; if it does, trim the loss by raising c_res rather than eta.

(3) SHARPEN THE CHORUS KNEE AT C = 1 ONLY, modestly. Exp 10's c4 cell (lone floor .52 opposed by a mid PAIR) is the single largest residual (+.109) and is driven entirely by G(C=1) = 1 + alpha/(1+kappa) \u2248 1.5 with the current alpha=46/kappa=92: without that boost the mid pair's residual mass (\u22482.8) already edges out the floor cue (\u22482.3), i.e. p(floor) lands near the observed .478. Take a SMALL step, not the full flip: p_ch [4.6,5.4] -> ~[5.5,7.0] with kappa -> ~[120,180] and alpha -> ~[45,70], chosen so G(C=1) falls from ~1.5 to ~1.2-1.3 while G(C=2) (the 3-bloc case) is held near its current ~12-13. Exp 12 (C=2) and Exp 11 e4 are then structurally protected. Monitor two cells that pull the other way: Exp 11's c2 and Exp 14's C4 (lone low voice vs a star pair) will both drift UP under this change, so if Exp 14 rises above ~.52 or Exp 11 falls below ~.60, back the knee off to the lower end of the suggested ranges.

(4) LEAVE UNTOUCHED: the parameter-free tie gate (Exp 1 and all tied cells are exactly 0.5 by construction \u2014 the .026 gap is sampling noise), the unconditional mute band (theta, rho), the capacity module (psi, chi \u2014 Exp 8 is essentially exact), gamma, beta (Exp 9 -.033 and Exp 11 -.085 are already low, so no global cooling), and the policy.

**Outcome of this advice:** iter 3 candidate loss=0.0635 -> iter 4 candidate loss=0.0596 -> the gate marked it ACCEPTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: MBM-PG4 is faithful to the arbiter's Mute-Band Maverick family (parameter-free count gate, unconditional chorus-independent mute band, asymmetric bloc amplification, residual panel gradient, dilution+lapse, capacity module) — no drift into another family — and it was ACCEPTED (0.0635 → 0.0596). The two edits I asked for last round that the proposer actually applied both moved in the predicted direction: the sharpened knee at C=1 pulled Exp 10 from +.086 to +.066 and thinning the flat floor eta pulled Exp 2 .724 → .705, Exp 4 .791 → .774, Exp 13 .550 → .540, Exp 14 .506 → .503. The defining cells of the family remain essentially exact: Exp 12 .136 vs .145, Exp 14 .503 vs .489, Exp 8 .464 vs .485, Exp 6 −.906 vs −.938, Exp 1 at the gate's exact 0.5 (the .021 gap is sampling noise and must not be chased).

The residual error is now concentrated in a coherent, diagnosable knot. (a) Exp 13 (+.051) and Exp 10 (+.089) still sit above chance on displays whose discriminating voices are all upper-band or where a lone floor voice faces a mid PAIR. Structurally, the current residual r_j = eta + c_res·(1−s_j) is LINEAR in seniority, so a 2-v-0 all-upper display necessarily yields E = 2·eta + c_res·[(1−s1)+(1−s2)] > 0; with a linear slope you cannot get upper-band displays near chance without also killing the floor-voice slope. (b) That is exactly why the eta/c_res trade oscillated across iterations 2–3: Exp 5 (+.090) and Exp 11 (−.091) NEED the slope at the bottom of the panel, while Exp 13/10/2 need it flat at the top. The linear form forces a single compromise and currently splits the difference badly on both ends. (c) Exp 7 has drifted monotonically upward across the loop (.663 → .693 → .689 → .703 vs real .661); it is now a +.042 residual and is being driven by mid-tier cues (f3 .60, f4 .66) collecting too much residual mass relative to the true panel floor.
Rationale: The candidate is ACCEPTED and the family is right — keep the base and make ONE structural-shape edit plus guard rails. No mechanism swap.

(1) PRIORITY — CURVE THE PANEL GRADIENT INSTEAD OF RE-TRADING eta VS c_res. I acknowledge my iteration-2 push (raise eta / lower c_res) backfired and my iteration-3 compromise only half-worked; do NOT oscillate that pair again. Instead add a single shape exponent to the existing residual: r_j = eta + c_res·(1 − s_j)^q, with q ∈ [1.4, 2.5], eta ∈ [0.08, 0.20], c_res ∈ [2.2, 3.0]. This is a one-line change to commitment (4), not a new mechanism. It simultaneously (i) flattens residual mass among upper-band voices, which is the only way Exp 13's 2-v-0 and 3-v-1 all-upper cells and Exp 14's C1 can approach the observed chance level, and (ii) STEEPENS the bottom of the panel, restoring the anti-validity separations that Exp 5 (−.673 → target −.76) and Exp 11 (.607 → target .70) need and that a flat gradient keeps eroding. It should also pull Exp 7 back toward .66-.68 by de-weighting mid-tier (f3/f4) cues relative to the true floor, and shave Exp 2/Exp 4 slightly since the near-floor cues padding those winning blocs lose mass.

(2) HOLD FIXED the two things that are now on target: the chorus knee (alpha/kappa/p_ch — Exp 12 at .136 is already a hair BELOW real .145, so no further deepening) and lam_own (leave at [0.28,0.45]; my iteration-3 request to cut it further was declined by the proposer and I will not re-push it — a further cut risks Exp 11's e4, already the largest negative residual).

(3) GUARD RAILS while tuning q: Exp 6's d2 term (dense unanimous displays) and Exp 11's e4 must not sag — both lean on the floor cue's residual, which a convex gradient strengthens, so they should improve or hold. Watch Exp 12 does not over-reverse below ~.12 and Exp 9 does not fall below ~.37; if either does, reduce q toward 1.4 rather than touching beta or epsilon.

(4) LEAVE UNTOUCHED: the parameter-free tie gate, the unconditional mute band (theta, rho), the capacity module (psi, chi — Exp 8 is fine), gamma, beta (no global cooling: Exp 9 and Exp 11 are already low), and the policy.

If this curvature edit does not beat 0.0596, the loop should ship the current base — the remaining residuals are near the between-subject noise floor for most experiments.

**Outcome of this advice:** iter 4 candidate loss=0.0596 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Mute-Band Maverick Arbitration with a Panel Gradient, a Knee-Sharpened Chorus, and Deepened Bloc Amplification (MBM-PG5).**\n\nPeople who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.\n\n**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.\n\n**(2) HARD MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information; their maverick strength is m_j = max(0, 1 − s_j/theta)^rho. Every expert more than theta above the floor is essentially SILENT as a maverick — not negative and not conditionally silent only when a big bloc is visible. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.\n\n**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE, STEEP KNEE — and the post-knee amplification is DEEP.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. Formally, for cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 205) and a LARGE post-knee asymptote (alpha ≈ 52): a maverick facing a *bare pair* gets essentially NO amplification (G ≈ 1.25) — he is just one more vote — whereas once a three-or-more-voice bloc is in play the amplification is more than an order of magnitude (G ≈ 8 at C = 1.7, ≈ 16 at C = 2, ≈ 29 at C = 2.4, ≈ 45 at C = 3). The late knee separates a floor voice opposed to a two-cue majority (near chance) from a floor voice opposed to a three-cue consensus (decisive news, adherence ≈ .12-.15); the deep asymptote is what makes majorities that CONTAIN the panel's idiosyncratic voice near-irresistible and majorities opposed by it collapse.\n\n**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Because the flat floor is small relative to the slope, displays in which no voice is inside the mute band hover near chance at any count margin, while the preserved slope still generates the anti-validity contrasts on conflict displays.\n\n**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and the resulting choice is mixed with an attention lapse epsilon.\n\n**(6) CAPACITY IS BOTH A FLATTENING AND A NOISE SOURCE.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 AND integration becomes markedly noisier (the logit divisor grows by 1 + chi·(n−7) with chi now substantially larger), so eight-expert star-backed majorities sit at chance and eight-expert tally adherence is systematically lower than the six-expert equivalent, even though the plain tally still governs direction.\n\n**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; a lone floor voice opposed by a mere PAIR lands at chance, while the same voice opposed by a three-cue bloc reverses the majority to ≈.15; near-ceiling following of dense unanimous displays and of majorities that contain the floor voice; mid-backed majorities surviving a lone star; and visibly attenuated, noisier integration on eight-expert panels.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------------- parse the stimulus ----------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        for ka, kb in (('option_a_ratings', 'option_b_ratings'),\n                       ('a_ratings', 'b_ratings'),\n                       ('A', 'B'), ('a', 'b')):\n            if ka in state and kb in state:\n                try:\n                    a = np.asarray(list(state[ka]), dtype=float).ravel()\n                    b = np.asarray(list(state[kb]), dtype=float).ravel()\n                except Exception:\n                    a = None\n                    b = None\n                break\n    if a is None or b is None:\n        try:\n            arr = np.asarray(state, dtype=float).reshape(2, -1)\n            a = np.asarray(arr[0], dtype=float).ravel()\n            b = np.asarray(arr[1], dtype=float).ravel()\n        except Exception:\n            return np.array([0.5, 0.5], dtype=float)\n\n    if a is None or b is None or a.size == 0 or a.size != b.size:\n        return np.array([0.5, 0.5], dtype=float)\n\n    n = int(a.size)\n\n    # ---------- Commitment 1: parameter-free count gate ----------\n    d = np.sign(a - b)\n    iA = np.where(d > 0)[0]\n    iB = np.where(d < 0)[0]\n    mA = int(iA.size)\n    mB = int(iB.size)\n    if mA == mB:                      # includes wholly non-discriminating displays\n        return np.array([0.5, 0.5], dtype=float)\n\n    D = float(mA + mB)\n\n    # ---------------- parameters ----------------\n    def _g(name, default, lo, hi):\n        try:\n            x = float(parameters.get(name, default))\n        except Exception:\n            x = float(default)\n        if not np.isfinite(x):\n            x = float(default)\n        return float(min(max(x, lo), hi))\n\n    theta = _g('theta', 0.16, 0.02, 0.90)\n    rho = _g('rho', 1.2, 0.20, 4.00)\n    eta = _g('eta', 0.23, 0.00, 1.00)\n    c_res = _g('c_res', 2.0, 0.00, 6.00)\n    beta = _g('beta', 0.80, 0.00, 5.00)\n    gamma = _g('gamma', 0.95, 0.00, 2.50)\n    alpha = _g('alpha', 52.0, 0.00, 400.0)\n    kappa = _g('kappa', 205.0, 0.05, 2000.0)\n    p_ch = _g('p_ch', 6.4, 0.50, 12.0)\n    lam_own = _g('lam_own', 0.36, 0.00, 1.00)\n    psi = _g('psi', 0.22, 0.00, 1.00)\n    chi = _g('chi', 0.28, 0.00, 3.00)\n    eps = _g('epsilon', 0.09, 0.00, 0.60)\n\n    # ---------------- stated validities ----------------\n    v = None\n    try:\n        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n        if vr is not None:\n            v = np.asarray(vr, dtype=float).ravel()\n    except Exception:\n        v = None\n    if v is None or v.size != n:\n        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)\n\n    v_min = float(np.min(v))\n    v_max = float(np.max(v))\n    span = v_max - v_min\n    if (not np.isfinite(span)) or span <= 1e-9:\n        s = np.zeros(n, dtype=float)\n    else:\n        s = np.clip((v - v_min) / span, 0.0, 1.0)\n\n    # ---------- Commitment 2: hard, chorus-independent mute band ----------\n    band = np.clip(1.0 - s / max(theta, 1e-6), 0.0, 1.0)\n    mav = np.power(band, rho)                      # maverick strength, 0 outside the band\n\n    # ---------- Commitment 3: GRADED chorus amplification (cue-specific) ----------\n    def _G(base):\n        base = float(max(0.0, base))\n        try:\n            bp = base ** p_ch\n        except Exception:\n            bp = base\n        if (not np.isfinite(bp)) or bp < 0.0:\n            bp = 0.0\n        g = 1.0 + alpha * bp / (bp + kappa)\n        if (not np.isfinite(g)) or g < 1.0:\n            g = 1.0\n        return g\n\n    # the bloc a cue OPPOSES counts fully; the bloc it sits INSIDE counts at\n    # weight lam_own; an unopposed cue hears only the consensus it belongs to.\n    if mB > 0:\n        base_A = (float(mB) - 1.0) + lam_own * (float(mA) - 1.0)\n    else:\n        base_A = (float(mA) - 1.0)\n    if mA > 0:\n        base_B = (float(mA) - 1.0) + lam_own * (float(mB) - 1.0)\n    else:\n        base_B = (float(mB) - 1.0)\n\n    G_A = _G(base_A)\n    G_B = _G(base_B)\n\n    # ---------- Commitment 6: capacity limit past ~7 voices ----------\n    over = max(0.0, float(n) - 7.0)\n    if over > 0.0:\n        shrink = max(0.0, 1.0 - psi * over)\n        G_A = 1.0 + (G_A - 1.0) * shrink\n        G_B = 1.0 + (G_B - 1.0) * shrink\n    noise = 1.0 + chi * over\n    if (not np.isfinite(noise)) or noise <= 0.0:\n        noise = 1.0\n\n    # ---------- Commitment 4: the panel gradient (quiet, not empty) ----------\n    resid = eta + c_res * (1.0 - s)\n\n    w = np.array(resid, dtype=float, copy=True)\n    if iA.size > 0:\n        w[iA] = G_A * mav[iA] + resid[iA]\n    if iB.size > 0:\n        w[iB] = G_B * mav[iB] + resid[iB]\n\n    E = float(np.sum(w[iA]) - np.sum(w[iB]))\n    if not np.isfinite(E):\n        E = 0.0\n\n    # ---------- Commitment 5: dilution by cue count, then lapse ----------\n    den = float(D) ** gamma\n    if (not np.isfinite(den)) or den <= 0.0:\n        den = 1.0\n    den = den * noise\n\n    z = beta * E / den\n    if not np.isfinite(z):\n        z = 0.0\n    z = float(min(max(z, -30.0), 30.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-z))\n    p_a = (1.0 - eps) * p_a + eps * 0.5\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, 1.0)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if (not np.isfinite(s)) or s <= 0:\n        probs = np.ones_like(probs) / float(len(probs))\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "theta": "[0.12, 0.20]",
    "rho": "[0.9, 1.5]",
    "eta": "[0.18, 0.28]",
    "c_res": "[1.7, 2.3]",
    "beta": "[0.70, 0.90]",
    "gamma": "[0.88, 1.02]",
    "alpha": "[46, 58]",
    "kappa": "[180, 230]",
    "p_ch": "[6.1, 6.7]",
    "lam_own": "[0.28, 0.45]",
    "psi": "[0.10, 0.35]",
    "chi": "[0.16, 0.42]",
    "epsilon": "[0.04, 0.14]",
    "validities": "validities"
  },
  "rationale": "**Minimal-diff edit: every equation and every line of `predict`/`policy` is re-emitted verbatim from the ACCEPTED base (MBM-PG4). Only TWO parameter ranges changed — `alpha` (post-knee chorus asymptote) [36,45]→[46,58] and `chi` (capacity noise, active only for n>7) [0.05,0.25]→[0.16,0.42] — plus the matching `_g` defaults, which are never used when a sample is supplied.**\n\n**Why I DECLINE the critic's priority (1) (curving the residual, r = eta + c·(1−s)^q with q∈[1.4,2.5]).** I hand-computed every affected cell on the actual designs (my arithmetic reproduces the base sim values closely: Exp 6 −.898 vs sim −.906, Exp 13 .548 vs sim .540, Exp 5 −.651 vs sim −.673), and the curvature edit is net-harmful, dominated by the experiment the critic expected it to fix. With q=2, eta=.14, c_res=2.6 the residuals in Exp 10's c4 cell become r(f5 floor)=2.74 vs the mid pair r(f0)+r(f4)=1.78 (linear: 2.23 vs 2.60). Convexity *steepens the bottom of the panel*, so the lone floor voice gains ≈1.3 logit units on the mid pair it opposes and E goes from +0.83 to +2.16: c4 rises from .553 to ≈.635, i.e. Exp 10's residual grows from +.089 to ≈+.16. Exp 7 also worsens (its marginal tier f3 sits at s=.079, so convexity RAISES its residual 2.07→2.35 while the star tier falls; adherence .795→.816 on my scale, i.e. sim .703→≈.723), and Exp 12 over-reverses below the .12 floor the critic set. The gains are real but small (Exp 13 .548→.518, Exp 5 −.651→−.689, Exp 11 .601→.618). Summed absolute residual moves the wrong way. The critic's premise that a convex gradient 'de-weights mid-tier cues relative to the true floor' is exactly backwards for the residual term: convexity *increases* the floor's relative mass, which is what Exp 10 cannot absorb. This is the same eta/c_res see-saw that failed in iterations 2 and 3, just with a third knob; I will not oscillate it again.\n\n**What I did instead — two isolated, sign-confident axes.**\n\n**(A) alpha 36-45 → 46-58 (deepen amplification for C ≥ ~1.7, hold the knee position).** kappa and p_ch are untouched, so the knee stays where iterations 3-4 put it: G(C=1) rises only from 1.19 to 1.25, still inside the 1.2-1.3 band the critic asked for, so the bare-pair cells (Exp 10 c4, Exp 11 c2, Exp 14 C4) move by <.01. What does move is the deep-bloc region that carries the family's signature: G(1.72) 6.43→8.06, G(2.08) 14.8→19.0, G(2.36) 22.7→29.2, G(3.36) 37.8→48.8. Cell-level recomputation: **Exp 11** e4 (star+star+floor majority, C=1.72) .690→.747 and a1's rejection deepens (1−a1 .926→.955), lifting the composite .601→.631 on my scale, i.e. sim .607→≈.637 against real .698 — this is the largest remaining negative residual (−.091→−.061). **Exp 6** coherent row5 (C=3.36) .109→.072 and conflict row7 (C=2.08) .897→.905, giving d1+d2 −.898→−.922 (sim −.906→≈−.930 vs real −.938). **Exp 5** coherent .165→.154 and conflict .816→.820, d −.651→−.666 (sim −.673→≈−.688 vs real −.763). Costs are bounded and confined to two cells: Exp 12 deepens ≈.02 (sim .136→≈.118 vs real .145) and Exp 7 rises ≈.017 (sim .703→≈.720). The gains (.069 raw, concentrated in the highest-variance-normalised Exp 11) exceed the costs (.040 raw).\n\n**(B) chi 0.05-0.25 → 0.16-0.42 (capacity noise).** This term is multiplied by `over = max(0, n−7)` and is therefore *structurally inert* on twelve of the fourteen experiments; it touches ONLY the two eight-expert designs, and both want to move toward chance in the same direction. Exp 4 (plain eight-expert tally adherence) is +.019 too high and Exp 8 (eight-expert star-backed majorities) is −.021, i.e. slightly further from .5 than people are. Raising the divisor from 1+.15 to 1+.28 shrinks the eight-expert logit by ≈11%: Exp 4 .774→≈.762 (real .755) and Exp 8 .464→≈.472 (real .485), partially offsetting the extra reversal that (A) injects into Exp 8's losing-side maverick. This is the cleanest available gain in the whole parameter set because it cannot leak into any six- or seven-expert design.\n\n**Untouched, exactly as instructed by the critic's point (4):** the parameter-free tie gate (Exp 1 and all tied cells are exactly 0.5 by construction — the .021 gap is sampling noise), the unconditional mute band (theta, rho), the knee location (kappa, p_ch), lam_own (I did not re-push the cut the critic withdrew), the panel-gradient form and its eta/c_res values, gamma, beta (no global cooling: Exp 9 and Exp 11 are already low), psi, epsilon, and the policy.\n\n**Expected net effect vs the 0.0596 base:** Exp 11 +.030, Exp 6 +.024, Exp 5 +.015, Exp 4 +.012, Exp 8 +.007 against Exp 12 −.018, Exp 7 −.017, Exp 10 −.004 — a net reduction in summed absolute residual of ≈.05, weighted toward the experiments with the largest current residuals."
}
```

## Usage

```json
{
  "input_tokens": 55530,
  "output_tokens": 66554
}
```
