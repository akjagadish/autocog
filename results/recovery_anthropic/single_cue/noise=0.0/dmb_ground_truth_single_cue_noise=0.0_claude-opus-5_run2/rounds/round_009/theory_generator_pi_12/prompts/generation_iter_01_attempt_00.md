# generation_iter_01_attempt_00

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
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 1 (= `pi_10`).

Replace pi_10 entirely. The replacement should be the strongest genuine rival to FM-ISN-CC5 that the data actually suggest: a NON-COMPENSATORY, SINGLE-CUE 'Take-the-Worst' mixture rather than a weighted-balance integrator.

Sketch — **Floor-Voice Lexicographic Rule (TTW-Mixture)**:
(1) On each trial the subject looks for ONE expert only: the panel's minimum-validity voice (panel-relative, identified from the instructions; Exp 13 shows the floor is defined relative to the displayed panel, not by an absolute closeness-to-.5 criterion).
(2) If that floor expert discriminates, the subject chooses its side with a single fixed probability p_f (~0.82-0.90 before lapse). Crucially this rate is COUNT-INVARIANT and CONTEXT-INVARIANT: it does not change with the number of opposing cues, their seniority, the total number of discriminating attributes D, whether the counts are tied, or whether the floor sits inside a unanimous bloc. This is a flat, parameter-free-in-shape prediction and is the sharp contrast with pi_11, whose Luce denominator (c_n + sum w), D^gamma dilution and rho_tie confidence gate all make these cells differ (pi_11 systematically under-shoots ties at .80/.83 where the data give .84/.87, and predicts solo > multi-cue displays).
(3) If the floor expert does NOT discriminate, the rule fails and the subject essentially guesses: near-exact chance, with at most a whisper of a fallback (a small probability delta of consulting the SECOND-lowest expert, delta <= 0.15, and no independent tally pull). This is the second sharp contrast with pi_11, whose continuous (1-s)^kappa weight hands the second-from-bottom voice ~0.2-0.26 of full loudness and therefore over-predicts Exp 9 (.749 vs .661) and would predict clear anti-validity in Exp 10 where the data are exactly chance (.485) and in Exp 3's ties (.474).
(4) A small lapse/side-noise term epsilon (~0.03-0.10) mixed with a coin flip accounts for residual scatter; no capacity term is needed (Exp 6 with 8 experts behaves like the 6-7 expert panels).

The predict() implementation should therefore be: locate argmin(validities); if it discriminates return [p_f, 1-p_f] oriented to its side (identical value for every D, every margin, every tie status, every unanimity condition); else, with weight delta orient to the second-lowest discriminating expert and otherwise return chance; finally mix with epsilon. Parameter ranges: p_f in [0.80, 0.92], delta in [0.0, 0.15], epsilon in [0.02, 0.12]. This theory is clearly distinguishable from pi_11 by: (a) tie-vs-majority and solo-vs-dense equality (pi_11 predicts gradients), (b) second-from-bottom cells (pi_11 predicts moderate anti-validity, TTW predicts chance), and (c) count-margin invariance of floor-following.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_11` (overall score: 0.925)

**Description**
**Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC5).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 15–17 and a very thin flat floor eta ≈ 0.008–0.018. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.34–0.46. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a moderately reduced gain rho_tie ≈ 0.60–0.78, so tie displays hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) CAPACITY IS AT MOST A WHISPER.** Holding a whole validity ordering in mind gets marginally harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi ≈ 0–0.10, i.e. essentially absent. Eight-expert panels retain their tally adherence, and star-backed dense majorities stay at chance for the independent reason that their weights are all near eta.

**(5) LAPSE.** A small attention lapse epsilon ≈ 0.05–0.10 mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.

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

    eta = _g('eta', 0.013, 0.0, 0.50)        # thin flat listening floor
    kappa = _g('kappa', 16.0, 1.0, 30.0)     # steepness of inverse-seniority descent
    beta = _g('beta', 8.75, 0.0, 40.0)       # integration slope
    c_n = _g('c_n', 1.35, 0.05, 20.0)        # constant of indecision
    gamma = _g('gamma', 0.40, 0.0, 2.0)      # mild count dilution
    rho_tie = _g('rho_tie', 0.69, 0.0, 1.0)  # confidence gain when counts are tied
    eps = _g('epsilon', 0.075, 0.0, 0.60)    # attention lapse
    chi = _g('chi', 0.05, 0.0, 2.0)          # (near-absent) capacity flattening past 7 voices

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

    # ---------- Commitment 4: (near-absent) capacity flattening ----------
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
- eta: [0.008, 0.018]
- kappa: [15.0, 17.0]
- beta: [8.3, 9.2]
- c_n: [1.2, 1.5]
- gamma: [0.34, 0.46]
- rho_tie: [0.60, 0.78]
- epsilon: [0.05, 0.10]
- chi: [0.00, 0.10]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.4900 (var=0.0081)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.7231 (var=0.0047)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.8098 (var=0.0017)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.7412 (var=0.0014)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.7333 (var=0.0048)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-1.0337 (var=0.0305)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.7494 (var=0.0033)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.4931 (var=0.0095)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.4028 (var=0.0021)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=0.0325 (var=0.0149)
- Experiment 11: real=0.6983 (var=0.0122) vs this=0.6767 (var=0.0057)
- Experiment 12: real=0.1450 (var=0.0099) vs this=0.1800 (var=0.0078)
- Experiment 13: real=0.4892 (var=0.0049) vs this=0.5038 (var=0.0056)
- Experiment 14: real=0.4892 (var=0.0049) vs this=0.4942 (var=0.0053)
- Experiment 15: real=0.0067 (var=0.0123) vs this=0.0067 (var=0.0177)
- Experiment 16: real=0.0517 (var=0.0159) vs this=-0.0250 (var=0.0228)
- Experiment 17: real=-0.0167 (var=0.0075) vs this=-0.0161 (var=0.0115)
- Experiment 18: real=0.0150 (var=0.0121) vs this=0.0158 (var=0.0178)
- Experiment 19: real=0.8417 (var=0.0217) vs this=0.8033 (var=0.0147)
- Experiment 20: real=0.8700 (var=0.0105) vs this=0.8325 (var=0.0064)


---

### `pi_9` (overall score: 0.699)

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
- Experiment 19: real=0.8417 (var=0.0217) vs this=0.3642 (var=0.0331)
- Experiment 20: real=0.8700 (var=0.0105) vs this=0.5250 (var=0.0134)


---

### `pi_10` (overall score: 0.647)

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
- Experiment 19: real=0.8417 (var=0.0217) vs this=0.1808 (var=0.0243)
- Experiment 20: real=0.8700 (var=0.0105) vs this=0.4950 (var=0.0087)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0309 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0309 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1 (most recent): 0.4719 (var=0.0076) (Δ vs real -0.0019)
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
- pi_11: 0.4900 (var=0.0081)

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
  - iter 1 (most recent): 0.7081 (var=0.0062) (Δ vs real +0.0544)
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
- pi_11: 0.7231 (var=0.0047)

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
  - iter 1 (most recent): 0.8044 (var=0.0027) (Δ vs real +0.0106)
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
- pi_11: 0.8098 (var=0.0017)

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
  - iter 1 (most recent): 0.7500 (var=0.0022) (Δ vs real -0.0046)
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
- pi_11: 0.7412 (var=0.0014)

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
  - iter 1 (most recent): -0.7267 (var=0.0077) (Δ vs real +0.0367)
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
- pi_11: -0.7333 (var=0.0048)

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
  - iter 1 (most recent): -0.8912 (var=0.0238) (Δ vs real +0.0463)
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
- pi_11: -1.0337 (var=0.0305)

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
  - iter 1 (most recent): 0.7066 (var=0.0033) (Δ vs real +0.0459)
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
- pi_11: 0.7494 (var=0.0033)

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
  - iter 1 (most recent): 0.4800 (var=0.0096) (Δ vs real -0.0050)
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
- pi_11: 0.4931 (var=0.0095)

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
  - iter 1 (most recent): 0.4100 (var=0.0027) (Δ vs real +0.0050)
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
- pi_11: 0.4028 (var=0.0021)

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
  - iter 1 (most recent): 0.0050 (var=0.0108) (Δ vs real +0.0275)
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
- pi_11: 0.0325 (var=0.0149)

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
  - iter 1 (most recent): 0.6988 (var=0.0038) (Δ vs real +0.0004)
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
- pi_11: 0.6767 (var=0.0057)

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
  - iter 1 (most recent): 0.1363 (var=0.0089) (Δ vs real -0.0087)
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
- pi_11: 0.1800 (var=0.0078)

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
  - iter 1 (most recent): 0.4975 (var=0.0044) (Δ vs real +0.0083)
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
- pi_11: 0.5038 (var=0.0056)

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
  - iter 1 (most recent): 0.4688 (var=0.0066) (Δ vs real -0.0204)
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
- pi_11: 0.4942 (var=0.0053)

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
  - iter 1 (most recent): -0.0142 (var=0.0174) (Δ vs real -0.0208)
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
- pi_11: 0.0067 (var=0.0177)

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
  - iter 1 (most recent): -0.0117 (var=0.0154) (Δ vs real -0.0633)
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
- pi_11: -0.0250 (var=0.0228)

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
  - iter 1 (most recent): 0.0178 (var=0.0118) (Δ vs real +0.0344)
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
- pi_11: -0.0161 (var=0.0115)

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
  - iter 1 (most recent): -0.0033 (var=0.0129) (Δ vs real -0.0183)
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
- pi_11: 0.0158 (var=0.0178)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Design-time panel (hard-coded from the experimental design):
    # validities = [0.74, 0.94, 0.58, 0.76, 0.92, 0.54, 0.72]
    floor_i = 5      # v = 0.54  -> panel floor voice
    second_i = 2     # v = 0.58  -> second-from-bottom voice
    star_set = frozenset([1, 4])       # 0.94, 0.92
    mid_set = frozenset([0, 3, 6])     # 0.74, 0.76, 0.72

    tie = [0, 0]
    cells = {'C3': [0, 0], 'C4': [0, 0], 'C5': [0, 0], 'C6': [0, 0]}

    for a_r, b_r, resp in zip(data['option_a_ratings'],
                              data['option_b_ratings'],
                              data['response']):
        try:
            a = [int(x) for x in a_r]
            b = [int(x) for x in b_r]
            r = int(resp)
        except Exception:
            continue
        if len(a) != 7 or len(b) != 7:
            continue
        winA = [i for i in range(7) if a[i] > b[i]]
        winB = [i for i in range(7) if b[i] > a[i]]
        if len(winA) == 0 and len(winB) == 0:
            continue

        # ---- tally-tie cells (C1: 1v1 floor vs star, C2: 2v2 with floor) ----
        if len(winA) == len(winB):
            if floor_i in winA:
                focal_is_A = True
            elif floor_i in winB:
                focal_is_A = False
            else:
                continue
            tie[1] += 1
            if (r == 0) == focal_is_A:
                tie[0] += 1
            continue

        # ---- 1-vs-2 seniority-titration cells ----
        if len(winA) == 1 and len(winB) == 2:
            solo = winA[0]
            opp = frozenset(winB)
            focal_is_A = True
        elif len(winB) == 1 and len(winA) == 2:
            solo = winB[0]
            opp = frozenset(winA)
            focal_is_A = False
        else:
            continue

        if opp == star_set:
            tag = 'star'
        elif opp <= mid_set:
            tag = 'mid'
        else:
            continue

        if solo == floor_i:
            key = 'C4' if tag == 'star' else 'C3'
        elif solo == second_i:
            key = 'C6' if tag == 'star' else 'C5'
        else:
            continue

        cells[key][1] += 1
        if (r == 0) == focal_is_A:
            cells[key][0] += 1

    def rate(pair):
        return (pair[0] / float(pair[1])) if pair[1] > 0 else None

    tr = rate(tie)
    if tr is None:
        tr = 0.5

    diffs = []
    for hi, lo in (('C4', 'C3'), ('C6', 'C5')):
        h = rate(cells[hi])
        l = rate(cells[lo])
        if (h is not None) and (l is not None):
            diffs.append(h - l)
    d = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return float(tr - d)

```

**Observed (real) value:** 0.8417 (var=0.0217)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8842 (var=0.0211) (Δ vs real +0.0425)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1808 (var=0.0243)
- pi_11: 0.8033 (var=0.0147)
- pi_1: 0.1425 (var=0.0248)
- pi_2: 0.5217 (var=0.0209)
- pi_3: 0.5075 (var=0.0253)
- pi_4: 0.5142 (var=0.0270)
- pi_5: 0.5567 (var=0.0153)
- pi_6: 0.3808 (var=0.0239)
- pi_7: 0.2292 (var=0.0292)
- pi_8: 0.2967 (var=0.0221)
- pi_9: 0.3642 (var=0.0331)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.78, 0.95, 0.55, 0.71, 0.88, 0.62, 0.83], dtype=float)

    hits = 0.0
    total = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        n = int(a.size)
        if n == V.size:
            v = V
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        d = np.sign(a - b)
        iA = np.where(d > 0)[0]
        iB = np.where(d < 0)[0]
        mA = int(iA.size)
        mB = int(iB.size)
        D = mA + mB
        # only TALLY-TIED, discriminating displays
        if D == 0 or mA != mB:
            continue

        vmin = float(np.min(v))
        vmax = float(np.max(v))
        span = vmax - vmin
        if not np.isfinite(span) or span <= 1e-9:
            continue
        s = np.clip((v - vmin) / span, 0.0, 1.0)
        w = np.power(np.clip(1.0 - s, 0.0, 1.0), 16.0)

        E = float(np.sum(w[iA])) - float(np.sum(w[iB]))
        if not np.isfinite(E) or abs(E) < 1e-9:
            continue
        # favored side under inverse-seniority (low-validity / "floor voice") reading
        favored = 0 if E > 0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        total += 1.0
        if resp == favored:
            hits += 1.0

    if total == 0.0:
        return 0.5
    return float(hits / total)
```

**Observed (real) value:** 0.8700 (var=0.0105)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8717 (var=0.0046) (Δ vs real +0.0017)
**Other theories' values on this metric (for reference):**
- pi_11: 0.8325 (var=0.0064)
- pi_10: 0.4950 (var=0.0087)
- pi_1: 0.1417 (var=0.0134)
- pi_2: 0.4933 (var=0.0087)
- pi_3: 0.4958 (var=0.0082)
- pi_4: 0.5042 (var=0.0121)
- pi_5: 0.8900 (var=0.0037)
- pi_6: 0.5250 (var=0.0092)
- pi_7: 0.4975 (var=0.0093)
- pi_8: 0.4933 (var=0.0103)
- pi_9: 0.5250 (var=0.0134)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Take-the-Worst: the Floor-Voice Lexicographic Rule (TTW-FLAT).**

When people are told the accuracies of a panel of binary raters and asked which product is better, they do not integrate the panel at all. They run a NON-COMPENSATORY, SINGLE-CUE search, and the cue they search for is the panel's *least* accurate voice. Five commitments.

**(1) ONE VOICE IS CONSULTED, AND IT IS THE PANEL FLOOR.** Search is lexicographic in ASCENDING stated validity. The subject looks for the single expert with the lowest validity *on the displayed panel* (panel-relative, read off the instructions; there is no absolute 'closeness-to-.5' criterion). Told that several experts are nearly infallible and one is barely better than a coin, subjects treat the near-infallible ratings as redundant, already-known public information and the marginal rater as the only voice carrying private, decision-relevant news. That voice, and only that voice, is read.

**(2) FLOOR-FOLLOWING IS A FLAT, COUNT-INVARIANT AND CONTEXT-INVARIANT RATE.** If the floor expert discriminates, the subject picks its side with one fixed probability p_f (~0.86–0.92 before lapse). This rate is a constant of the person, not of the display: it does NOT change with how many cues oppose the floor voice, with how senior those opponents are, with the total number of discriminating attributes D, with whether the feature-wise counts are tied or lopsided, or with whether the floor sits alone or inside a unanimous bloc. This flatness is the theory's sharpest, shape-free prediction: solo-vs-dense, tie-vs-majority, star-opposition-vs-mid-opposition, and opposed-vs-unanimous all come out at exactly the same number.

**(3) IF THE FLOOR IS SILENT, THE RULE FAILS AND THE SUBJECT GUESSES.** When the floor expert gives both products the same rating, the non-compensatory rule has no output. There is NO fallback to a tally, NO compensatory balance, and NO validity-ordered take-the-best. At most there is a whisper: with a small probability delta (<= 0.15) the subject continues the descent one step and orients to the lowest-validity cue that does discriminate; otherwise the response is a coin flip. Hence displays whose discriminating voices are all upper-band sit at essentially exact chance at every count margin, second-from-bottom voices produce only a whisper (never moderate anti-validity), and tally adherence is a *by-product* of the floor happening to side with the majority, never an independent pull.

**(4) NO CAPACITY TERM.** Because only one cue is ever read, panel size is irrelevant: eight-expert panels behave exactly like six- and seven-expert panels.

**(5) LAPSE.** A small attention/side-noise term epsilon (~0.02–0.10) mixes the whole result with a coin flip.

**Cross-design signature:** identical floor-following rate on 1v1 ties, 1v2, 1v3 and unanimous displays; exact chance (± the delta whisper) whenever the floor is silent, including all-upper displays, star-backed dense majorities and second-from-bottom cells; reversal of any majority opposed by the floor voice; near-ceiling adherence to any majority containing the floor voice; and a flat line under seniority titration of the opposing bloc.

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

    p_f = _g('p_f', 0.89, 0.50, 1.0)      # flat floor-following rate
    delta = _g('delta', 0.06, 0.0, 0.30)  # whisper of a one-step descent
    eps = _g('epsilon', 0.06, 0.0, 0.60)  # lapse / side noise

    # ---------------- stated validities (panel, from instructions) ----------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)
    v = np.where(np.isfinite(v), v, 0.75)

    # ---------------- feature-wise discrimination ----------------
    d = np.sign(a - b)                       # +1 -> favours A, -1 -> favours B
    disc = np.nonzero(d)[0]
    if disc.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- Commitment 1: locate the PANEL FLOOR voice ----------
    v_min = float(np.min(v))
    floor_idx = np.nonzero(v <= v_min + 1e-12)[0]

    net_floor = float(np.sum(d[floor_idx])) if floor_idx.size > 0 else 0.0
    floor_speaks = bool(np.any(d[floor_idx] != 0)) if floor_idx.size > 0 else False

    if floor_speaks:
        if net_floor > 0:
            # ---- Commitment 2: flat, count- and context-invariant rate ----
            p_a = p_f
        elif net_floor < 0:
            p_a = 1.0 - p_f
        else:
            # degenerate: tied floor voices contradict each other -> no output
            p_a = 0.5
    else:
        # ---- Commitment 3: the rule failed -> guess, with a whisper ----
        vd = v[disc]
        j = int(disc[int(np.argmin(vd))])
        # if several discriminating cues share the minimal validity, take their net
        lowest = float(np.min(vd))
        tied = disc[np.abs(v[disc] - lowest) <= 1e-12]
        net_low = float(np.sum(d[tied]))
        if net_low > 0:
            dirn = 1.0
        elif net_low < 0:
            dirn = -1.0
        else:
            dirn = float(np.sign(d[j]))
        p_a = 0.5 + dirn * 0.5 * delta

    # ---------- Commitment 5: lapse ----------
    p_a = (1.0 - eps) * p_a + eps * 0.5
    if not np.isfinite(p_a):
        p_a = 0.5
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
- p_f: [0.84, 0.94]
- delta: [0.0, 0.12]
- epsilon: [0.02, 0.10]
- validities: validities

`rationale`: pi_10 (CTN-BG) failed catastrophically exactly where the data are most diagnostic: Exp 19 (.181 vs real .842), Exp 20 (.495 vs real .870), Exp 17/18 (predicted chorus gradients of +.20/+.16 where the data are flat at -.02/+.02). Every one of those failures traces to the same two mechanisms — a chorus-size amplification gate and a parameter-free count gate that forces exact chance on tally ties. The data say the opposite on both counts: tally-tied displays containing the floor voice are followed at .87 (Exp 20), and floor-following is completely insensitive to the size and seniority of the opposing bloc, to unanimity, and to tie status (Exps 17, 18, 19). Rather than patch the integrator, I implement the arbiter's prescribed rival family: a genuinely NON-COMPENSATORY, SINGLE-CUE lexicographic rule that searches the validity ordering from the BOTTOM up.

Why this is a real rival and not a re-parameterisation of pi_11: pi_11 is still a weighted balance, so its Luce denominator, D^gamma dilution and rho_tie gate make solo/dense, tie/majority and opposed/unanimous cells DIFFER — which is precisely where it mis-fits (it under-shoots the tally-tie cells of Exps 19/20 at .80/.83 against .84/.87, over-predicts Exp 2 at .723 vs .654 and Exp 7 at .749 vs .661). TTW-FLAT makes these cells numerically IDENTICAL by construction, a shape-free prediction with no free parameter controlling it.

Hand-checking the design cells cell-by-cell with p_eff = (1-eps)p_f + eps/2 ≈ 0.867 and fallback F = 0.5 + delta/2 ≈ 0.528: Exp 1 ties → 1-F = .472 (real .474); Exp 5 coherent−conflict → (1-p_eff)−p_eff = −.74 (real −.763); Exp 6 → d1+d2 ≈ −.92 (real −.938, and better than pi_11's −1.034); Exp 7 → (p_eff+F)/2 = .70 (real .661, better than pi_11's .749); Exp 8 star-backed dense majorities → 1−F = .47 (real .485); Exp 9 → .41 (real .405); Exp 10 → exactly 0 because the floor carries both C3 and C4 (real −.023); Exp 11 → .71 (real .698); Exp 12 → 1−p_eff = .13 (real .145); Exps 13/14 all-upper cells → .50/.49 (real .489/.489); Exps 15/16/17/18 → ~0 by count- and context-invariance (real .007/.052/−.017/.015); Exp 19 → p_eff − 0 = .87 (real .842, pi_11 .803); Exp 20 → p_eff = .87 (real .870, pi_10 .495). Tally adherence in Exps 3/4 emerges as a by-product (~.81/.72 vs real .794/.755) without any independent tally pull.

The three parameters are all rate parameters with no shape freedom — the theory cannot manufacture gradients across count, margin, seniority or unanimity even if it wanted to — so it is maximally falsifiable and maximally experiment-invariant: the same p_f reproduces floor-following across 5-, 6-, 7- and 8-expert panels with no capacity term.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The TTW-FLAT candidate is a faithful implementation of the arbiter's prescribed non-compensatory single-cue family (panel-floor lexicographic search, flat count/context-invariant following rate, fallback-to-chance-with-whisper, no capacity term, lapse), and it fits the 20-experiment battery remarkably well: 16 of 20 metrics land within ~0.02 of the real value. The diagnostic cells that killed every previous integrator are now nailed: Exp 19 (.884 vs .842, vs pi_10's .181), Exp 20 (.872 vs .870, vs pi_10's .495), Exp 12 (.136 vs .145), Exp 11 (.699 vs .698), Exp 9 (.410 vs .405), Exp 8 (.480 vs .485), Exp 1 (.472 vs .474), Exp 13 (.498 vs .489), Exp 10 (.005 vs -.023), Exp 18 (-.003 vs .015). The flatness commitments (Exps 15-18 near zero; Exp 6 8-expert panel behaving like 6-7) are exactly what the data show, so the count/context-invariance claim survives and is the reason this candidate generalises where the weighted-balance rivals do not.

Four residuals remain, and three of them share a signature. (a) Exp 2: .708 vs .654 (+.054); (b) Exp 7: .707 vs .661 (+.046); (c) Exp 16: -.012 vs +.052 (-.064); (d) Exp 19 slightly hot at .884 vs .842 (+.042). Exps 2 and 7 are precisely the cells whose value is a 50/50 mix of a floor-speaks cell and a floor-SILENT fallback cell. With p_eff ~.867 pinned by Exp 20 (.870) and Exp 12 (1-p_eff = .136 vs .145), the Exp 7 real value of .661 implies a fallback rate F ~= 0.45, i.e. floor-silent displays are answered at or marginally BELOW chance rather than at 0.5 + delta/2 = .528. The current positive delta whisper is therefore pushing the wrong way; Exps 13/14 (.489/.489 real, both nominally at-or-below chance) point the same direction. Exp 16's small positive solo-vs-dense contrast (+.052) is the only genuinely off-family residual, but it is within ~1 SD of the between-subject scatter (var=.015) and is not worth a mechanism to chase.
Rationale: The candidate was ACCEPTED and should remain the base; I am asking only for a narrow, in-family re-tune of the SAME mechanism (do not change family, do not reintroduce any tally pull, chorus gate, D-dilution or capacity term \u2014 those are exactly what the arbiter excluded and what the data reject in Exps 17-20). Two specific knobs:

1) The fallback whisper `delta` is currently forced non-negative, so floor-silent displays sit at F = .528. The pooled evidence (Exp 7 implies F ~= .45 given p_eff ~= .87 from Exp 20; Exp 2 over-predicted by +.054; Exps 13/14 real values .489/.489) says F should be at or a hair BELOW chance. Widen the range to a signed whisper, e.g. delta in [-0.10, +0.06], keeping the identical code path (orient to the lowest-validity discriminating cue with weight delta; a negative delta simply means the residual descent orients weakly AWAY from it). This should pull Exp 2 toward ~.68, Exp 7 toward ~.67 and leave Exps 8/13/14/15 essentially unchanged, because none of the floor-speaks cells touch delta.

2) Tighten `p_f` rather than move it. Exp 20 (.870), Exp 12 (.145) and Exp 11 (.698) want p_eff ~= .86; Exp 19 (.842) wants a shade less; Exps 5/6 (-.763, -.938, currently under-shot at -.727/-.891) want a shade more. These flanking pressures cancel, so do NOT swing p_f in either direction \u2014 narrow the search to p_f in [0.86, 0.91] with epsilon in [0.02, 0.08] so the fitter cannot drift to the .92-.94 end that is driving the Exp 19 overshoot.

Leave everything else untouched. Explicitly do not add a solo-display boost to chase Exp 16 (+.052 with var=.015): that would manufacture exactly the D-gradient the arbiter ruled out and would immediately damage Exp 15 (real .007) and Exp 18 (real .015). If the signed-delta change does not beat 0.0309, the gate will discard it and the current candidate stands.

**Outcome of this advice:** iter 1 candidate loss=0.0309 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Take-the-Worst: the Floor-Voice Lexicographic Rule (TTW-FLAT, signed-whisper variant).**\n\nWhen people are told the accuracies of a panel of binary raters and asked which product is better, they do not integrate the panel at all. They run a NON-COMPENSATORY, SINGLE-CUE search, and the cue they search for is the panel's *least* accurate voice. Five commitments.\n\n**(1) ONE VOICE IS CONSULTED, AND IT IS THE PANEL FLOOR.** Search is lexicographic in ASCENDING stated validity. The subject looks for the single expert with the lowest validity *on the displayed panel* (panel-relative, read off the instructions; there is no absolute 'closeness-to-.5' criterion). Told that several experts are nearly infallible and one is barely better than a coin, subjects treat the near-infallible ratings as redundant, already-known public information and the marginal rater as the only voice carrying private, decision-relevant news. That voice, and only that voice, is read.\n\n**(2) FLOOR-FOLLOWING IS A FLAT, COUNT-INVARIANT AND CONTEXT-INVARIANT RATE.** If the floor expert discriminates, the subject picks its side with one fixed probability p_f (~0.86-0.91 before lapse). This rate is a constant of the person, not of the display: it does NOT change with how many cues oppose the floor voice, with how senior those opponents are, with the total number of discriminating attributes D, with whether the feature-wise counts are tied or lopsided, or with whether the floor sits alone or inside a unanimous bloc. This flatness is the theory's sharpest, shape-free prediction: solo-vs-dense, tie-vs-majority, star-opposition-vs-mid-opposition, and opposed-vs-unanimous all come out at exactly the same number.\n\n**(3) IF THE FLOOR IS SILENT, THE RULE FAILS AND THE SUBJECT GUESSES \u2014 WITH A SIGNED WHISPER THAT IS, IF ANYTHING, SLIGHTLY AVERSIVE.** When the floor expert gives both products the same rating, the non-compensatory rule has no output. There is NO fallback to a tally, NO compensatory balance, and NO validity-ordered take-the-best. At most there is a whisper: the subject may continue the descent one step and orient weakly with respect to the lowest-validity cue that *does* discriminate, with a SIGNED weight delta of small magnitude (|delta| <= 0.10). Critically delta is allowed to be negative: having failed to find the voice it was looking for, the subject's residual reading of the next-lowest voice is a hair *away* from it, so floor-silent displays land at or marginally BELOW chance rather than marginally above. Hence displays whose discriminating voices are all upper-band sit at essentially exact chance (a touch under) at every count margin, second-from-bottom voices produce only a whisper (never moderate anti-validity), and tally adherence is a *by-product* of the floor happening to side with the majority, never an independent pull.\n\n**(4) NO CAPACITY TERM.** Because only one cue is ever read, panel size is irrelevant: eight-expert panels behave exactly like six- and seven-expert panels.\n\n**(5) LAPSE.** A small attention/side-noise term epsilon (~0.02-0.08) mixes the whole result with a coin flip.\n\n**Cross-design signature:** identical floor-following rate on 1v1 ties, 1v2, 1v3 and unanimous displays; chance-or-a-hair-below whenever the floor is silent, including all-upper displays, star-backed dense majorities and second-from-bottom cells; reversal of any majority opposed by the floor voice; near-ceiling adherence to any majority containing the floor voice; and a flat line under seniority titration of the opposing bloc.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------------- parse the stimulus ----------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        for ka, kb in (('option_a_ratings', 'option_b_ratings'),\n                       ('a_ratings', 'b_ratings'),\n                       ('A', 'B'), ('a', 'b')):\n            if ka in state and kb in state:\n                try:\n                    a = np.asarray(list(state[ka]), dtype=float).ravel()\n                    b = np.asarray(list(state[kb]), dtype=float).ravel()\n                except Exception:\n                    a = None\n                    b = None\n                break\n    if a is None or b is None:\n        try:\n            arr = np.asarray(state, dtype=float).reshape(2, -1)\n            a = np.asarray(arr[0], dtype=float).ravel()\n            b = np.asarray(arr[1], dtype=float).ravel()\n        except Exception:\n            return np.array([0.5, 0.5], dtype=float)\n\n    if a is None or b is None or a.size == 0 or a.size != b.size:\n        return np.array([0.5, 0.5], dtype=float)\n\n    n = int(a.size)\n\n    # ---------------- parameters ----------------\n    def _g(name, default, lo, hi):\n        try:\n            x = float(parameters.get(name, default))\n        except Exception:\n            x = float(default)\n        if not np.isfinite(x):\n            x = float(default)\n        return float(min(max(x, lo), hi))\n\n    p_f = _g('p_f', 0.88, 0.50, 1.0)        # flat floor-following rate\n    delta = _g('delta', -0.02, -0.30, 0.30)  # SIGNED whisper of a one-step descent\n    eps = _g('epsilon', 0.05, 0.0, 0.60)     # lapse / side noise\n\n    # ---------------- stated validities (panel, from instructions) ----------\n    v = None\n    try:\n        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n        if vr is not None:\n            v = np.asarray(vr, dtype=float).ravel()\n    except Exception:\n        v = None\n    if v is None or v.size != n:\n        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)\n    v = np.where(np.isfinite(v), v, 0.75)\n\n    # ---------------- feature-wise discrimination ----------------\n    d = np.sign(a - b)                       # +1 -> favours A, -1 -> favours B\n    disc = np.nonzero(d)[0]\n    if disc.size == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    # ---------- Commitment 1: locate the PANEL FLOOR voice ----------\n    v_min = float(np.min(v))\n    floor_idx = np.nonzero(v <= v_min + 1e-12)[0]\n\n    net_floor = float(np.sum(d[floor_idx])) if floor_idx.size > 0 else 0.0\n    floor_speaks = bool(np.any(d[floor_idx] != 0)) if floor_idx.size > 0 else False\n\n    if floor_speaks:\n        if net_floor > 0:\n            # ---- Commitment 2: flat, count- and context-invariant rate ----\n            p_a = p_f\n        elif net_floor < 0:\n            p_a = 1.0 - p_f\n        else:\n            # degenerate: tied floor voices contradict each other -> no output\n            p_a = 0.5\n    else:\n        # ---- Commitment 3: the rule failed -> guess, with a SIGNED whisper ----\n        vd = v[disc]\n        j = int(disc[int(np.argmin(vd))])\n        # if several discriminating cues share the minimal validity, take their net\n        lowest = float(np.min(vd))\n        tied = disc[np.abs(v[disc] - lowest) <= 1e-12]\n        net_low = float(np.sum(d[tied]))\n        if net_low > 0:\n            dirn = 1.0\n        elif net_low < 0:\n            dirn = -1.0\n        else:\n            dirn = float(np.sign(d[j]))\n        p_a = 0.5 + dirn * 0.5 * delta\n\n    # ---------- Commitment 5: lapse ----------\n    p_a = (1.0 - eps) * p_a + eps * 0.5\n    if not np.isfinite(p_a):\n        p_a = 0.5\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, 1.0)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if (not np.isfinite(s)) or s <= 0:\n        probs = np.ones_like(probs) / float(len(probs))\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "p_f": "[0.86, 0.91]",
    "delta": "[-0.10, 0.06]",
    "epsilon": "[0.02, 0.08]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff edit on the ACCEPTED TTW-FLAT base, implementing exactly the two knobs the critic asked for and nothing else \u2014 no new mechanism, no tally pull, no chorus gate, no D-dilution, no capacity term.\n\nEdit 1 (one line in `predict`, one line in `parameters`): the fallback whisper `delta` was clamped to be non-negative (`_g('delta', 0.06, 0.0, 0.30)`), forcing floor-SILENT displays to sit at F = 0.5 + delta/2 = .528. The code path is otherwise identical, so simply widening the clamp to `_g('delta', -0.02, -0.30, 0.30)` and the sampled range to [-0.10, +0.06] lets the residual one-step descent orient weakly AWAY from the second-lowest voice. The pooled evidence points there: with p_eff \u2248 .87 pinned by Exp 20 (.870) and Exp 12 (1-p_eff = .136 vs real .145), Exp 7's real .661 = (p_eff + F)/2 implies F \u2248 .45, and Exps 13/14 (real .489/.489, both a hair below chance) and Exp 2 (over-predicted +.054) say the same. A signed, slightly negative delta pulls Exp 2 toward ~.68 and Exp 7 toward ~.67, and nudges Exps 13/14 from .498/.469 toward ~.48, while every floor-speaks cell (Exps 5, 6, 11, 12, 19, 20) is untouched because delta never enters those branches. Exps 8/10/15/17/18 involve only fallback-vs-fallback or floor-vs-floor contrasts, so they stay near their current near-zero/near-.48 values.\n\nEdit 2 (two parameter ranges): `p_f` narrowed from [0.84, 0.94] to [0.86, 0.91] and `epsilon` from [0.02, 0.10] to [0.02, 0.08]. The critic's diagnosis is that the flanking pressures cancel \u2014 Exps 20/12/11 want p_eff \u2248 .86, Exp 19 a shade less, Exps 5/6 a shade more \u2014 so p_f should not be swung, only prevented from drifting to the .92-.94 end that drives the Exp 19 overshoot (.884 vs .842). Tightening the upper end trims Exp 19 toward ~.86 and costs at most ~.01 on Exps 5/6, which are currently under-shot by only .037/.046.\n\nExplicitly NOT done: no solo-display boost for Exp 16 (+.052, var=.015, within ~1 SD of between-subject scatter). Adding one would manufacture precisely the D-gradient the arbiter ruled out and would immediately break Exp 15 (real .007) and Exp 18 (real .015), both of which the flat rule already nails. The theory keeps its three shape-free rate parameters and therefore remains maximally falsifiable and experiment-invariant across 5-, 6-, 7- and 8-expert panels."
}
```

## Usage

```json
{
  "input_tokens": 61045,
  "output_tokens": 4550
}
```
