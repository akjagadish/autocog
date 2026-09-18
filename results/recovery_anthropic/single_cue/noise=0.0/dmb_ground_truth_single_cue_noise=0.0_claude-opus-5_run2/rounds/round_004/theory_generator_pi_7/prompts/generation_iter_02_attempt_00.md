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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Replace pi_5 with a *graded* inverse-validity tallying account that removes coarse tiering and the unconditional majority discount, and that explains the large-panel washout by smooth dilution rather than by a hard capacity cliff. Sketch:

**Gated Linear Inverse-Validity Integration with Panel-Size Dilution (GLIV-D).**
1. Count gate (shared with pi_6, strongly supported by Exp3, Exp1's tied cells and Exp2's C3): if nA == nB the display is non-diagnostic and P(A)=0.5 exactly, with no validity tie-break and no take-the-best fallback.
2. When a strict count winner exists, evidence is E = sum_j w_j * sign(a_j - b_j) with a *linear* (or at most mildly concave, exponent 0.8-1.2) panel-relative inverse-validity weight w_j = 1 - lam * (v_j - v_min)/(v_max - v_min), lam in ~0.85-1.20. This is the crucial departure from pi_6: mid-panel experts must retain roughly half-weight so that two mids (~.74/.70) exactly offset one floor voice (~.52) \, forcing Exp2's C4 to ~0.50 rather than 0.83, while still making the top expert near-zero/slightly negative (preserving Exp7/Exp8's large negative contrasts and Exp9's 0.66).
3. Graded dilution instead of a 7.5-expert step: slope beta is divided by D^gamma * (n/6)^tau (tau ~0.6-1.2, gamma ~0.7-1.0). This must reproduce, without any threshold, that star-backed dense majorities on n=6/7 panels are reversed or well below chance (Exp1, Exp7, Exp8) while the same structural profiles on n=8 panels regress to chance (Exp10 obs 0.485), and that overall tally adherence on mixed dense displays stays ~0.75-0.80 (Exp5 0.794, Exp6 0.755).
4. Only a mild release of the discount when a single cue discriminates (lam_eff scaled by (D-1)/((D-1)+kappa), kappa small), plus a lapse ~0.10-0.20 keeping predictions off the 0/1 rails \, the observed values are all intermediate (0.40-0.79, or -0.02), so parameter ranges should be chosen so that no cell is pushed past ~0.90 or below ~0.10.

Calibration targets the new theory must hit simultaneously: Exp1 ~0.40-0.45; Exp2 ~-0.10 to +0.05; Exp3 ~0.48-0.52; Exp4 ~0.62-0.70; Exp5 ~0.78-0.82; Exp6 ~0.73-0.78; Exp7 ~-0.75; Exp8 ~-0.90; Exp9 ~0.65-0.70; Exp10 ~0.45-0.52. It should be designed to be *distinguishable* from pi_6 by two clean signatures: (i) mid-panel-backed majorities facing a lone star should be clearly above chance under GLIV-D but near chance under pi_6's saturating inversion, and (ii) the n=7 -> n=8 transition should be smooth under GLIV-D but a step under pi_6's capacity gate.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.785)

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


---

### `pi_4` (overall score: 0.552)

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


---

### `pi_3` (overall score: 0.395)

**Description**
**Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- k: [1.5, 3.5]
- nu: [0.12, 0.32]
- epsilon: [0.0, 0.12]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5125 (var=0.0055)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.6669 (var=0.0098)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.6379 (var=0.0030)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.6565 (var=0.0039)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.0042 (var=0.0075)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.0675 (var=0.0417)
- Experiment 7: real=0.6606 (var=0.0038) vs this=0.4981 (var=0.0037)
- Experiment 8: real=0.4850 (var=0.0085) vs this=0.5913 (var=0.0076)
- Experiment 9: real=0.4050 (var=0.0022) vs this=0.4828 (var=0.0041)
- Experiment 10: real=-0.0225 (var=0.0100) vs this=-0.1062 (var=0.0243)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1528 -> ACCEPTED
- iter 2: loss=0.1266 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1266 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.5131 (var=0.0071) (Δ vs real +0.0394)
  - iter 2 (most recent): 0.5106 (var=0.0053) (Δ vs real +0.0369)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)
- pi_5: 0.5169 (var=0.0089)
- pi_6: 0.5106 (var=0.0056)

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
  - iter 1: 0.7944 (var=0.0093) (Δ vs real +0.1406)
  - iter 2 (most recent): 0.6919 (var=0.0200) (Δ vs real +0.0381)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)
- pi_5: 0.9281 (var=0.0024)
- pi_6: 0.7025 (var=0.0100)

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
  - iter 1: 0.7125 (var=0.0039) (Δ vs real -0.0812)
  - iter 2 (most recent): 0.6906 (var=0.0049) (Δ vs real -0.1031)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)
- pi_5: 0.8167 (var=0.0020)
- pi_6: 0.7831 (var=0.0039)

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
  - iter 1: 0.8721 (var=0.0022) (Δ vs real +0.1175)
  - iter 2 (most recent): 0.8240 (var=0.0027) (Δ vs real +0.0694)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)
- pi_5: 0.7783 (var=0.0043)
- pi_6: 0.7175 (var=0.0042)

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
  - iter 1: -0.7958 (var=0.0045) (Δ vs real -0.0325)
  - iter 2 (most recent): -0.8046 (var=0.0051) (Δ vs real -0.0413)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)
- pi_5: -0.7671 (var=0.0062)
- pi_6: -0.7525 (var=0.0071)

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
  - iter 1: -0.8469 (var=0.0249) (Δ vs real +0.0906)
  - iter 2 (most recent): -0.7944 (var=0.0248) (Δ vs real +0.1431)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)
- pi_5: -0.8519 (var=0.0288)
- pi_6: -0.9162 (var=0.0494)

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
  - iter 1: 0.7084 (var=0.0019) (Δ vs real +0.0478)
  - iter 2 (most recent): 0.7134 (var=0.0029) (Δ vs real +0.0528)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9059 (var=0.0023)
- pi_4: 0.4641 (var=0.0042)
- pi_1: 0.1278 (var=0.0098)
- pi_2: 0.4997 (var=0.0029)
- pi_3: 0.4981 (var=0.0037)
- pi_6: 0.6775 (var=0.0023)

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
  - iter 1: 0.5331 (var=0.0213) (Δ vs real +0.0481)
  - iter 2 (most recent): 0.4556 (var=0.0131) (Δ vs real -0.0294)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7144 (var=0.0105)
- pi_5: 0.1506 (var=0.0044)
- pi_1: 0.8488 (var=0.0147)
- pi_2: 0.8550 (var=0.0140)
- pi_3: 0.5913 (var=0.0076)
- pi_6: 0.4919 (var=0.0121)

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
  - iter 1: 0.4541 (var=0.0037) (Δ vs real +0.0491)
  - iter 2 (most recent): 0.4453 (var=0.0032) (Δ vs real +0.0403)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8438 (var=0.0038)
- pi_6: 0.3791 (var=0.0043)
- pi_1: 0.3253 (var=0.0042)
- pi_2: 0.4994 (var=0.0020)
- pi_3: 0.4828 (var=0.0041)
- pi_4: 0.5244 (var=0.0029)

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
  - iter 1: -0.0912 (var=0.0396) (Δ vs real -0.0688)
  - iter 2 (most recent): -0.0700 (var=0.0453) (Δ vs real -0.0475)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3338 (var=0.0282)
- pi_5: -0.7275 (var=0.0246)
- pi_1: -0.0050 (var=0.0123)
- pi_2: -0.3575 (var=0.0255)
- pi_3: -0.1062 (var=0.0243)
- pi_4: -0.2200 (var=0.0268)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Superlinear Redundancy-Discounted Tallying with Seniority-Scaled Lone-Dissent Salience and Panel-Size Precision Dilution (SR-DT-2).**

People told the validities of a panel of binary expert ratings do not weight experts *by* validity; they weight them by how much *news* a rating carries, which is a decreasing function of how senior the expert is inside the displayed panel. Six commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert is evaluated, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not (equal win-counts, including wholly non-discriminating displays), the display is construed as non-diagnostic: choice is exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free: on tally-tied, side-counterbalanced profiles, adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) Seniority is measured against the chance baseline and the panel's best voice, and the discount is SUPERLINEAR in seniority.** s_j = (v_j - 0.5)/(v_max - 0.5), clipped to [0,1]; news value w_j = 1 - lambda * s_j^omega with omega > 1. The discount is negligible across the bottom half of the panel, about half weight for genuine mid-panel experts, and only for the top one or two voices does it drive weight to zero or below. Two mid-panel experts therefore exactly offset one floor-level voice, while a single top expert is worth nothing or less.

**(3) Redundancy requires same-side corroboration.** An expert is discounted only because he is heard as restating what colleagues who point the same way already say: lambda_side = lambda_n * (m_side - 1)/((m_side - 1) + kappa). A lone voice cannot be redundant with anybody.

**(4) A lone voice's impact scales with its seniority (dissent salience).** Being the only expert who differentiates the products is *arresting in proportion to the speaker's standing*: an isolated near-perfect authority contradicting a chorus is striking and is heard at more than face value, w = 1 + (phi - 1) * s_j with phi slightly above 1, whereas an isolated barely-diagnostic voice is merely idiosyncratic and carries ordinary face value (s ~ 0, w ~ 1). This is the crucial asymmetry: a solitary star can hold its own against a four-cue majority (Exp2's 1-vs-4 conflict cells fall near chance), while a solitary floor-level voice cannot beat a mid-panel pair (Exp10's C4 stays near chance and does not flip), a dissociation that a validity-blind lone-voice bonus cannot produce.

**(5) Precision falls with both conflict and panel size.** Confidence is beta * E / (D^gamma * (n/6)^tau): weighted evidence spread across many mutually contradicting cues is compressed superlinearly (gamma > 1), and *decision precision itself* degrades smoothly as the panel grows beyond the reference six-expert display (tau > 0). Panel size therefore lowers accuracy of execution as well as fidelity of the validity ordering.

**(6) Ordering fidelity decays smoothly with panel span.** lambda_n = lambda * exp(-rho * max(0, n - 7)): the seniority discount fades once the panel exceeds the ~7-item span, without any cliff. Combined with (5), 8-expert panels show both a weakened inversion (star-backed dense majorities regress to chance rather than being reversed) and lower overall tally adherence, while 6-expert panels show the sharpest tallying and the strongest inversion. An attention lapse epsilon mixes every decision with a coin flip.

**Cross-design signature:** exact chance on tally-tied displays of any validity composition; strong reversal of chorus-backed star majorities on six/seven-expert panels; near-chance for the same structure on eight-expert panels together with visibly reduced (but still above-chance) pure tally adherence there; near-ceiling following of a lone *high-validity* discriminating cue; near-chance when a lone star opposes a broad majority; mid-panel-backed majorities facing a lone floor voice sitting essentially at chance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    def _pair(st):
        a = None
        b = None
        if isinstance(st, dict):
            for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                           ('a_ratings', 'b_ratings'),
                           ('A', 'B'), ('a', 'b')):
                if ka in st and kb in st:
                    try:
                        a = np.asarray(list(st[ka]), dtype=float).ravel()
                        b = np.asarray(list(st[kb]), dtype=float).ravel()
                    except Exception:
                        a = None
                        b = None
                    break
        if a is None:
            try:
                arr = np.asarray(st, dtype=float)
                arr = arr.reshape(2, -1)
                a = np.asarray(arr[0], dtype=float).ravel()
                b = np.asarray(arr[1], dtype=float).ravel()
            except Exception:
                return None, None
        return a, b

    a, b = _pair(state)
    if a is None or b is None:
        return np.array([0.5, 0.5], dtype=float)
    if a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: unweighted count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes the wholly non-discriminating case
        return np.array([0.5, 0.5], dtype=float)
    D = float(mA + mB)

    # ---------- parameters ----------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.60, 0.0, 3.0)
    omega = _g('omega', 1.70, 0.5, 3.5)
    beta = _g('beta', 17.0, 0.0, 40.0)
    gamma = _g('gamma', 1.10, 0.0, 2.5)
    kappa = _g('kappa', 0.05, 1e-6, 1.0)
    rho = _g('rho', 0.70, 0.0, 2.0)
    tau = _g('tau', 2.50, 0.0, 5.0)
    phi = _g('phi', 1.40, 0.5, 2.5)
    eps = _g('epsilon', 0.13, 0.0, 0.6)

    # ---------- stated validities ----------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- Commitment 2: seniority anchored at chance & the panel's best voice ----------
    vmax = float(np.max(v))
    span = vmax - 0.5
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - 0.5) / span, 0.0, 1.0)
    s_pow = np.power(np.clip(s, 0.0, 1.0), omega)   # superlinear discount profile

    # ---------- Commitment 6: span-graded ordering fidelity ----------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * float(np.exp(-rho * over))
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ---------- Commitment 3: redundancy needs same-side corroboration ----------
    def _side_lam(m):
        mm = float(m)
        if mm <= 1.0:
            return 0.0                       # a lone voice cannot be redundant
        val = lam_n * (mm - 1.0) / ((mm - 1.0) + kappa)
        if (not np.isfinite(val)) or val < 0.0:
            return 0.0
        return float(val)

    lam_A = _side_lam(mA)
    lam_B = _side_lam(mB)

    if mA > 0:
        wA = 1.0 - lam_A * s_pow[iA]
    else:
        wA = np.array([], dtype=float)
    if mB > 0:
        wB = 1.0 - lam_B * s_pow[iB]
    else:
        wB = np.array([], dtype=float)

    # ---------- Commitment 4: seniority-scaled lone-dissent salience ----------
    if mA == 1:
        wA = 1.0 + (phi - 1.0) * s[iA]
    if mB == 1:
        wB = 1.0 + (phi - 1.0) * s[iB]

    E = float(np.sum(wA) - np.sum(wB))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: dilution by conflict AND by panel size ----------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    size_fac = (float(n) / 6.0) ** tau
    if (not np.isfinite(size_fac)) or size_fac <= 0.0:
        size_fac = 1.0
    den = den * size_fac

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- attention lapse ----------
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
- lam: [1.35, 1.90]
- omega: [1.45, 2.05]
- beta: [14.0, 20.0]
- gamma: [0.95, 1.25]
- kappa: [0.02, 0.15]
- rho: [0.55, 0.90]
- tau: [2.0, 3.0]
- phi: [1.15, 1.65]
- epsilon: [0.08, 0.18]
- validities: validities

`rationale`: Minimal-diff edit on the ACCEPTED base (loss 0.1528), implementing the critic's three prioritized tweaks, with one principled modification to the third.

**(1) Panel-size SLOPE dilution (critic's top priority).** Added `size_fac = (n/6)^tau` to the denominator, so z = beta*E/(D^gamma*(n/6)^tau), tau in [2.0,3.0]. Previously panel size only faded the *discount* (lam_n), never precision, which is exactly why the two n=8 experiments split: Exp4 raced to ceiling (0.872 vs 0.755) while Exp8 needed chance. Hand-computation on Exp4's mean trial (z ~ 2.7 before) gives, with the new divisor (8/6)^2.5 ~ 2.05 and the raised beta, z ~ 1.7 -> adherence ~0.80 (down from 0.872), and nudges Exp8 from 0.533 toward ~0.51 (real 0.485). n=6 experiments are untouched by construction ((6/6)^tau = 1); n=7 is mildly softened ((7/6)^2.5 ~ 1.47), which slightly improves Exp10 (-0.091 -> ~-0.08 vs real -0.023).

**(2) Compensating slope at n=6 (critic's item 2).** beta range shifted [10,16] -> [14,20] and epsilon narrowed [0.09,0.21] -> [0.08,0.18]. Exp3 needs about +0.08 of tally adherence at n=6: with the ~1.3x slope its underlying z rises 1.10 -> ~1.45, adherence ~0.71 -> ~0.78 (real 0.794). Because edit (1) already removes the excess precision at n=8, the extra slope no longer inflates Exp4. It also sharpens Exp9 toward the observed below-chance 0.405 and Exp6 toward -0.938.

**(3) Lone-dissent salience, but SCALED BY SENIORITY (deliberate refinement of the critic's suggestion).** The critic proposed either a delta>0 softening of the lone-voice release or a flat phi>1 bonus for m_side==1. The delta option is the wrong sign (it would *discount* the lone star and make the majority even easier to follow), and a *flat* phi is provably harmful: in Exp10's C4 cell the lone voice is the panel's floor expert (v=.52) opposing a mid pair, so a flat phi=1.4 flips that cell from ~0.37 to ~0.72 and pushes the Exp10 contrast from -0.09 to +0.22 -- destroying the candidate's single best discriminating win. I therefore made the salience proportional to the lone speaker's standing: w = 1 + (phi-1)*s_j. This is theoretically the right construal (an isolated *authority* contradicting a chorus is arresting; an isolated barely-diagnostic voice is merely idiosyncratic) and it separates the two cells cleanly: Exp2's lone star (s=1) gets the full phi bonus, taking its 1-vs-4 conflict cell from ~0.68 to ~0.50 and the Exp2 metric from 0.794 toward ~0.71 (real 0.654), while Exp10's lone floor voice (s ~ 0.04) is left essentially unchanged, preserving the near-zero C4-minus-C3 contrast. It also raises adherence on Exp3's single-star discriminating trials, reinforcing edit (2).

Everything else is byte-identical to the accepted base: the parameter-free count gate (Exp1 stays at chance), chance-anchored superlinear seniority, same-side corroboration, D^gamma conflict dilution, exponential span-graded fidelity, and the lapse.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a clear overall success within the arbiter's prescribed family: it reproduces the two hardest signatures in the corpus — the large negative coherence-contrast on n=6/7 panels (Exp5 -0.796 vs -0.763; Exp6 -0.847 vs -0.938), the marginal-tier preference on Exp7 (0.708 vs 0.661), the near-chance star-backed-majority adherence on the n=8 panel (Exp8 0.533 vs 0.485), the below-chance marginal-net metric of Exp9 (0.454 vs 0.405), the chance-level tally-tie gate of Exp1 (0.513 vs 0.474) and, crucially, Exp10's near-zero contrast (-0.091 vs -0.023) where both pi_5 (-0.73) and pi_6 (+0.33) failed catastrophically in opposite directions. The chance-anchored superlinear seniority profile plus same-side corroboration is doing exactly what the rationale claims: mid experts retain roughly half weight, so the C4 mid-pair-vs-floor cell lands near chance rather than at either rail, and lone dissenters are resisted rather than inverted. Residual misfit is concentrated in three cells, and it has a clean, interpretable structure. (i) Exp4 (n=8, pure tally adherence) is badly OVER-predicted: 0.872 vs 0.755. Because ordering fidelity decays as exp(-rho*(n-7)), the inversion largely switches off on 8-expert panels and the model degenerates into near-pure tallying with a big slope, giving near-ceiling adherence. (ii) Exp3 (n=6, pure tally adherence) is UNDER-predicted: 0.713 vs 0.794. So the model has the n=6 vs n=8 adherence ordering backwards relative to the data: humans follow the tally *more* on the smaller panel and regress toward chance on the larger one, whereas the candidate does the reverse. (iii) Exp2 conflict adherence is over-predicted (0.794 vs 0.654): the lone-dissenter release (lam_side=0 at m=1) plus a strong slope makes a lone star too easy to outvote. Together these mean the fade-with-n mechanism is currently the *only* thing that changes with panel size, and it changes the wrong quantity — it removes the discount instead of removing decision precision.
Rationale: The candidate was ACCEPTED and should remain the base; the request is for a targeted, in-family tune, not a redesign. Keep every commitment intact (count gate, chance-anchored superlinear seniority s=(v-0.5)/(v_max-0.5) with omega>1, same-side corroboration lambda_side, conflict dilution D^gamma, span-graded fidelity, lapse). Three minor adjustments, in priority order: (1) Add the arbiter's panel-size SLOPE dilution that the current version omitted: z = beta*E / (D^gamma * (n/6)^tau) with tau ~0.6-1.2. Right now panel size only fades the discount (lam_n), never the precision, which is why the n=8 experiments split — Exp4 goes to ceiling (0.872 vs 0.755) while Exp8 needs chance. A slope divided by (8/6)^tau pulls Exp4 down toward ~0.78-0.80 and nudges Exp8 from 0.533 toward 0.49, without touching the n=6/7 fits (Exp5/6/7/9/10) at all. (2) Compensate on n=6 by raising beta modestly (shift the range up to roughly [13, 19]) and/or narrowing epsilon to ~[0.08, 0.16]; Exp3 needs +0.08 of tally adherence at n=6, and once (1) is in place the extra slope no longer inflates Exp4. (3) For Exp2, soften the full lone-voice release rather than removing it: use lam_side = lam_n * (m-1+delta)/((m-1+delta)+kappa) with a small delta>0, or equivalently give a lone dissenting cue a weight slightly ABOVE 1 (a dissent-salience factor phi in ~[1.0,1.4] applied only when m_side==1). Either knob lets a solitary star resist a 4-cue majority harder, moving Exp2 from 0.79 toward the observed 0.65, and it will also shave the small over-prediction on Exp7. Do NOT increase lam/omega globally to fix Exp2 — that would re-break Exp10's near-zero contrast, which is currently the candidate's best discriminating win. Also leave the parameter-free count gate untouched: Exp1 is already fine.

**Outcome of this advice:** iter 1 candidate loss=0.1528 -> iter 2 candidate loss=0.1266 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-2 candidate is the running best (0.1266 vs 0.1528) and my previous top-priority advice (add a panel-size SLOPE dilution term (n/6)^tau, raise beta, add a lone-dissent factor) was validated by the gate. The two biggest previous errors were repaired: Exp2 fell 0.794 -> 0.692 (real 0.654) thanks to the seniority-scaled phi, and Exp4 fell 0.872 -> 0.824 (real 0.755) thanks to the size dilution, while Exp8 moved 0.533 -> 0.456 (real 0.485) and Exp10 stayed near-zero (-0.070 vs -0.023) — still the corpus's cleanest discriminating win versus pi_5 (-0.73) and pi_6 (+0.33). The signature negative coherence contrasts remain intact (Exp5 -0.805 vs -0.763; Exp7 0.713 vs 0.661; Exp9 0.445 vs 0.405) and the count gate keeps Exp1 at chance.

Residual error is now concentrated in two places, both about UNANIMOUS / non-conflicting dense displays rather than about the inversion itself.
(i) Exp6 is the largest residual (-0.794 vs -0.938, Δ +0.143) and it got WORSE than iter 1 (-0.847). Exp6's metric is d1 (coherence contrast, computed only on trials where both sides discriminate) plus d2 (sparse-unanimous minus dense-unanimous adherence). d1 is well captured elsewhere (Exp5, Exp7), so the shortfall is most plausibly in d2: in the current model an unanimous display with many cues gets a LARGER same-side redundancy discount (lam_side grows with m_side) while the D^gamma term keeps per-cue evidence flat, so dense unanimous adherence ends up BELOW sparse unanimous — i.e. model d2 > 0 where the data want d2 <= 0. Raising tau to [2,3] also diluted n=7 by ~1.5x, further flattening Exp6 relative to iter 1, which is exactly the wrong direction for this experiment.
(ii) Exp3 (n=6 pure tally adherence) is under-predicted and, despite the beta raise, drifted DOWN 0.713 -> 0.691 (real 0.794). Inspection of Exp3's profiles shows most majorities are star-backed, so the seniority discount plus the new phi bonus on lone opposing stars actively suppresses tally adherence there. Exp4 (n=8) is still over (0.824 vs 0.755). So the n=6 (needs more adherence) / n=8 (needs less) tension that I flagged in iter 1 is reduced but not resolved.
Rationale: Keep the accepted base and all six commitments (count gate, chance-anchored superlinear seniority, same-side corroboration, D^gamma conflict dilution, span-graded fidelity, lapse, seniority-scaled lone salience) — this is squarely in the arbiter's graded inverse-validity + panel-dilution family and is faithful to it. Three minor, in-family knobs, in priority order:

(1) NEW (top priority, targets Exp6 and Exp3 simultaneously): make the redundancy discount require OPPOSITION, not just same-side corroboration. Multiply lam_side by an opposition factor, lam_eff = lam_side * m_lose/(m_lose + kappa2) with kappa2 small (~0.05-0.3), so that an unopposed (unanimous) display carries essentially no inverse-validity discount and is followed strongly regardless of how many cues back it. Rationale: an expert can only be heard as 'redundant hype' when someone points the other way; with no dissent the tally is unambiguous. This raises dense-unanimous adherence, which (a) drives Exp6's d2 from positive toward ~0/negative and should recover most of the +0.143 gap without touching d1 (d1 is computed only on trials with n_lose>0, so Exp5/Exp7/Exp8/Exp9 conflict cells are untouched), and (b) lifts Exp3 toward 0.79. Note this is the opposite sign of the small over-negativity on Exp5 (-0.805 vs -0.763), so it should help there too.

(2) Re-reference the panel-size precision dilution instead of strengthening it further. Adding (n/6)^tau was accepted, so do NOT remove it — but tau in [2,3] over-penalises n=7, which is where Exp6 and Exp10 live and where Exp6 regressed. Use size_fac = (max(n,7)/7)^tau (i.e. no extra dilution until the panel exceeds 7) or equivalently keep (n/6)^tau but drop tau to ~[0.8,1.5] and add a separate n>7 term. This restores slope at n=7 (Exp6 more negative) while keeping the n=8 correction that fixed Exp4/Exp8. Since fix (1) will push Exp4 up again, the n>7 dilution exponent can carry the extra load (tau_8 effective ~2-3 at n=8).

(3) Trim phi slightly (range ~[1.05,1.40] instead of [1.15,1.65]) and keep beta where it is. Exp2 is now only +0.038 off, so phi has done its job; but the lone-star bonus is part of what is depressing Exp3's tally adherence, and a smaller phi combined with (1) should keep Exp2 near 0.65-0.70 while letting Exp3 climb.

Do NOT touch the parameter-free count gate (Exp1 fine), do not raise lam/omega globally (that would re-break Exp10's near-zero contrast, the candidate's best discriminator), and do not revisit the delta-style softening of the lone-voice release that I mooted in iter 1 — the seniority-scaled phi already covers that ground.

**Outcome of this advice:** iter 2 candidate loss=0.1266 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Superlinear Redundancy-Discounted Tallying with Baselined Seniority-Scaled Dissent Salience and Two-Regime Panel-Size Precision Dilution (SR-DT-3).**\n\nPeople told the validities of a panel of binary expert ratings do not weight experts *by* validity; they weight them by how much *news* a rating carries, which is a decreasing function of how senior the expert is inside the displayed panel. Six commitments.\n\n**(1) The count comparator is a gate, not a weight.** Before any expert is evaluated, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not (equal win-counts, including wholly non-discriminating displays), the display is construed as non-diagnostic: choice is exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.\n\n**(2) Seniority is measured against the chance baseline and the panel's best voice, and the discount is SUPERLINEAR in seniority.** s_j = (v_j - 0.5)/(v_max - 0.5), clipped to [0,1]; news value w_j = 1 - lambda * s_j^omega with omega > 1. The discount is negligible across the bottom half of the panel, about half weight for genuine mid-panel experts, and only for the top one or two voices does it drive weight to zero or below. Two mid-panel experts therefore roughly offset one floor-level voice, while a single top expert is worth nothing or less.\n\n**(3) Redundancy requires same-side corroboration.** An expert is discounted only because he is heard as restating what colleagues pointing the same way already say: lambda_side = lambda_n * (m_side - 1)/((m_side - 1) + kappa). A lone voice cannot be redundant with anybody.\n\n**(4) A lone voice is heard at MORE than face value, with a seniority-graded bonus on top of a constant dissent baseline.** Being the only expert who differentiates the products is intrinsically arresting — any solitary discriminating voice is taken at slightly above face value (baseline phi0 >= 1) — and the surplus grows with the speaker's standing: w = phi0 + (phi - phi0) * s_j. This yields a graded dissociation rather than an all-or-none one: a solitary near-perfect authority can hold its own against a four-cue majority (Exp2's 1-vs-4 conflict cell sits at chance), while a solitary *floor-level* voice, though slightly amplified by its isolation, only just balances a mid-panel pair (Exp10's C4 sits at chance instead of being outvoted). A pure face-value construal for lone low-validity cues is too weak; a flat large bonus is too strong; the baseline-plus-seniority form is the unique intermediate that satisfies both cells.\n\n**(5) Precision falls with conflict and, in two regimes, with panel size.** Confidence is beta * E / (D^gamma * (n/n_ref)^tau * (1 + eta * max(0, n - 7))). Weighted evidence spread across many mutually contradicting cues is compressed superlinearly (gamma > 1); decision precision degrades mildly and smoothly with panel size across the ordinary range (small tau), and then sharply once the panel exceeds the ~7-item working-memory span (eta), where the subject can no longer hold the whole display in mind at once. This is a *graded* two-regime dilution, not a cliff.\n\n**(6) Ordering fidelity decays smoothly with panel span.** lambda_n = lambda * exp(-rho * max(0, n - 7)): the seniority discount fades once the panel exceeds span. Combined with (5), 8-expert panels show a weakened inversion (star-backed dense majorities regress to chance rather than being reversed) *and* markedly lower overall tally adherence, while six- and seven-expert panels show the sharpest tallying and the strongest inversion. An attention lapse epsilon mixes every decision with a coin flip.\n\n**Cross-design signature:** exact chance on tally-tied displays of any validity composition; strong reversal of chorus-backed star majorities on six/seven-expert panels; near-chance for the same structure on eight-expert panels together with visibly reduced (but still above-chance) pure tally adherence there; near-ceiling following of a lone high-validity discriminating cue; near-chance when a lone star opposes a broad majority; and mid-panel-backed majorities facing a lone floor voice sitting essentially at chance.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------- parse the stimulus ----------\n    def _pair(st):\n        a = None\n        b = None\n        if isinstance(st, dict):\n            for ka, kb in (('option_a_ratings', 'option_b_ratings'),\n                           ('a_ratings', 'b_ratings'),\n                           ('A', 'B'), ('a', 'b')):\n                if ka in st and kb in st:\n                    try:\n                        a = np.asarray(list(st[ka]), dtype=float).ravel()\n                        b = np.asarray(list(st[kb]), dtype=float).ravel()\n                    except Exception:\n                        a = None\n                        b = None\n                    break\n        if a is None:\n            try:\n                arr = np.asarray(st, dtype=float)\n                arr = arr.reshape(2, -1)\n                a = np.asarray(arr[0], dtype=float).ravel()\n                b = np.asarray(arr[1], dtype=float).ravel()\n            except Exception:\n                return None, None\n        return a, b\n\n    a, b = _pair(state)\n    if a is None or b is None:\n        return np.array([0.5, 0.5], dtype=float)\n    if a.size == 0 or a.size != b.size:\n        return np.array([0.5, 0.5], dtype=float)\n\n    n = int(a.size)\n\n    # ---------- Commitment 1: unweighted count gate ----------\n    d = np.sign(a - b)\n    iA = np.where(d > 0)[0]\n    iB = np.where(d < 0)[0]\n    mA = int(iA.size)\n    mB = int(iB.size)\n    if mA == mB:                       # includes the wholly non-discriminating case\n        return np.array([0.5, 0.5], dtype=float)\n    D = float(mA + mB)\n\n    # ---------- parameters ----------\n    def _g(name, default, lo, hi):\n        try:\n            x = float(parameters.get(name, default))\n        except Exception:\n            x = float(default)\n        if not np.isfinite(x):\n            x = float(default)\n        return float(min(max(x, lo), hi))\n\n    lam = _g('lam', 1.60, 0.0, 3.0)\n    omega = _g('omega', 1.70, 0.5, 3.5)\n    beta = _g('beta', 17.0, 0.0, 40.0)\n    gamma = _g('gamma', 1.10, 0.0, 2.5)\n    kappa = _g('kappa', 0.05, 1e-6, 1.0)\n    rho = _g('rho', 0.70, 0.0, 2.0)\n    tau = _g('tau', 0.20, 0.0, 3.0)\n    eta = _g('eta', 1.70, 0.0, 6.0)\n    phi = _g('phi', 1.40, 0.5, 2.5)\n    phi0 = _g('phi0', 1.08, 0.5, 2.0)\n    eps = _g('epsilon', 0.13, 0.0, 0.6)\n\n    # ---------- stated validities ----------\n    v = None\n    try:\n        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n        if vr is not None:\n            v = np.asarray(vr, dtype=float).ravel()\n    except Exception:\n        v = None\n    if v is None or v.size != n:\n        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)\n\n    # ---------- Commitment 2: seniority anchored at chance & the panel's best voice ----------\n    vmax = float(np.max(v))\n    span = vmax - 0.5\n    if (not np.isfinite(span)) or span <= 1e-9:\n        s = np.zeros(n, dtype=float)\n    else:\n        s = np.clip((v - 0.5) / span, 0.0, 1.0)\n    s_pow = np.power(np.clip(s, 0.0, 1.0), omega)   # superlinear discount profile\n\n    # ---------- Commitment 6: span-graded ordering fidelity ----------\n    over = max(0.0, float(n) - 7.0)\n    lam_n = lam * float(np.exp(-rho * over))\n    if (not np.isfinite(lam_n)) or lam_n < 0.0:\n        lam_n = 0.0\n\n    # ---------- Commitment 3: redundancy needs same-side corroboration ----------\n    def _side_lam(m):\n        mm = float(m)\n        if mm <= 1.0:\n            return 0.0                       # a lone voice cannot be redundant\n        val = lam_n * (mm - 1.0) / ((mm - 1.0) + kappa)\n        if (not np.isfinite(val)) or val < 0.0:\n            return 0.0\n        return float(val)\n\n    lam_A = _side_lam(mA)\n    lam_B = _side_lam(mB)\n\n    if mA > 0:\n        wA = 1.0 - lam_A * s_pow[iA]\n    else:\n        wA = np.array([], dtype=float)\n    if mB > 0:\n        wB = 1.0 - lam_B * s_pow[iB]\n    else:\n        wB = np.array([], dtype=float)\n\n    # ---------- Commitment 4: baselined, seniority-scaled lone-dissent salience ----------\n    phi_eff = phi if phi > phi0 else phi0\n    if mA == 1:\n        wA = phi0 + (phi_eff - phi0) * s[iA]\n    if mB == 1:\n        wB = phi0 + (phi_eff - phi0) * s[iB]\n\n    E = float(np.sum(wA) - np.sum(wB))\n    if not np.isfinite(E):\n        E = 0.0\n\n    # ---------- Commitment 5: dilution by conflict AND by panel size (two regimes) ----------\n    den = float(D) ** gamma\n    if (not np.isfinite(den)) or den <= 0.0:\n        den = 1.0\n    size_fac = ((float(n) / 6.0) ** tau) * (1.0 + eta * max(0.0, float(n) - 7.0))\n    if (not np.isfinite(size_fac)) or size_fac <= 0.0:\n        size_fac = 1.0\n    den = den * size_fac\n\n    z = beta * E / den\n    if not np.isfinite(z):\n        z = 0.0\n    z = float(min(max(z, -30.0), 30.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-z))\n\n    # ---------- attention lapse ----------\n    p_a = (1.0 - eps) * p_a + eps * 0.5\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, 1.0)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if (not np.isfinite(s)) or s <= 0:\n        probs = np.ones_like(probs) / float(len(probs))\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "lam": "[1.35, 1.90]",
    "omega": "[1.45, 2.05]",
    "beta": "[14.0, 20.0]",
    "gamma": "[0.95, 1.25]",
    "kappa": "[0.02, 0.15]",
    "rho": "[0.55, 0.90]",
    "tau": "[0.0, 0.4]",
    "eta": "[1.2, 2.2]",
    "phi": "[1.15, 1.55]",
    "phi0": "[1.00, 1.15]",
    "epsilon": "[0.08, 0.18]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff edit on the ACCEPTED base (loss 0.1266). Two lines of `predict` change plus two new scalars; every commitment, name and equation is otherwise byte-identical.\n\n**(A) Two-regime panel-size dilution — implements the critic's item (2), and fixes Exp4/Exp8.** `size_fac` becomes `(n/6)^tau * (1 + eta*max(0, n-7))` with tau now in [0.0,0.4] and a new eta in [1.2,2.2]. At n=6 it is exactly 1 (as before); at n=7 it is now ~1.00-1.06 instead of ~1.47, restoring the slope the critic identified as having flattened Exp6; at n=8 it is ~1.8-3.0 (mean ~2.7) versus the old ~2.05, i.e. the supra-span penalty is *strengthened* exactly where the two n=8 experiments need it. Hand-computation: Exp4's mean underlying z (1.92 at 0.824 adherence) falls by ~1.4x to ~1.40 -> adherence ~0.762 (real 0.755, was 0.824, the largest single residual); Exp8's z of -0.20 shrinks to -0.145 -> 0.469 (real 0.485, was 0.456). Exp6 gains the restored n=7 slope: recomputing all six cells of its metric by hand gives d1 ~ -0.83 (from -0.773), moving the total from -0.794 toward ~-0.84 (real -0.938). Exp1/2/3/5/7/9 (n=5,6) are untouched by construction.\n\n**(B) A baseline on the lone-dissent salience — fixes Exp10, the corpus's key discriminator.** The lone-voice weight becomes `w = phi0 + (phi - phi0)*s_j` (phi0 in [1.00,1.15]) instead of `1 + (phi-1)*s_j`. Theoretically: isolation is itself arresting, so *any* solitary discriminating voice is heard slightly above face value, with the surplus still graded by standing. Quantitatively this is the only cell in the corpus that moves much: in Exp10's C4 (mid pair .74/.70 versus the lone floor expert .52 on an n=7 panel) the restored n=7 slope from edit (A) would otherwise have driven the contrast from -0.070 to ~-0.079; with phi0 ~1.08 the lone floor voice weighs ~1.11 against the mid pair's ~1.10, so C4 lands at ~0.48-0.52 and the C4-C3 contrast lands near -0.01 (real -0.0225). Everywhere the lone voice is the *star* (Exp2 cell 1, Exp5, Exp6 pair 2, Exp7 trial 3) s~1 so the weight is still phi and nothing changes; where the lone voice is mid/low the small lift pushes Exp2 (0.692 -> ~0.68, real 0.654) and Exp9 (0.445 -> ~0.43, real 0.405) in the right direction, at a cost of only ~+0.013 on Exp7.\n\n**(C) Why I did NOT add the critic's top-priority opposition gate (lam_eff = lam_side * m_lose/(m_lose+kappa2)).** Its stated justification was that the model's Exp6 d2 (sparse-unanimous minus dense-unanimous) is positive. I recomputed Exp6's two unanimous cells by hand with the design validities [0.94,0.58,0.86,0.63,0.72,0.80,0.55]: the sparse cell (A wins f2,f4; D=2) gives p~0.865 and the dense cell (A wins f0,f3,f5,f6; D=4) gives p~0.919, i.e. the model's d2 is already **-0.054**, the correct sign. Zeroing the discount on unanimous displays would send both to the lapse ceiling (~0.935) and push d2 to 0, *worsening* Exp6 by ~0.05 and cancelling the entire gain from edit (A) on the very metric the gate was meant to repair; it would also make Exp10's C4 mid-pair evidence larger and drive that contrast to ~-0.14. Its remaining benefits (Exp3 +0.02, Exp5/Exp7 +0.015 each) do not cover that. I therefore kept commitment (3) as pure same-side corroboration and used the critic's items (2) and (3) instead. I also left phi's upper bound trimmed only slightly ([1.15,1.55]) rather than to 1.40, because phi is what keeps Exp2's 1-vs-4 star cell at chance and Exp3's lone-star unanimous cell at ceiling. The parameter-free count gate (Exp1) is untouched, and lam/omega are unchanged so Exp5/6/7/9's inversion signatures are preserved."
}
```

## Usage

```json
{
  "input_tokens": 39827,
  "output_tokens": 34604
}
```
