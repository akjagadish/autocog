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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a new theory: Bounded Outlier-Distrust Tallying (BODT). Core claims: (1) Subjects integrate all binary ratings as sign votes with subjective weights that are NOT a monotone function of stated validity — the weight of an expert is computed relative to panel structure (this is the insight pi_7 got right and FDEI structurally cannot express). (2) Experts with validity inside a trusted band [v_lo, v_hi] receive approximately equal plateau weight; chance experts (v near 0.5) receive a small weight whose sign and magnitude are set by a consensus gate (near zero on coherent panels, mildly positive when an above-ceiling outlier destabilizes the scale). (3) THE KEY DEPARTURE FROM CREDT: distrust of an above-ceiling outlier is a bounded, saturating SHRINKAGE of its weight toward a floor, not an unbounded sign flip. The discount grows with the outlier's uncorroborated edge (peaked near a small gap ~0.05, receding for dense ladders and for large edges on coherent panels) and is amplified on chance-laden panels, but it saturates so that w_top never falls below roughly -0.3 to -0.5 plateau units in moderate regimes, and reaches strongly negative values only in the composition-amplified regime (a far outlier floating above a majority-chance panel, as in Exp8). This single change predicts: deference ~0.45-0.50 (not 0.24) in the gap=0.05 spurious-precision regime (Exp2), an attenuated contrast signature in Exp1, mild anti-deference ~0.55-0.60 in Exp9/Exp11, pro-99% deference ~0.60 on dense ladders (Exp12), and near-chance tally following ~0.52 on |d|=1 trials (Exp6), while retaining the strong-side collapse of Exp8. (4) Choice remains a low-gain softmax plus a uniform lapse, with a moderately elevated lapse rate to absorb the residual over-polarization pi_7 exhibited. BODT differs from pi_4 (FDEI) in abandoning monotone diagnosticity weighting entirely (fixing Exps 7, 8, 9, 11), and differs from pi_7 (CREDT) in replacing unbounded peaked distrust with saturating bounded distrust (fixing the magnitude overshoots in Exps 1 and 2 while preserving every sign prediction pi_7 got right).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.695)

**Description**
Consensus-Relative Edge-Distrust Tallying (CREDT), corroborated-precision edition. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score. The subjective weight of an expert is set by: (1) a trust plateau over stated validity [v_lo, v_hi] with the ceiling just above 0.98; (2) a small negative chance flank (w(0.5) = -kappa_lo); (3) consensus gating of that flank by the panel's distrust signal; and (4) the novel claim: distrust of the panel-max expert is distrust of UNCOPROBORATED PRECISION and is NON-MONOTONE in the edge (gap) the top claim holds over the next rung. A dense ladder (gap ~ 0.01) corroborates near-certainty (the panel itself approaches the claim) -> trust. A small uncorroborated edge (gap ~ 0.05, e.g. 0.99 atop 0.95) reads as spurious precision -- the same underlying competence with inflated confidence -> maximal distrust. A large edge (gap >= ~0.09) on a coherent, competent panel reads as a categorically better instrument and is taken at face value -> trust returns. But a far outlier on an incoherent, chance-laden scale (a 0.99 floating above a panel stuffed with coin-flip experts) is unearned certainty -> collapse again, via a composition-amplified route that fires only on chance-laden panels. Choice is a low-gain softmax plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Consensus-Relative Edge-Distrust Tallying (CREDT),
    # corroborated-precision edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU UP TO NEAR-CERTAINTY: all v in [v_lo, v_hi]
    #       (v_hi ~ 0.981, i.e. a 98% expert is still INSIDE the
    #       plateau) get approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK: v near 0.5 gets a small NEGATIVE weight,
    #       reaching -kappa_lo at v = 0.5, rising linearly to the
    #       plateau at v_lo.
    # (iii) CONSENSUS GATING of the flank: when an above-ceiling
    #       claim is present, gate = lambda_ext + (1-lambda_ext)
    #       * sig_gap -- distrust generalizes to the scale and
    #       chance cues are counted (weight -> w_p) rather than
    #       inverted; at-or-below-ceiling panels keep the full
    #       negative flank. UNCHANGED from the accepted base.
    #  (iv) ABOVE-CEILING DISTRUST OF THE PANEL-MAX EXPERT IS
    #       NON-MONOTONE IN THE RUNG GAP (distrust of uncorroborated
    #       precision): the discount is rho * ( P(gap) + C * sig_gap ),
    #       where P is an asymmetric PEAKED bump (maximal at gap ~
    #       g_star ~ 0.05, near zero for dense ladders gap <= 0.02,
    #       receding for large edges gap >= 0.09 on coherent panels)
    #       and C * sig_gap is a composition-amplified route that
    #       re-imposes full collapse for far outliers on CHANCE-LADEN
    #       panels (C fires only when the fraction of chance-flank
    #       cues exceeds c_half). The extremity term d_w * (v_top -
    #       v_hi) is unchanged.
    # Choice is a low-gain softmax (inverse temperature beta) over
    # the two mirrored scores, plus a uniform lapse (epsilon).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    rho = float(parameters["rho"])
    d_w = float(parameters["d_w"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- Panel structure: top validity and its gap to the
    # next-highest DISTINCT validity in the panel. ----
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    # If every expert shares the top validity, the panel offers no
    # corroboration ladder; treat the next rung as chance (0.5).
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel (Exp 8) stays inside the trusted plateau;
    # only 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi
    if outlier:
        x = (gap - g0) / s_gap
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * x))
    else:
        sig_gap = 0.0

    # ---- (iii): consensus gate for the chance flank. UNCHANGED
    # from the accepted base (same g0 / s_gap, same lambda_ext). ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii) + (iii): plateau with a consensus-gated
    # chance flank. UNCHANGED from the accepted base. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # Consensus gating: with an above-ceiling claim in the
        # panel, chance cues are counted (weight -> w_p) rather
        # than inverted; with no above-ceiling claim the full
        # negative flank is retained.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iv): above-ceiling discount of the panel-max expert,
    # NON-MONOTONE in the rung gap (the only edited block). ----
    if outlier:
        # (a) Panel composition: fraction of chance-flank cues.
        c_frac = float(np.mean(flank))
        # Composition amplifier: distrust of a far-outlying top
        # claim re-imposes itself only when the panel's validity
        # scale is chance-laden (incoherent), e.g. a 0.99 floating
        # above four coin-flip experts.
        x_c = (c_frac - c_half) / s_c
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * x_c))
        # (b) Peaked edge-distrust bump: maximal when the top
        # expert's edge over the next rung is small but
        # uncorroborated (spurious precision, gap ~ g_star ~ 0.05),
        # near zero for dense ladders (corroborated precision) and
        # receding for large edges on coherent panels (a
        # categorically better instrument, taken at face value).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = (
            w_p
            - d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)
            - rho * (P + C * float(sig_gap))
        )
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of isolated 0.99 super-experts).
        w[v >= v_top - 1e-9] = w_top

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.19, 0.21]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- rho: [1.30, 1.42]
- d_w: [0.35, 0.45]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.011, 0.013]
- s_r: [0.030, 0.036]
- c_half: [0.34, 0.38]
- s_c: [0.05, 0.07]
- c_amp: [1.22, 1.36]
- beta: [0.41, 0.43]
- epsilon: [0.04, 0.06]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2530 (var=0.0065)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.1225 (var=0.0025)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.6267 (var=0.0043)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.5425 (var=0.0051)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=-0.1425 (var=0.0147)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.1235 (var=0.0028)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.5787 (var=0.0053)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.4813 (var=0.0186)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.5715 (var=0.0054)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.6094 (var=0.0057)
- Experiment 11: real=0.0206 (var=0.0003) vs this=0.0758 (var=0.0002)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.2367 (var=0.0031)


---

### `pi_3` (overall score: 0.694)

**Description**
Validity-Weighted Tallying (graded-evidence regime): people integrate evidence from ALL binary expert ratings, casting each feature a sign vote (+1 win / -1 loss / 0 tie) weighted by w_j = validity_j^gamma, summing into a zero-sum evidence score, with choice a softmax (inverse temperature beta) over the two scores plus an independent uniform lapse (epsilon). The theoretical claim added this round is about where humans sit in this family: in the UNSATURATED softmax regime. Because the evidence scores are integer-scale sign tallies, beta >= 3 already drives tanh(beta*m) ~ 1 for every unit margin, making the model behaviorally deterministic and pinning it to the tallying corner; the graded dose-response in tally margin that both experiments demand (Exp 1 metric at 44% of the deterministic-tally value; Exp 2's small positive b_d) lives at beta ~ 0.2-0.3, where p(A) is approximately linear in the tally margin. Gamma stays in a mild, provably sub-lexicographic band so that no subject leaps into the TTB regime.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Weighted Tallying.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a vote sign(a_j - b_j) for A's score and
    # the opposite for B's score; votes are weighted by w_j =
    # validity_j ** gamma. gamma = 0 -> unweighted Tallying;
    # gamma -> large -> lexicographic TTB (top cue dominates).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Validity-based weights. Since validities are in [0.5, 1.0],
    # gamma = 0 gives all-ones weights (pure Tallying); increasing
    # gamma increasingly privileges high-validity cues, with the
    # top cue dominating as gamma grows (TTB limit).
    w = np.power(val, gamma)

    # Weighted evidence scores. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform,
    # which is the correct behavior for an undiscriminating tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- gamma: [0.05, 0.7]
- beta: [0.03, 0.55]
- epsilon: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2182 (var=0.0156)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.0861 (var=0.0026)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.6456 (var=0.0096)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.6062 (var=0.0085)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=0.2958 (var=0.0606)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.4190 (var=0.0107)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.3831 (var=0.0085)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.6062 (var=0.0141)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.5220 (var=0.0036)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.6206 (var=0.0104)
- Experiment 11: real=0.0206 (var=0.0003) vs this=0.0148 (var=0.0004)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.5867 (var=0.0071)


---

### `pi_6` (overall score: 0.517)

**Description**
Broad-Spectrum Skepticism Tallying (BSST), low-gain/held-cliff calibration. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) is a wide inverted-U over STATED validity with skepticism on both flanks: (i) chance experts (v = 0.5) carry a small negative weight (~0.23 trusted-cue units) — a coin-flipper's endorsement is mildly anti-diagnostic; (ii) the entire band v ~ [0.55, 0.95] is a flat trust plateau (~0.9), making mid-band experts nearly exchangeable and generating moderate unweighted-tally behavior; (iii) distrust of claimed certainty is CONCENTRATED in a very sharp cliff between 0.98 and 0.99: w(0.98) stays mildly positive while w(0.99) plunges to ~ -1.1 trusted-cue units. The calibration claim added this round is about the GAIN-FLANK DECOUPLING: the softmax gain beta can be lowered (0.34) while the chance-flank magnitude kappa_lo is raised (0.23) so that the product beta*kappa_lo (~0.077) — the quantity that pins the chance-cue signature — is held invariant, but the product beta*|w(0.99)| (~0.37) is allowed to FALL because kappa_hi is held at its iter-2 value rather than deepened. This is the one combination never tried in nine iterations: every prior beta reduction (iters 5, 9) was bundled with a kappa_hi increase that exactly cancelled the gain on the 0.99-driven cells.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Broad-Spectrum Skepticism Tallying (BSST), sharp-late-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a wide inverted-U 'skepticism' function of the
    # expert's STATED validity v:
    #   v <= v_rise :  w = -kappa_lo * (v_rise - v)/(v_rise - 0.5)
    #                  (chance experts: linear lobe, w(0.5) = -kappa_lo;
    #                   a coin-flipper's endorsement is mildly
    #                   anti-diagnostic noise)
    #   v >  v_rise :  w = w_plateau - (w_plateau + kappa_hi)
    #                        * sigmoid((v - v_c)/s_hi)
    #                  (flat trust plateau over the mid-band -- the
    #                   logistic is ~0 for every v <= 0.95, so all
    #                   mid-band experts get the full plateau weight;
    #                   then an extremely SHARP distrust cliff packed
    #                   between 0.98 and 0.99: w(0.98) stays mildly
    #                   positive, w(0.99) plunges to ~ -1.1 trusted-cue
    #                   units, saturating at -kappa_hi as v -> 1)
    # Votes are summed into a zero-sum evidence score; choice is a
    # low-gain softmax with inverse temperature beta plus a uniform
    # lapse epsilon.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    v_rise = float(parameters["v_rise"])
    w_plateau = float(parameters["w_plateau"])
    v_c = float(parameters["v_c"])
    s_hi = float(parameters["s_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.empty(n_features, dtype=float)

    # (i) Low flank: chance experts carry a small negative weight,
    # linearly reaching -kappa_lo at v = 0.5 and 0 at v_rise.
    lo = v < v_rise
    denom_lo = max(v_rise - 0.5, 1e-6)
    w[lo] = -kappa_lo * np.clip(v_rise - v[lo], 0.0, None) / denom_lo

    # (ii)+(iii) Flat trust plateau followed by an extremely sharp
    # distrust cliff concentrated between v = 0.98 and v = 0.99.
    # In the mid-band the logistic term is ~0, so w ~ w_plateau
    # (nearly equal weights -> tally-like counting); at 0.98 the
    # weight is only mildly discounted, at 0.99 it is strongly
    # anti-diagnostic, and it saturates at -kappa_hi as v -> 1.
    hi = ~lo
    # Numerically stable logistic.
    x = (v[hi] - v_c) / s_hi
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    w[hi] = w_plateau - (w_plateau + kappa_hi) * sig

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- v_rise: [0.515, 0.54]
- w_plateau: [0.88, 0.93]
- v_c: [0.9855, 0.9875]
- s_hi: [0.0025, 0.0033]
- kappa_lo: [0.22, 0.235]
- kappa_hi: [1.68, 1.72]
- beta: [0.33, 0.35]
- epsilon: [0.05, 0.07]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.1973 (var=0.0076)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.0956 (var=0.0023)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.4811 (var=0.0035)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.3850 (var=0.0041)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=-0.1350 (var=0.0257)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.4360 (var=0.0065)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.6756 (var=0.0042)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.4213 (var=0.0156)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.8115 (var=0.0019)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.2988 (var=0.0058)
- Experiment 11: real=0.0206 (var=0.0003) vs this=-0.0227 (var=0.0002)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.2667 (var=0.0041)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0730 -> ACCEPTED
- iter 2: loss=0.0556 -> ACCEPTED
- iter 3: loss=0.0554 -> ACCEPTED
- iter 4: loss=0.0669 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0554 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    rows_a = data["option_a_ratings"].tolist()
    rows_b = data["option_b_ratings"].tolist()
    resp = data["response"].tolist()
    n_rows = len(rows_a)
    contribs = []
    for i in range(n_rows):
        ca = rows_a[i]
        cb = rows_b[i]
        if isinstance(ca, str):
            ca = ca.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if isinstance(cb, str):
            cb = cb.replace("[", " ").replace("]", " ").replace(",", " ").split()
        a = [int(float(v)) for v in ca]
        b = [int(float(v)) for v in cb]
        r = int(resp[i])
        n = len(a)
        if n == 0 or n != len(b):
            continue
        ttb = -1
        for j in range(n):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb < 0:
            continue
        aw = 0
        bw = 0
        for j in range(n):
            if a[j] > b[j]:
                aw = aw + 1
            elif b[j] > a[j]:
                bw = bw + 1
        if aw == bw:
            if r == ttb:
                contribs.append(0.5)
            else:
                contribs.append(-0.5)
        else:
            if bw > aw:
                tal = 1
            else:
                tal = 0
            if tal == ttb:
                continue
            if r == ttb:
                contribs.append(1.0)
            else:
                contribs.append(-1.0)
    if len(contribs) == 0:
        return 0.0
    total = 0.0
    for c in contribs:
        total = total + c
    return float(total / len(contribs))
```

**Observed (real) value:** -0.2180 (var=0.0140)
**Previous candidate values (this loop):**
  - iter 1: -0.2472 (var=0.0067) (Δ vs real -0.0292)
  - iter 2: -0.2377 (var=0.0073) (Δ vs real -0.0197)
  - iter 3: -0.2280 (var=0.0083) (Δ vs real -0.0100)
  - iter 4 (most recent): -0.2352 (var=0.0061) (Δ vs real -0.0172)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)
- pi_7: -0.2530 (var=0.0065)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dual-dissociation regression metric.
    # Per trial, compute:
    #   d = tally margin  = (#features A wins) - (#features B wins)
    #   s = top-cue direction = sign(A[0] - B[0])   (feature 0 has the
    #       highest validity 0.92, and every designed pair discriminates on it)
    # Regress chose-A on [1, d, s] via OLS and return b_d - b_s.
    # Tallying: choice probability depends only on d  -> b_d > 0, b_s = 0 -> metric > 0.
    # TTB: choice probability depends only on s       -> b_d = 0, b_s > 0 -> metric < 0.
    # The two theories push this scalar in opposite directions, and the
    # partial regression cleanly removes the design's d-s correlation.
    df = data.reset_index(drop=True)

    A = np.stack([np.asarray(list(r), dtype=float) for r in df["option_a_ratings"].values])
    B = np.stack([np.asarray(list(r), dtype=float) for r in df["option_b_ratings"].values])

    # Tally margin: strict feature-wise wins; ties count for neither side.
    d = (A > B).sum(axis=1).astype(float) - (B > A).sum(axis=1).astype(float)

    # Direction of the highest-validity (top) cue, which always discriminates
    # in this design: +1 if it favors A, -1 if it favors B.
    s = np.sign(A[:, 0] - B[:, 0])

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = 1.0 - df["response"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(y)), d, s])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_d = float(coef[1])
    b_s = float(coef[2])

    # Tallying's signature (slope in d) minus TTB's signature (slope in s).
    return b_d - b_s
```

**Observed (real) value:** 0.0771 (var=0.0031)
**Previous candidate values (this loop):**
  - iter 1: 0.1103 (var=0.0017) (Δ vs real +0.0332)
  - iter 2: 0.1110 (var=0.0019) (Δ vs real +0.0339)
  - iter 3: 0.1038 (var=0.0025) (Δ vs real +0.0267)
  - iter 4 (most recent): 0.1066 (var=0.0020) (Δ vs real +0.0296)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)
- pi_7: 0.1225 (var=0.0025)

### Experiment 3
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-following rate on decisive trials.

    For every trial, compute the raw (unweighted) tally margin
    d = (# cues where A wins) - (# cues where B wins).
    On trials with d != 0, check whether the subject's choice agrees
    with the option favored by the raw cue count (A if d > 0, B if
    d < 0). The metric is the fraction of such agreement over all
    decisive trials (d = 0 trials are excluded because the tallying
    heuristic itself is indifferent there).
    """
    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.5

    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy(dtype=float)

    d = np.empty(n, dtype=float)
    for i in range(n):
        a = np.asarray(a_lists[i], dtype=float)
        b = np.asarray(b_lists[i], dtype=float)
        d[i] = float(np.sum(a > b) - np.sum(b > a))

    decisive = d != 0.0
    if not np.any(decisive):
        return 0.5

    dd = d[decisive]
    rr = resp[decisive]
    # Option favored by the raw tally: A (response 0) if d > 0,
    # otherwise B (response 1).
    agree = np.where(dd > 0.0, rr == 0.0, rr == 1.0)
    return float(np.mean(agree))
```

**Observed (real) value:** 0.6170 (var=0.0034)
**Previous candidate values (this loop):**
  - iter 1: 0.6570 (var=0.0037) (Δ vs real +0.0400)
  - iter 2: 0.6470 (var=0.0053) (Δ vs real +0.0300)
  - iter 3: 0.6396 (var=0.0044) (Δ vs real +0.0226)
  - iter 4 (most recent): 0.6404 (var=0.0040) (Δ vs real +0.0233)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)
- pi_7: 0.6267 (var=0.0043)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Hard-ceiling test on the |d|=1 family (FACT 1 of the design).

    Metric = proportion of choices that agree with the UNWEIGHTED tally
    count-winner, restricted to trials whose raw tally margin is exactly
    +/-1 (the 8 trial types t1-t8; 48 of the 96 trials per subject).

    Tallying (any beta >= 1, any epsilon) predicts ~1 - epsilon/2 here,
    i.e. >= ~0.85 for any moderate lapse. The validity-weighted
    competitor is structurally capped: its choice probability on a |d|=1
    trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign diff;
    the mean of |m| over the 8 |d|=1 trial types is <= 1 for every
    gamma in [0.05, 0.7] (equality only at gamma=0), and beta <= 0.55,
    so by concavity of tanh its mean agreement on this family is
    <= 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE parameter box.
    """
    def _mat(col):
        vals = []
        for v in col:
            if isinstance(v, np.ndarray):
                vals.append(v)
            else:
                vals.append(list(v))
        return np.asarray(vals, dtype=float)

    A = _mat(data["option_a_ratings"])
    B = _mat(data["option_b_ratings"])
    resp = np.asarray(data["response"], dtype=float)

    if A.ndim != 2 or A.shape[0] == 0 or A.shape != B.shape:
        return 0.5

    # Raw (unweighted) tally margin per trial.
    d = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # The |d| = 1 family: 8 unique pairs x 6 reps = 48 trials/subject.
    mask = np.abs(d) == 1
    if mask.sum() < 10:
        # Defensive fallback: use all non-tied trials.
        mask = d != 0
    if mask.sum() == 0:
        return 0.5

    dm = d[mask]
    rm = resp[mask]
    # Count-winner is A when d>0 (response 0), B when d<0 (response 1).
    consistent = np.where(dm > 0, rm == 0, rm == 1)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.5133 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.5400 (var=0.0044) (Δ vs real +0.0267)
  - iter 2: 0.5387 (var=0.0048) (Δ vs real +0.0254)
  - iter 3: 0.5367 (var=0.0036) (Δ vs real +0.0233)
  - iter 4 (most recent): 0.5188 (var=0.0057) (Δ vs real +0.0054)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)
- pi_7: 0.5425 (var=0.0051)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Chance-cue sensitivity on "chance-only" (Z-family) trials.
    # Design validities: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]
    # -> features 0-3 are diagnostic, features 4-7 are the four v=0.5 experts.
    # A Z-family trial is one where ALL diagnostic cues tie, so the pair is
    # discriminated ONLY by the 50%-accurate experts. On such a trial let
    # k = net number of chance cues won by A (signed). The competing theory
    # (llr-rectified weights) annihilates v=0.5 cues exactly, so p(A)=0.5
    # regardless of k -> subjects match the k-implied side at chance (metric 0).
    # My theory gives v=0.5 cues weight 0.5^gamma, so choices should follow
    # sign(k) above chance (metric > 0).
    DIAG = [0, 1, 2, 3]
    CHANCE = [4, 5, 6, 7]

    a_ratings = data["option_a_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    b_ratings = data["option_b_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    responses = data["response"].to_numpy()

    correct = []
    for av, bv, r in zip(a_ratings, b_ratings, responses):
        if av.shape[0] != 8 or bv.shape[0] != 8:
            continue
        # Keep only trials where every diagnostic cue ties (chance-only trials)
        if not np.array_equal(av[DIAG], bv[DIAG]):
            continue
        # Net chance-cue advantage for A
        k = int(np.sign(av[CHANCE] - bv[CHANCE]).sum())
        if k == 0:
            continue
        chose_a = (int(r) == 0)  # response: 0 = A, 1 = B
        followed = (k > 0 and chose_a) or (k < 0 and not chose_a)
        correct.append(1.0 if followed else 0.0)

    if len(correct) == 0:
        return 0.0
    # 2 * (accuracy - 0.5): 0 = chance-level following of chance cues,
    # positive = subjects systematically follow the 50%-accurate experts.
    return 2.0 * (float(np.mean(correct)) - 0.5)
```

**Observed (real) value:** -0.1683 (var=0.0124)
**Previous candidate values (this loop):**
  - iter 1: -0.1925 (var=0.0175) (Δ vs real -0.0242)
  - iter 2: -0.1333 (var=0.0312) (Δ vs real +0.0350)
  - iter 3: -0.1467 (var=0.0169) (Δ vs real +0.0217)
  - iter 4 (most recent): -0.1883 (var=0.0166) (Δ vs real -0.0200)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)
- pi_7: -0.1425 (var=0.0147)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Strong-Side Choice Rate on Multi-Weak Conflict trials (SSR).

    Selects the five conflict cells in which a high-validity side (99%
    experts, or 75% experts for R3) faces TWO OR MORE opposing weak
    experts: K2 (1x99 vs 2 weak), K3 (1x99 vs 3 weak), M3 (2x99 vs 3
    weak), M4 (2x99 vs 4 weak), R3 (2x75 vs 3 weak) -- 5 unique pairs
    x 2 mirrors x 4 reps = 40 trials per subject. Returns the
    proportion of trials on which the subject chose the strong side.
    """
    hits = 0
    n = 0

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in a]
        b = [int(x) for x in b]
        if len(a) < 8 or len(b) < 8:
            continue

        # Validity-class vote balances (features 0-1 = 0.99 experts,
        # 2-3 = 0.75 experts, 4-7 = weak experts).
        d_hi = (a[0] + a[1]) - (b[0] + b[1])
        d_mid = (a[2] + a[3]) - (b[2] + b[3])
        d_lo = (a[4] + a[5] + a[6] + a[7]) - (b[4] + b[5] + b[6] + b[7])

        # K2/K3/M3/M4: super experts alone on one side (no 75s involved)
        # opposing >= 2 weak experts.
        if d_mid == 0 and d_hi != 0 and abs(d_lo) >= 2:
            target_is_a = d_hi > 0
        # R3: two 75% experts vs three weak experts.
        elif d_hi == 0 and abs(d_mid) == 2 and abs(d_lo) == 3:
            target_is_a = d_mid > 0
        else:
            # Excludes K1 (|d_lo|=1), R1, W2/W3 (no strong side),
            # W4a (|d_mid|=1), G1 (mixed 99+75 side), X1 (all tied).
            continue

        n += 1
        chose_a = (int(r) == 0)
        if chose_a == target_is_a:
            hits += 1

    if n == 0:
        return 0.5
    return hits / n

```

**Observed (real) value:** 0.1180 (var=0.0074)
**Previous candidate values (this loop):**
  - iter 1: 0.1475 (var=0.0030) (Δ vs real +0.0295)
  - iter 2: 0.1345 (var=0.0021) (Δ vs real +0.0165)
  - iter 3: 0.1375 (var=0.0034) (Δ vs real +0.0195)
  - iter 4 (most recent): 0.1080 (var=0.0016) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)
- pi_7: 0.1235 (var=0.0028)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Super-expert contrarianism index.

    Over all trials in which the feature-0 expert (the 99%-validity
    expert) discriminates between the two options (a[0] != b[0]),
    compute the proportion of choices that go AGAINST the option
    endorsed by that expert.

    Advocated theory (inverted trust): the 99% expert's endorsement is
    anti-diagnostic (w(0.99) ~ -1 trusted-cue unit), so subjects
    predominantly choose AGAINST it -> index ~0.8-0.9.

    Competing theory (flattened diagnosticity): the 99% expert gets the
    LARGEST positive weight, so subjects predominantly FOLLOW it ->
    index ~0.2-0.35.
    """
    n_anti = 0
    n_sel = 0
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(ra, dtype=int).ravel()
        b = np.asarray(rb, dtype=int).ravel()
        if a.size == 0 or b.size == 0 or a[0] == b[0]:
            continue  # expert 0 does not discriminate on this trial
        # Option index endorsed by the 99%-validity (feature-0) expert.
        endorsed = 0 if a[0] == 1 else 1
        r = int(resp)
        n_sel += 1
        if r != endorsed:
            n_anti += 1
    if n_sel == 0:
        return 0.5
    return float(n_anti) / float(n_sel)

```

**Observed (real) value:** 0.5796 (var=0.0066)
**Previous candidate values (this loop):**
  - iter 1: 0.5276 (var=0.0060) (Δ vs real -0.0520)
  - iter 2: 0.5031 (var=0.0041) (Δ vs real -0.0764)
  - iter 3: 0.5000 (var=0.0049) (Δ vs real -0.0796)
  - iter 4 (most recent): 0.4836 (var=0.0045) (Δ vs real -0.0960)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)
- pi_7: 0.5787 (var=0.0053)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(follow the 98% expert | 98%-vs-90% conflict trials).

    A 'D-cell' trial is one where the top expert (v=0.98, feature 0) and the
    second expert (v=0.90, feature 1) endorse OPPOSITE options, while the
    third expert (v=0.75, feature 2) is tied. On these trials the metric is
    the fraction of choices that follow the option endorsed by the 98%
    expert. FDEI (compressed diagnosticity) treats 0.98 and 0.90 as nearly
    exchangeable, so this should be ~0.50 (coin flip). SITT (inverted trust)
    assigns the 98% expert a large NEGATIVE weight, so subjects should
    systematically choose AGAINST the 98% endorsement (~0.05-0.10).
    """
    if data is None or len(data) == 0:
        return 0.5

    follow_flags = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.ndim != 1 or a.shape[0] < 3 or a.shape != b.shape:
            continue

        d0 = a[0] - b[0]  # 98% expert's relative endorsement of A
        d1 = a[1] - b[1]  # 90% expert's relative endorsement of A
        d2 = a[2] - b[2]  # 75% expert must be tied (no interference)

        # Need a genuine 98-vs-90 conflict with the 75% expert tied.
        if d0 == 0 or d1 == 0 or d2 != 0:
            continue
        if np.sign(d0) == np.sign(d1):
            continue

        endorsed_a = d0 > 0          # 98% expert endorses option A
        chose_a = (int(row["response"]) == 0)
        follow_flags.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follow_flags) == 0:
        return 0.5
    return float(np.mean(follow_flags))
```

**Observed (real) value:** 0.4625 (var=0.0103)
**Previous candidate values (this loop):**
  - iter 1: 0.4238 (var=0.0121) (Δ vs real -0.0388)
  - iter 2: 0.4662 (var=0.0119) (Δ vs real +0.0037)
  - iter 3: 0.4562 (var=0.0160) (Δ vs real -0.0063)
  - iter 4 (most recent): 0.4587 (var=0.0167) (Δ vs real -0.0038)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)
- pi_7: 0.4813 (var=0.0186)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-super-expert choice rate on the 10 pre-registered 'cliff conflict'
    trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m).

    On each of these trials exactly one option is endorsed by the
    99%-validity experts (features 0 and 1) while the rival option is
    endorsed only by experts of validity <= 0.98, and the 99%-endorsed
    side has no 0.90-class support (this excludes the G1/G1m agreement
    anchors).  The metric is the fraction of these trials on which the
    subject chooses the option NOT endorsed by the 99% experts.

    BSST (sharp distrust cliff between 0.98 and 0.99, w(0.99) ~ -1.1
    trusted-cue units, w(0.98) ~ +0.6, w(0.90) ~ +0.9, w(0.50) ~ -0.23):
    the 99%-endorsed side is disfavored on every one of these cells, with
    predicted anti-99% choice rates of roughly S1 .86-.90, S2 .90-.95,
    D1 .72-.79, D2 .85-.94, V1 .70-.76  ->  overall ~0.84.

    FDEI (delta ~ 0.15 compression makes 0.98 and 0.99 experts nearly
    exchangeable, w99 - w98 in [-0.21, +0.15]): it is locked near 0.50 on
    the D cells, mildly PRO-99% on the S cells (the 99-side also carries
    a tiny validity edge), and strongly pro-99% on V1 (three chance cues
    are annihilated exactly)  ->  overall anti-99% rate ~0.30-0.53,
    center ~0.44.  Sign-disjoint from BSST with a ~0.40 mean gap.
    """

    def as_tuple(cell):
        try:
            return tuple([int(x) for x in cell])
        except Exception:
            return None

    # key: (A ratings, B ratings) -> favored option (0 = A, 1 = B),
    # where 'favored' is the side NOT endorsed by the 99% experts.
    TRIALS = {
        # S1 / S1m  (A or B wins both 0.99; rival wins both 0.90)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0), (0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0)): 1,
        ((0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0)): 0,
        # S2 / S2m  (one side wins 0.99+0.99+0.98; rival 0.90+0.90+0.55)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)): 1,
        ((0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1)): 0,
        # D1 / D1m  (one side wins a 0.99; rival wins a 0.98)
        ((0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)): 1,
        # D2 / D2m  (one side wins both 0.99; rival wins both 0.98)
        ((0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)): 1,
        # V1 / V1m  (one side wins a 0.99; rival wins all three 0.50)
        ((1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1)): 1,
        ((0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)): 0,
    }

    hits = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        ka = as_tuple(a)
        kb = as_tuple(b)
        if ka is None or kb is None:
            continue
        fav = TRIALS.get((ka, kb))
        if fav is None:
            continue
        try:
            ri = int(r)
        except Exception:
            continue
        hits.append(1.0 if ri == fav else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5600 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1: 0.5235 (var=0.0033) (Δ vs real -0.0365)
  - iter 2: 0.5490 (var=0.0057) (Δ vs real -0.0110)
  - iter 3: 0.5555 (var=0.0057) (Δ vs real -0.0045)
  - iter 4 (most recent): 0.5585 (var=0.0066) (Δ vs real -0.0015)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)
- pi_7: 0.5715 (var=0.0054)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Follow-the-99%-expert rate": across all trials where the single
    # 0.99-validity expert (feature index 0 in the fixed validity vector
    # [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .]) discriminates
    # between A and B, compute the proportion of choices that go to the
    # option the 0.99 expert rated 1 (i.e., the option the top expert
    # "endorsed").
    #
    # FDEI: w(0.99) is the LARGEST positive weight in the whole vector
    # (its compression never inverts the validity ordering), so the
    # 0.99-favored option is followed above chance on essentially every
    # such trial -> metric ~ 0.60-0.65.
    # BSST: w(0.99) is strongly NEGATIVE (~ -1.1 trusted-cue units, the
    # distrust cliff), so the 0.99-favored option is chosen BELOW chance
    # on every such trial (the background cues are balanced/weak in this
    # design) -> metric ~ 0.15-0.35.
    # The sign of (metric - 0.5) alone separates the two theories on
    # every trial in the 0.99-discriminating set.

    TOP = 0  # index of the 0.99-validity expert, fixed by the design

    resp = data['response'].to_numpy()
    follows = []

    for i in range(len(data)):
        a_raw = data['option_a_ratings'].iloc[i]
        b_raw = data['option_b_ratings'].iloc[i]
        try:
            av = np.asarray(list(a_raw), dtype=int).ravel()
            bv = np.asarray(list(b_raw), dtype=int).ravel()
        except Exception:
            continue
        if av.size <= TOP or bv.size <= TOP:
            continue

        d = int(av[TOP]) - int(bv[TOP])
        if d == 0:
            continue  # 0.99 expert tied on this trial -> uninformative

        favored_a = d > 0          # option the 0.99 expert endorsed
        chose_a = (int(resp[i]) == 0)
        follows.append(1.0 if (chose_a == favored_a) else 0.0)

    if len(follows) == 0:
        return 0.5  # neutral fallback (should not occur: 8 trials x 4 reps)

    return float(np.mean(follows))
```

**Observed (real) value:** 0.5988 (var=0.0026)
**Previous candidate values (this loop):**
  - iter 1: 0.6212 (var=0.0105) (Δ vs real +0.0225)
  - iter 2: 0.5894 (var=0.0077) (Δ vs real -0.0094)
  - iter 3: 0.5806 (var=0.0075) (Δ vs real -0.0181)
  - iter 4 (most recent): 0.5769 (var=0.0099) (Δ vs real -0.0219)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)
- pi_7: 0.6094 (var=0.0057)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Theory-discriminant index for the 10-expert panel
    validities = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50].

    Both theories reduce to a FIXED weight vector over the 10 experts on
    this panel. At each theory's parameter-box midpoint we compute its
    predicted p(choose A) for the observed trial, take the difference
    d_t = p_CREDT(t) - p_FDEI(t), and average d_t * y_t over all trials,
    where y_t = +1 if the subject chose A and -1 if B.

    E[metric] = (1/T) * sum_t d_t * (2*p_true(t) - 1).
    If CREDT generated the data this is strongly POSITIVE (~ +0.045);
    if FDEI generated the data it is NEGATIVE (~ -0.022). The contrast
    is driven by the cells where the boxes are disjoint: the Z-family
    (chance-cue ladders: CREDT 0.59-0.74 vs FDEI exactly 0.50), the
    0.99-expert trials (CREDT ~0.28-0.36 vs FDEI ~0.55-0.60), the
    X-family sign flips, and K1 (four trusted experts vs four
    coin-flippers: CREDT ~0.50 vs FDEI ~0.78).
    """
    validities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50])

    # ---------- CREDT weights at parameter-box midpoints ----------
    w_p, v_lo, v_hi, kappa_lo = 0.90, 0.53, 0.9815, 0.20
    g0, s_gap = 0.043, 0.013
    rho, d_w, lambda_ext = 1.36, 0.40, 0.40
    g_star, s_l, s_r = 0.048, 0.012, 0.033
    c_half, s_c, c_amp = 0.36, 0.06, 1.29
    beta_c, eps_c = 0.42, 0.05

    v = np.clip(validities, 0.5, 1.0)
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second
    outlier = v_top > v_hi
    if outlier:
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * ((gap - g0) / s_gap)))
    else:
        sig_gap = 0.0
    gate = (lambda_ext + (1.0 - lambda_ext) * sig_gap) if outlier else 0.0

    w_c = np.full(10, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        w_fl = -kappa_lo + (w_p + kappa_lo) * np.clip(v[flank] - 0.5, 0.0, None) / (v_lo - 0.5)
        w_c[flank] = (1.0 - gate) * w_fl + gate * w_p
    if outlier:
        c_frac = float(flank.mean())
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * (c_frac - c_half) / s_c))
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = w_p - d_w * (v_top - v_hi) / (1.0 - v_hi) - rho * (P + C * sig_gap)
        w_c[v >= v_top - 1e-9] = w_top

    # ---------- FDEI weights at parameter-box midpoints ----------
    delta, tau = 0.15, 0.02
    beta_f, eps_f = 0.32, 0.025
    vv = np.clip(validities, 0.5, 1.0 - 1e-9)
    llr = np.log(vv / (1.0 - vv))
    w_f = np.power(np.maximum(llr - tau, 0.0), delta)

    def _p_choose_a(s, beta, eps):
        # low-gain softmax over [s, -s] plus uniform lapse
        z = beta * s
        p_core = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - eps) * p_core + eps * 0.5

    total = 0.0
    n = 0
    for a_cell, b_cell, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        votes = np.sign(a - b)
        s_c = float(np.dot(w_c, votes))
        s_f = float(np.dot(w_f, votes))
        p_c = _p_choose_a(s_c, beta_c, eps_c)
        p_f = _p_choose_a(s_f, beta_f, eps_f)
        d_t = p_c - p_f
        y = 1.0 if int(resp) == 0 else -1.0
        total += d_t * y
        n += 1

    if n == 0:
        return 0.0
    return float(total / n)
```

**Observed (real) value:** 0.0206 (var=0.0003)
**Previous candidate values (this loop):**
  - iter 1: 0.0502 (var=0.0003) (Δ vs real +0.0297)
  - iter 2: 0.0468 (var=0.0002) (Δ vs real +0.0262)
  - iter 3: 0.0527 (var=0.0004) (Δ vs real +0.0321)
  - iter 4 (most recent): 0.0655 (var=0.0003) (Δ vs real +0.0449)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0758 (var=0.0002)
- pi_4: -0.0398 (var=0.0003)
- pi_1: -0.0446 (var=0.0003)
- pi_2: 0.0431 (var=0.0003)
- pi_3: 0.0148 (var=0.0004)
- pi_5: -0.0288 (var=0.0001)
- pi_6: -0.0227 (var=0.0002)

### Experiment 12
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Super-expert deference rate.

    Among all trials on which the highest-validity expert (feature 0, the
    99% expert) discriminates between option A and option B, return the
    fraction of choices that FOLLOW that expert's recommendation (i.e.
    chose the option the 99% expert rated 1).

    FDEI (flattened-diagnosticity integration) gives the 0.99 expert its
    largest positive weight, so it predicts deference well above chance
    (~0.57 overall, ~0.69 on isolated super-expert trials, but BELOW 0.5
    when a 0.94+0.65+0.50 coalition opposes it).

    CREDT, on this panel (0.99 sitting 0.05 above the 0.94 rung with a
    partially chance-laden scale), assigns the panel-max expert a large
    NEGATIVE weight, so it predicts systematic ANTI-deference (~0.21).
    """
    df = data.reset_index(drop=True)

    def _first_rating(cell):
        # Cells may be list / tuple / np.ndarray of 0/1 ints, or a string
        # serialization like "[1, 0, 1, ...]". Robustly extract element 0.
        if isinstance(cell, str):
            s = cell.strip().strip("[]")
            parts = s.replace(",", " ").split()
            return int(float(parts[0]))
        seq = list(cell)
        return int(seq[0])

    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]

    n = 0
    follow = 0
    for i in range(len(df)):
        try:
            a0 = _first_rating(a_col.iloc[i])
            b0 = _first_rating(b_col.iloc[i])
        except Exception:
            continue
        # Skip trials where the 99% expert does not discriminate.
        if a0 not in (0, 1) or b0 not in (0, 1) or a0 == b0:
            continue
        # Option endorsed by the 99% expert: 0 = A, 1 = B.
        winner = 0 if a0 == 1 else 1
        resp = int(r_col.iloc[i])
        n += 1
        if resp == winner:
            follow += 1

    if n == 0:
        return 0.5
    return float(follow) / float(n)
```

**Observed (real) value:** 0.4525 (var=0.0028)
**Previous candidate values (this loop):**
  - iter 1: 0.3958 (var=0.0051) (Δ vs real -0.0567)
  - iter 2: 0.4554 (var=0.0049) (Δ vs real +0.0029)
  - iter 3: 0.4537 (var=0.0042) (Δ vs real +0.0012)
  - iter 4 (most recent): 0.4163 (var=0.0044) (Δ vs real -0.0363)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6017 (var=0.0039)
- pi_7: 0.2367 (var=0.0031)
- pi_1: 0.8358 (var=0.0153)
- pi_2: 0.6479 (var=0.0042)
- pi_3: 0.5867 (var=0.0071)
- pi_5: 0.1121 (var=0.0017)
- pi_6: 0.2667 (var=0.0041)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Bounded Outlier-Distrust Tallying (BODT), engaged-shoulder edition. Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders gap ~0.01; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. This round's calibration claims: (a) the distrust bump's RIGHT SHOULDER is wide enough to cover the moderate-gap regime (gap ~0.09, a 0.99 expert floating above a 0.90 rung on a panel larded with chance cues), which the previous calibration left in a dead zone between the peaked bump and the far gate — this is where super-expert contrarianism actually lives (Exp7); (b) the composition route's far gate is lowered just enough to partially engage on such chance-laden moderate-gap panels while remaining essentially closed at gap ~0.05 (Exp12 stays in its prescribed 0.45-0.50 deference band) and fully open only at gap >= ~0.15 (Exp6's far-outlier collapse preserved); (c) the chance-flank magnitude kappa_lo is raised to restore the anti-following of coin-flip experts under the now-softer softmax gain (Exp5), rather than re-sharpening beta; (d) the dense-ladder calibration (s_l, g_star), the peak magnitude (rho_m, w_floor, d_w), and the gain/lapse balance (beta, epsilon) are held exactly at their accepted values — nine of twelve experiments are already at target and must not be disturbed. Choice is a low-gain softmax plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Bounded Outlier-Distrust Tallying (BODT), engaged-shoulder edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above
    #       0.98, so a 98% expert stays INSIDE the plateau) get
    #       approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the plateau at v_lo). When an above-ceiling claim is
    #       present, the flank is gated: gate = lambda_ext +
    #       (1-lambda_ext)*sig_gap, so chance cues are COUNTED
    #       (weight -> w_p) rather than inverted.
    # (iii) BOUNDED OUTLIER DISTRUST (the core BODT claim): the
    #       discount of the panel-max expert is a SATURATING
    #       SHRINKAGE toward a floor, not an unbounded sign flip:
    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)
    #       where P is a PEAKED bump in the rung gap (maximal at
    #       gap ~ g_star ~ 0.05 -- spurious precision; a SMALL but
    #       nonzero discount on dense ladders gap ~ 0.01 via the
    #       widened left width s_l; receding for large edges on
    #       coherent panels via the right width s_r, now WIDENED so
    #       the shoulder covers the moderate-gap regime gap ~ 0.09).
    #       Because of the exponential saturation, w_top_mod never
    #       leaves [w_floor, w_p].
    #  (iv) COMPOSITION-AMPLIFIED ROUTE: an additive discount
    #         comp = c_amp * sig_gap * far(gap) * gate(c_frac)
    #       where far(gap) is a logistic gate whose center g_far is
    #       now LOW enough (~0.10-0.12) to partially engage on
    #       chance-laden moderate-gap panels (gap ~ 0.09 with
    #       c_frac > c_half), while remaining essentially closed at
    #       gap ~ 0.05 and fully open only for far outliers
    #       (gap >= ~0.15) on majority-chance panels.
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. Choice is a low-gain softmax (inverse temperature
    # beta) over the two mirrored scores, plus a uniform lapse
    # (epsilon).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    w_floor = float(parameters["w_floor"])
    rho_m = float(parameters["rho_m"])
    d_w = float(parameters["d_w"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- Panel structure: top validity and its gap to the
    # next-highest DISTINCT validity in the panel. ----
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    # If every expert shares the top validity, the panel offers no
    # corroboration ladder; treat the next rung as chance (0.5).
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel stays inside the trusted plateau; only
    # 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): plateau with a consensus-gated chance flank. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted (weight -> w_p) rather than inverted.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iii) + (iv): bounded outlier distrust, with a
    # composition-amplified route that now partially engages on
    # chance-laden MODERATE-gap panels. ----
    if outlier:
        # (a) Peaked bump in the uncorroborated edge: maximal near
        # gap ~ g_star (spurious precision), a SMALL nonzero
        # discount on dense ladders (widened s_l), receding for
        # large edges on coherent panels (widened right shoulder
        # s_r covers the moderate-gap regime gap ~ 0.09).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every P; even at peak
        # distrust the outlier is only demoted toward the floor
        # (~ -0.25..-0.30 plateau units), never unboundedly flipped.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))

        # (c) Composition-amplified route: a logistic far gate
        # centered at g_far ~ 0.10-0.12, so it PARTIALLY engages on
        # chance-laden moderate-gap panels (gap ~ 0.09, c_frac >
        # c_half) and is fully open only for far outliers
        # (gap >= ~0.15) on majority-chance panels.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        far = _sigmoid((gap - g_far) / s_far)
        comp = c_amp * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of isolated 0.99 super-experts).
        w[v >= v_top - 1e-9] = w_top

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.24, 0.27]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.018, 0.024]
- s_r: [0.045, 0.055]
- w_floor: [-0.30, -0.25]
- rho_m: [0.9, 1.2]
- d_w: [0.15, 0.22]
- g_far: [0.10, 0.12]
- s_far: [0.025, 0.035]
- c_half: [0.32, 0.35]
- s_c: [0.05, 0.07]
- c_amp: [2.0, 2.3]
- beta: [0.34, 0.39]
- epsilon: [0.05, 0.07]
- validities: validities

`rationale`: Minimal-diff edit of the ACCEPTED iter-2 BODT base, applying the critic's three in-family adjustments verbatim; the mechanism (plateau, consensus gate, saturating shrinkage, composition route, softmax+lapse) is byte-for-byte unchanged — only three parameter ranges move. (1) ENGAGE MODERATE-GAP DISTRUST FOR EXP7 (primary fix, residual -0.076, the largest in the battery). The critic correctly diagnosed that the iter-2 composition-route strengthening was a NO-OP on Exp7: its panel is a 0.99 expert sitting gap=0.09 above a 0.90 rung with c_frac=0.375, below the old far gate (g_far 0.13-0.17, far(0.09) ~ 0.12), and the old right shoulder gave P(0.09) only ~0.44. Two levers move, exactly as prescribed: s_r 0.030-0.036 -> 0.045-0.055, so at box midpoint P(gap=0.09) ~ exp(-0.5*(0.042/0.05)^2) ~ 0.70 and the saturating shrinkage alone pulls w_top_mod from ~0.45 down to ~0.29; and g_far 0.13-0.17 -> 0.10-0.12, so far(0.09) ~ 0.3-0.4 and, with c_frac 0.375 > c_half 0.32-0.35 (comp_gate ~ 0.6-0.7) and sig_gap ~ 1, the composition route contributes ~0.4-0.6 additional discount — pushing w_top on Exp7's panel to roughly -0.3..-0.4 and moving anti-deference from 0.503 toward the real 0.580. Tradeoff guard, per the critic: at Exp12's gap=0.05 the far gate stays essentially closed (far(0.05) ~ 0.08-0.12, and its c_frac sits below c_half so comp_gate is small), so the comp contribution there is ~0.05 or less — Exp12's deference should remain in the prescribed 0.45-0.50 band; if simulation dips below 0.45 the box's upper g_far end (0.12) plus the untouched rho_m/w_floor calibration is the intended fallback, and I deliberately did NOT trim rho_m, preserving the peak calibration that put Exp12 dead center last round. (2) RESTORE THE CHANCE-FLANK MAGNITUDE (Exp5, secondary): kappa_lo 0.21-0.24 -> 0.24-0.27 to compensate for the softer gain (beta ~ 0.36) that diluted anti-following of the four v=0.5 experts on the ungated [0.95, 0.85, 0.75, 0.65, 0.5x4] panel — moving the metric from -0.133 back toward the real -0.168. This is the correct lever rather than re-sharpening beta: the critic explicitly warned against oscillating on beta/epsilon now that Exp3/Exp4/Exp8 are at target, and a stronger kappa_lo may also trim Exp2's residual over-positive tally signature (+0.034) since anti-followed near-chance cues reduce the effective tally slope. (3) EVERYTHING ELSE HELD: s_l, g_star, rho_m, w_floor, d_w, c_amp, c_half, beta, epsilon are all unchanged — Exp9 (0.549 vs 0.560), Exp10 (0.589 vs 0.599), Exp12 (0.455 vs 0.453), Exp6 (0.135 vs 0.118), Exp8 (0.466 vs 0.463), Exp3 (0.647 vs 0.617), Exp4 (0.539 vs 0.513), Exp1 (-0.238 vs -0.218) are all at or near target and the dense-ladder and peak calibrations that got them there are protected. The only side-effects I accept are the small drifts the critic pre-authorized: Exp6's far-outlier panel (gap=0.24) is insensitive to g_far's lowering (far was already ~1) and to s_r (P(0.24) ~ 0 either way); Exp11's tiny +0.026 residual may drift slightly with kappa_lo, which is acceptable given its minuscule variance and correct sign. Expected net effect: the Exp7 residual shrinks from -0.076 toward ~0.01-0.02, Exp5 moves ~0.03 toward the real value, and the other ten experiments hold — pushing the aggregate loss below the 0.0556 floor.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: BODT is a clear improvement over the CREDT base (pi_7) and was rightly accepted: the flagship Exp12 spurious-precision panel moved from 0.237 to 0.396 (real 0.453), Exp11's contrast signature improved from 0.076 to 0.050 (real 0.021), and no experiment's sign prediction flipped. The bounded-shrinkage edit did what the arbiter prescribed in direction. However, the calibration is not yet right, and the residuals are SYSTEMATIC, not noise: (a) Exp12 deference is 0.396 vs real 0.453 — below the arbiter's explicitly prescribed 0.45-0.50 band, so the moderate-regime distrust at gap~0.05 is still slightly too strong (w_top lands near -0.15 to -0.2 after the extremity term when it should be near 0 to -0.1); (b) Exp7 anti-deference dropped from pi_7's excellent 0.579 to 0.528 (real 0.580) and Exp9 anti-99 dropped from 0.572 to 0.524 (real 0.560) — the elevated lapse (epsilon 0.06-0.09) plus the softened distrust is compressing the anti-99 signals toward 0.5; (c) Exp10 (dense ladder) is at 0.621 vs real 0.599, so dense-ladder panels actually want slightly MORE distrust than the model currently gives — note this is the same direction as Exp9, so both dense-gap experiments consistently ask for a small anti-99 push at gap~0.01, where the peaked bump P is currently ~0; (d) Exp3 (0.657 vs 0.617), Exp2 (0.110 vs 0.077) and Exp4 (0.540 vs 0.513) show mild residual over-polarization on non-outlier panels, i.e., the global gain/lapse balance is a touch too sharp. The needed correction is a coherent re-balancing INSIDE the BODT family: reduce distrust specifically at gap~0.05, increase it slightly at dense-ladder gaps, strengthen the composition route, and trade a little beta against epsilon.
Rationale: Keep the BODT structure exactly as is (plateau, consensus gate, saturating shrinkage, composition route, softmax+lapse) — only recalibrate. Four targeted, in-family adjustments: (1) WEAKEN PEAK DISTRUST AT gap~0.05 to hit the prescribed 0.45-0.50 deference on Exp12: lower rho_m toward ~0.9-1.2 and/or raise w_floor toward -0.25..-0.30, and trim d_w toward 0.15-0.22 (the extremity term currently subtracts ~0.10-0.15 on top of the saturated shrinkage, pushing w_top too negative on the 0.99-over-0.94 panel). (2) ADD A SMALL DENSE-LADDER DISTRUST so Exp9 (anti-99 0.524 -> ~0.56) and Exp10 (0.621 -> ~0.60) both improve: currently P(gap=0.01) ~ 0 because s_l is tiny; widen s_l toward ~0.018-0.024 (or shift g_star slightly below 0.05) so dense-gap outliers receive a small bump while the rho_m reduction in (1) more than cancels it at gap~0.05 — this is exactly the sign pattern the two dense-gap experiments jointly demand. (3) RESTORE THE COMPOSITION-ROUTE SIGNAL for Exp7 (0.528 vs real 0.580): raise c_amp modestly (toward 2.0-2.3) or lower c_half toward 0.32-0.35 so the far-outlier/chance-panel distrust survives the elevated lapse; do not touch the far gate (it correctly protects Exp12). (4) RE-BALANCE GAIN VS LAPSE: the over-polarized non-outlier panels (Exp3 0.657 vs 0.617, Exp2 0.110 vs 0.077, Exp4 0.540 vs 0.513) want slightly lower beta (toward 0.34-0.39), while the compressed anti-signals (Exp7/Exp9) want epsilon trimmed back toward 0.05-0.07 — lowering beta weakens the anti-99 rates a little, so compensate via the stronger core distrust signals from (2) and (3) rather than via the lapse. These are all minor parameter/shape tweaks within the prescribed BODT mechanism; no structural change is needed. The current candidate is the new accepted base, so the next iteration should build directly on it with these recalibrations.

**Outcome of this advice:** iter 1 candidate loss=0.0730 -> iter 2 candidate loss=0.0556 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 recalibration was a clear success overall (loss 0.0730 -> 0.0556) and most of my iter-1 advice landed as intended: Exp12 deference moved 0.396 -> 0.455 (real 0.453, dead center of the arbiter's 0.45-0.50 band), Exp9 anti-99 0.524 -> 0.549 (real 0.560), Exp10 0.621 -> 0.589 (real 0.599), Exp3 0.657 -> 0.647 and Exp4 0.540 -> 0.539 (both relaxing toward real), Exp8 essentially perfect (0.466 vs 0.463), Exp6 improved to 0.135 (real 0.118), Exp1 improved to -0.238 (real -0.218). The dense-ladder distrust (s_l widening) and the gain/lapse rebalance did exactly what they were supposed to do. However, two residuals are systematic and one of them moved in the WRONG direction despite my advice: (a) Exp7 anti-deference FELL from 0.528 to 0.503 (real 0.580) — now the largest residual in the whole battery (-0.076). Diagnosis: the composition-route strengthening (c_amp up, c_half down) was a NO-OP on Exp7, because Exp7's panel is a 0.99 expert sitting ~0.09 above a 0.90 second rung with 3/8 chance cues — its rung gap is well below the far gate g_far (0.13-0.17), so `far(gap)` ~ 0.12 and the composition discount never engages. The only change that actually acted on Exp7 was the beta cut (0.40 -> 0.36), which compresses any anti-99 signal toward 0.5. The proposer's rationale assumed Exp7 had gap ~0.24 (that is Exp6's structure, not Exp7's). (b) Exp5 anti-chance-following weakened from -0.193 to -0.133 (real -0.168) with variance jumping to 0.0312 — the same beta cut diluted the kappa_lo-driven anti-following of chance cues on the ungated [0.95, 0.85, 0.75, 0.65, 0.5x4] panel. (c) Exp2 remains over-polarized (+0.034, essentially unmoved by the beta cut), and Exp11's contrast signature is still a touch too CREDT-like (+0.026). The mechanism family is right and now well-calibrated on nine of twelve experiments; what remains is a diagnosable engagement failure of the moderate-gap distrust on chance-laden panels (Exp7) plus a chance-flank magnitude that no longer matches the softer gain.
Rationale: Keep the BODT mechanism byte-for-byte (plateau, consensus gate, saturating shrinkage, composition route, softmax+lapse); only parameter ranges move. Three targeted, in-family adjustments: (1) ENGAGE MODERATE-GAP DISTRUST FOR EXP7 (primary fix, current residual -0.076): the distrust bump's right shoulder currently gives P(gap~0.09) only ~0.44 (s_r 0.030-0.036 with g_star ~0.048), and the composition route is blocked by the far gate. Widen s_r to ~0.045-0.055 so the right shoulder covers gap~0.09 with P ~ 0.6-0.7 while P(gap~0.05) stays saturated at ~1 and the dense-ladder left side (s_l, gap~0.01) is untouched; additionally lower g_far modestly to ~0.10-0.12 so the composition route partially fires on chance-laden moderate-gap panels (Exp7: c_frac 0.375 > c_half). Guard the tradeoff: Exp12's gap=0.05 panel must stay in the 0.45-0.50 deference band — at g_far ~0.11 its far-gate value is only ~0.1, giving a small comp (~0.15); if simulated Exp12 deference dips below 0.45, keep g_far at the higher end and lean on s_r alone, or trim rho_m slightly (toward 0.9) to compensate. Do NOT touch s_l, g_star, or the dense-ladder calibration — Exp9/Exp10/Exp12 are all at target and the previous widening worked. (2) RESTORE THE CHANCE-FLANK MAGNITUDE (Exp5, secondary): the beta cut diluted anti-chance-following (now -0.133 vs real -0.168); raise kappa_lo from 0.21-0.24 to ~0.24-0.27 to compensate for the softer gain. This is the correct lever rather than raising beta back — do not oscillate on beta/epsilon, they are now correctly set for the non-outlier panels (Exp3/Exp4 both improved this round). A slightly stronger kappa_lo may also trim Exp2's over-positive tally signature (b_d - b_s at +0.034) if that panel contains near-chance experts, since anti-followed chance cues reduce the effective tally slope. (3) LEAVE Exp6/Exp8/Exp11 ALONE except as side-effects: Exp6 (0.135 vs 0.118) and Exp8 (0.466 vs 0.463) are at target and their panels (gap 0.24; no outlier) are insensitive to s_r and g_far changes in the ranges proposed; Exp11's small +0.026 residual may drift slightly with kappa_lo — accept that, since its variance is tiny and the sign is correct. The failure this round was not the mechanism but a lever that never engaged (far gate above Exp7's gap); pointing the existing composition route and bump shoulder at the gap~0.09 / majority-plausible-chance regime is exactly the in-family correction the arbiter's 'amplified on chance-laden panels' clause prescribes.

**Outcome of this advice:** iter 2 candidate loss=0.0556 -> iter 3 candidate loss=0.0554 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate was accepted by the gate, but only marginally (0.0556 -> 0.0554), and the headline target of my iter-2 advice — Exp7 — did not move at all: anti-deference went 0.503 -> 0.500 against a real value of 0.580 (residual now -0.080, by far the largest in the battery; its squared residual dominates roughly 60% of the remaining loss). Everything else held or improved as designed: Exp12 is dead-on (0.454 vs 0.453, inside the prescribed 0.45-0.50 band), Exp5 recovered to -0.147 (real -0.168), Exp1 improved to -0.228 (real -0.218), Exp9 is at 0.556 (real 0.560), Exp8 at 0.456 (real 0.463), Exp6 at 0.138 (real 0.118). So the mechanism family is right and eleven of twelve experiments are essentially calibrated; the single failure is a MAGNITUDE failure of the moderate-gap composition distrust on Exp7.

Why did the iter-2 levers (s_r widening, g_far lowering) fail despite engaging? Working through Exp7's trial structure: the panel is a 0.99 expert at gap 0.09 over a 0.90 rung with c_frac 0.375. With the new box midpoints, the widened shoulder gives P(0.09) ~ 0.70 and the partially-open far gate gives far(0.09) ~ 0.34, so w_top lands near -0.3 to -0.4. But on Exp7's discriminating trials the 99% expert is frequently corroborated by one or two plateau-weighted mid cues (weight +0.9 each) or opposed by a single gated chance cue (+0.9). A trial-by-trial computation shows w_top = -0.3..-0.4 yields an average anti-deference of only ~0.52 (before lapse compression), exactly what the simulation produced. Reaching the real 0.58 requires w_top of roughly -0.8 to -1.0 on this panel — which is precisely what pi_7's unbounded distrust delivered (w_top ~ -0.87, metric 0.579). Within BODT, the only route to that magnitude is the composition-amplified term: comp must be ~1.0-1.2 on Exp7's panel, roughly DOUBLE its current value (~0.5). The shoulder-widening route is exhausted; the missing magnitude must come from c_amp x far x comp_gate.
Rationale: Keep the BODT mechanism byte-for-byte (plateau, consensus gate, saturating shrinkage, composition route, softmax+lapse); this is a pure magnitude recalibration of the composition-amplified route, which is the in-family knob the arbiter's 'amplified on chance-laden panels' clause explicitly provides. Acknowledging my own track record: the iter-2 direction (engage moderate-gap distrust) was right and the candidate was accepted, but the step size was far too small — do not re-widen s_r or re-move g_star/s_l; the shoulder is done. The new adjustments: (1) PRIMARY — DOUBLE THE COMPOSITION DISCOUNT ON EXP7'S PANEL: raise c_amp from 2.0-2.3 to ~3.0-3.6 AND lower g_far from 0.10-0.12 to ~0.085-0.10 (optionally widen s_far to 0.035-0.045), so that on a gap~0.09, c_frac~0.375 panel the product c_amp*far(0.09)*comp_gate delivers comp ~ 1.0-1.2 and w_top lands near -0.8..-1.0. A useful cross-check: pi_7 achieved the real 0.579 with w_top ~ -0.87 on this panel, so target that w_top and verify via simulation that Exp7 anti-deference lands 0.56-0.59. If comp_gate is the binding constraint, lowering c_half toward 0.30-0.33 is an acceptable secondary lever. (2) GUARD EXP12: at gap 0.05, far rises to only ~0.2-0.27 and Exp12's c_frac sits below c_half (comp_gate ~ 0.2), so its comp stays <= ~0.15-0.20; if simulated deference dips below the 0.45 band floor, compensate with a slight rho_m trim (0.9-1.2 -> 0.85-0.95), which raises w_top_mod almost exclusively in the saturated P~1 regime (exactly Exp12's gap=0.05 peak) while barely touching the gap~0.09 shoulder — this is the surgical counterweight, preferable to touching w_floor or d_w. (3) ACCEPT THE TWO SIDE-EFFECTS: Exp6 currently sits at 0.138 vs real 0.118, i.e. slightly too pro-strong-side, so the additional comp on its gap=0.24, c_frac=0.5 panel moves it TOWARD the real value — this is a bonus, not a cost. Exp11 (currently +0.032) will drift ~+0.01 further in the CREDT-like direction because its panel also has gap 0.09 with c_frac 0.5; accept this — its variance is tiny, its sign is correct, and the Exp7 gain (squared residual ~0.0063 -> ~0.0005) dwarfs the Exp11 cost. (4) DO NOT TOUCH anything else: kappa_lo (Exp5 now -0.147 vs real -0.168, close enough), beta/epsilon (settled; do not oscillate), s_l/g_star/w_floor/d_w (dense-ladder and peak calibrations are at target). The candidate was accepted, so build directly on this iter-3 base; the expected net effect is the aggregate loss dropping well below 0.0554, driven almost entirely by the Exp7 residual collapsing from -0.080 to under ~0.02.

**Outcome of this advice:** iter 3 candidate loss=0.0554 -> iter 4 candidate loss=0.0669 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-4 candidate was REJECTED by the gate (loss 0.0554 -> 0.0669), and my iter-3 advice must be judged a failure: the predicted Exp7 collapse of the residual did not happen. Reading the trajectory honestly: (1) Exp7 anti-deference went 0.500 -> 0.4836 (real 0.5796) — the residual GREW to -0.096, now overwhelmingly the largest in the battery. (2) The predicted costs did materialize: Exp12 deference fell out of the prescribed band (0.454 -> 0.416, real 0.4525), Exp11 drifted further CREDT-ward (+0.032 -> +0.045, real 0.021). (3) The predicted bonuses also materialized: Exp4 improved (0.537 -> 0.519, real 0.513), Exp6 improved (0.138 -> 0.108, real 0.118), Exp9 improved marginally. So the levers physically moved — they just did not act on Exp7. Two critical inferences follow. FIRST, THE NOISE FLOOR: Exp5 moved -0.147 -> -0.188 (real -0.168) even though its panel ([0.95, 0.85, 0.75, 0.65, 0.5x4], no outlier) is structurally untouched by all four moved parameters (c_amp, g_far, s_far, rho_m all live in the outlier branch). Point-estimate noise is therefore ~±0.04, which means Exp7's -0.016 wiggle is noise — but the INFORMATIVE fact is that a predicted +0.08 signal failed to appear at all. SECOND, THE MONOTONICITY PARADOX: across three successive distrust-increasing edits, Exp7 anti-deference moved monotonically DOWN (0.528 -> 0.503 -> 0.500 -> 0.484). If the composition route delivered comp ~1.0 on Exp7's panel as my box-midpoint arithmetic assumed (gap 0.09, c_frac 0.375, comp_gate ~0.66 -> w_top ~ -0.76), a trial-by-trial computation gives anti-deference ≈ 0.58 — essentially the real value. It did not happen. The only consistent explanation is that the composition gate is CLOSED on Exp7's actual panel: its chance fraction sits BELOW c_half (0.32-0.35), so comp_gate ≈ 0.2 and comp ≈ 0.3-0.4 regardless of c_amp — the amplitude doublings of iter-3 and iter-4 were multiplying a near-zero gate product. Under that hypothesis the only iter-4 lever that actually acted on Exp7 was the rho_m trim, which RAISED w_top_mod (less distrust) — exactly the wrong direction, matching the small observed decline. My repeated assumption that Exp7 is a gap~0.09 / c_frac~0.375 panel is falsified by two consecutive null results; the binding constraint on Exp7 is the GATE THRESHOLD, not the magnitude.
Rationale: The gate already reverted the base to iter-3; do NOT re-apply the iter-4 moves (c_amp 3.0-3.6, g_far 0.085-0.10, s_far 0.035-0.045, rho_m 0.85-0.95 as a package) — that direction was rejected. The new direction is a THRESHOLD fix, not another magnitude fix, and it stays entirely inside the prescribed BODT family (it moves an existing parameter of the composition route). (1) PRIMARY — OPEN THE COMPOSITION GATE ON EXP7: lower c_half from 0.32-0.35 to ~0.20-0.24. The evidence says Exp7's c_frac sits just below the current threshold (comp_gate ≈ 0.24, comp ≈ 0.35, w_top ≈ -0.1, anti-deference ≈ 0.50 — matching the observed 0.500/0.484 almost exactly). With c_half ≈ 0.22, comp_gate(0.25) ≈ 0.6-0.7 and the EXISTING iter-3 amplitude (c_amp 2.8-3.2) delivers comp ≈ 0.7-1.0 on a gap~0.09 panel, landing w_top near -0.5..-0.8 and anti-deference near 0.55-0.58. Before committing, VERIFY from Exp7's actual validities what gap and c_frac really are (my structural assumptions have now been wrong twice — compute, don't guess), and confirm by simulation that anti-deference lands 0.55-0.59. (2) CALIBRATE THE AMPLITUDE BETWEEN THE TWO FLANKING ITERATIONS to avoid oscillation: c_amp 2.8-3.2 (iter-3 had 2.0-2.3, iter-4 overshot at 3.0-3.6 with an open far gate); g_far 0.095-0.105 with s_far 0.028-0.035, chosen so far(0.09) ≈ 0.35-0.45 (engages Exp7/Exp4) while far(0.05) ≤ ~0.15 (protects Exp12 — the iter-4 regression there came from the far gate opening at gap 0.05, plus noise). (3) GUARD EXP12: if its c_frac is also near 0.25, lowering c_half opens its gate too; the far gate above keeps its comp ≤ ~0.2, but if simulated deference dips below 0.45, apply the surgical counterweight — trim rho_m toward 0.85-0.90, which raises w_top_mod almost exclusively in the saturated P≈1 regime (gap ≈ 0.05) while costing little at gap 0.09 where the now-open comp absorbs it. (4) ACCEPT the Exp11 side-cost (its gap-0.09 / high-c_frac panel sees somewhat more distrust; it will drift CREDT-ward from +0.032 — its variance is tiny and the Exp7 gain, squared residual ~0.0063 -> ~0.001, dwarfs it), and note Exp4 and Exp6 both IMPROVED under gap-0.09/gap-0.24 comp increases, so they benefit from this edit too. (5) DO NOT TOUCH: kappa_lo (the Exp5 swing was pure noise — no lever acts on that panel), beta/epsilon (settled; the iter-4 Exp5 'regression' is not evidence against them), s_l/g_star/w_floor/d_w (dense-ladder and peak calibrations are at target: Exp9 -0.005, Exp10 -0.018, Exp12 +0.001 on the iter-3 base). Expected net effect on the reverted iter-3 base: Exp7 residual collapses from -0.080 toward -0.01..-0.03, Exp4/Exp6 hold their gains, Exp12 stays inside 0.45-0.50, and the aggregate loss drops well below 0.0554. If verification shows instead that Exp7's gap is ≈ 0.04-0.05 (degenerate with Exp12's regime, gate irrelevant), then the honest conclusion is that BODT's current shape cannot separate Exp7 from Exp12 by composition — in that case do not force it; protect Exp12's band and harvest the reliably-signed smaller residuals (Exp4, Exp6, Exp11, Exp2/Exp3 over-polarization) rather than risking a third rejection on Exp7.

**Outcome of this advice:** iter 4 candidate loss=0.0669 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
```json
{
  "theory": "Bounded Outlier-Distrust Tallying (BODT), protected-peak harvest edition. Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — a far outlier floating above a majority-chance panel. This round's calibration claims: (a) the composition-amplified route's amplitude is the correct lever for the one experiment that sits fully inside that regime (Exp6: far outlier at gap 0.24 above a majority-chance panel), and is raised to put its strong-side collapse dead on target; (b) the chance-flank magnitude is the correct lever for the two largest reliably-signed residuals on non-outlier panels (Exp2's over-positive tally signature and Exp5's under-negative chance-cue anti-following), both of which ask for a stronger negative flank; (c) the peak-regime calibration (rho_m, w_floor, d_w, g_star, s_l, s_r) and the gain/lapse balance (beta, epsilon) are held exactly at their accepted values, protecting the nine at-target experiments — in particular Exp12's spurious-precision band. Choice is a low-gain softmax plus a uniform lapse.",
  "predict": "def predict(parameters, state, history):\n    # Bounded Outlier-Distrust Tallying (BODT), protected-peak harvest edition.\n    # Stimulus: array of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are binary expert ratings (0/1).\n    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a\n    # subjective trust function of the expert's STATED validity v,\n    # evaluated RELATIVE to the panel:\n    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above\n    #       0.98, so a 98% expert stays INSIDE the plateau) get\n    #       approximately equal positive weight w_p.\n    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small\n    #       negative weight (-kappa_lo at v=0.5, rising linearly to\n    #       the plateau at v_lo). When an above-ceiling claim is\n    #       present, the flank is gated: gate = lambda_ext +\n    #       (1-lambda_ext)*sig_gap, so chance cues are COUNTED\n    #       (weight -> w_p) rather than inverted.\n    # (iii) BOUNDED OUTLIER DISTRUST (the core BODT claim): the\n    #       discount of the panel-max expert is a SATURATING\n    #       SHRINKAGE toward a floor, not an unbounded sign flip:\n    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)\n    #       where P is a PEAKED bump in the rung gap (maximal at\n    #       gap ~ g_star ~ 0.05 -- spurious precision; a SMALL but\n    #       nonzero discount on dense ladders gap ~ 0.01 via the\n    #       widened left width s_l; receding for large edges on\n    #       coherent panels via the right width s_r, widened so\n    #       the shoulder covers the moderate-gap regime gap ~ 0.09).\n    #       Because of the exponential saturation, w_top_mod never\n    #       leaves [w_floor, w_p].\n    #  (iv) COMPOSITION-AMPLIFIED ROUTE: an additive discount\n    #         comp = c_amp * sig_gap * far(gap) * gate(c_frac)\n    #       where far(gap) is a logistic gate centered at g_far\n    #       ~0.10-0.12 and gate(c_frac) fires when the fraction of\n    #       chance-flank cues exceeds c_half. This route reaches\n    #       strongly negative w_top only for far outliers floating\n    #       above majority-chance panels (the arbiter's amplified\n    #       regime, e.g. Exp6); this round its amplitude c_amp is\n    #       raised so that regime's collapse lands on target.\n    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)\n    # is retained. Choice is a low-gain softmax (inverse temperature\n    # beta) over the two mirrored scores, plus a uniform lapse\n    # (epsilon).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    w_p = float(parameters[\"w_p\"])\n    v_lo = float(parameters[\"v_lo\"])\n    v_hi = float(parameters[\"v_hi\"])\n    kappa_lo = float(parameters[\"kappa_lo\"])\n    g0 = float(parameters[\"g0\"])\n    s_gap = float(parameters[\"s_gap\"])\n    lambda_ext = float(parameters[\"lambda_ext\"])\n    g_star = float(parameters[\"g_star\"])\n    s_l = float(parameters[\"s_l\"])\n    s_r = float(parameters[\"s_r\"])\n    w_floor = float(parameters[\"w_floor\"])\n    rho_m = float(parameters[\"rho_m\"])\n    d_w = float(parameters[\"d_w\"])\n    g_far = float(parameters[\"g_far\"])\n    s_far = float(parameters[\"s_far\"])\n    c_half = float(parameters[\"c_half\"])\n    s_c = float(parameters[\"s_c\"])\n    c_amp = float(parameters[\"c_amp\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    def _sigmoid(x):\n        # Numerically stable logistic via the identity\n        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).\n        return 0.5 * (1.0 + np.tanh(0.5 * x))\n\n    a, b = stim[0], stim[1]\n\n    # Feature-wise votes: +1 where A wins the feature, -1 where B\n    # wins, 0 on ties (a tied cue is uninformative for the pair).\n    votes = np.sign(a - b)\n\n    v = np.clip(val, 0.5, 1.0)\n\n    # ---- Panel structure: top validity and its gap to the\n    # next-highest DISTINCT validity in the panel. ----\n    v_top = float(v.max())\n    below = v[v < v_top - 1e-9]\n    # If every expert shares the top validity, the panel offers no\n    # corroboration ladder; treat the next rung as chance (0.5).\n    v_second = float(below.max()) if below.size > 0 else 0.5\n    gap = v_top - v_second\n\n    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981\n    # a 0.98-topped panel stays inside the trusted plateau; only\n    # 99%+ claims enter the distrust branch.\n    outlier = v_top > v_hi\n\n    if outlier:\n        sig_gap = _sigmoid((gap - g0) / s_gap)\n    else:\n        sig_gap = 0.0\n\n    # ---- (ii): consensus gate for the chance flank. ----\n    if outlier:\n        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)\n    else:\n        gate = 0.0\n\n    # ---- (i) + (ii): plateau with a consensus-gated chance flank. ----\n    w = np.full(n_features, w_p, dtype=float)\n    flank = v < v_lo\n    if np.any(flank):\n        denom = max(v_lo - 0.5, 1e-6)\n        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(\n            v[flank] - 0.5, 0.0, None\n        ) / denom\n        # With an above-ceiling claim in the panel, chance cues are\n        # counted (weight -> w_p) rather than inverted.\n        w[flank] = (1.0 - gate) * w_flank + gate * w_p\n\n    # ---- (iii) + (iv): bounded outlier distrust, with a\n    # composition-amplified route for far outliers floating\n    # above majority-chance panels. ----\n    if outlier:\n        # (a) Peaked bump in the uncorroborated edge: maximal near\n        # gap ~ g_star (spurious precision), a SMALL nonzero\n        # discount on dense ladders (widened s_l), receding for\n        # large edges on coherent panels (right shoulder s_r\n        # covers the moderate-gap regime gap ~ 0.09).\n        width = s_l if gap < g_star else s_r\n        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))\n\n        # (b) BOUNDED, SATURATING shrinkage toward the floor:\n        # w_top_mod in [w_floor, w_p] for every P; even at peak\n        # distrust the outlier is only demoted toward the floor\n        # (~ -0.25..-0.30 plateau units), never unboundedly flipped.\n        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))\n\n        # (c) Composition-amplified route: a logistic far gate\n        # centered at g_far ~ 0.10-0.12, so it engages only for\n        # far outliers on majority-chance panels (the arbiter's\n        # amplified regime); its amplitude c_amp is the one\n        # lever moved this round.\n        c_frac = float(np.mean(flank))\n        comp_gate = _sigmoid((c_frac - c_half) / s_c)\n        far = _sigmoid((gap - g_far) / s_far)\n        comp = c_amp * float(sig_gap) * float(far) * float(comp_gate)\n\n        # (d) Small bounded extremity term.\n        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)\n\n        w_top = w_top_mod - ext - comp\n        # All experts tied at the panel maximum share the distrust\n        # (e.g., a pair of isolated 0.99 super-experts).\n        w[v >= v_top - 1e-9] = w_top\n\n    # Weighted evidence score. B's score is the mirror of A's\n    # because every vote is zero-sum across the two options.\n    s_a = float(np.dot(w, votes))\n    scores = np.array([s_a, -s_a], dtype=float)\n\n    # Numerically stable low-gain softmax. When all cues tie\n    # (s_a == 0) the softmax is exactly uniform.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    # Independent lapse: with probability epsilon output a uniform\n    # pick over the two options.\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # guard against float drift\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "w_p": "[0.88, 0.92]",
    "v_lo": "[0.52, 0.54]",
    "v_hi": "[0.9805, 0.9825]",
    "kappa_lo": "[0.28, 0.31]",
    "g0": "[0.041, 0.045]",
    "s_gap": "[0.012, 0.014]",
    "lambda_ext": "[0.35, 0.45]",
    "g_star": "[0.046, 0.050]",
    "s_l": "[0.018, 0.024]",
    "s_r": "[0.045, 0.055]",
    "w_floor": "[-0.30, -0.25]",
    "rho_m": "[0.9, 1.2]",
    "d_w": "[0.15, 0.22]",
    "g_far": "[0.10, 0.12]",
    "s_far": "[0.025, 0.035]",
    "c_half": "[0.32, 0.35]",
    "s_c": "[0.05, 0.07]",
    "c_amp": "[2.6, 2.9]",
    "beta": "[0.34, 0.39]",
    "epsilon": "[0.05, 0.07]",
    "validities": "validities"
  },
  "rationale": "MINIMAL-DIFF EDIT of the ACCEPTED iter-3 BODT base: the mechanism (plateau, consensus gate, saturating shrinkage, composition route, softmax+lapse) is byte-for-byte unchanged — only TWO parameter ranges move (kappa_lo 0.24-0.27 -> 0.28-0.31; c_amp 2.0-2.3 -> 2.6-2.9). Everything else, including the peak calibration (rho_m, w_floor, d_w, g_star, s_l, s_r), the far gate (g_far, s_far), the composition threshold (c_half, s_c), and the gain/lapse balance (beta, epsilon), is held exactly at its accepted value.\n\n(1) WHY I AM NOT FOLLOWING THE PRIMARY ADVICE (lower c_half to 0.20-0.24), and am instead invoking the critic's OWN pre-authorized fallback clause. The critic instructed me to VERIFY Exp7's panel structure from the trajectory rather than guess, and the verification is decisive. Under the gap-0.09 hypothesis, the iter-2 -> iter-3 edit (s_r 0.033 -> 0.050 raising P(0.09) from 0.44 to 0.70; g_far 0.15 -> 0.11 raising far(0.09) from 0.12 to 0.34) should have moved w_top on Exp7's panel by roughly -0.5 and anti-deference by +0.05-0.08; the observed movement was -0.003 (noise). Under the same hypothesis, iter-4 (c_amp 3.3, g_far 0.0925) should have delivered comp ~1.0, w_top ~ -0.76, and anti-deference ~0.58; the observed value was 0.484 — the WRONG direction. Two consecutive nulls are only consistent with Exp7's rung gap sitting BELOW g_star (~0.048), where neither the right shoulder s_r nor the far gate ever engages — i.e., Exp7 is degenerate with Exp12's spurious-precision regime, exactly the case the critic's fallback clause anticipates: 'BODT's current shape cannot separate Exp7 from Exp12 by composition — do not force it; protect Exp12's band and harvest the reliably-signed smaller residuals.' Three further quantitative guards confirm this: (a) the pi_7 anchor (w_top ~ -0.85 -> observed 0.579) versus iter-3 (w_top ~ 0 -> 0.500) gives an anti-deference slope of only ~0.09-0.12 per unit w_top, so reaching 0.58 requires Delta w_top ~ -0.7 at gap ~0.04-0.05 — but the bounded shrinkage floor (w_floor -0.275) caps the peak-regime discount, and lowering w_floor / deepening the peak hits Exp12's IDENTICAL gap regime with a deference slope of ~0.16-0.26 per unit (the iter-1 anchor: w_top ~ -0.175 gave Exp12 0.396 and Exp7 0.528 — the loop's iters 2-3 already slid along exactly that trade-off frontier to the current optimum, and the gate rewarded it); (b) lowering c_half to 0.20-0.24 cannot reach Exp7 (its far gate stays shut at gap ~0.04) but DOES open the composition gate on Exp12's own panel (c_frac ~ 0.25, comp_gate 0.19 -> 0.63), adding ~0.07 of comp discount and pushing Exp12's deference toward ~0.44, below the arbiter's prescribed 0.45-0.50 band — i.e., the primary advice touches exactly the experiment the fallback clause says to protect while missing its target; (c) Exp5's iter-3 -> iter-4 swing (-0.147 -> -0.188) on a panel structurally untouched by every moved parameter establishes a ~±0.04 noise floor, so chasing Exp7's residual with structurally inert levers is how iter-4 got rejected.\n\n(2) THE TWO LEVERS I DO MOVE, both squarely inside the prescribed BODT family, both targeting reliably-signed residuals. LEVER A — c_amp 2.0-2.3 -> 2.6-2.9 (Exp6, residual +0.020, stable across iters 1-3): Exp6 is the composition-amplified regime par excellence — a far outlier (gap 0.24) floating above a majority-chance panel (c_frac 0.5, comp_gate 0.94, far gate fully open) — the one experiment where the arbiter's clause 'reaches strongly negative values only in the composition-amplified regime' applies at full strength. Its residual says the strong-side collapse is slightly too weak. Trial-level arithmetic at box midpoints: comp 2.02 -> 2.59 moves w_top from -1.21 to -1.78, and the five metric cells (K2 .124->.099, K3 .084->.065, M3 .052->.037, M4 .041->.033, R3 .351 unchanged) average to 0.117 versus the real 0.118 — dead center, with the box edges bracketing 0.113-0.120. The iter-4 observation (0.108 at c_amp 3.3) independently confirms the slope (~-0.027 per comp unit) and that 2.6-2.9 does not overshoot. Cross-checks on every other outlier panel: Exp9/Exp10 (dense ladders, far(0.01) ~ 0.03) and Exp12 (far(0.05) ~ 0.12, comp_gate ~ 0.19) see comp changes of at most ~0.01 in w_top — noise; Exp11 sees w_top move ~ -0.19, but that metric demonstrated near-total insensitivity to w_top between iter-2 (w_top +0.14 -> 0.0468) and iter-3 (w_top -0.465 -> 0.0527), so the expected drift is ~+0.002. LEVER B — kappa_lo 0.24-0.27 -> 0.28-0.31 (Exp2 +0.027 and Exp5 +0.022, the two largest reliably-signed residuals after Exp7): both live on NON-outlier panels where the chance flank is un-gated, and both ask for the SAME lever — a stronger negative chance flank. Anchors: the iter-2 -> iter-3 kappa_lo increase (+0.03) moved Exp2 -0.007 and Exp5 -0.014; the proposed +0.04 (midpoint 0.295) projects Exp2 0.104 -> ~0.095 (real 0.077) and Exp5 -0.147 -> ~-0.166 (real -0.168). This is the lever the critic itself prescribed in iter-2 ('raise kappa_lo ... rather than re-sharpening beta'), and it leaves beta/epsilon untouched, protecting the at-target non-outlier calibrations. Gated panels (Exp6/7/9/10/11/12) are insensitive to kappa_lo by construction; Exp8's possible chance background moves at most ±0.005; Exp1/Exp3 drift favorably if their panels carry chance cues (anti-followed near-chance cues reduce effective tally slope, trimming their small over-polarization residuals).\n\n(3) WHAT IS DELIBERATELY NOT MOVED, and why. g_far stays 0.10-0.12: raising it back to 0.13-0.15 was considered and rejected — it trims Exp11 by only ~0.006 (given that metric's demonstrated w_top-insensitivity) while nudging Exp12 ~+0.004 off center; a wash not worth the risk. rho_m/w_floor/d_w stay put: any deepening of the peak regime trades Exp7 against Exp12 at roughly par or worse (the iter-1 anchor shows the pair was already optimized along that frontier), and the rho_m trim is what pushed iter-4's Exp7 in the wrong direction. s_l/g_star/s_r stay put: the dense-ladder and shoulder calibrations are at target (Exp9 -0.004, Exp10 -0.018, Exp12 +0.001 on the iter-3 base). beta/epsilon stay put: the beta opposition (Exp2/3/4 want less gain; Exp6/7/9/10 want more) nets to approximately zero and the critic explicitly warned against oscillating on them. Expected net effect on the iter-3 base: Exp6 +0.020 -> ~0.000, Exp2 +0.027 -> ~+0.018, Exp5 +0.022 -> ~+0.005, small favorable drifts on Exp1/Exp3, everything else within noise — pushing the aggregate loss below the 0.0554 floor without exposing any at-target experiment to a structural change."
}
```
```

## Usage

```json
{
  "input_tokens": 45106,
  "output_tokens": 78937,
  "reasoning_tokens": 73667,
  "provider": "Sail Research"
}
```
