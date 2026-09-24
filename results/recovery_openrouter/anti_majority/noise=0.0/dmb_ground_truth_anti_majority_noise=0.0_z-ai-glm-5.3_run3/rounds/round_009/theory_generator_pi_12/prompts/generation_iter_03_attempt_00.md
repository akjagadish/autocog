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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 (JU-AEI) — whose two distinctive mechanisms (habitual gamma support on (2.0, 2.2] and per-trial gamma jitter) are both empirically falsified — with a new theory that is a stronger competitor to pi_7 rather than a patch of pi_11. Proposed theory: Heterogeneous-Precision Anti-Endorsement Integration (HP-AEI). (i) RETAIN the empirically validated core unchanged: endorsements read as evidence against the carrying option, E = sum_j v_j^gamma (b_j - a_j) with RAW unnormalized validity-power weights, p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), A_eff = (1-eps)(2m-1) near 0.69, identical vectors give exactly 50%. (ii) REPLACE the population structure: gamma ~ U[0, 2] with density mildly tilted toward [0.7, 1.8] (NO mass above 2.0 — respecting Experiment 1's hard behavioral ceiling and Experiment 20's crossing statistic, which pi_7 matches exactly), and — the novel ingredient — beta log-uniform over a substantially WIDER box, roughly [1.0, 7.0], with a ~25-30% shallow-precision subpopulation at beta in [1.0, 2.0]. NO per-trial gamma jitter and NO per-trial beta noise (Experiment 2 falsified the former; the loop evidence cited in pi_11's own description shows per-trial precision noise flattens posteriors undesirably). (iii) WHY this works where both current theories fail: the shallow-beta cluster produces psychometrics that a fixed-gamma per-subject MLE can only fit by pushing gamma-hat upward — manufacturing the ~0.30 APPARENT tail mass of Experiment 16 and the 0.4335 flip-rate tilt of Experiment 18 WITHOUT any true gamma > 2 behavioral signature, thereby simultaneously satisfying Experiment 1's ceiling (max s ~ 1.0), Experiment 20's crossing statistic (~-0.21), and Experiment 16's estimator. The wider beta box also softens the Bayes factors of Experiments 13 and 17 toward their observed mid-range values (52.6, 12.9 instead of pi_7's overpredictions 96.7, 23.8) and Exp 15's positive offset. (iv) Falsifiable signatures: apparent-tail mass (fixed-gamma MLE) ~0.25-0.35 with TRUE tail mass exactly 0; population beta MLEs stratified into a visible shallow mode; per-subject normalized slopes spanning [~4, ~40] rather than pi_7's all-steep [>9.5]; flip rates on gamma-invariant sign-change trials in [0.42, 0.46]; hard ceiling s <= 1.5 preserved on top-rung designs. This gives a theory that inherits every one of pi_7's wins on the shared core while explaining the four residual experiments (13, 16, 17, 18) that pi_7 systematically miscalibrates, making it a genuine task-invariant competitor rather than a degenerate variant.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.866)

**Description**
Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

`parameters`:
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1580 (var=0.0120) vs this=0.1660 (var=0.0056)
- Experiment 2: real=0.1814 (var=0.0118) vs this=0.1764 (var=0.0097)
- Experiment 3: real=-0.7167 (var=0.0939) vs this=-0.7067 (var=0.0731)
- Experiment 4: real=-0.5230 (var=0.0540) vs this=-0.5575 (var=0.0274)
- Experiment 5: real=0.5944 (var=0.0376) vs this=0.5915 (var=0.0172)
- Experiment 6: real=-0.6758 (var=0.0393) vs this=-0.6850 (var=0.0317)
- Experiment 7: real=0.7044 (var=0.0382) vs this=0.7111 (var=0.0259)
- Experiment 8: real=0.3467 (var=0.0108) vs this=0.3521 (var=0.0065)
- Experiment 9: real=-1.7633 (var=3.8102) vs this=-1.3031 (var=2.4039)
- Experiment 10: real=0.9113 (var=0.0227) vs this=0.9161 (var=0.0235)
- Experiment 11: real=52.5984 (var=6.4849) vs this=96.6930 (var=15.9176)
- Experiment 12: real=-0.4144 (var=0.4154) vs this=-0.2520 (var=0.3854)
- Experiment 13: real=6.0286 (var=0.5984) vs this=5.3165 (var=0.3027)
- Experiment 14: real=0.2983 (var=0.0006) vs this=0.0990 (var=0.0001)
- Experiment 15: real=12.9184 (var=1.0896) vs this=23.7874 (var=2.2777)
- Experiment 16: real=0.4335 (var=0.0024) vs this=0.5335 (var=0.0013)
- Experiment 17: real=0.8355 (var=2.6525) vs this=0.3381 (var=0.3081)
- Experiment 18: real=-0.2077 (var=0.1170) vs this=-0.2078 (var=0.1272)
- Experiment 19: real=1.0000 (var=0.0729) vs this=1.2500 (var=0.0858)
- Experiment 20: real=0.0229 (var=0.0000) vs this=0.0431 (var=0.0000)


---

### `pi_10` (overall score: 0.771)

**Description**
Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69 — the 'steep normalized slope x moderate asymptote' family. The population is a three-segment mixture with SEGMENT-SPECIFIC PRECISION, all segments now at moderate-to-high precision (no soft-beta segment remains, since weakly identified slopes were shown to leak Exp 10/14 posteriors below threshold and to donate RFI-ward on Exp 9): (1) a 40% low bulk on gamma in [0, 1.0] with beta log-uniform on [2.2, 4.0] — the validated shared-core bulk, concentrated below the ~0.73/1.0 flip points of Exp 16's ladder, with the beta floor raised so every low-bulk subject's normalized slope is sharply identified; (2) a 25% mid segment on gamma in [1.05, 1.35] with beta log-uniform on [2.2, 3.2] — a band where RSD's uniform density (0.5) beats SGAI's (0.4) and EPAI's (0.375), sitting fully inside GSP's gamma box with GSP-compatible beta, so these subjects donate RSD-ward on Exp 15 while pulling Exp 11 and Exp 9 toward their observed values; (3) a 35% heavy tail on gamma in [2.0, 2.10] with beta log-uniform on [2.2, 2.8] — an extreme edge-hug of the RSD gamma boundary (average distance <= 0.05) at sharpened precision, so tail subjects' gamma posteriors concentrate just above 2.0 (feeding Exp 14), their slopes stay above the s>9.5 threshold (feeding Exp 10), and their near-deterministic past-flip-point behavior pulls Exp 16 down toward 0.4335. No per-trial gamma jitter. Falsifiable signatures: ~35% of subjects carry gamma within 0.10 of 2.0 with per-subject posteriors extending smoothly past 2.0; Exps 13/15 small and positive; Exp 11 in the 50-100 range; Exp 14 at or above 0.21; Exp 16 in [0.42, 0.45]; Exp 10 at or above 0.85.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-AEI-r6 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: three-segment mixture with SEGMENT-SPECIFIC
    # beta. All segments now carry moderate-to-high precision (no
    # soft-beta segment: weakly identified slopes were shown to leak
    # Exp 10/14 posteriors below threshold and donate RFI-ward on Exp 9).
    # gamma_u ~ U[0, 1] is mapped through:
    #   u in [0.00, 0.40): gamma ~ U[0.0, 1.0], beta log-U[2.2, 4.0]
    #            (40%; the validated low bulk, UNTOUCHED in gamma support.
    #            Beta floor raised 1.8 -> 2.2 per the pre-specified Exp 10
    #            contingency: sharpens per-subject slope identification
    #            without touching any other mechanism.)
    #   u in [0.40, 0.65): gamma ~ U[1.05, 1.35], beta log-U[2.2, 3.2]
    #            (25%; ENLARGED from 18%. On [1.05, 1.3] RSD's uniform
    #            density 0.5 beats SGAI's 0.4 and EPAI's 0.375, so these
    #            subjects donate RSD-ward on Exp 15; the band sits fully
    #            inside GSP's gamma box with GSP-compatible beta, pulling
    #            Exp 11 down toward the observed 52.6 and Exp 9 GSP-ward
    #            toward -1.76; it is above the 1.0 flip point
    #            (Exp-16-neutral); gamma near 1.2 matches the
    #            validity-proportional metrics of Exps 2 and 5.)
    #   u in [0.65, 1.00]: gamma ~ U[2.0, 2.10], beta log-U[2.2, 2.8]
    #            (35%; TRIMMED from 42% and RE-SHARPENED. Extreme edge-hug
    #            of the RSD gamma boundary (average distance <= 0.05), with
    #            beta raised from [1.9, 2.4] to [2.2, 2.8]: sharper gamma
    #            identification concentrates tail posteriors just above
    #            2.0 (Exp 14 recovers), keeps slopes above the s>9.5
    #            threshold (Exp 10), makes tail subjects near-deterministic
    #            past the flip points (Exp 16 pulled down toward 0.4335),
    #            and removes RFI-fittable soft slopes (Exp 9 back toward
    #            -1.76). The fraction trim cuts the residual Exp 13 edge
    #            cost ~17% while the mid segment's positive donations
    #            hold the total near the observed +6.)
    u = float(parameters["gamma_u"])
    if u < 0.40:
        gamma = 1.0 * (u / 0.40)
        log_b = float(parameters["log_beta_low"])
    elif u < 0.65:
        gamma = 1.05 + 0.30 * ((u - 0.40) / 0.25)
        log_b = float(parameters["log_beta_mid"])
    else:
        gamma = 2.0 + 0.10 * ((u - 0.65) / 0.35)
        log_b = float(parameters["log_beta_tail"])

    beta = float(np.exp(log_b))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1-m) with the conventional polarity. Continuous, no empty middle,
    # no mirror cluster. NO per-trial gamma jitter (delta = 0): the
    # stochastic attention gate is a confirmed dead lever.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

`parameters`:
- gamma_u: [0, 1]
- log_beta_low: [0.788, 1.386]
- log_beta_mid: [0.788, 1.163]
- log_beta_tail: [0.788, 1.030]
- epsilon: [0.03, 0.10]
- m: [0.75, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1580 (var=0.0120) vs this=0.1513 (var=0.0077)
- Experiment 2: real=0.1814 (var=0.0118) vs this=0.1714 (var=0.0079)
- Experiment 3: real=-0.7167 (var=0.0939) vs this=-0.7750 (var=0.0581)
- Experiment 4: real=-0.5230 (var=0.0540) vs this=-0.5479 (var=0.0211)
- Experiment 5: real=0.5944 (var=0.0376) vs this=0.5701 (var=0.0179)
- Experiment 6: real=-0.6758 (var=0.0393) vs this=-0.6917 (var=0.0411)
- Experiment 7: real=0.7044 (var=0.0382) vs this=0.7067 (var=0.0235)
- Experiment 8: real=0.3467 (var=0.0108) vs this=0.3629 (var=0.0063)
- Experiment 9: real=-1.7633 (var=3.8102) vs this=-1.1568 (var=1.5167)
- Experiment 10: real=0.9113 (var=0.0227) vs this=0.8791 (var=0.0271)
- Experiment 11: real=52.5984 (var=6.4849) vs this=114.0736 (var=12.7298)
- Experiment 12: real=-0.4144 (var=0.4154) vs this=-0.2310 (var=0.5023)
- Experiment 13: real=6.0286 (var=0.5984) vs this=-9.6424 (var=0.6044)
- Experiment 14: real=0.2983 (var=0.0006) vs this=0.1215 (var=0.0002)
- Experiment 15: real=12.9184 (var=1.0896) vs this=7.5133 (var=1.5781)
- Experiment 16: real=0.4335 (var=0.0024) vs this=0.4754 (var=0.0011)
- Experiment 17: real=0.8355 (var=2.6525) vs this=-0.2872 (var=0.6994)
- Experiment 18: real=-0.2077 (var=0.1170) vs this=0.1782 (var=0.1651)
- Experiment 19: real=1.0000 (var=0.0729) vs this=2.1250 (var=0.4528)
- Experiment 20: real=0.0229 (var=0.0000) vs this=0.1754 (var=0.0001)


---

### `pi_11` (overall score: 0.723)

**Description**
Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI, clip-restoration pass). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with A_eff = (1-eps)(2m-1) near 0.69 (steep normalized slope x moderate asymptote). The population: habitual gamma ~ U[0, 2.2] (single continuous uniform — no gaps, no cliffs, no narrow tail segment), beta log-uniform on [1.8, 6.0] independent of gamma. The novel mechanism is per-trial stochastic validity-attention jitter: with per-trial probability delta ~ 0.13-0.16, the effective gamma for that trial is redrawn uniformly from a SYMMETRIC window [gamma_habit - 0.50, gamma_habit + 0.50] clipped to [0, 2.4] — the arbiter's own prescribed geometry. The critical factorization this theory embodies: TRUE habitual mass above 2.2 is poison (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment and destroys Exp 18's negative offset), while per-trial EXCURSIONS reaching 2.4 from habits below 2.2 are safe and are the only channel that moves the apparent-tail estimator (Exp 14) above 0.17. There is NO per-trial log-beta jitter: five passes of loop evidence establish that any per-trial precision noise flattens per-subject gamma posteriors toward the estimator's EB prior mean and caps the tail estimate at ~0.12, while gamma-only jitter at delta ~0.14 simultaneously satisfies the BF softening (Exps 15/9/12) and the flip-rate anchor (Exp 16). Falsifiable signatures: apparent tail-mass estimates ~0.17-0.20 on shallow-slope designs; positive RSD-vs-EPAI log BF ~[4, 10]; flip rates in [0.43, 0.45]; crossing statistic in [-0.15, -0.10]; steep-slope posterior >= 0.88; no plateau in recovered gamma rung ladders.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI),
    # clip-restoration pass: habitual support held at U[0, 2.2], the
    # per-trial gamma-excursion clip restored to the arbiter's own [0, 2.4],
    # gamma-only jitter (NO per-trial log-beta noise), symmetric +/-0.50
    # window, delta in [0.13, 0.16].
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"JU-AEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- (1) GAMMA SUPPORT: single continuous uniform on [0, 2.2] ----
    # Held exactly at [0, 2.2]: TRUE habitual mass above 2.2 is poison
    # (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment on
    # Exp 13 and inflates Exp 18's p_hi term). Apparent tail mass is
    # manufactured purely by the per-trial excursions below.
    gamma_habit = 2.2 * float(parameters["gamma_u"])

    # ---- (2) PRECISION: log-uniform raw inverse temperature, ----
    # ---- independent of gamma (no segment-coupled beta boxes)  ----
    # NO per-trial log-beta jitter: five loop passes establish that any
    # per-trial precision noise flattens per-subject gamma posteriors
    # toward the Exp 14 estimator's EB prior mean (0.075), capping the
    # tail estimate at ~0.12, and pushes Exp 16's flip rate toward 0.5.
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # ---- (3) PER-TRIAL STOCHASTIC VALIDITY-ATTENTION JITTER ----
    # Gamma-only. With per-trial probability delta in [0.13, 0.16] the
    # effective gamma is redrawn uniformly from a SYMMETRIC window
    # [gamma_habit - 0.50, gamma_habit + 0.50], clipped to [0, 2.4]
    # (the arbiter's own prescribed geometry). The 2.4 up-clip is the
    # single change most implicated in the previous pass's rejection:
    # excursions reaching 2.4 from habits in [1.7, 2.2] are what the
    # fixed-gamma per-subject MLE reads as apparent tail mass (the only
    # channel that has ever moved Exp 14 above 0.17), while habitual
    # RESIDENCE above 2.2 (removed here by keeping the support at 2.2)
    # is what flipped Exp 13's sign in the iter-6 pass.
    delta = float(parameters["delta"])
    if np.random.random() < delta:
        lo = max(0.0, gamma_habit - 0.50)
        hi = min(2.4, gamma_habit + 0.50)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma_habit

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma_t)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster. UNCHANGED (Exps 7/8 anchors are exact).
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

`parameters`:
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- delta: [0.13, 0.16]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1580 (var=0.0120) vs this=0.1593 (var=0.0065)
- Experiment 2: real=0.1814 (var=0.0118) vs this=0.1764 (var=0.0135)
- Experiment 3: real=-0.7167 (var=0.0939) vs this=-0.7000 (var=0.0756)
- Experiment 4: real=-0.5230 (var=0.0540) vs this=-0.5691 (var=0.0263)
- Experiment 5: real=0.5944 (var=0.0376) vs this=0.5665 (var=0.0263)
- Experiment 6: real=-0.6758 (var=0.0393) vs this=-0.6587 (var=0.0352)
- Experiment 7: real=0.7044 (var=0.0382) vs this=0.6744 (var=0.0349)
- Experiment 8: real=0.3467 (var=0.0108) vs this=0.3475 (var=0.0083)
- Experiment 9: real=-1.7633 (var=3.8102) vs this=-1.2961 (var=2.5233)
- Experiment 10: real=0.9113 (var=0.0227) vs this=0.9004 (var=0.0321)
- Experiment 11: real=52.5984 (var=6.4849) vs this=41.8821 (var=6.2799)
- Experiment 12: real=-0.4144 (var=0.4154) vs this=-0.2418 (var=0.3912)
- Experiment 13: real=6.0286 (var=0.5984) vs this=-0.9777 (var=0.5911)
- Experiment 14: real=0.2983 (var=0.0006) vs this=0.1003 (var=0.0002)
- Experiment 15: real=12.9184 (var=1.0896) vs this=1.5576 (var=1.0977)
- Experiment 16: real=0.4335 (var=0.0024) vs this=0.4498 (var=0.0010)
- Experiment 17: real=0.8355 (var=2.6525) vs this=-0.0354 (var=0.4369)
- Experiment 18: real=-0.2077 (var=0.1170) vs this=-0.0808 (var=0.2909)
- Experiment 19: real=1.0000 (var=0.0729) vs this=2.7500 (var=0.4086)
- Experiment 20: real=0.0229 (var=0.0000) vs this=0.0649 (var=0.0000)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0958 -> ACCEPTED
- iter 2: loss=0.1044 -> REJECTED
- iter 3: loss=0.0959 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0958 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation score: on trials where Take-The-Best (top-validity
    # cascade) and Tallying (cue-count majority) make DIFFERENT modal
    # predictions, how often does the subject side with the TTB option?
    # TTB predicts > 0.5 on every such trial (constant sigmoid(beta)
    # confidence, including on tally-tie trials 7/8 where Tallying must
    # guess at exactly 50%). Tallying predicts < 0.5 on conflict trials
    # (majority option wins) and exactly 0.5 on tally ties. So the metric
    # is positive under TTB and negative under Tallying.
    V = np.array([0.95, 0.85, 0.65, 0.60, 0.50])
    order = np.argsort(-V, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    matches = []
    for i in range(len(a_col)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)

        # TTB winner: first discriminating cue in validity order.
        ttb = -1
        for j in order:
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb == -1:
            continue  # TTB guesses; not diagnostic

        # Tallying winner: strict cue-count majority.
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins > b_wins:
            tal = 0
        elif b_wins > a_wins:
            tal = 1
        else:
            tal = -1  # tally tie -> Tallying guesses

        # Keep only dissociation trials: the two theories' modal
        # predictions differ (tally tie, or tally majority against TTB).
        if tal == ttb:
            continue

        matches.append(1.0 if resp[i] == ttb else 0.0)

    if len(matches) == 0:
        return 0.0
    # Center at 0.5: positive => TTB-like, negative => Tallying-like.
    return float(np.mean(matches)) - 0.5
```

**Observed (real) value:** 0.1580 (var=0.0120)
**Previous candidate values (this loop):**
  - iter 1: 0.1783 (var=0.0062) (Δ vs real +0.0203)
  - iter 2: 0.1497 (var=0.0058) (Δ vs real -0.0083)
  - iter 3 (most recent): 0.1700 (var=0.0091) (Δ vs real +0.0120)
**Other theories' values on this metric (for reference):**
- pi_1: 0.3480 (var=0.0087)
- pi_2: -0.2963 (var=0.0068)
- pi_3: 0.2473 (var=0.0026)
- pi_4: 0.1530 (var=0.0069)
- pi_5: 0.2150 (var=0.0067)
- pi_6: 0.1693 (var=0.0082)
- pi_7: 0.1660 (var=0.0056)
- pi_8: 0.1537 (var=0.0082)
- pi_9: 0.1513 (var=0.0063)
- pi_10: 0.1513 (var=0.0077)
- pi_11: 0.1593 (var=0.0065)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the TALLY winner | Tallying and Take-The-Best disagree).

    On every trial we compute (a) the tally winner (option winning more
    strict feature-wise comparisons) and (b) the TTB winner (winner on
    the first discriminating cue in descending-validity order, cues here
    are already ordered by validity). We keep only CONFLICT trials where
    both heuristics have a definite winner and they point to opposite
    options, and return the proportion of choices falling on the tally
    winner's side.

    Under Tallying this proportion is high (sigmoid(2*beta) on the
    margin-2 grid conflicts, sigmoid(6*beta) on the extreme-conflict
    trials where one option wins 7 of 8 cues); under TTB it is low
    (1 - sigmoid(beta)). Works on the pooled DataFrame and on any
    single subject's slice (each subject has 7 unique conflict trials
    x 4 reps = 28 conflict observations).
    """
    validities = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    cue_order = np.argsort(-validities, kind="stable")

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying needs a strict majority; skip tied tallies.
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Take-The-Best winner: first discriminating cue in validity order.
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        # Keep only genuine conflicts between the two heuristics.
        if ttb_winner == tally_winner:
            continue

        resp = int(row["response"])
        hits.append(1.0 if resp == tally_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.1814 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1: 0.1693 (var=0.0092) (Δ vs real -0.0121)
  - iter 2: 0.2050 (var=0.0122) (Δ vs real +0.0236)
  - iter 3 (most recent): 0.1729 (var=0.0062) (Δ vs real -0.0086)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8664 (var=0.0110)
- pi_1: 0.1536 (var=0.0134)
- pi_3: 0.2557 (var=0.0053)
- pi_4: 0.2579 (var=0.0137)
- pi_5: 0.1479 (var=0.0061)
- pi_6: 0.1793 (var=0.0080)
- pi_7: 0.1764 (var=0.0097)
- pi_8: 0.1571 (var=0.0041)
- pi_9: 0.1714 (var=0.0079)
- pi_10: 0.1714 (var=0.0079)
- pi_11: 0.1764 (var=0.0135)

### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Depth/conflict invariance contrast for TTB vs PWEI.

    Three structurally identified trial classes (validity order is fixed:
    [0.90, 0.89, 0.88, 0.87, 0.70, 0.60]):

    HIGH  - the top cue (0.90) discriminates and every other discriminating
            cue agrees with it (design trials 1, 2, 9, 10; 24 obs/subject).
            PWEI evidence margin |D| is maximal (~1 or w_top).
    DEPTH5- cues 1-4 all tie; only the 0.70-validity cue discriminates
            (design trials 11, 12; 12 obs/subject). PWEI's weight for
            0.70^gamma is ~1e-5 of total, so PWEI predicts ~50% choices.
    REV   - the top cue discriminates one way, the three sub-top cues
            (0.89, 0.88, 0.87) discriminate the other way, tail cues tie
            (design trials 3, 4; 12 obs/subject). In PWEI's steep-gamma
            regime the sub-top block outweighs the top cue, so PWEI
            chooses AGAINST the TTB winner.

    For each class compute p = P(choose the TTB winner). TTB's structural
    invariant is a single constant confidence p* on every discriminating
    trial, so 2*p_HIGH - p_DEPTH5 - p_REV = 0 exactly, at ANY (beta, eps).
    PWEI predicts p_HIGH ~ 0.9, p_DEPTH5 ~ 0.5, p_REV ~ 0.1-0.37, giving a
    value ~0.9-1.2 across its entire parameter box.
    """
    VALS = np.array([0.90, 0.89, 0.88, 0.87, 0.70, 0.60])
    order = np.argsort(-VALS, kind="stable")  # validity-descending cue indices

    n_high, k_high = 0, 0
    n_d5, k_d5 = 0, 0
    n_rev, k_rev = 0, 0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)  # +1 favors A, -1 favors B, 0 tie, per cue

        # First discriminating cue in validity order (TTB's search).
        first = None
        for j in order:
            if s[j] != 0.0:
                first = j
                break
        if first is None:
            continue  # identical options: both theories say 50%

        ttb_winner_is_a = s[first] > 0
        chose_ttb = (int(resp) == 0) == ttb_winner_is_a

        if first == 4:
            # DEPTH5: only the 0.70-validity cue discriminates.
            n_d5 += 1
            k_d5 += int(chose_ttb)
        elif first == 0 and s[1] == -s[0] and s[2] == -s[0] and s[3] == -s[0] \
                and s[4] == 0.0 and s[5] == 0.0:
            # REV: top cue vs the three sub-top cues, tail ties.
            n_rev += 1
            k_rev += int(chose_ttb)
        elif first == 0:
            # HIGH candidate: all other discriminating cues agree with top.
            others = [s[j] for j in range(1, 6) if s[j] != 0.0]
            if all(o == s[0] for o in others):
                n_high += 1
                k_high += int(chose_ttb)
        # All other trial types (mixed conflicts, cue-2-led trials, ties)
        # are deliberately excluded to keep the classes pure.

    p_high = k_high / n_high if n_high > 0 else 0.5
    p_d5 = k_d5 / n_d5 if n_d5 > 0 else 0.5
    p_rev = k_rev / n_rev if n_rev > 0 else 0.5

    return 2.0 * p_high - p_d5 - p_rev
```

**Observed (real) value:** -0.7167 (var=0.0939)
**Previous candidate values (this loop):**
  - iter 1: -0.7800 (var=0.0602) (Δ vs real -0.0633)
  - iter 2: -0.6467 (var=0.0893) (Δ vs real +0.0700)
  - iter 3 (most recent): -0.7717 (var=0.0644) (Δ vs real -0.0550)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0033 (var=0.0461)
- pi_3: 1.0483 (var=0.0759)
- pi_2: 0.7083 (var=0.0512)
- pi_4: -0.5050 (var=0.1068)
- pi_5: -0.7033 (var=0.1239)
- pi_6: -0.6950 (var=0.0763)
- pi_7: -0.7067 (var=0.0731)
- pi_8: -0.7200 (var=0.0585)
- pi_9: -0.7233 (var=0.0493)
- pi_10: -0.7750 (var=0.0581)
- pi_11: -0.7000 (var=0.0756)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # OLS slope of "chose the TTB (first-discriminating-cue) winner" on the
    # PWEI power-weighted evidence margin measured TOWARD that winner.
    # TTB: choice prob is one constant p* on every discriminating trial
    #      (any beta, eps)  =>  slope exactly 0.
    # PWEI: prob is a psychometric function of the weighted margin D
    #      =>  strongly positive slope (~0.5 over the design's margin range).
    val = np.array([0.95, 0.92, 0.91, 0.90, 0.89, 0.88], dtype=float)
    gamma = 40.0  # mid of PWEI's calibrated regime [35, 50]
    w = np.power(val, gamma)
    w = w / w.sum()
    cue_order = np.argsort(-val, kind="stable")

    xs = []
    ys = []
    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)
        if not np.any(s != 0):
            # Identical options: pure guessing for both theories; excluded.
            continue
        # TTB winner: first discriminating cue in descending-validity order.
        winner = None
        for j in cue_order:
            if s[j] != 0:
                winner = 0 if s[j] > 0 else 1
                break
        if winner is None:
            continue
        # Power-weighted evidence for option A.
        D = float(np.sum(w * s))
        # Margin re-signed so that positive = evidence favors the TTB winner.
        x = D if winner == 0 else -D
        y = 1.0 if int(r) == winner else 0.0
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    if sxx <= 1e-12:
        return 0.0
    sxy = float(np.sum((xs - xs.mean()) * (ys - ys.mean())))
    return float(sxy / sxx)
```

**Observed (real) value:** -0.5230 (var=0.0540)
**Previous candidate values (this loop):**
  - iter 1: -0.5133 (var=0.0263) (Δ vs real +0.0098)
  - iter 2: -0.5039 (var=0.0290) (Δ vs real +0.0192)
  - iter 3 (most recent): -0.5540 (var=0.0208) (Δ vs real -0.0310)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5548 (var=0.0148)
- pi_1: 0.0081 (var=0.0061)
- pi_2: 0.5897 (var=0.0315)
- pi_4: -0.4252 (var=0.0349)
- pi_5: -0.5389 (var=0.0255)
- pi_6: -0.5237 (var=0.0265)
- pi_7: -0.5575 (var=0.0274)
- pi_8: -0.5635 (var=0.0198)
- pi_9: -0.5423 (var=0.0159)
- pi_10: -0.5479 (var=0.0211)
- pi_11: -0.5691 (var=0.0263)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Lower-cue evidence-usage slope.

    For every trial with a discriminating cue, determine the Take-The-Best
    winner (first cue in descending-validity order that differs). Pool trials
    into cells keyed by the inverted-polarity integrated evidence favoring
    that winner, E_w = sum_j w_j * (loser_j - winner_j), w_j proportional to
    validity. Then compute the OLS slope of P(choose the TTB winner) on E_w
    across cells.

    TTB: the winner's choice probability is the SAME constant p* on every
    discriminating trial (only the top discriminating cue is used), so the
    psychometric function over E_w is exactly flat -> slope = 0 for ANY
    (beta, epsilon).

    SPI: choice probability is a strictly increasing sigmoid of E_w
    (rho < 0.5 preserves monotonicity, lapse is affine), and the design's
    cells sweep E_w from -1 (unanimous endorsement) to +0.58 -> large
    positive slope.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.70, 0.65, 0.60], dtype=float)
    w = validities / validities.sum()
    cue_order = np.argsort(-validities, kind="stable")

    a_cells = data["option_a_ratings"].tolist()
    b_cells = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    n = len(data)
    E_list = []
    Y_list = []

    for i in range(n):
        a = np.asarray(a_cells[i], dtype=float).ravel()
        b = np.asarray(b_cells[i], dtype=float).ravel()
        if a.shape[0] != validities.shape[0] or b.shape[0] != validities.shape[0]:
            continue

        # TTB cascade: first discriminating cue in validity order.
        winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner < 0:
            continue  # no discriminating cue: uninformative for the contrast

        # Inverted-polarity integrated evidence favoring the TTB winner:
        # the loser's endorsements (1s) count FOR the winner (sparse option).
        if winner == 0:
            E = float(np.sum(w * (b - a)))
        else:
            E = float(np.sum(w * (a - b)))

        E_list.append(E)
        Y_list.append(1.0 if int(resp[i]) == winner else 0.0)

    if len(E_list) < 4:
        return 0.0

    E = np.asarray(E_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)

    # Pool trials into cells by (rounded) evidence value; mirrored twins and
    # repetitions of the same pair share the same E_w and are averaged.
    keys = np.round(E, 4)
    uniq = np.unique(keys)
    if uniq.shape[0] < 2:
        return 0.0

    cell_x = np.empty(uniq.shape[0], dtype=float)
    cell_y = np.empty(uniq.shape[0], dtype=float)
    for idx in range(uniq.shape[0]):
        mask = keys == uniq[idx]
        cell_x[idx] = E[mask].mean()
        cell_y[idx] = Y[mask].mean()

    x = cell_x - cell_x.mean()
    y = cell_y - cell_y.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0

    return float(np.sum(x * y) / denom)
```

**Observed (real) value:** 0.5944 (var=0.0376)
**Previous candidate values (this loop):**
  - iter 1: 0.6478 (var=0.0199) (Δ vs real +0.0534)
  - iter 2: 0.5583 (var=0.0250) (Δ vs real -0.0361)
  - iter 3 (most recent): 0.6028 (var=0.0195) (Δ vs real +0.0084)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0009 (var=0.0089)
- pi_4: 0.4379 (var=0.0326)
- pi_2: -0.6876 (var=0.0238)
- pi_3: 0.0089 (var=0.0056)
- pi_5: 0.5894 (var=0.0647)
- pi_6: 0.5414 (var=0.0227)
- pi_7: 0.5915 (var=0.0172)
- pi_8: 0.5704 (var=0.0179)
- pi_9: 0.5813 (var=0.0162)
- pi_10: 0.5701 (var=0.0179)
- pi_11: 0.5665 (var=0.0263)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design constants: 8 cues with the experiment's validities.
    validities = np.array([0.95, 0.92, 0.62, 0.60, 0.62, 0.60, 0.62, 0.60])

    # Parse per-trial rating vectors (cells are lists -> project safely).
    a_lists = [list(r) for r in data["option_a_ratings"]]
    b_lists = [list(r) for r in data["option_b_ratings"]]
    A = np.asarray(a_lists, dtype=float)
    B = np.asarray(b_lists, dtype=float)
    resp = np.asarray(data["response"]).astype(int)

    n_feat = A.shape[1]
    if n_feat == len(validities):
        cue_order = np.argsort(-validities, kind="stable")
    else:
        cue_order = np.arange(n_feat)

    n_a = A.sum(axis=1)
    n_b = B.sum(axis=1)

    # TTB winner per trial: first discriminating cue in validity order.
    winner = np.full(A.shape[0], -1, dtype=int)
    for j in cue_order:
        open_rows = winner == -1
        if not open_rows.any():
            break
        wa = open_rows & (A[:, j] > B[:, j])
        wb = open_rows & (B[:, j] > A[:, j])
        winner[wa] = 0
        winner[wb] = 1

    abs_diff = np.abs(n_a - n_b)
    denser = np.where(n_a > n_b, 0, 1)  # option carrying MORE 1-ratings

    # Extreme-count trials only (|n_A - n_B| >= 5): the sign of SPI's
    # evidence is robust across its whole parameter box on these cells.
    valid = (winner >= 0) & (abs_diff >= 5)
    ttb_picks_denser = (winner == denser)

    chose_winner = (resp == winner)

    grp_d = valid & ttb_picks_denser      # TTB winner is the DENSER option
    grp_s = valid & (~ttb_picks_denser)    # TTB winner is the SPARSER option

    nd = int(grp_d.sum())
    ns = int(grp_s.sum())
    if nd == 0 or ns == 0:
        return 0.0

    p_denser = float(chose_winner[grp_d].mean())
    p_sparser = float(chose_winner[grp_s].mean())

    return p_denser - p_sparser
```

**Observed (real) value:** -0.6758 (var=0.0393)
**Previous candidate values (this loop):**
  - iter 1: -0.6808 (var=0.0306) (Δ vs real -0.0050)
  - iter 2: -0.6725 (var=0.0291) (Δ vs real +0.0033)
  - iter 3 (most recent): -0.7279 (var=0.0389) (Δ vs real -0.0521)
**Other theories' values on this metric (for reference):**
- pi_4: -0.5012 (var=0.0594)
- pi_1: 0.0138 (var=0.0135)
- pi_2: 0.7354 (var=0.0335)
- pi_3: -0.0125 (var=0.0078)
- pi_5: -0.6912 (var=0.1205)
- pi_6: -0.6900 (var=0.0330)
- pi_7: -0.6850 (var=0.0317)
- pi_8: -0.6883 (var=0.0128)
- pi_9: -0.6725 (var=0.0255)
- pi_10: -0.6917 (var=0.0411)
- pi_11: -0.6587 (var=0.0352)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Per-subject choice CONSISTENCY (extremity) toward the sparse option on
    # near-unanimous trials (one option all-zeros, the other endorsed on at
    # least n_features-1 cues), averaged over subjects.
    #
    # RFI-delta: polarity is a stable per-subject trait with only a small
    # attentional slip (delta <= 0.12) and small lapse (eps <= 0.08), so EVERY
    # subject -- majority (anti-endorsement) and minority (conventional)
    # alike -- is strongly and stably consistent on these trials:
    # per-subject p(sparse) is either ~[0.80, 0.92] or its mirror ~[0.08, 0.20],
    # i.e. |2p-1| ~ [0.6, 0.85] for essentially all subjects, with tight
    # between-subject spread.
    #
    # SPI: the wide lapse/polarity-mixing boxes (eps in [0.05, 0.35],
    # rho in [0, 0.35], beta in [0.5, 20]) mean per-subject attenuation varies
    # enormously; the average subject sits much closer to chance (mean
    # |2p-1| far below RFI's), and the population spread is wide.
    # The absolute value makes the statistic polarity-symmetric, so RFI's
    # small conventional minority does NOT inflate the between-subject
    # variance (both clusters have the same extremity).

    a_ratings = data['option_a_ratings'].tolist()
    b_ratings = data['option_b_ratings'].tolist()
    resp = data['response'].astype(float).values
    subj = data['subject_id'].values

    n = len(data)
    a_sum = np.empty(n, dtype=float)
    b_sum = np.empty(n, dtype=float)
    n_feat = np.empty(n, dtype=float)
    for i in range(n):
        av = list(a_ratings[i])
        bv = list(b_ratings[i])
        a_sum[i] = float(np.sum(av))
        b_sum[i] = float(np.sum(bv))
        n_feat[i] = float(len(av))

    # Near-unanimous trials: sparse side has zero endorsements, dense side
    # is endorsed on all or all-but-one cue (E in ~[0.8, 1.0] for any
    # weighting gamma in [0, 2]).
    thresh = n_feat - 1.0
    sparse_is_a = (a_sum == 0.0) & (b_sum >= thresh)
    sparse_is_b = (b_sum == 0.0) & (a_sum >= thresh)
    extreme = sparse_is_a | sparse_is_b

    # 1 if the subject chose the all-zeros (sparse) option, else 0.
    chose_sparse = np.where(sparse_is_a, 1.0 - resp, resp)

    vals = []
    for s in np.unique(subj[extreme]):
        m = extreme & (subj == s)
        if m.sum() == 0:
            continue
        p = float(np.mean(chose_sparse[m]))
        vals.append(abs(2.0 * p - 1.0))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.7044 (var=0.0382)
**Previous candidate values (this loop):**
  - iter 1: 0.7144 (var=0.0354) (Δ vs real +0.0100)
  - iter 2: 0.6567 (var=0.0316) (Δ vs real -0.0478)
  - iter 3 (most recent): 0.7256 (var=0.0221) (Δ vs real +0.0211)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7611 (var=0.0118)
- pi_4: 0.5167 (var=0.0365)
- pi_1: 0.6944 (var=0.0505)
- pi_2: 0.7267 (var=0.0418)
- pi_3: 0.6067 (var=0.0143)
- pi_6: 0.7544 (var=0.0264)
- pi_7: 0.7111 (var=0.0259)
- pi_8: 0.7178 (var=0.0239)
- pi_9: 0.6922 (var=0.0227)
- pi_10: 0.7067 (var=0.0235)
- pi_11: 0.6744 (var=0.0349)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Polarity-agnostic preference EXTREMITY on high-contrast trials.
    #
    # Select trials where the two options' endorsement counts differ by >= 3
    # (design trials R3..R8 and mirrors: |E| in [~0.35, 1.0] for every gamma
    # in [0,2]; the count-sparser option is the inverted-evidence-favored
    # option on all of them). Per subject, compute p_hat = rate of choosing
    # the count-sparser option on those 48 trials, then take |p_hat - 0.5|.
    # The metric is invariant to a subject's polarity (a conventional-polarity
    # subject and a red-flag subject both yield large extremity), which
    # neutralizes RFI's ~5% minority cluster, and directly measures the
    # ATTENUATION away from a pure deterministic preference: SPI's large
    # lapse (eps in [0.05,0.35], mean ~0.20) plus polarity mixture
    # (rho in [0,0.35], mean ~0.175) pulls per-subject p toward 0.5
    # (predicted extremity ~0.22), whereas RFI-delta's small slip
    # (delta ~0.10) and tiny lapse (eps <= 0.08) keep every subject
    # near saturation (predicted extremity ~0.35, per-subject range
    # [0.75, 0.92] in p). Applied to the pooled DataFrame the function
    # returns the mean of the per-subject values, so the pooled point
    # estimate is exactly the population mean of the per-subject metric.
    sids = []
    chose_sparse = []
    for r in data.itertuples(index=False):
        a = np.asarray(r.option_a_ratings, dtype=float)
        b = np.asarray(r.option_b_ratings, dtype=float)
        sa = int(a.sum())
        sb = int(b.sum())
        # Keep only high-contrast trials (count difference >= 3).
        if abs(sa - sb) < 3:
            continue
        resp = int(r.response)  # 0 = chose A, 1 = chose B
        sparse_is_a = sa < sb
        picked_sparse = (resp == 0) if sparse_is_a else (resp == 1)
        sids.append(r.subject_id)
        chose_sparse.append(1.0 if picked_sparse else 0.0)
    if len(sids) == 0:
        return 0.0
    tmp = pd.DataFrame({"sid": sids, "cs": chose_sparse})
    per_subj = tmp.groupby("sid")["cs"].mean()
    extremity = (per_subj - 0.5).abs()
    return float(extremity.mean())
```

**Observed (real) value:** 0.3467 (var=0.0108)
**Previous candidate values (this loop):**
  - iter 1: 0.3679 (var=0.0060) (Δ vs real +0.0213)
  - iter 2: 0.3150 (var=0.0078) (Δ vs real -0.0317)
  - iter 3 (most recent): 0.3408 (var=0.0074) (Δ vs real -0.0058)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2367 (var=0.0084)
- pi_5: 0.3758 (var=0.0028)
- pi_1: 0.3283 (var=0.0115)
- pi_2: 0.3421 (var=0.0094)
- pi_3: 0.1929 (var=0.0036)
- pi_6: 0.3467 (var=0.0078)
- pi_7: 0.3521 (var=0.0065)
- pi_8: 0.3438 (var=0.0047)
- pi_9: 0.3658 (var=0.0051)
- pi_10: 0.3629 (var=0.0063)
- pi_11: 0.3475 (var=0.0083)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.98, 0.95, 0.88, 0.85, 0.82, 0.80,
                  0.78, 0.76, 0.74, 0.72, 0.62, 0.62], dtype=float)
    WTIE = np.power(V, 0.75)

    df = data.reset_index(drop=True)

    # ---------- 1. classify trials into evidence cells ----------
    # Cell = the dense-minus-sparse endorsement pattern; its evidence under
    # any gamma is x(gamma) = sum_j v_j^gamma * dvec_j  (>= 0 on this design).
    subj_cells = {}
    for i in range(len(df)):
        a = np.asarray(df.at[i, "option_a_ratings"], dtype=float)
        b = np.asarray(df.at[i, "option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.shape[0] != V.shape[0]:
            continue
        if np.array_equal(a, b):
            continue  # identical vectors: uninformative
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if ca < cb:
            sparse, dvec = 0, b - a
        elif cb < ca:
            sparse, dvec = 1, a - b
        else:
            wa = float(np.dot(WTIE, a))
            wb = float(np.dot(WTIE, b))
            if wa < wb:
                sparse, dvec = 0, b - a
            else:
                sparse, dvec = 1, a - b
        y = 1.0 if int(df.at[i, "response"]) == sparse else 0.0
        key = tuple(int(round(v)) for v in dvec)
        sid = df.at[i, "subject_id"]
        cells = subj_cells.setdefault(sid, {})
        ent = cells.setdefault(key, [0, 0.0])
        ent[0] += 1
        ent[1] += y

    if not subj_cells:
        return 0.0

    # ---------- 2. per-subject polarity orientation ----------
    # RFI-delta has a ~5.5% conventional-polarity minority; orienting each
    # subject by their overall sparse-choice rate maps that minority onto
    # the majority cluster (removing the huge mixture variance a signed
    # statistic would otherwise suffer). Under GSP every subject's overall
    # sparse rate stays well above 0.5 (min ~0.65), so orientation never
    # misfires there.
    for cells in subj_cells.values():
        tot_n = sum(e[0] for e in cells.values())
        tot_k = sum(e[1] for e in cells.values())
        if tot_n > 0 and tot_k < 0.5 * tot_n:
            for e in cells.values():
                e[1] = e[0] - e[1]

    # ---------- 3. reference mixtures ----------
    # Both theories reduce to the SAME per-subject family
    #     p_i = q*sigmoid(s*x_i(gamma)) + (1-q)*(1 - sigmoid(s*x_i(gamma)))
    # with theory-specific priors:
    #   RFI: s = 2*beta/sum(v^gamma), beta~U[3,8]  -> s in [0.5, 1.69];
    #        q = (1-eps)(1-delta)+eps/2 in [0.85, 0.92] (TIGHT); gamma in [0,1]
    #   GSP: s = beta_raw in [1.1, 3.2] (log-uniform);
    #        q = (1-eps)*m+eps/2 in [0.72, 0.98] (WIDE); gamma in [0, 1.5]
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    keys = sorted({k for cells in subj_cells.values() for k in cells})
    ncell = len(keys)
    if ncell == 0:
        return 0.0
    key_mat = np.array([list(k) for k in keys], dtype=float)  # (C, 12)

    GAMMAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    XG = np.array([key_mat @ np.power(V, g) for g in GAMMAS])  # (7, C)

    # RFI mixture components
    BETAS = np.array([3.0, 4.25, 5.5, 6.75, 8.0])
    Q_RFI = np.array([0.86, 0.885, 0.91])
    QW_RFI = np.array([0.25, 0.5, 0.25])
    rg, rs, rq, rw = [], [], [], []
    for gi in range(5):  # gamma in [0, 1]
        Sg = float(np.sum(np.power(V, GAMMAS[gi])))
        for be in BETAS:
            s = 2.0 * be / Sg
            for qi in range(3):
                rg.append(gi)
                rs.append(s)
                rq.append(Q_RFI[qi])
                rw.append(QW_RFI[qi] / 25.0)
    rg = np.array(rg); rs = np.array(rs); rq = np.array(rq); rw = np.array(rw)

    # GSP mixture components
    SG = np.exp(np.linspace(np.log(1.1), np.log(3.2), 7))
    Q_GSP = np.array([0.735, 0.79, 0.845, 0.90, 0.955])
    gg, gs, gq, gw = [], [], [], []
    w_unit = 1.0 / (7 * 7 * 5)
    for gi in range(7):  # gamma in [0, 1.5]
        for s in SG:
            for q in Q_GSP:
                gg.append(gi); gs.append(s); gq.append(q); gw.append(w_unit)
    gg = np.array(gg); gs = np.array(gs); gq = np.array(gq); gw = np.array(gw)

    def comp_logps(gidx, slopes, qs):
        z = slopes[:, None] * XG[gidx, :]
        sig = sigmoid(z)
        p = qs[:, None] * sig + (1.0 - qs[:, None]) * (1.0 - sig)
        p = np.clip(p, 0.01, 0.99)
        return np.log(p), np.log(1.0 - p)

    LP_r, L1_r = comp_logps(rg, rs, rq)
    LP_g, L1_g = comp_logps(gg, gs, gq)
    LW_r = np.log(rw)
    LW_g = np.log(gw)

    def logmix(LP, L1, LW, k_vec, n_vec):
        ll = (k_vec[None, :] * LP + (n_vec - k_vec)[None, :] * L1).sum(axis=1) + LW
        m = np.max(ll)
        return float(m + np.log(np.sum(np.exp(ll - m))))

    # ---------- 4. per-subject marginal log-likelihood ratio ----------
    vals = []
    for sid in sorted(subj_cells, key=str):
        cells = subj_cells[sid]
        k_vec = np.zeros(ncell)
        n_vec = np.zeros(ncell)
        for ci, key in enumerate(keys):
            ent = cells.get(key)
            if ent is not None:
                n_vec[ci] = float(ent[0])
                k_vec[ci] = float(ent[1])
        if n_vec.sum() <= 0:
            continue
        vals.append(logmix(LP_r, L1_r, LW_r, k_vec, n_vec)
                    - logmix(LP_g, L1_g, LW_g, k_vec, n_vec))

    if not vals:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -1.7633 (var=3.8102)
**Previous candidate values (this loop):**
  - iter 1: -1.1512 (var=3.8140) (Δ vs real +0.6121)
  - iter 2: -1.6557 (var=3.7529) (Δ vs real +0.1075)
  - iter 3 (most recent): -1.2571 (var=2.2075) (Δ vs real +0.5062)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6498 (var=1.2048)
- pi_6: -0.6480 (var=2.0074)
- pi_1: -2.7859 (var=6.5552)
- pi_2: -1.5565 (var=3.7090)
- pi_3: -1.7169 (var=0.8671)
- pi_4: -1.4521 (var=2.5200)
- pi_7: -1.3031 (var=2.4039)
- pi_8: -0.6950 (var=1.2398)
- pi_9: -0.7021 (var=1.9618)
- pi_10: -1.1568 (var=1.5167)
- pi_11: -1.2961 (var=2.5233)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------
    # Theory-neutral psychometric slope test.
    #
    # Both theories are EXACTLY re-parameterizable, on every trial, as
    #   p(choose A) = 0.5 + A * ( sigmoid(s * E_norm(gamma)) - 0.5 ),
    #     E_norm(gamma) = sum_j v_j^gamma (b_j - a_j) / sum_j v_j^gamma,
    # where
    #   GSP  (raw weights):    s = beta_raw * sum(v^gamma) in [9.1, 38.4],
    #                          A = (1-eps)(2m-1) in [0.44, 0.98];
    #   RFI  (normalized wts): s = beta in [3, 8],
    #                          A = +-(1-eps)(1-2 delta), |A| in [0.70, 0.84].
    # The two slope supports are DISJOINT: every GSP subject's effective
    # normalized evidence slope lies above ~9, every RFI subject's below 8.
    # The attenuation (m, eps, delta) and the polarity sign are absorbed
    # by the free coefficient A in [-1, 1], which removes exactly the
    # level/attenuation heterogeneity that destroyed the level-based
    # statistics tried before.
    #
    # Per subject we fit (gamma, s, A) by binomial maximum likelihood on
    # ALL 96 trials (grid search; exact likelihood, no approximation),
    # and report the posterior probability that s > 9.5 under a
    # log-uniform prior on s in [1.2, 48] (profiled over gamma and A).
    #   GSP subjects -> posterior mass almost entirely above 9.5
    #                  (expected value ~0.75-0.85);
    #   RFI subjects -> posterior mass below 9.5
    #                  (expected value ~0.15-0.45).
    # The pooled metric is the mean of the per-subject posterior
    # probabilities; a single subject's slice returns that subject's
    # posterior probability.
    # ------------------------------------------------------------------

    VALS = np.array([0.99, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78,
                     0.76, 0.74, 0.72, 0.50, 0.50], dtype=float)

    GAMMA_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    S_GRID = np.logspace(np.log10(1.2), np.log10(48.0), 40)
    A_GRID = np.linspace(-0.98, 0.98, 41)
    S_THRESH = 9.5

    n_g, n_s, n_a = len(GAMMA_GRID), len(S_GRID), len(A_GRID)

    # per-gamma cue weights and their totals
    W = np.power(VALS[None, :], GAMMA_GRID[:, None])      # (n_g, F)
    WSUM = W.sum(axis=1)                                  # (n_g,)
    ABOVE = S_GRID > S_THRESH                            # (n_s,)

    def _vec(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        return np.asarray(list(x), dtype=float)

    def _subject_value(sub):
        a_list = sub["option_a_ratings"].tolist()
        b_list = sub["option_b_ratings"].tolist()
        r = sub["response"].to_numpy()

        # aggregate identical trials: key -> [n_chose_A, n_total]
        agg = {}
        for i in range(len(sub)):
            a = _vec(a_list[i])
            b = _vec(b_list[i])
            key = (tuple(int(round(v)) for v in a),
                   tuple(int(round(v)) for v in b))
            y = 1 if int(r[i]) == 0 else 0   # response 0 = chose A
            if key in agg:
                agg[key][0] += y
                agg[key][1] += 1
            else:
                agg[key] = [y, 1]

        nll = np.zeros((n_g, n_s, n_a), dtype=float)
        for (ta, tb), (k, n) in agg.items():
            da = np.asarray(ta, dtype=float)
            db = np.asarray(tb, dtype=float)
            d = db - da                     # endorsement difference (B - A)
            if not np.any(d != 0.0):
                continue                    # tie trials: no information
            e = (W @ d) / WSUM              # (n_g,) normalized evidence for A
            for gi in range(n_g):
                x = S_GRID * e[gi]           # (n_s,)
                ax = np.abs(x)
                ex = np.exp(-ax)
                sg = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
                dev = sg - 0.5                               # (n_s,)
                p = 0.5 + A_GRID[None, :] * dev[:, None]     # (n_s, n_a)
                p = np.clip(p, 1e-9, 1.0 - 1e-9)
                nll[gi] -= k * np.log(p) + (n - k) * np.log1p(-p)

        # profile the negative log-likelihood over gamma and A
        prof = nll.min(axis=(0, 2))         # (n_s,)
        w = np.exp(-(prof - prof.min()))   # likelihood weights (log-uniform s prior)
        return float(w[ABOVE].sum() / w.sum())

    subj = data["subject_id"].to_numpy()
    vals = []
    for s in np.unique(subj):
        sub = data[subj == s]
        vals.append(_subject_value(sub))
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.9113 (var=0.0227)
**Previous candidate values (this loop):**
  - iter 1: 0.8223 (var=0.0584) (Δ vs real -0.0891)
  - iter 2: 0.7936 (var=0.0447) (Δ vs real -0.1178)
  - iter 3 (most recent): 0.8536 (var=0.0343) (Δ vs real -0.0577)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7849 (var=0.0519)
- pi_5: 0.5780 (var=0.0892)
- pi_1: 0.9526 (var=0.0112)
- pi_2: 0.9697 (var=0.0075)
- pi_3: 0.5167 (var=0.0397)
- pi_4: 0.6639 (var=0.0486)
- pi_7: 0.9161 (var=0.0235)
- pi_8: 0.8983 (var=0.0297)
- pi_9: 0.8413 (var=0.0356)
- pi_10: 0.8791 (var=0.0271)
- pi_11: 0.9004 (var=0.0321)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd
from collections import defaultdict


def metric(data: pd.DataFrame) -> float:
    """Total log marginal-likelihood ratio (log Bayes factor) of RSD-vgp over GSP.

    Both theories share the identical choice core
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E(gamma) = sum_j v_j**gamma * (b_j - a_j),
    and differ ONLY in the population supports of (gamma, log beta):
        RSD-vgp: gamma ~ U[0,2], log beta ~ U[ln1.8, ln6]
        GSP    : gamma ~ U[0,1.5], log beta ~ U[ln1.1, ln3.2]
    For every subject the exact binomial likelihood of their choices over the
    distinct trial types is integrated (grid) against each theory's prior; the
    attenuation A_eff gets a shared neutral prior (the two theories' A_eff
    ranges nearly coincide, so it must not drive the comparison). Subjects are
    independent, so on the pooled DataFrame the value is the sum of the
    per-subject log Bayes factors (the joint log BF of the whole dataset); on a
    single-subject slice it is that subject's individual log Bayes factor.
    Positive values favor RSD-vgp, negative values favor GSP.
    """
    VAL = np.array([1.0, 0.99, 0.92, 0.85, 0.75, 0.65,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    df = data
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    a_list = df['option_a_ratings'].tolist()
    b_list = df['option_b_ratings'].tolist()
    resp = np.asarray(df['response'].values, dtype=int)
    subj = df['subject_id'].values

    # Hashable string keys for the trial types (avoids unhashable list cells).
    abmap = {}
    key_list = []
    for i in range(n_rows):
        a = a_list[i]
        b = b_list[i]
        ka = ''.join(['1' if int(x) else '0' for x in a])
        kb = ''.join(['1' if int(x) else '0' for x in b])
        key = ka + '#' + kb
        key_list.append(key)
        if key not in abmap:
            abmap[key] = (np.array([int(x) for x in a], dtype=float),
                          np.array([int(x) for x in b], dtype=float))

    tkeys = sorted(abmap.keys())
    tidx = {k: i for i, k in enumerate(tkeys)}
    T = len(tkeys)

    K = defaultdict(lambda: np.zeros(T))  # A-choices per trial type
    N = defaultdict(lambda: np.zeros(T))  # presentations per trial type
    for i in range(n_rows):
        s = subj[i]
        ti = tidx[key_list[i]]
        N[s][ti] += 1.0
        if resp[i] == 0:
            K[s][ti] += 1.0

    # Parameter grids (cover both theories' supports).
    gam = np.arange(0.02, 2.0001, 0.02)      # validity exponent gamma
    lbg = np.arange(0.02, 1.9401, 0.02)      # log raw inverse temperature
    Agr = np.arange(0.42, 0.9901, 0.03)     # effective attenuation A_eff
    G, L, NA = len(gam), len(lbg), len(Agr)

    # E_t(gamma) = sum_j v_j^gamma (b_j - a_j), shape (G, T)
    W = VAL[None, :] ** gam[:, None]
    E = np.empty((G, T))
    for key in tkeys:
        a, b = abmap[key]
        E[:, tidx[key]] = W @ (b - a)

    # s = sigmoid(beta * E) - 0.5, shape (L, G, T)
    beta = np.exp(lbg)
    u = beta[:, None, None] * E[None, :, :]
    au = np.abs(u)
    ex = np.exp(-au)
    sig = np.where(u >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
    s = sig - 0.5

    # p(A) = 0.5 + A_eff * s, shape (NA, L, G, T)
    P = 0.5 + Agr[:, None, None, None] * s[None, :, :, :]
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    logP = np.log(P)
    log1P = np.log1p(-P)
    del P, s, sig, u, au, ex, E, W

    # Log priors.
    logwA = -np.log(NA)  # shared neutral attenuation prior
    logpR_g = np.full(G, np.log(0.5))                          # RSD: U[0,2]
    logpG_g = np.where(gam <= 1.5, np.log(1.0 / 1.5), -np.inf)  # GSP: U[0,1.5]
    loR, hiR = np.log(1.8), np.log(6.0)
    loG, hiG = np.log(1.1), np.log(3.2)
    logpR_b = np.where((lbg >= loR) & (lbg <= hiR),
                       -np.log(hiR - loR), -np.inf)              # RSD: U[ln1.8, ln6]
    logpG_b = np.where((lbg >= loG) & (lbg <= hiG),
                       -np.log(hiG - loG), -np.inf)             # GSP: U[ln1.1, ln3.2]

    PR = logpR_b[None, :, None] + logpR_g[None, None, :] + logwA  # (L, G)
    PG = logpG_b[None, :, None] + logpG_g[None, None, :] + logwA  # (L, G)

    def lse(x):
        m = np.max(x)
        if not np.isfinite(m):
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))

    total_llr = 0.0
    for sid in list(N.keys()):
        kk = K[sid]
        nn = N[sid]
        if nn.sum() <= 0:
            continue
        # binomial log-likelihood on the (NA, L, G) parameter grid
        LL = (np.einsum('t,ijkt->ijk', kk, logP)
              + np.einsum('t,ijkt->ijk', nn - kk, log1P))
        total_llr += lse(LL + PR[None, :, :]) - lse(LL + PG[None, :, :])

    return float(total_llr)
```

**Observed (real) value:** 52.5984 (var=6.4849)
**Previous candidate values (this loop):**
  - iter 1: 36.3328 (var=7.1234) (Δ vs real -16.2656)
  - iter 2: 5.6564 (var=1.8493) (Δ vs real -46.9421)
  - iter 3 (most recent): 20.0709 (var=1.6430) (Δ vs real -32.5275)
**Other theories' values on this metric (for reference):**
- pi_7: 96.6930 (var=15.9176)
- pi_6: -21.1257 (var=0.4647)
- pi_1: -150.7700 (var=0.2496)
- pi_2: 220.1524 (var=4.1508)
- pi_3: -87.8944 (var=0.2065)
- pi_4: 18.4709 (var=6.0537)
- pi_5: -41.9588 (var=1.0440)
- pi_8: 67.3084 (var=8.4566)
- pi_9: 113.8881 (var=19.4047)
- pi_10: 114.0736 (var=12.7298)
- pi_11: 41.8821 (var=6.2799)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Per-subject BOUNDED log Bayes factor (GSP vs RSD-vgp), averaged
    # over subjects:
    #     score_i = tanh( log m_GSP(data_i) - log m_RSD(data_i) )
    #
    # Both theories share the EXACT per-trial likelihood
    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),
    #     E    = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement),
    #     A_eff = (1 - eps) * (2m - 1),
    # so eps and m enter ONLY through A_eff and each subject's 96 choices
    # identify just (gamma, ln beta, A_eff). The theories differ ONLY in
    # the population prior over (gamma, ln beta):
    #     GSP : gamma ~ U[0, 1.5],  ln beta ~ U[ln 1.1, ln 3.2]
    #     RSD : gamma ~ U[0, 2.0],  ln beta ~ U[ln 1.8, ln 6.0]
    # (RSD's two-component gamma mixture is algebraically uniform.)
    # Each marginal is computed by grid quadrature over the theory's exact
    # box, using THIS experiment's true validities
    #     [1, 1, 1, .5, .5, .5, .5, .5, .5, .6, .6, .7]
    # (verified by reproducing the design's own mid-box predictions:
    # B1 = .77, F0 = .42, C2 = .74 at gamma=.75, beta=2, A_eff=.70).
    # A_eff receives a shared uniform prior on [0.42, 0.98], which covers
    # both theories' induced ranges and cancels in the ratio.
    # tanh() bounds each subject's contribution to (-1, 1), capping the
    # heavy log-BF tails that destroyed the between-subject variance
    # budget of the unbounded version.
    # ------------------------------------------------------------------

    V = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7])

    gam = np.linspace(0.0, 2.0, 81)                    # gamma grid (union)
    lnb = np.linspace(np.log(1.1), np.log(6.0), 61)    # ln beta grid (union)
    ave = np.linspace(0.42, 0.98, 12)                   # A_eff grid (shared)
    betas = np.exp(lnb)
    ng, nb, na = len(gam), len(lnb), len(ave)

    gsp_g = gam <= 1.5 + 1e-9
    gsp_b = lnb <= np.log(3.2) + 1e-9
    rsd_b = lnb >= np.log(1.8) - 1e-9
    all_g = np.ones(ng, dtype=bool)

    def subject_score(sub: pd.DataFrame) -> float:
        # ---- pool mirrored twins into canonical (X, Y) cells, X <= Y ----
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                           sub["option_b_ratings"].tolist(),
                           sub["response"].tolist()):
            ta = tuple((int(v) for v in a))
            tb = tuple((int(v) for v in b))
            if ta <= tb:
                X, Y, choseX = ta, tb, (int(r) == 0)
            else:
                X, Y, choseX = tb, ta, (int(r) == 1)
            key = (X, Y)
            cnt = agg.get(key)
            if cnt is None:
                cnt = agg[key] = [0, 0]
            cnt[0] += 1
            if choseX:
                cnt[1] += 1

        D, kX, nT = [], [], []
        for (X, Y), cnt in agg.items():
            d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)
            if not np.any(d != 0.0):
                continue            # exact tie: p = 0.5, cancels in the BF
            D.append(d)
            kX.append(float(cnt[1]))
            nT.append(float(cnt[0]))
        if not D:
            return 0.0
        D = np.vstack(D)                                # (C, F)
        kX = np.asarray(kX)
        kY = np.asarray(nT) - kX

        nf = D.shape[1]
        if nf <= len(V):
            v = V[:nf]
        else:
            v = np.concatenate([V, np.full(nf - len(V), 0.6)])

        W = np.power(v[None, :], gam[:, None])          # (ng, F)
        E = D @ W.T                                     # (C, ng)
        x = E[:, :, None] * betas[None, None, :]         # (C, ng, nb)
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))             # stable logistic
        C = sig.shape[0]
        sig = sig.reshape(C, ng * nb)

        LL = np.empty((na, ng * nb))
        for ia in range(na):
            p = 0.5 + ave[ia] * (sig - 0.5)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            ll = kX[:, None] * np.log(p) + kY[:, None] * np.log1p(-p)
            LL[ia] = ll.sum(axis=0)
        LL = LL.reshape(na, ng, nb)

        def logm(gmask, bmask):
            # grid-average of exp(LL) over the theory's box = marginal
            # likelihood under the uniform prior on that box
            sub_ = LL[:, gmask, :][:, :, bmask]
            m = float(sub_.max())
            return m + float(np.log(np.mean(np.exp(sub_ - m))))

        lg = logm(gsp_g, gsp_b)
        lr = logm(all_g, rsd_b)
        return float(np.tanh(lg - lr))

    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return subject_score(data)
    scores = [subject_score(sub)
              for _, sub in data.groupby("subject_id", sort=False)]
    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.4144 (var=0.4154)
**Previous candidate values (this loop):**
  - iter 1: -0.1552 (var=0.3670) (Δ vs real +0.2592)
  - iter 2: -0.1112 (var=0.4668) (Δ vs real +0.3032)
  - iter 3 (most recent): -0.1479 (var=0.3119) (Δ vs real +0.2665)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2650 (var=0.2313)
- pi_7: -0.2520 (var=0.3854)
- pi_1: 0.7899 (var=0.0150)
- pi_2: -0.9161 (var=0.0663)
- pi_3: 0.7753 (var=0.0098)
- pi_4: 0.0191 (var=0.3421)
- pi_5: 0.5030 (var=0.2038)
- pi_8: -0.3417 (var=0.3534)
- pi_9: -0.2905 (var=0.4581)
- pi_10: -0.2310 (var=0.5023)
- pi_11: -0.2418 (var=0.3912)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Joint log Bayes factor between the two population models.
    # Both theories share the SAME per-trial likelihood
    #   p(A) = 0.5 + A_eff * (sigmoid(beta * E_gamma) - 0.5),
    #   E_gamma = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1).
    # They differ ONLY in the population priors over (gamma, beta, A_eff):
    #   RSD-vgp: gamma ~ U[0,2]; beta ~ logU[1.8,6]; m~U[0.74,1], eps~U[0.03,0.11]
    #   EPAI-xp: gamma ~ 0.55*U[0,1.2] + 0.45*U[1.2,2.4]; beta ~ logU[2,7];
    #            m~U[0.78,0.96], eps~U[0.03,0.10]
    # The metric marginalizes the likelihood over each theory's prior by
    # deterministic quadrature and returns sum_i ln[ m_RSD(data_i)/m_EPAI(data_i) ]
    # over subjects (subjects are independent, so the joint log Bayes factor
    # is exactly the sum of the per-subject log Bayes factors).
    val = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    if data is None or len(data) == 0:
        return 0.0

    # ---------- map trials to unique stimulus types ----------
    a_lists = data['option_a_ratings'].tolist()
    b_lists = data['option_b_ratings'].tolist()
    keys = [(tuple(x), tuple(y)) for x, y in zip(a_lists, b_lists)]

    type_index = {}
    diff_rows = []
    for kk in keys:
        if kk not in type_index:
            type_index[kk] = len(diff_rows)
            diff_rows.append(np.asarray(kk[1], dtype=float) - np.asarray(kk[0], dtype=float))
    D = np.vstack(diff_rows)                      # [T, F]  b - a per type
    T = D.shape[0]
    tidx = np.array([type_index[kk] for kk in keys], dtype=int)

    resp_ser = pd.to_numeric(pd.Series(data['response'].values), errors='coerce')
    resp = resp_ser.to_numpy(dtype=float)
    if np.isnan(resp).any():
        vals = [str(r).strip().upper() for r in data['response'].values]
        resp = np.array([0.0 if v in ('0', 'A') else 1.0 for v in vals])

    uq, sidx = np.unique(np.asarray(data['subject_id'].values), return_inverse=True)
    n_sub = int(len(uq))

    K = np.zeros((n_sub, T))                     # A-choice counts per type
    N = np.zeros((n_sub, T))                     # total trials per type
    isA = (resp == 0).astype(float)
    np.add.at(K, (sidx, tidx), isA)
    np.add.at(N, (sidx, tidx), 1.0)

    # ---------- gamma quadrature: 96 midpoint bins on [0, 2.4] ----------
    ge = np.linspace(0.0, 2.4, 97)
    gm = 0.5 * (ge[:-1] + ge[1:])
    dg = float(ge[1] - ge[0])
    wR_g = np.where(gm <= 2.0, 0.5, 0.0) * dg                 # RSD: U[0,2]
    wE_g = np.where(gm <= 1.2, 0.55 / 1.2, 0.45 / 1.2) * dg    # EPAI mixture
    wR_g = wR_g / wR_g.sum()
    wE_g = wE_g / wE_g.sum()
    G = len(gm)

    # ---------- beta quadrature: 12 midpoint bins in log space ----------
    le = np.linspace(np.log(1.8), np.log(7.0), 13)
    bm = 0.5 * (le[:-1] + le[1:])
    beta = np.exp(bm)

    def _log_overlap(lo, hi):
        lo_c = np.clip(le[:-1], lo, hi)
        hi_c = np.clip(le[1:], lo, hi)
        w = np.maximum(hi_c - lo_c, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    wR_b = _log_overlap(np.log(1.8), np.log(6.0))   # logU[1.8, 6]
    wE_b = _log_overlap(np.log(2.0), np.log(7.0))   # logU[2, 7]
    B = len(beta)

    # ---------- evidence and sigmoid - 0.5 ----------
    Wg = np.power(val[None, :], gm[:, None])         # [G, F] v^gamma
    Ev = Wg @ D.T                                    # [G, T]
    x = beta[:, None, None] * Ev[None, :, :]         # [B, G, T]
    x = np.clip(x, -60.0, 60.0)
    u = 1.0 / (1.0 + np.exp(-x)) - 0.5               # sigmoid(beta*E) - 0.5
    C = B * G
    u_flat = u.reshape(C, T)                         # c = b*G + g

    w_cR = (wR_b[:, None] * wR_g[None, :]).ravel()
    w_cE = (wE_b[:, None] * wE_g[None, :]).ravel()

    # ---------- A_eff quadrature from each theory's (m, eps) uniforms ----------
    def _a_grid(m_lo, m_hi, e_lo, e_hi):
        mm = m_lo + (m_hi - m_lo) * (np.arange(7) + 0.5) / 7.0
        ee = e_lo + (e_hi - e_lo) * (np.arange(4) + 0.5) / 4.0
        A = (2.0 * mm - 1.0)[:, None] * (1.0 - ee)[None, :]
        return A.ravel()

    def _prep(A_vals, w_c):
        pu = 0.5 + A_vals[:, None, None] * u_flat[None, :, :]   # p(A)
        D1 = np.log(pu) - np.log1p(-pu)                        # log p - log(1-p)
        L0 = np.log1p(-pu)                                     # log(1-p)
        nA = A_vals.size
        D1m = D1.reshape(nA * C, T)
        L0m = L0.reshape(nA * C, T)
        logw = (np.full(nA, -np.log(nA))[:, None] + np.log(w_c)[None, :]).ravel()
        return D1m, L0m, logw

    A_R = _a_grid(0.74, 1.00, 0.03, 0.11)
    A_E = _a_grid(0.78, 0.96, 0.03, 0.10)
    D1R, L0R, lwR = _prep(A_R, w_cR)
    D1E, L0E, lwE = _prep(A_E, w_cE)

    def _logsumexp(z):
        m = np.max(z)
        return float(m + np.log(np.sum(np.exp(z - m))))

    total = 0.0
    for i in range(n_sub):
        # log likelihood of subject i's counts at every grid point:
        # sum_t [ K_t*log p_t + (N_t-K_t)*log(1-p_t) ]
        lr = _logsumexp(D1R @ K[i] + L0R @ N[i] + lwR)
        le_i = _logsumexp(D1E @ K[i] + L0E @ N[i] + lwE)
        total += lr - le_i
    return float(total)
```

**Observed (real) value:** 6.0286 (var=0.5984)
**Previous candidate values (this loop):**
  - iter 1: 13.1201 (var=0.2089) (Δ vs real +7.0915)
  - iter 2: 14.0839 (var=0.3100) (Δ vs real +8.0553)
  - iter 3 (most recent): 9.2959 (var=0.2065) (Δ vs real +3.2672)
**Other theories' values on this metric (for reference):**
- pi_7: 5.3165 (var=0.3027)
- pi_8: -30.9335 (var=3.3944)
- pi_1: 136.1063 (var=0.3030)
- pi_2: -106.1209 (var=4.0607)
- pi_3: 135.2209 (var=0.2267)
- pi_4: 38.8942 (var=0.7056)
- pi_5: 7.2875 (var=0.2574)
- pi_6: 15.9085 (var=0.2273)
- pi_9: 6.8983 (var=0.2368)
- pi_10: -9.6424 (var=0.6044)
- pi_11: -0.9777 (var=0.5911)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Estimated population fraction of subjects whose validity exponent
    gamma lies above 2.0 (the hard support edge of RSD-vgp's gamma box),
    measured with the anti-endorsement raw-weight measurement model that
    BOTH theories share per-trial, and stabilized by a weak empirical-Bayes
    smoothing prior so the statistic is well-behaved on single subjects."""
    VAL = np.array([1.0, 1.0, 1.0, 0.74, 0.72, 0.68, 0.63,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    KAPPA = 15.0      # smoothing prior strength (in subjects)
    P0 = 0.075        # smoothing prior mean tail mass
    TAIL = 2.0        # gamma support edge of the competing theory

    if data is None or len(data) == 0:
        return P0

    # ---- tidy parse (list cells -> hashable tuples) ----
    ka = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    kb = [tuple(int(v) for v in x) for x in data['option_b_ratings']]
    resp = np.asarray(data['response'].values).astype(int)
    choseA = (resp == 0).astype(int)

    df = pd.DataFrame({'s': np.asarray(data['subject_id'].values),
                       'ka': ka, 'kb': kb, 'cA': choseA})
    cnt = df.groupby(['s', 'ka', 'kb']).agg(
        n=('cA', 'size'), k=('cA', 'sum')).reset_index()

    uniq = sorted(set(zip(cnt['ka'].values, cnt['kb'].values)))
    tidx = {t: i for i, t in enumerate(uniq)}
    subjects = list(pd.unique(cnt['s']))
    sidx = {s: i for i, s in enumerate(subjects)}
    S = len(subjects)
    T = len(uniq)
    if S == 0 or T == 0:
        return P0

    K = np.zeros((S, T))   # A-choices per (subject, trial type)
    N = np.zeros((S, T))   # trials per (subject, trial type)
    for row in cnt.itertuples(index=False):
        K[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.k)
        N[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.n)

    # ---- diffuse parameter grids (theory-neutral, covers both supports) ----
    gam = np.linspace(0.0, 2.6, 53)                       # validity exponent
    Agrid = np.linspace(0.35, 0.95, 7)                    # effective attenuation
    Bgrid = np.exp(np.linspace(np.log(1.8), np.log(7.0), 7))  # inverse temp
    Wg = VAL[None, :] ** gam[:, None]                     # (53, 13)

    # per-trial-type choice probabilities under the shared core:
    # p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j)
    LP = np.empty((T, 53, 7, 7))
    L1P = np.empty((T, 53, 7, 7))
    for t, (a, b) in enumerate(uniq):
        av = np.array(a, dtype=float)
        bv = np.array(b, dtype=float)
        E = Wg @ (bv - av)                                # (53,)
        x = Bgrid[None, :] * E[:, None]                   # (53, 7)
        ex = np.exp(-np.abs(x))
        sig = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
        P = 0.5 + Agrid[None, :, None] * (sig[:, None, :] - 0.5)  # (53,7,7)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
        LP[t] = np.log(P)
        L1P[t] = np.log1p(-P)

    tail_mask = gam > (TAIL + 1e-9)

    # ---- per-subject posterior P(gamma > 2 | own 96 trials) ----
    s_vals = np.empty(S)
    chunk = 64
    for i0 in range(0, S, chunk):
        Kc = K[i0:i0 + chunk]
        Nc = N[i0:i0 + chunk]
        logL = (np.tensordot(Kc, LP, axes=(1, 0)) +
                np.tensordot(Nc - Kc, L1P, axes=(1, 0)))   # (s,53,7,7)
        m = logL.max(axis=(1, 2, 3), keepdims=True)
        w = np.exp(logL - m)
        post = w.sum(axis=(2, 3))                          # marginal over A,beta
        tot = post.sum(axis=1)
        tot = np.where(tot <= 0.0, 1.0, tot)
        s_vals[i0:i0 + chunk] = post[:, tail_mask].sum(axis=1) / tot

    raw = float(np.mean(s_vals))
    # empirical-Bayes shrinkage toward the prior mean; with one subject the
    # posterior mean is prior-dominated (tight across subjects), with many
    # subjects the data dominate (population tail mass).
    return (KAPPA * P0 + S * raw) / (KAPPA + S)
```

**Observed (real) value:** 0.2983 (var=0.0006)
**Previous candidate values (this loop):**
  - iter 1: 0.0941 (var=0.0002) (Δ vs real -0.2042)
  - iter 2: 0.0843 (var=0.0001) (Δ vs real -0.2141)
  - iter 3 (most recent): 0.0857 (var=0.0001) (Δ vs real -0.2127)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1396 (var=0.0003)
- pi_7: 0.0990 (var=0.0001)
- pi_1: 0.0175 (var=0.0000)
- pi_2: 0.7299 (var=0.0001)
- pi_3: 0.0173 (var=0.0000)
- pi_4: 0.1168 (var=0.0002)
- pi_5: 0.0871 (var=0.0002)
- pi_6: 0.0638 (var=0.0001)
- pi_9: 0.0951 (var=0.0001)
- pi_10: 0.1215 (var=0.0002)
- pi_11: 0.1003 (var=0.0002)

### Experiment 15
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # ==================================================================
    # Population log Bayes factor: log P(data | RSD-vgp) - log P(data | SGAI-r7).
    # Both theories share the identical per-trial choice kernel
    #     p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    #     E = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1),
    # and differ ONLY in their population priors over (gamma, beta, eps, m)
    # plus SGAI's per-trial gamma jitter (prob ~0.11, +/-0.4 around the
    # habitual gamma, capped at [0, 2.2]).  So the marginal likelihood
    # under each theory's stated population prior isolates exactly the
    # claims that separate them:
    #   RSD-vgp: gamma ~ U[0,2] exactly; beta log-U[1.8,6]; eps U[.03,.11];
            #   m U[.74,1]; no jitter.
    #   SGAI-r7: 80% majority (gamma 20%[.8,1.3]/48%[1.3,1.7]/24%[1.7,2.0]/
    #   8%(2.0,2.2]; beta log-U[1.8,6]; eps U[.03,.10]; m U[.8,.98]) plus
    #   20% low-precision cluster (gamma U[0,1]; beta log-U[1.2,1.8];
    #   eps U[.15,.25]); per-trial gamma jitter.
    # The metric integrates the shared kernel over these priors on a grid
    # (SGAI's jitter is folded into the per-trial kernel as a 0.89/0.11
    # mixture over the +/-0.4 gamma window) and returns the total log
    # likelihood ratio.  On a pooled DataFrame this is the dataset log Bayes
    # factor (sum of independent per-subject contributions); on a single
    # subject's slice it is that subject's log-BF contribution.
    # ==================================================================
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    v = np.array([1.0, 1.0, 1.0,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.6])
    nf = len(v)

    def _to_arr(x):
        if isinstance(x, np.ndarray):
            a = x.astype(float)
            if a.size == nf:
                return a
        try:
            a = np.asarray(list(x), dtype=float)
            if a.size == nf:
                return a
        except (TypeError, ValueError):
            pass
        s = str(x).strip().strip("[]()")
        return np.array([float(tok) for tok in s.replace(",", " ").split()])

    A = np.vstack([_to_arr(r) for r in df["option_a_ratings"]])
    B = np.vstack([_to_arr(r) for r in df["option_b_ratings"]])
    resp = df["response"].to_numpy(dtype=float)  # 0 = chose A, 1 = chose B

    # ---- unique trial types ----------------------------------------
    keys = [(tuple(A[i]), tuple(B[i])) for i in range(len(df))]
    index_of = {}
    type_idx = np.empty(len(keys), dtype=int)
    for i, k in enumerate(keys):
        if k not in index_of:
            index_of[k] = len(index_of)
        type_idx[i] = index_of[k]
    types = list(index_of.keys())
    Amat = np.array([np.array(t[0], dtype=float) for t in types])
    Bmat = np.array([np.array(t[1], dtype=float) for t in types])
    T = len(types)

    subj_codes = pd.factorize(df["subject_id"])[0]
    S = int(subj_codes.max()) + 1
    nA = np.zeros((S, T))
    nB = np.zeros((S, T))
    np.add.at(nA, (subj_codes, type_idx), (resp == 0).astype(float))
    np.add.at(nB, (subj_codes, type_idx), (resp == 1).astype(float))

    # ---- master parameter grids ------------------------------------
    gmaster = np.round(np.arange(0.0, 2.2 + 1e-9, 0.05), 6)       # 45 gammas
    bmaster = np.exp(np.linspace(np.log(1.2), np.log(6.0), 48))  # 48 betas

    # E[g, t] = sum_j v_j^g * (B_tj - A_tj)
    Vg = v[None, :] ** gmaster[:, None]
    E = Vg @ (Bmat - Amat).T

    def _sigmoid(x):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    # s[g, b, t] = sigmoid(b * E) - 0.5   (kernel without jitter)
    X = bmaster[None, :, None] * E[:, None, :]
    s_master = _sigmoid(X) - 0.5

    # SGAI per-trial gamma jitter folded into the kernel:
    # s_jit(g) = 0.89*s(g) + 0.11*mean_{gt in [g-0.4, g+0.4] cap [0,2.2]} s(gt)
    delta = 0.11
    s_jit = np.empty_like(s_master)
    for gi in range(len(gmaster)):
        g = gmaster[gi]
        lo = int(np.searchsorted(gmaster, max(0.0, g - 0.4) - 1e-9))
        hi = int(np.searchsorted(gmaster, min(2.2, g + 0.4) + 1e-9))
        if hi <= lo:
            hi = lo + 1
        s_jit[gi] = ((1.0 - delta) * s_master[gi]
                     + delta * s_master[lo:hi].mean(axis=0))

    def _logml(smat, g_idx, g_w, b_idx, b_w, alphas, a_w):
        # marginal log-likelihood for every subject under one population
        # component given by (gamma grid/weights, beta grid/weights,
        # A_eff grid/weights); smat supplies the (possibly jittered) kernel.
        s_sel = smat[np.ix_(g_idx, b_idx)]                 # (ng, nb, T)
        p = 0.5 + alphas[None, None, :, None] * s_sel[:, :, None, :]
        p = np.clip(p, 1e-12, 1.0 - 1e-12)                 # (ng, nb, na, T)
        LP = np.log(p).reshape(-1, T)
        LQ = np.log1p(-p).reshape(-1, T)
        ll = nA @ LP.T + nB @ LQ.T                         # (S, ncombo)
        w = (g_w[:, None, None] * b_w[None, :, None]
             * a_w[None, None, :]).reshape(-1)
        ll = ll + np.log(np.maximum(w, 1e-300))[None, :]
        m = np.max(ll, axis=1)
        return m + np.log(np.sum(np.exp(ll - m[:, None]), axis=1))

    # ---- RSD-vgp population ----------------------------------------
    g_idx = np.arange(0, 41)                    # gamma ~ U[0, 2]
    g_w = np.full(41, 1.0 / 41)
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.11, 4)
    mm = np.linspace(0.74, 1.00, 7)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_R = _logml(s_master, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 low-precision cluster (weight 0.2) -----------------
    g_idx = np.arange(0, 21)                    # gamma ~ U[0, 1]
    g_w = np.full(21, 1.0 / 21)
    b_idx = np.arange(0, 12)                    # beta log-U[1.2, 1.8]
    b_w = np.full(12, 1.0 / 12)
    eps = np.linspace(0.15, 0.25, 4)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_L = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 majority cluster (weight 0.8) ----------------------
    g_idx = np.concatenate([np.arange(16, 26), np.arange(26, 34),
                            np.arange(34, 40), np.arange(40, 45)])
    g_w = np.concatenate([np.full(10, 0.20 / 10), np.full(8, 0.48 / 8),
                          np.full(6, 0.24 / 6), np.full(5, 0.08 / 5)])
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.10, 3)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_M = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    lml_S = np.logaddexp(np.log(0.2) + lml_L, np.log(0.8) + lml_M)

    llr = lml_R - lml_S   # per-subject log Bayes factor (RSD vs SGAI)
    return float(np.sum(llr))
```

**Observed (real) value:** 12.9184 (var=1.0896)
**Previous candidate values (this loop):**
  - iter 1: 25.7574 (var=1.8275) (Δ vs real +12.8390)
  - iter 2: 11.0161 (var=0.6873) (Δ vs real -1.9023)
  - iter 3 (most recent): 25.9021 (var=1.8123) (Δ vs real +12.9837)
**Other theories' values on this metric (for reference):**
- pi_7: 23.7874 (var=2.2777)
- pi_9: -14.1580 (var=0.3187)
- pi_1: -19.5136 (var=0.1437)
- pi_2: 11.8045 (var=1.9732)
- pi_3: -21.6258 (var=0.1020)
- pi_4: 6.5010 (var=0.5862)
- pi_5: -8.3945 (var=0.2766)
- pi_6: 28.0330 (var=1.5827)
- pi_8: 17.1418 (var=1.6894)
- pi_10: 7.5133 (var=1.5781)
- pi_11: 1.5576 (var=1.0977)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Fixed design validities (15 experts, design-time constants).
    V = np.array([1.0, 1.0, 1.0, 0.8, 0.75, 0.65, 0.6,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().lstrip('[').rstrip(']')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            arr = arr.ravel()
        return arr

    # Raw (unnormalized) validity weights at a low-gamma and a high-gamma
    # regime. Under the shared anti-endorsement core, a trial's evidence is
    # E(gamma) = sum_j v_j^gamma (b_j - a_j); the two critical 'flip' rungs
    # of this design are exactly the trials where sign(E) changes between
    # these two regimes (flip points gamma* ~ 0.73 and ~ 1.0).
    w_lo = np.power(V, 0.5)
    w_hi = np.power(V, 1.5)

    # Bayesian (Beta) regularization of the sparse-side rate: 96 pseudo-
    # trials centered at 0.5. Per subject only 16 critical trials exist
    # (binomial SE ~ 0.12), so the raw per-subject rate is unstable; the
    # shrinkage stabilizes individual-subject estimates while leaving the
    # pooled estimate (16 x 25 = 400 critical trials) essentially
    # unchanged.
    KAPPA = 96.0
    M0 = 0.5

    k = 0  # sparse-side choices on flip trials
    n = 0  # flip trials seen

    a_col = data['option_a_ratings'].values
    b_col = data['option_b_ratings'].values
    r_col = data['response'].values

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != V.size or b.size != V.size:
            continue
        e_lo = float(np.sum(w_lo * (b - a)))
        e_hi = float(np.sum(w_hi * (b - a)))
        if e_lo == 0.0 or e_hi == 0.0:
            continue  # tie trial, uninformative
        if (e_lo > 0.0) == (e_hi > 0.0):
            continue  # not a flip trial (sign of evidence is gamma-invariant)
        # Flip trial: low-gamma subjects prefer one side, high-gamma
        # subjects the other. The low-gamma-preferred side is the one with
        # the smaller raw validity weight (the 'count-sparser' side).
        n += 1
        sparse_is_a = (e_lo > 0.0)
        chose_a = (int(float(r_raw)) == 0)
        if chose_a == sparse_is_a:
            k += 1

    if n + KAPPA <= 0:
        return float(M0)
    return float((k + KAPPA * M0) / (n + KAPPA))
```

**Observed (real) value:** 0.4335 (var=0.0024)
**Previous candidate values (this loop):**
  - iter 1: 0.4967 (var=0.0011) (Δ vs real +0.0632)
  - iter 2: 0.4576 (var=0.0007) (Δ vs real +0.0241)
  - iter 3 (most recent): 0.4844 (var=0.0011) (Δ vs real +0.0509)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3750 (var=0.0007)
- pi_7: 0.5335 (var=0.0013)
- pi_1: 0.8058 (var=0.0004)
- pi_2: 0.1931 (var=0.0005)
- pi_3: 0.8549 (var=0.0001)
- pi_4: 0.5201 (var=0.0006)
- pi_5: 0.5212 (var=0.0004)
- pi_6: 0.5759 (var=0.0005)
- pi_8: 0.4018 (var=0.0010)
- pi_10: 0.4754 (var=0.0011)
- pi_11: 0.4498 (var=0.0010)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_CACHE = {}

_VALIDITIES = np.array([1.0, 1.0, 1.0,
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                        0.72, 0.63])


def _to_array(x):
    """Coerce a rating cell (list/tuple/ndarray/str) to a float ndarray."""
    if isinstance(x, str):
        x = [t for t in x.strip('[]() ').split(',') if t.strip() != '']
    return np.asarray(list(x), dtype=float)


def metric(data: pd.DataFrame) -> float:
    """
    Mean per-subject log Bayes factor, RSD-vgp vs HT-AEI-r6.

    Both theories share the exact per-trial likelihood
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E    = sum_j v_j^gamma * (b_j - a_j),
    so the ONLY thing that distinguishes them is the population
    distribution of (gamma, beta). This metric integrates the exact
    binomial likelihood of each subject's 96 choices over each theory's
    stated population prior:
      RSD-vgp : gamma ~ U[0, 2] exactly; beta log-U[1.8, 6.0].
      HT-AEI  : 40% gamma U[0,1]  & beta log-U[2.2, 4.0];
                25% gamma U[1.05,1.35] & beta log-U[2.2, 3.2];
                35% gamma U[2.0,2.1]  & beta log-U[2.2, 2.8].
    with a common neutral uniform prior on A_eff in [0.42, 0.97]
    (the shared range of (1-eps)(2m-1) under both theories).

    Returns the mean over subjects of log m_RSD - log m_HT.
    Positive = data favor RSD-vgp; negative = data favor HT-AEI-r6.
    On a single subject's slice it returns that subject's log Bayes
    factor, so the pooled value is exactly the mean of the per-subject
    values and the between-subject variance is well defined.
    """
    df = data.reset_index(drop=True).copy()

    def _key(x):
        return ''.join([str(int(round(v))) for v in _to_array(x)])

    df['stim'] = (df['option_a_ratings'].apply(_key) + '|' +
                  df['option_b_ratings'].apply(_key))

    stims = sorted(set(df['stim'].tolist()))
    ckey = tuple(stims)

    if ckey not in _CACHE:
        # ---------------- parameter grids ----------------
        g_grid = np.round(np.arange(0.0, 2.1001, 0.01), 10)      # 211 pts
        b_grid = np.exp(np.linspace(np.log(1.8), np.log(6.0), 41))
        A_grid = np.linspace(0.42, 0.97, 15)

        dg = float(g_grid[1] - g_grid[0])
        dlb = float(np.log(b_grid[1]) - np.log(b_grid[0]))
        dA = float(A_grid[1] - A_grid[0])

        # ---------------- population priors ----------------
        # gamma densities (per unit gamma)
        pr_rsd = np.where(g_grid <= 2.0 + 1e-9, 0.5, 0.0)
        pr_ht = (0.40 * np.where(g_grid <= 1.0 + 1e-9, 1.0, 0.0)
                 + 0.25 * np.where((g_grid >= 1.05 - 1e-9) &
                                   (g_grid <= 1.35 + 1e-9),
                                   1.0 / 0.30, 0.0)
                 + 0.35 * np.where((g_grid >= 2.0 - 1e-9) &
                                   (g_grid <= 2.10 + 1e-9), 10.0, 0.0))

        def _logu(lo, hi):
            # log-uniform density on the beta grid
            return np.where((b_grid >= lo - 1e-9) & (b_grid <= hi + 1e-9),
                            1.0 / np.log(hi / lo), 0.0)

        # joint (gamma, beta) grid log-masses for RSD
        dens_rsd = pr_rsd[:, None] * _logu(1.8, 6.0)[None, :]
        lw_rsd = np.full(dens_rsd.shape, -np.inf)
        msk = dens_rsd > 0
        lw_rsd[msk] = np.log(dens_rsd[msk]) + np.log(dg) + np.log(dlb)

        # joint (gamma, beta) grid log-masses for HT (segment-specific beta)
        bw_ht = np.zeros((len(g_grid), len(b_grid)))
        bw_ht[g_grid <= 1.0 + 1e-9] = _logu(2.2, 4.0)
        bw_ht[(g_grid >= 1.05 - 1e-9) & (g_grid <= 1.35 + 1e-9)] = _logu(2.2, 3.2)
        bw_ht[(g_grid >= 2.0 - 1e-9) & (g_grid <= 2.10 + 1e-9)] = _logu(2.2, 2.8)
        dens_ht = pr_ht[:, None] * bw_ht
        lw_ht = np.full(dens_ht.shape, -np.inf)
        msk = dens_ht > 0
        lw_ht[msk] = np.log(dens_ht[msk]) + np.log(dg) + np.log(dlb)

        # common neutral A_eff prior (uniform on the shared range)
        log_a_w = float(np.log(dA / (0.97 - 0.42)))

        # ---------------- per-stimulus likelihood tables ----------------
        first = df.drop_duplicates(subset='stim').set_index('stim')
        a_mat = np.vstack([_to_array(first.loc[s, 'option_a_ratings'])
                           for s in stims])
        b_mat = np.vstack([_to_array(first.loc[s, 'option_b_ratings'])
                           for s in stims])
        D = b_mat - a_mat                                  # (n_u, 14)
        Wg = np.power(_VALIDITIES[:, None], g_grid[None, :])  # (14, n_g)
        E = D @ Wg                                        # (n_u, n_g)

        X = E[:, :, None] * b_grid[None, None, :]          # (n_u, n_g, n_b)
        S = 1.0 / (1.0 + np.exp(-np.clip(X, -60.0, 60.0)))
        core = S - 0.5
        P = 0.5 + A_grid[None, None, None, :] * core[:, :, :, None]
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        logP = np.log(P)
        log1mP = np.log1p(-P)

        _CACHE[ckey] = (logP, log1mP, lw_rsd, lw_ht, log_a_w)
    else:
        logP, log1mP, lw_rsd, lw_ht, log_a_w = _CACHE[ckey]

    stim_idx = {s: i for i, s in enumerate(stims)}
    n_u = len(stims)

    vals = []
    for sid, sdf in df.groupby('subject_id', sort=True):
        nA = np.zeros(n_u)
        nB = np.zeros(n_u)
        for s, v in sdf.groupby('stim')['response']:
            v = np.asarray(v, dtype=float)
            i = stim_idx[s]
            nb = float(np.sum(v))          # response 1 = chose B
            nB[i] = nb
            nA[i] = float(len(v)) - nb    # response 0 = chose A

        # exact binomial log-likelihood on the (gamma, beta, A_eff) grid
        LL = (np.tensordot(nA, logP, axes=([0], [0])) +
              np.tensordot(nB, log1mP, axes=([0], [0])))   # (n_g, n_b, n_A)

        def _marg(lw):
            tot = LL + lw[:, :, None] + log_a_w
            mx = float(np.max(tot))
            return mx + float(np.log(np.sum(np.exp(tot - mx))))

        vals.append(_marg(lw_rsd) - _marg(lw_ht))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.8355 (var=2.6525)
**Previous candidate values (this loop):**
  - iter 1: 0.4479 (var=0.6379) (Δ vs real -0.3875)
  - iter 2: 0.0764 (var=0.3742) (Δ vs real -0.7591)
  - iter 3 (most recent): 0.1307 (var=0.4124) (Δ vs real -0.7048)
**Other theories' values on this metric (for reference):**
- pi_7: 0.3381 (var=0.3081)
- pi_10: -0.2872 (var=0.6994)
- pi_1: 0.2008 (var=0.0004)
- pi_2: -2.3152 (var=0.1061)
- pi_3: 0.2408 (var=0.0009)
- pi_4: 0.1507 (var=0.3046)
- pi_5: 0.0018 (var=0.2637)
- pi_6: 0.2901 (var=0.3029)
- pi_8: -0.0356 (var=1.2437)
- pi_9: 0.2908 (var=1.3375)
- pi_11: -0.0354 (var=0.4369)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Crossing-point population statistic.
    # Per subject: fit the shared core model p(B) = 0.5 - A*(sigmoid(beta*E(gamma)) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j), by grid MLE over (gamma, beta, A),
    # then score s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95) under the flat-prior
    # profile-likelihood posterior over gamma. Metric = mean of s over subjects.
    a_list = data["option_a_ratings"].to_numpy()
    b_list = data["option_b_ratings"].to_numpy()
    A_mat = np.vstack([np.asarray(r, dtype=float) for r in a_list])
    B_mat = np.vstack([np.asarray(r, dtype=float) for r in b_list])
    y = data["response"].to_numpy(dtype=float)
    subj = data["subject_id"].to_numpy()

    n_feat = A_mat.shape[1]
    if n_feat >= 8:
        # design validities: six 1.00 cues, twenty-four 0.50 cues, 0.64, 0.76
        V = np.array([1.0] * 6 + [0.5] * (n_feat - 8) + [0.64, 0.76], dtype=float)
    else:
        V = np.ones(n_feat, dtype=float)

    gam = np.linspace(0.0, 2.5, 126)
    bet = np.geomspace(1.6, 7.5, 8)
    amps = np.linspace(0.25, 0.97, 9)

    Wg = V[None, :] ** gam[:, None]  # (G, F) validity-power weights per gamma

    scores = []
    for s_id in pd.unique(data["subject_id"]):
        m = subj == s_id
        D = (B_mat[m] - A_mat[m]).astype(float)  # (n, F)
        yy = y[m]  # 1 = chose B
        E = D @ Wg.T  # (n, G) raw evidence for A over B
        X = E[:, :, None] * bet[None, None, :]  # (n, G, B)
        d = 0.5 * np.tanh(0.5 * X)  # sigmoid(X) - 0.5, stable, in [-0.5, 0.5]
        ll_prof = np.full(gam.shape[0], -np.inf)  # profile log-likelihood over gamma
        for a in amps:
            P = np.clip(0.5 - a * d, 1e-9, 1.0 - 1e-9)  # p(choose B)
            ll = np.sum(
                yy[:, None, None] * np.log(P)
                + (1.0 - yy)[:, None, None] * np.log(1.0 - P),
                axis=0,
            )  # (G, B)
            ll_prof = np.maximum(ll_prof, ll.max(axis=1))
        w = np.exp(ll_prof - ll_prof.max())
        w = w / w.sum()  # posterior over gamma (flat prior on [0, 2.5])
        p_hi = float(w[gam > 1.95].sum())          # tail mass above RSD's hard cap
        p_gap = float(w[(gam > 1.40) & (gam <= 1.95)].sum())  # HT's empty-gap mass
        scores.append(p_hi - p_gap)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.2077 (var=0.1170)
**Previous candidate values (this loop):**
  - iter 1: -0.1359 (var=0.1346) (Δ vs real +0.0718)
  - iter 2: -0.1817 (var=0.1301) (Δ vs real +0.0261)
  - iter 3 (most recent): -0.2455 (var=0.1583) (Δ vs real -0.0377)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1782 (var=0.1651)
- pi_7: -0.2078 (var=0.1272)
- pi_1: -0.0043 (var=0.0003)
- pi_2: 0.9966 (var=0.0002)
- pi_3: -0.0004 (var=0.0000)
- pi_4: -0.2504 (var=0.1229)
- pi_5: 0.0592 (var=0.0565)
- pi_6: -0.0842 (var=0.0441)
- pi_8: -0.0112 (var=0.3413)
- pi_9: -0.3784 (var=0.2838)
- pi_11: -0.0808 (var=0.2909)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Top-rung dense-preference ceiling probe (RSD-vgp vs JU-AEI).

    Rung trials: one option ('sparse' side) endorsed on exactly the five
    unit-validity cues (features 0-4) and nothing else; the other ('dense'
    side) endorsed on m of the 23 half-validity cues and nothing else.
    The anti-endorsement evidence for the sparse side is
        E(gamma) = m * 2**(-gamma) - 5,
    strictly decreasing in gamma (unit cues have weight 1 for every
    gamma; half cues have weight 2**(-gamma)).

    Per-subject score:  s = r(19) + r(20) + r(21), the summed rate of
    dense-side choices on the three highest non-control rungs (m = 19,
    20, 21; the m = 23 cell is the double-floor control and is excluded).

    HARD CEILING UNDER RSD-vgp: every RSD-vgp subject has gamma <= 2, and
    at gamma = 2 the three evidences are exactly (-0.25, 0, +0.25); by the
    odd symmetry of the logistic,
        p19 + p20 + p21 = 1.5  EXACTLY at gamma = 2,
    and strictly less for every gamma < 2, for every beta, lapse and
    polarity parameter (A_eff >= 0 multiplies a nonpositive sum). So
    s <= 1.5 is a hard, parameter-free, per-subject ceiling under
    RSD-vgp; observed values above it are pure binomial noise.

    UNDER JU-AEI: 9.1% of subjects have habitual gamma in (2.0, 2.2];
    all three top-rung evidences are then negative and s rises up to
    ~2.6 (gamma_h = 2.2, high beta, high A_eff, jitter-marginalized).

    Applied to a single subject's slice: returns that subject's s.
    Applied to pooled multi-subject data: returns the population MAXIMUM
    of s -- the natural extreme-value test of a hard bound.
    """
    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.0

    subj = df["subject_id"].to_numpy()
    resp = df["response"].to_numpy()

    m_arr = np.full(n, -1, dtype=int)
    dense_arr = np.full(n, -1, dtype=int)

    for i in range(n):
        try:
            va = np.asarray(list(df.at[i, "option_a_ratings"]), dtype=int)
            vb = np.asarray(list(df.at[i, "option_b_ratings"]), dtype=int)
        except Exception:
            continue
        if va.size != vb.size or va.size < 6:
            continue
        ua = int(va[:5].sum())
        ha = int(va[5:].sum())
        ub = int(vb[:5].sum())
        hb = int(vb[5:].sum())
        # rung trial: sparse side = exactly the 5 unit cues, dense side =
        # only half-cue endorsements (m of them). Mirrored twins both match.
        if ua == 5 and ha == 0 and ub == 0 and hb >= 6:
            m_arr[i] = hb                      # B is the dense side
            dense_arr[i] = 1 if int(resp[i]) == 1 else 0
        elif ub == 5 and hb == 0 and ua == 0 and ha >= 6:
            m_arr[i] = ha                      # A is the dense side
            dense_arr[i] = 1 if int(resp[i]) == 0 else 0

    sel = m_arr >= 0
    if not sel.any():
        return 0.0

    frame = pd.DataFrame({
        "subject_id": subj[sel],
        "m": m_arr[sel],
        "dense": dense_arr[sel],
    })

    observed = sorted(int(x) for x in pd.unique(frame["m"]))
    target = [x for x in (19, 20, 21) if x in observed]
    if len(target) < 3:
        # fallback: three largest rung levels strictly below the top rung
        below = [x for x in observed if x < observed[-1]]
        pool = below if len(below) >= 3 else observed
        target = pool[-3:]

    sub = frame[frame["m"].isin(target)]
    if len(sub) == 0:
        return 0.0

    per = (
        sub.groupby(["subject_id", "m"])["dense"].mean()
           .groupby(level=0).sum()
    )
    if len(per) == 0:
        return 0.0
    return float(per.max())
```

**Observed (real) value:** 1.0000 (var=0.0729)
**Previous candidate values (this loop):**
  - iter 1: 1.3750 (var=0.1293) (Δ vs real +0.3750)
  - iter 2: 1.6250 (var=0.1041) (Δ vs real +0.6250)
  - iter 3 (most recent): 1.5000 (var=0.0803) (Δ vs real +0.5000)
**Other theories' values on this metric (for reference):**
- pi_7: 1.2500 (var=0.0858)
- pi_11: 2.7500 (var=0.4086)
- pi_1: 1.3750 (var=0.1039)
- pi_2: 3.0000 (var=0.0741)
- pi_3: 1.1250 (var=0.0364)
- pi_4: 2.1250 (var=0.1669)
- pi_5: 0.8750 (var=0.0511)
- pi_6: 1.1250 (var=0.0864)
- pi_8: 2.6250 (var=0.3314)
- pi_9: 2.2500 (var=0.2235)
- pi_10: 2.1250 (var=0.4528)

### Experiment 20
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ---------------- fixed design constants ----------------
    # Actual per-expert validities of this experiment (31 experts:
    # six unit cues, twenty-five 0.50 cues).
    VALS = np.array([1.0] * 6 + [0.5] * 25)

    DELTA = 0.145        # JU-AEI per-trial validity-attention jitter probability (mid of [0.13, 0.16])
    GMAX_RSD = 2.0       # RSD-vgp: hard upper edge of the gamma support
    GMAX_JU = 2.2        # JU-AEI: hard upper edge of the habitual gamma support
    TAIL_LOC = 2.0       # RSD's hard edge = the population tail boundary
    TAIL_SCALE = 0.12    # softness of the apparent-tail indicator
    W_LLR = 0.1          # weight of the profile log-likelihood-ratio component

    # Empirical-Bayes aggregation: prior worth K pseudo-subjects at R_PRIOR.
    # A single subject's slice cannot identify a POPULATION parameter, so the
    # per-slice estimate is prior-dominated (stable across subjects); pooled
    # data dominate as n grows. One formula for every pool size.
    K_PRIOR = 29.0
    R_PRIOR = 0.05

    # ---------------- grids ----------------
    gs = np.linspace(0.0, 2.4, 49)          # gamma grid (step 0.05), up to the jitter clip 2.4
    ng = gs.size
    step = gs[1] - gs[0]
    half_idx = int(round(0.5 / step))      # +/-0.50 jitter half-window -> 10 grid steps
    idx = np.arange(ng)
    lo_idx = np.maximum(0, idx - half_idx)  # window clipped to [0, 2.4]
    hi_idx = np.minimum(ng - 1, idx + half_idx)
    wcnt = (hi_idx - lo_idx + 1).astype(float)

    betas = np.exp(np.linspace(np.log(1.8), np.log(6.0), 12))   # shared beta box
    aeffs = np.linspace(0.45, 0.95, 11)                         # shared A_eff box

    W = np.power(VALS[None, :], gs[:, None])                    # (ng, n_features)
    mask_rsd = gs <= (GMAX_RSD + 1e-9)
    mask_ju = gs <= (GMAX_JU + 1e-9)

    def _sigmoid(x):
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    def _to_vec(x):
        if isinstance(x, str):
            import ast
            x = ast.literal_eval(x)
        return np.asarray(x, dtype=float).ravel()

    def _subject_score(df_s):
        # ---- pool this subject's trials into unique (A, B) cells ----
        cache = {}
        d_list, nA_list, nB_list = [], [], []
        for a_raw, b_raw, resp in zip(df_s['option_a_ratings'].values,
                                      df_s['option_b_ratings'].values,
                                      df_s['response'].values):
            a = _to_vec(a_raw)
            b = _to_vec(b_raw)
            key = (a.tobytes(), b.tobytes())
            if key not in cache:
                cache[key] = len(d_list)
                d_list.append(b - a)          # anti-endorsement evidence direction
                nA_list.append(0)
                nB_list.append(0)
            ci = cache[key]
            if int(resp) == 0:
                nA_list[ci] += 1
            else:
                nB_list[ci] += 1
        n_cells = len(d_list)
        E_mat = np.empty((ng, max(n_cells, 1)))
        for ci in range(n_cells):
            E_mat[:, ci] = W @ d_list[ci]    # E(gamma) per cell, RAW weights
        E_mat = E_mat[:, :n_cells]
        nA = np.asarray(nA_list, dtype=float)[None, :]
        nB = np.asarray(nB_list, dtype=float)[None, :]

        best_rsd = -np.inf   # best fixed-gamma fit with gamma <= 2.0 (RSD-vgp)
        best_ju = -np.inf    # best jittered fit with habit <= 2.2 (JU-AEI)
        best_free = -np.inf  # agnostic fixed-gamma fit over the full grid [0, 2.4]
        ghat = gs[0]

        for beta in betas:
            s0 = _sigmoid(beta * E_mat)                    # (ng, C)
            for a_eff in aeffs:
                q = 0.5 + a_eff * (s0 - 0.5)               # p(A), gamma fixed
                np.clip(q, 1e-9, 1.0 - 1e-9, out=q)
                ll = nA * np.log(q) + nB * np.log(1.0 - q)  # (ng, C)
                ll_free = ll.sum(axis=1)                    # (ng,)
                jf = int(np.argmax(ll_free))
                if ll_free[jf] > best_free:
                    best_free = ll_free[jf]
                    ghat = gs[jf]                           # apparent-tail MLE
                mr = ll_free[mask_rsd].max()
                if mr > best_rsd:
                    best_rsd = mr
                # JU marginal psychometric: (1-delta)*q(habit)
                #   + delta*mean of q over the clipped +/-0.5 window
                cs = np.cumsum(q, axis=0)
                pad = np.vstack([np.zeros((1, q.shape[1])), cs])
                qbar = (pad[hi_idx + 1, :] - pad[lo_idx, :]) / wcnt[:, None]
                qj = (1.0 - DELTA) * q + DELTA * qbar
                np.clip(qj, 1e-9, 1.0 - 1e-9, out=qj)
                ll_ju = (nA * np.log(qj) + nB * np.log(1.0 - qj)).sum(axis=1)
                mj = ll_ju[mask_ju].max()
                if mj > best_ju:
                    best_ju = mj

        # (1) apparent-tail membership: does the subject's fixed-gamma MLE
        #     validity exponent exceed RSD's hard 2.0 edge?
        m_soft = 1.0 / (1.0 + np.exp(-(ghat - TAIL_LOC) / TAIL_SCALE))
        # (2) profile log-likelihood advantage of the jittered model
        s_llr = best_ju - best_rsd
        return float(m_soft + W_LLR * s_llr)

    # ---------------- aggregate over subjects (empirical Bayes) ----------------
    r_vals = []
    for sid in pd.unique(data['subject_id'].values):
        df_s = data[data['subject_id'] == sid]
        r_vals.append(_subject_score(df_s))
    r_vals = np.asarray(r_vals, dtype=float)
    n = r_vals.size
    if n == 0:
        return float(R_PRIOR)
    pe = (n * float(r_vals.mean()) + K_PRIOR * R_PRIOR) / (n + K_PRIOR)
    return float(pe)
```

**Observed (real) value:** 0.0229 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0524 (var=0.0000) (Δ vs real +0.0295)
  - iter 2: 0.0525 (var=0.0000) (Δ vs real +0.0296)
  - iter 3 (most recent): 0.0369 (var=0.0000) (Δ vs real +0.0140)
**Other theories' values on this metric (for reference):**
- pi_11: 0.0649 (var=0.0000)
- pi_7: 0.0431 (var=0.0000)
- pi_1: 0.0067 (var=0.0000)
- pi_2: 0.9578 (var=0.0000)
- pi_3: 0.0072 (var=0.0000)
- pi_4: 0.0461 (var=0.0000)
- pi_5: 0.0182 (var=0.0000)
- pi_6: 0.0187 (var=0.0000)
- pi_8: 0.1479 (var=0.0002)
- pi_9: 0.0846 (var=0.0000)
- pi_10: 0.1754 (var=0.0001)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Heterogeneous-Precision Anti-Endorsement Integration (HP-AEI). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) near 0.69 for the bulk. The population structure is the theory's distinctive claim: (i) gamma is supported on [0, 2] with NO mass above 2.0 (respecting the hard behavioral ceiling of Experiment 19 and the crossing statistic of Experiment 18), with density mildly tilted toward the interior band [0.7, 1.8] (32% on [0, 0.7], 62% on [0.7, 1.8], only 6% on [1.8, 2.0]); (ii) precision is HETEROGENEOUS on a substantially wider box: beta is log-uniform over [1.0, 7.0], with a ~25% shallow-precision subpopulation on beta in [1.0, 2.0] and the remaining 75% log-uniform on [2.0, 7.0]. Shallow-precision subjects compensate with more consistent polarity (m remapped into [0.86, 1.0]), so their anti-endorsement belief is strong but their psychometric slope is shallow. There is NO per-trial gamma jitter and NO per-trial beta noise (both empirically falsified). The key mechanistic insight: a shallow-true-beta subject produces a flattened choice sequence that a fixed-gamma per-subject MLE with a beta floor of ~1.8 can only accommodate by pushing gamma-hat upward — manufacturing APPARENT tail mass (~0.25-0.35 estimated at gamma > 2) with TRUE tail mass exactly zero. This simultaneously preserves every win of the shared core (Exps 1-8, 18) while softening the over-sharp Bayes factors (Exps 11, 13, 15, 17) toward their observed mid-range values and pulling the flip-rate tilt (Exp 16) down toward 0.4335 via the interior gamma tilt. Falsifiable signatures: apparent-tail mass (fixed-gamma MLE) ~0.25-0.35 with true tail mass exactly 0; population beta MLEs stratified into a visible shallow mode near 1.4; per-subject normalized slopes spanning roughly [4, 40] rather than all-steep; flip rates on gamma-invariant sign-change trials in [0.42, 0.46]; hard per-subject ceiling s <= 1.5 on top-rung designs preserved.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Heterogeneous-Precision Anti-Endorsement Integration (HP-AEI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HP-AEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- (1) VALIDITY EXPONENT: tilted interior support, hard cap 2.0 ----
    # gamma_u ~ U[0, 1] mapped through a three-segment mixture:
    #   u in [0.00, 0.32): gamma ~ U[0.0, 0.7]   (32%; near-flat weighting)
    #   u in [0.32, 0.94): gamma ~ U[0.7, 1.8]   (62%; the interior bulk --
    #        density 0.564 vs 0.457 below and 0.30 above: a MILD tilt
    #        toward the band that straddles the flip points of Exp 16)
    #   u in [0.94, 1.00]: gamma ~ U[1.8, 2.0]   ( 6%; trimmed from the
    #        uniform's 10% -- NO mass above 2.0, preserving Exp 19's
    #        hard per-subject ceiling and Exp 18's crossing statistic)
    u = float(parameters["gamma_u"])
    if u < 0.32:
        gamma = 0.7 * (u / 0.32)
    elif u < 0.94:
        gamma = 0.7 + 1.1 * ((u - 0.32) / 0.62)
    else:
        gamma = 1.8 + 0.2 * ((u - 0.94) / 0.06)

    # ---- (2) PRECISION: heterogeneous, wide log-uniform box ----
    # beta_u ~ U[0, 1] mapped through a two-segment mixture:
    #   u in [0.00, 0.25): beta log-U[1.0, 2.0]  (25%; the SHALLOW-
    #        precision subpopulation. Their flattened psychometrics are
    #        what a fixed-gamma MLE with beta floor ~1.8 reads as
    #        apparent tail mass, and what softens the RSD-vs-GSP / 
    #        RSD-vs-SGAI Bayes factors toward their observed values.)
    #   u in [0.25, 1.00]: beta log-U[2.0, 7.0]  (75%; the steep bulk,
    #        extended above pi_7's 6.0 ceiling.)
    # NO per-trial beta noise and NO per-trial gamma jitter: both are
    # empirically dead levers (per-trial precision noise flattens
    # posteriors toward the EB prior mean; gamma jitter donates
    # likelihood to exclusive high-gamma segments).
    bu = float(parameters["beta_u"])
    ln2 = float(np.log(2.0))
    ln7 = float(np.log(7.0))
    if bu < 0.25:
        shallow = True
        log_b = (bu / 0.25) * ln2          # log-uniform on [1.0, 2.0]
    else:
        shallow = False
        log_b = ln2 + ((bu - 0.25) / 0.75) * (ln7 - ln2)  # log-U[2.0, 7.0]
    beta = float(np.exp(log_b))

    epsilon = float(parameters["epsilon"])

    # ---- (3) POLARITY: graded, with precision-consistency coupling ----
    # Shallow-precision subjects hold a MORE consistent anti-endorsement
    # polarity (m remapped into [0.86, 1.0]): low precision costs slope,
    # not conviction, which holds the extremity anchors (Exps 7/8) while
    # the shallow beta does the Bayes-factor softening.
    m_raw = float(parameters["m"])
    if shallow:
        m = 0.86 + 0.14 * ((m_raw - 0.74) / 0.26)
    else:
        m = m_raw

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

`parameters`:
- gamma_u: [0, 1]
- beta_u: [0, 1]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

`rationale`: HP-AEI implements the arbiter's prescription exactly while keeping every empirically validated component of the running-best core (pi_7) frozen. (1) RETAINED CORE: anti-endorsement reading (E = sum_j v_j^gamma (b_j - a_j) with RAW unnormalized validity-power weights), the graded-polarity mixture with A_eff near 0.69, the small lapse, and exact 50% on identical vectors. This alone anchors Exps 1-8 and 18, where pi_7 is already essentially exact (Exp 18: -0.2078 vs real -0.2077). (2) GAMMA POPULATION: support hard-capped at 2.0 with a mild interior tilt (62% on [0.7, 1.8], only 6% on [1.8, 2.0] vs the uniform's 10%). This respects Experiment 19's parameter-free per-subject ceiling (s <= 1.5; real max 1.0) that every theory with mass above 2.0 violates (pi_11: 2.75, pi_10: 2.13), and the added interior mass straddling the ~0.73/~1.0 flip points pulls Experiment 16's flip-rate statistic DOWN from pi_7's 0.5335 toward the observed 0.4335 without any true gamma > 2 signature. (3) THE NOVEL INGREDIENT — HETEROGENEOUS PRECISION: beta log-uniform over a substantially wider box [1.0, 7.0] with a ~25% shallow-precision subpopulation on [1.0, 2.0]. Mechanistically, a shallow-beta subject's flattened choice sequence cannot be represented by any fixed-gamma MLE whose beta grid floors at ~1.8 (Exps 14, 18, 20 estimators) except by inflating gamma-hat — manufacturing the ~0.30 APPARENT tail mass of Experiment 16-class estimators (real 0.2983; pi_7 undershoots at 0.099) with TRUE tail mass exactly zero. The same shallow cluster donates likelihood GSP-ward/SGAI-ward on the Bayes-factor experiments (11, 12, 15, 17), softening pi_7's systematic overpredictions (Exp 11: 96.7 -> toward 52.6; Exp 15: 23.8 -> toward 12.9) because the competing theories' beta boxes ([1.1, 3.2], [1.2, 1.8]) cover the shallow region that pi_7's [1.8, 6.0] box excludes. (4) NO PER-TRIAL NOISE of any kind: Experiment 2 falsified gamma jitter, and five loop passes established that per-trial precision noise flattens posteriors toward the EB prior mean — both mechanisms that sank pi_11 (REJECTED trajectory) are absent. (5) PRECISION-CONSISTENCY COUPLING: shallow-precision subjects remap m into [0.86, 1.0], so their low slope costs no conviction — this holds the extremity/consistency anchors (Exps 7/8, real 0.7044/0.3467) that a naive shallow cluster would erode, while leaving the flat-psychometric signature (and hence the apparent-tail inflation) fully intact since the Exp 14 estimator marginalizes over A_eff anyway. Falsifiable signatures: apparent-tail mass ~0.25-0.35 with true tail mass 0; a visible shallow mode (~1.4) in population beta MLEs; per-subject normalized slopes spanning ~[4, 40]; flip rates in [0.42, 0.46]; s <= 1.5 ceiling on top-rung designs. Where the previous candidate pi_11 was a degenerate patch of the core (its two distinctive mechanisms both empirically falsified, aggregate loss 0.723 vs pi_7's 0.866), HP-AEI changes only the population structure — the single dimension along which pi_7 miscalibrates the four residual experiments (13, 16, 17 and the BF family) — making it a genuine task-invariant competitor rather than a variant.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: HP-AEI is a faithful implementation of the arbiter's prescribed family: the pi_7 core (raw anti-endorsement evidence, graded polarity, lapse, exact ties at 50%) is retained verbatim, gamma is hard-capped at 2.0 with an interior tilt, beta is heterogeneous on [1.0, 7.0] with a 25% shallow cluster, and there is no per-trial noise of any kind. The shared-core anchors hold: Exps 1, 2, 4, 6, 7, 8 are all essentially on top of the real values, and the hard behavioral ceiling of Exp 19 is preserved (1.375 <= 1.5). The softening goals were partially achieved: Exp 11 moved from pi_7's overprediction 96.7 to 36.3 (real 52.6 — right direction, now slightly undershooting), Exp 16 improved from 0.5335 to 0.4967 (real 0.4335), and Exp 17 improved from 0.338 to 0.448 (real 0.836). However, the theory's CENTRAL novel mechanism — shallow-true-beta subjects being read as high-gamma by fixed-gamma MLEs — is quantitatively falsified by its own flagship test: Exp 14's apparent-tail mass came out at 0.0941 versus the real 0.2983 and the theory's own stated signature of 0.25-0.35. That is statistically indistinguishable from pi_7 (0.0990). The precision-consistency coupling (remapping shallow subjects' m into [0.86, 1.0]) is the likely culprit: by preserving their conviction, it leaves the fixed-gamma estimator enough likelihood at moderate gamma that no gamma-hat inflation is needed. Meanwhile the shallow cluster has costs elsewhere: Exp 10 dropped to 0.8223 (real 0.9113; pi_7 was 0.9161 — real subjects are almost uniformly steep-slope), Exp 13 overshot to 13.12 (real 6.03; pi_7's 5.32 was closer), Exp 18 moved off pi_7's near-exact -0.2078 to -0.1359 (real -0.2077), and Exp 9 drifted to -1.15 (real -1.76). So the candidate trades small gains on three Bayes-factor experiments for regressions on four others, while the one decisive signature that justified the mechanism did not materialize.
Rationale: The gate accepted this candidate on aggregate loss, but the distinctive HP-AEI ingredient is not delivering its predicted effect, so tuning should continue on this same family rather than shipping. Concrete diagnosis and in-family adjustments (no change of mechanism family): (1) EXP 14 IS THE FAILURE TO FIX. The 25% shallow cluster on beta in [1.0, 2.0] with m remapped into [0.86, 1.0] produces apparent tail mass of only 0.094 vs the real 0.298. The m-remap is self-defeating: high consistency plus moderate slope is fit by the estimator at ordinary gamma, so no gamma-hat inflation occurs. Loosen the coupling — allow shallow subjects' m down to roughly 0.78-0.82 — and/or deepen the shallow box (e.g., 30-35% mass on beta in [0.8, 1.8]) so their psychometrics genuinely cannot be matched at beta >= 1.8 without inflating gamma-hat. Exps 7/8 currently overshoot slightly (0.7144 vs 0.7044; 0.3679 vs 0.3467), so there is a small amount of extremity budget to spend here without breaking those anchors. (2) EXP 10 TENSION: real subjects' posterior P(s > 9.5) is 0.911; the shallow cluster pulled the candidate to 0.822. Whatever shallow mass is retained must sit low enough in beta that its NORMALIZED slope (beta * sum(v^gamma)) still clears 9.5 for typical gamma — i.e., prefer a narrower, deeper shallow mode (e.g., log-uniform on [1.0, 1.6]) over a wide [1.0, 2.0] slab, and verify against Exp 10 before Exp 14. (3) EXP 13 OVERSHOOT (13.12 vs real 6.03, worse than pi_7's 5.32): trim the steep bulk's beta ceiling back toward 6.0 (the extension to 7.0 donates excess RSD-ward likelihood) and/or slightly reduce the interior gamma tilt. (4) EXP 16 (0.4967 vs 0.4335): the interior tilt helped but undershot; a modest shift of gamma mass from the [0.7, 1.8] band into [0, 0.7] (e.g., 32% -> 38-40%) would move flip rates toward the target [0.42, 0.46], but check Exp 3 (already at -0.78 vs real -0.717) — if Exp 3 degrades further, compensate by trimming the [1.8, 2.0] cap mass instead. (5) EXP 18 regression (-0.1359 vs real -0.2077, where pi_7 was essentially exact): the interior gamma tilt plus shallow beta flattened per-subject posteriors; the low-gamma mass added for Exp 16 should help here too, since the statistic penalizes mass in the (1.40, 1.95] gap — re-tune the tilt jointly against Exps 16 and 18. In short: keep the HP-AEI structure exactly as prescribed, but recalibrate the shallow cluster (deeper/lower beta, weaker m-coupling), pull the steep beta ceiling back toward 6.0, and re-balance the gamma tilt so that Exp 14's apparent-tail signature actually lands in 0.25-0.35 while Exps 10, 13, and 18 stay at or better than pi_7's values.

**Outcome of this advice:** iter 1 candidate loss=0.0958 -> iter 2 candidate loss=0.1044 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate implemented my five in-family adjustments, and the gate rejected it (loss 0.1044 vs the accepted base's 0.0958), so my iteration-1 advice was, on net, wrong and must be revised. What the experiment shows: (a) The flagship fix FAILED in the intended direction — Exp 14's apparent-tail mass moved from 0.0941 to 0.0843 (real 0.2983), i.e. loosening the m-coupling and deepening/narrowing the shallow-beta box did NOT produce gamma-hat inflation. The reason is now clear from the estimator's arithmetic: with raw weights and 13 features, sum(v^gamma) only falls from ~13 (gamma=0) to ~6.4 (gamma=2), so a shallow-beta subject with LOW gamma still has a normalized slope of 11-16 — well inside the fixed-gamma estimator's reachable range (beta >= 1.8 gives slopes 10-23), and the free A-grid absorbs the rest. Low beta only forces gamma-hat above 2 when it is COMBINED with high gamma, pushing the subject's slope below the estimator's floor of ~1.8*sum(v^2.6) ≈ 10. (b) The m-loosening was actively harmful everywhere else: it collapsed Exp 11 from 36.3 to 5.66 (real 52.6) by pushing shallow subjects deep into GSP's (gamma <= 1.5, beta <= 3.2) box, dragged Exps 7 (0.657 vs 0.704), 8 (0.315 vs 0.347), 10 (0.794 vs 0.911), 13 (14.08 vs 6.03), 19 (1.625 vs 1.0, via added binomial noise at low attenuation) and 12 away from their targets. (c) The changes that DID work and should be kept: the gamma tilt to 42/52/6 with the 1.95 cap improved Exps 16 (0.4576 vs real 0.4335), 18 (-0.1817 vs -0.2077), 9 (-1.66 vs -1.76), 1, 5, and — together with the m change — Exp 15 (11.0 vs real 12.9). So the correct reading is: keep the gamma-tilt/cap revision, revert the polarity loosening, and restructure the shallow cluster so that shallow beta is coupled to HIGH gamma rather than to the low-tilted population distribution.
Rationale: Verdict regenerate, building on the REJECTED-gate base (iter 1, loss 0.0958), with the following revised, in-family directions — explicitly not a repeat of the failed iteration-1 advice: (1) REVERT the polarity loosening. Restore the shallow subjects' m remap to [0.86, 1.0] (iter 1's setting). The A-grid of the Exps 14/18/20 estimators absorbs attenuation, so loosening m cannot force gamma inflation; it only adds noise (Exp 19's max-s inflated to 1.625) and donates likelihood to GSP (Exp 11 collapsed to 5.66 vs real 52.6). This reversion alone should recover Exps 7, 8, 10, 11, 13, 19 toward their iter-1 values. (2) RESTRUCTURE THE SHALLOW CLUSTER AS HIGH-GAMMA. The apparent-tail mechanism requires the subject's normalized slope beta*sum(v^gamma) to fall BELOW the Exp 14 estimator's floor (~10-11.6 at beta=1.8, gamma<=2.6) while staying ABOVE Exp 10's 9.5 threshold. The only region satisfying both is roughly beta in [1.45, 1.8] with gamma in [1.6, 1.95]: e.g. (beta=1.5, gamma=1.9) gives Exp 14 slope ~9.9 (below floor -> gamma-hat ~2.3-2.6, tail) and Exp 10 slope ~10.1 (above 9.5). So draw the 25-30% shallow subpopulation with beta log-U[~1.4, 1.8] and gamma from [~1.5, 1.95], instead of pairing low beta with the low-tilted gamma distribution (which produces slopes 11-16 that the estimator fits at ordinary gamma — exactly why iter 2's Exp 14 stayed at 0.084). This is a correlation between the two population draws, not a change of mechanism. (3) REBALANCE THE NON-SHALLOW GAMMA TILT to keep the total mass above the ~1.0 flip point near the level that produced iter 2's good Exp 16 (0.4576) and Exp 18 (-0.1817): since the shallow cluster now contributes ~25-30% above 1.5, shift the remaining 70-75% of gamma mass downward (e.g. ~50% on [0, 0.7], ~45% on [0.7, 1.4], ~5% on [1.4, 1.95]). The extra (1.40, 1.95] gap mass also pushes Exp 18's crossing statistic further toward the real -0.2077. (4) KEEP the 1.95 gamma cap (it protects Exps 19/20) and the steep bulk at beta log-U[2.0, 6.0]; if Exp 13 still overshoots (>= 13), the excess RSD-ward likelihood is coming from the steep bulk's gamma placement, not its beta ceiling — compress that bulk's gamma toward [0.7, 1.4] rather than touching beta again. (5) DIAGNOSTIC ORDER: verify Exp 10 (target >= 0.88) and Exp 11 (target >= 30) BEFORE fine-tuning Exp 14; the restructured shallow cluster exits GSP's gamma box (gamma > 1.5), so per-subject Bayes factors should stop donating to GSP. Accept that Exp 20 (real 0.0229) may drift up from 0.05 — no fixed per-subject (beta, gamma) model with raw weights can simultaneously show tail in Exp 14 and not in Exp 20 (the slope/floor ratios are nearly identical across the two designs), and no existing theory matches both either; Exp 14's error (0.20) dominates Exp 20's (0.03) in the loss budget.

**Outcome of this advice:** iter 2 candidate loss=0.1044 -> iter 3 candidate loss=0.0959 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-3 candidate implemented my iteration-2 advice faithfully: polarity remap reverted to [0.86, 1.0], the shallow cluster restructured as beta log-U[1.4,1.8] coupled to gamma U[1.5,1.95], steep-bulk beta ceiling pulled to 6.0, bulk gamma tilt rebalanced downward. The result is a statistical dead heat with the accepted base (0.0959 vs 0.0958) — my advice recovered the iter-2 regression but did not beat the base, so the gate rejected it and iter 4 must build on the iter-1 base. The critical empirical facts from this round: (1) THE FLAGSHIP MECHANISM FAILED A THIRD TIME. Exp 14's apparent-tail mass is 0.0857 vs real 0.2983 — statistically indistinguishable from pi_7 (0.0990) and from the two previous configurations (0.094, 0.084). The failure is now clearly STRUCTURAL, not parametric: the Exp 14 estimator's free A-grid [0.35, 0.95] and beta-grid [1.8, 7] nest the shared core exactly, and for any in-family subject the estimator can match the small-|E| psychometric by trading beta against A (local slope is proportional to A*beta), so apparent gamma-hat tracks TRUE gamma. Low true beta does not force gamma inflation; only true gamma > 2 would, which the arbiter forbids and which Exp 20's real value (0.0229) independently rules out. Exp 14's 0.298 is unreachable in this family; every theory except pi_2 sits at 0.06-0.14. (2) The high-gamma coupling was all cost, no benefit: it pushed Exp 11 from 36.3 to 20.1 (shallow beta < 1.8 falls outside RSD's beta box, so those subjects stop donating RSD-ward), inflated Exp 19 to the 1.50 ceiling (gamma ~1.95 makes the top-rung evidences near zero, maximizing dense-choice rates; real is 1.00), and worsened Exps 6 and 7. (3) The changes that DID help and are separable from the coupling: the steep-bulk beta ceiling at 6.0 (Exp 13: 13.12 -> 9.30), the 1.95 cap plus downward bulk tilt (Exp 16: 0.497 -> 0.484; Exp 18: -0.136 -> -0.246; Exp 10: 0.822 -> 0.854; Exp 20: 0.052 -> 0.037; Exp 9 improved slightly). These gains came from the bulk re-tilt and beta ceiling, not from the coupled cluster. (4) The largest remaining movable errors in the base are Exp 11 (-16.3), Exp 15 (+12.8), Exp 13 (+7.1), Exp 17 (-0.39), Exp 19 (+0.375), Exp 9 (+0.61) — a cluster of Bayes-factor metrics that all respond to WHERE the population's (gamma, beta) mass sits relative to the comparison theories' boxes.
Rationale: Verdict regenerate, building on the unchanged iter-1 base, with a NEW in-family direction that abandons the failed coupling rather than re-tuning it a fourth time. (1) STOP CHASING EXP 14. Three shallow-cluster configurations (low-beta/low-gamma, low-beta/loose-m, low-beta/high-gamma) all produced ~0.08-0.09 vs real 0.298. The estimator nests the shared core with free (gamma <= 2.6, beta <= 7, A <= 0.95); the A-grid absorbs the beta-attenuation tradeoff exactly, so apparent gamma-hat approximately equals true gamma for every in-family subject. Accept ~0.09-0.12 as the in-family floor and reallocate that budget — this is a concession to the arbiter's own no-mass-above-2.0 constraint, which Exp 20 (real 0.0229) confirms is correct. (2) KEEP the two iter-3 edits that are orthogonal to the coupling: steep-bulk beta ceiling at 6.0 and the 1.95 gamma cap with a downward bulk tilt (~50-55% on [0, 0.7], ~40-45% on [0.7, 1.4], ~5% on [1.4, 1.95]). These delivered the Exp 13/16/18/10/20 gains without the coupling's costs. (3) SHRINK the low-beta shallow cluster from ~25-27% to ~12-15% (beta log-U[1.1, 1.9], gamma from the tilted distribution, m remapped [0.86, 1.0]). Its low-gamma/low-beta members donate GSP-ward in Exp 11 (real 52.6 wants MORE RSD-ward donation than the base's 36.3), and a smaller cluster also raises Exp 10 toward the 0.88+ target by removing sub-9.5-slope subjects. (4) THE NEW LEVER — add a ~12-15% mid-high-gamma mode placed INSIDE RSD's beta box: gamma ~ U[1.5, 1.9], beta log-U[2.0, 3.2], m remapped high. This single component acts on four failing metrics simultaneously: (a) Exp 15 — SGAI's majority concentrates 48% of its density on gamma [1.3, 1.7] and 24% on [1.7, 2.0] (densities 1.2 and 0.8 per unit vs RSD's uniform 0.5), so subjects in [1.5, 1.9] donate SGAI-ward, pulling the base's +12.8 overshoot toward the real 12.9; (b) Exp 11 — gamma > 1.5 is outside GSP's box but inside RSD's, and beta >= 2.0 is inside RSD's box, so these subjects donate RSD-ward, moving 36.3 toward 52.6; (c) Exp 17 — HT-AEI-r6 has an EMPTY gamma gap on (1.35, 2.0), so this mode donates strongly RSD-ward there, moving the base's 0.448 toward the real 0.836; (d) Exp 18 — the mode supplies exactly the (1.40, 1.95] gap mass the crossing statistic needs (base -0.136 vs real -0.208). Crucially, the gamma ceiling of 1.9 keeps Exp 19 safe: at gamma = 1.9 the m=19 rung evidence is +0.09 (sparse side still slightly favored), so the population max-s stays near ~1.0-1.2 instead of the coupling's ceiling-pinning 1.50 — an improvement even over the base's 1.375. Avoid beta below 2.0 in this mode (the [1.8, 2.0] strip is RSD-only in Exp 13 and would worsen that residual overshoot). (5) COMPENSATE Exp 16 by keeping the bulk tilt low enough that total mass above the ~1.0 flip point stays near 33-36% (iter 2's tilt, which produced Exp 16 = 0.4576, is the calibration reference); the mid-high mode's above-1.0 mass must be paid for out of the bulk's [0.7, 1.4] band, not added on top. (6) DIAGNOSTIC ORDER before fine-tuning: Exp 11 >= 40, Exp 15 <= 18, Exp 17 >= 0.6, Exp 18 within [-0.24, -0.17], Exp 19 <= 1.25, Exp 10 >= 0.86. Do NOT repeat the two gate-rejected moves: the m-loosening (iter 2) and the high-gamma/beta<1.8 coupling (iter 3). If the mid-high mode overshoots Exp 18 negative or inflates Exp 19, trim its gamma ceiling toward 1.8 and its mass toward 10% rather than moving beta.

**Outcome of this advice:** iter 3 candidate loss=0.0959 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Heterogeneous-Precision Anti-Endorsement Integration, revision 2 (HP-AEI-r2). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) near 0.69 for the bulk. The population is a four-segment mixture over (gamma, beta), all with NO per-trial noise of any kind (both jitter mechanisms are empirically falsified): (1) a 48% steep bulk on gamma in [0, 0.7] with beta log-uniform on [2.0, 6.0]; (2) a 26% steep mid-bulk on gamma in [0.7, 1.4] with beta log-uniform on [2.0, 6.0] — together the downward-tilted steep bulk whose beta ceiling is pulled back to 6.0; (3) a TRIMMED 13% shallow-precision cluster on beta log-uniform [1.1, 1.9] with gamma uniform on [0, 1.4] and polarity remapped into [0.86, 1.0] (low precision costs slope, not conviction); (4) the NEW ingredient, a 13% mid-high-gamma mode on gamma uniform [1.5, 1.9] with beta log-uniform [2.0, 3.2] — placed fully INSIDE RSD-vgp's (gamma, beta) box but inside the gamma gap (1.35, 2.0) that HT-AEI leaves empty and above GSP's and SGAI's low-gamma concentrations. The gamma support is hard-capped at 1.9 (no mass at or above 2.0), preserving Experiment 19's per-subject behavioral ceiling and Experiment 20's near-zero apparent-tail statistic, while total mass above the ~1.0 evidence-flip point is held near ~32-34%, the calibration that anchors Experiment 16's flip-rate tilt. Falsifiable signatures: per-subject normalized slopes spanning ~[9, 40] with the shallow mode visible near beta ~1.4; flip rates on gamma-invariant sign-change trials in [0.42, 0.46]; a 13% population mode in the (1.4, 1.9] gamma band with beta in [2.0, 3.2]; hard per-subject ceiling s <= 1.5 on top-rung designs with population max near 1.0-1.25; apparent-tail mass (fixed-gamma MLE) accepted at the in-family floor ~0.09-0.12 with TRUE tail mass exactly zero.",
  "predict": "def predict(parameters, state, history):\n    # Heterogeneous-Precision Anti-Endorsement Integration, rev. 2 (HP-AEI-r2).\n    # Changes vs the accepted base (minimal diff): the population mixture\n    # over (gamma, beta) is restructured per the iteration-3 diagnosis --\n    # (a) steep-bulk beta ceiling pulled 7.0 -> 6.0; (b) gamma support\n    # hard-capped at 1.9 with a downward bulk tilt (48% on [0, 0.7],\n    # 26% on [0.7, 1.4]); (c) shallow cluster TRIMMED 25% -> 13% and\n    # re-boxed to beta log-U[1.1, 1.9] with gamma U[0, 1.4]; (d) NEW\n    # 13% mid-high-gamma mode: gamma U[1.5, 1.9], beta log-U[2.0, 3.2].\n    # The per-trial choice core is UNCHANGED from the accepted base.\n    # Stimulus: array-like of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are 0/1 cue values. History is ignored\n    # (no feedback in this task).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"HP-AEI-r2 expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    u = float(parameters[\"gamma_u\"])\n    bu = float(parameters[\"beta_u\"])\n    epsilon = float(parameters[\"epsilon\"])\n    m_raw = float(parameters[\"m\"])\n\n    ln2 = float(np.log(2.0))\n    ln6 = float(np.log(6.0))\n\n    # ---- (1) POPULATION MIXTURE over (gamma, beta) ----\n    # gamma_u ~ U[0, 1] selects one of four segments:\n    #   u in [0.00, 0.13): SHALLOW cluster (13%). beta log-U[1.1, 1.9],\n    #        gamma U[0, 1.4], polarity remapped into [0.86, 1.0].\n    #        Trimmed from the base's 25% on [1.0, 2.0]: the removed mass\n    #        was donating GSP-ward on Exp 11 and leaking Exp 10 below\n    #        the 9.5 slope threshold without ever inflating Exp 14.\n    #   u in [0.13, 0.61): steep bulk-low (48%). gamma U[0, 0.7],\n    #        beta log-U[2.0, 6.0].\n    #   u in [0.61, 0.87): steep bulk-mid (26%). gamma U[0.7, 1.4],\n    #        beta log-U[2.0, 6.0]. The beta ceiling is pulled back to\n    #        6.0 (the base's 7.0 extension donated excess RSD-ward\n    #        likelihood on Exp 13).\n    #   u in [0.87, 1.00]: MID-HIGH mode (13%). gamma U[1.5, 1.9],\n    #        beta log-U[2.0, 3.2], polarity remapped into [0.86, 1.0].\n    #        This is the new lever: gamma > 1.5 sits outside GSP's box\n    #        but inside RSD's (Exp 11 donation), inside HT-AEI's empty\n    #        (1.35, 2.0) gap (Exp 17 donation), straddles SGAI's dense\n    #        [1.3, 2.0] band (Exp 15 softening), and supplies exactly\n    #        the (1.40, 1.95] mass Exp 18's crossing statistic needs.\n    #        The 1.9 ceiling keeps every top-rung evidence positive\n    #        (Exp 19 safe) and keeps true tail mass at zero (Exp 20).\n    if u < 0.13:\n        gamma = 1.4 * (u / 0.13)\n        lo = float(np.log(1.1))\n        hi = float(np.log(1.9))\n        log_b = lo + bu * (hi - lo)\n        hi_consistency = True\n    elif u < 0.61:\n        gamma = 0.7 * ((u - 0.13) / 0.48)\n        log_b = ln2 + bu * (ln6 - ln2)\n        hi_consistency = False\n    elif u < 0.87:\n        gamma = 0.7 + 0.7 * ((u - 0.61) / 0.26)\n        log_b = ln2 + bu * (ln6 - ln2)\n        hi_consistency = False\n    else:\n        gamma = 1.5 + 0.4 * ((u - 0.87) / 0.13)\n        log_b = ln2 + bu * (float(np.log(3.2)) - ln2)\n        hi_consistency = True\n\n    beta = float(np.exp(log_b))\n\n    # ---- (2) POLARITY: graded, with precision/consistency coupling ----\n    # Shallow-precision and mid-high-mode subjects hold a MORE consistent\n    # anti-endorsement polarity (m remapped into [0.86, 1.0]): low slope\n    # costs no conviction, which holds the extremity anchors (Exps 7/8).\n    # The iter-2 loosening of this coupling was gate-REJECTED and is NOT\n    # repeated; the iter-3 high-gamma/beta<1.8 coupling is also NOT\n    # repeated (the mid-high mode keeps beta >= 2.0, inside RSD's box).\n    if hi_consistency:\n        m = 0.86 + 0.14 * ((m_raw - 0.74) / 0.26)\n    else:\n        m = m_raw\n\n    # RAW validity weighting: w_j = v_j^gamma, NO normalization.\n    # This preserves the absolute endorsement-margin scale: a margin-2\n    # conflict carries twice the evidence of a margin-1 conflict,\n    # independent of n_features (the cross-experiment scale separation\n    # that per-experiment normalization provably erases).\n    w = np.power(val, gamma)\n\n    a, b = stim[0], stim[1]\n    # Anti-endorsement (sparse-preferring) evidence for A over B:\n    # each endorsement carried by B counts FOR A, each endorsement\n    # carried by A counts AGAINST A. E = 0 on identical rating vectors.\n    E = float(np.sum(w * (b - a)))\n\n    # Numerically stable logistic (both branches exponentiate a\n    # non-positive argument, so no overflow is possible).\n    x = beta * E\n    if x >= 0.0:\n        sig = 1.0 / (1.0 + np.exp(-x))\n    else:\n        ex = np.exp(x)\n        sig = ex / (1.0 + ex)\n\n    # Graded polarity: with per-trial probability m the subject responds\n    # with the INVERTED (sparse-preferring) polarity, with probability\n    # (1 - m) with the conventional polarity. Continuous, no empty\n    # middle, no mirror cluster. NO per-trial gamma jitter and NO\n    # per-trial beta noise: both are empirically dead levers.\n    p_core = m * sig + (1.0 - m) * (1.0 - sig)\n\n    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.\n    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon\n\n    return np.array([p_a, 1.0 - p_a])",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = probs / probs.sum()  # Guard against float drift.\n    return np.random.choice(len(probs), p=probs)",
  "parameters": {
    "gamma_u": "[0, 1]",
    "beta_u": "[0, 1]",
    "epsilon": "[0.03, 0.11]",
    "m": "[0.74, 1.0]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit of the accepted iter-1 base (HP-AEI) implementing the iteration-3 critic's revised directions exactly, while explicitly NOT repeating the two gate-rejected moves (iter-2's m-loosening; iter-3's high-gamma/beta<1.8 coupling). The per-trial choice core (raw anti-endorsement evidence, graded polarity, lapse, exact 50% ties) is byte-identical to the base; ONLY the population mixture over (gamma, beta) changes. (1) STOP CHASING EXP 14, as advised: three shallow-cluster configurations all landed at 0.08-0.09 vs real 0.298 because the estimator's free A-grid absorbs the beta-attenuation tradeoff, so apparent gamma-hat tracks true gamma. I accept the in-family floor (~0.09-0.12) and reallocate that budget. (2) KEPT the two iter-3 edits that were orthogonal to the failed coupling and delivered gains without costs: the steep-bulk beta ceiling at 6.0 (Exp 13: 13.12 -> ~9.3) and the hard gamma cap with downward bulk tilt (Exps 16/18/10/20 all improved in iter 3). The bulk is 48% on [0, 0.7] + 26% on [0.7, 1.4], and the cap is tightened further to 1.9. (3) TRIMMED the shallow cluster 25% -> 13% on beta log-U[1.1, 1.9] with gamma U[0, 1.4]: its low-gamma members were donating GSP-ward on Exp 11 (real 52.6 wants MORE RSD-ward donation than the base's 36.3) and pulling Exp 10 below the 0.88 target; the m-remap to [0.86, 1.0] is retained. (4) THE NEW LEVER, exactly as prescribed: a 13% mid-high-gamma mode, gamma U[1.5, 1.9], beta log-U[2.0, 3.2], m remapped high. It acts on four failing metrics simultaneously: (a) Exp 15 -- the mode straddles SGAI's dense [1.3, 2.0] band, softening the base's +12.8 overshoot toward the real 12.9; (b) Exp 11 -- gamma > 1.5 is outside GSP's box but inside RSD's, and beta >= 2.0 is inside RSD's box, so these subjects donate RSD-ward, moving 36.3 toward 52.6; (c) Exp 17 -- HT-AEI-r6 has an empty gamma gap on (1.35, 2.0), so this mode donates strongly RSD-ward, moving 0.448 toward the real 0.836; (d) Exp 18 -- the mode supplies exactly the (1.40, 1.95] gap mass the crossing statistic needs (base -0.136 vs real -0.208). The 1.9 gamma ceiling keeps every top-rung evidence in Exp 19 positive (population max-s near 1.0-1.25, well under the 1.5 hard ceiling; the iter-3 coupling pinned 1.50 because gamma ~1.95 zeroes the evidences) and keeps true tail mass at zero for Exp 20. (5) EXP 16 COMPENSATION: total mass above the ~1.0 flip point is held near ~32% (26% bulk-mid contributing ~15%, shallow ~4%, mid-high 13%), close to iter-2's calibration that produced 0.4576 against the real 0.4335 -- the mid-high mode's above-1.0 mass is paid for out of the bulk's [0.7, 1.4] band (26% vs the base's 62% on [0.7, 1.8]), not added on top. (6) DIAGNOSTIC ORDER respected: the configuration is built to land Exp 10 >= 0.86 (only the 13% shallow cluster can fall near the 9.5 slope threshold, and only its high-gamma tail does), Exp 11 >= 40, Exp 15 <= 18, Exp 17 >= 0.6, Exp 18 in [-0.24, -0.17], Exp 19 <= 1.25. Where the base was already essentially exact (Exps 1-8 anchors via the untouched core), nothing in the edit touches those mechanisms, so the anchors should persist."
}
```
```

## Usage

```json
{
  "input_tokens": 90829,
  "output_tokens": 9976,
  "reasoning_tokens": 6157,
  "provider": "Sail Research"
}
```
