# generation_iter_09_attempt_00

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
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 2 (= `pi_8`).

Replace pi_8 (EPAI-xp) with a new theory — Stochastic-Gate Anti-Endorsement Integration (SGAI) — that keeps the empirically confirmed core (endorsements read as evidence AGAINST the endorsed option; E = sum_j v_j^gamma (b_j - a_j) with RAW unnormalized validity weights; p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2) but revises the population and adds within-subject noise, targeting exactly the three failures: (i) TRIAL-LEVEL ATTENTION GATE: on each trial, with probability delta in [0.10, 0.25], the subject re-samples cue attention, drawing an effective per-trial gamma_t from a wide distribution around their mean gamma (or falling back to raw cue counting); this within-subject jitter flattens per-subject gamma identification, which reproduces Experiment 2's inflated tail-mass estimate (~0.25-0.32 vs observed 0.298) WITHOUT placing genuine mass above gamma = 2.2, so Experiment 1's RSD-vs-EPAI log Bayes factor stays mildly positive (~+3 to +7, matching observed +6.03 and avoiding EPAI's -31). (ii) POPULATION PRIORS: gamma from a mixture with mean ~1.0 and support [0, 2.2] — predominantly below 2 with only a 5-10% sliver in (2.0, 2.2] (no mass above 2.2, so Exp 1 stays safe); beta log-uniform on [1.6, 6.0] (floor slightly BELOW RSD's 1.8, since Exps 13/14 hint at lower-precision subjects than either theory admits); m in [0.76, 0.98] and epsilon in [0.03, 0.12], keeping mean A_eff ~0.65-0.70 to preserve the good fits on Exps 9 and 10. (iii) LOW-PRECISION SUBPOPULATION: 10-15% of subjects with beta near the floor and elevated epsilon (0.15-0.25), absorbing the shallow-slope / GSP-ward drift seen in Experiments 13 (predicted ~45-60 vs observed 52.6, fixing RSD's 96.7 over-prediction) and 14 (predicted ~-0.38 to -0.45 vs observed -0.414). Because the per-trial core is unchanged, Experiments 3-10 and 12 should remain well fit; the new degrees of freedom (delta, gamma jitter width, low-precision mixture weight) are each pinned by a currently-missed experiment (2, 13/14, 11 respectively), giving the new theory falsifiable signatures distinct from both pi_7 (hard gamma <= 2, noise-free) and pi_8 (gamma mass in (2.0, 2.4], beta floor 2.0): per-subject psychometric functions with shallower mid-slope than their saturated asymptote implies, elevated trial-to-trial inconsistency on repeated identical trials beyond binomial noise, and a positive-but-attenuated Exp 1 log Bayes factor.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.898)

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


---

### `pi_8` (overall score: 0.819)

**Description**
Extended-Precision Anti-Endorsement Integration (EPAI-xp). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it — a 'fewer red flags' / rarity-implies-quality reading — so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, a raw inverse temperature beta, and a validity exponent gamma, combined as p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population (the 'xp' extension) is characterized by three refinements: (i) gamma follows a concave-then-heavy two-component mixture on [0, 2.4] — 55% uniform on [0, 1.2] and 45% uniform on [1.2, 2.4] — placing roughly 30-35% of subjects above gamma = 1.5 (more than RSD's 25%) while extending support past 2.0 to absorb the observation that steep-gamma subjects behave somewhat shallower than a hard uniform-[0,2] edge would predict; (ii) beta is log-uniform on [2.0, 7.0], raising the floor above RSD's 1.8 (no evidence of any subject below ~1.8) and extending the ceiling to 7 to steepen probe confidence; (iii) m in [0.78, 0.96] and epsilon in [0.03, 0.10] keep the mean effective attenuation A_eff = (1-eps)(2m-1) near 0.68-0.72, matching observed extremity anchors. The resulting subject family is 'steep normalized slope x moderate asymptote': essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) is far above the 9.5 threshold that separates raw-weight from normalized-weight integration, while asymptotic preference stays moderate. This yields falsifiable predictions distinct from RSD-vgp: a non-uniform gamma histogram with mass above 2.0, a beta floor at 2.0 rather than 1.8, per-subject flip-ladder inversion rates on F3/F4 of ~20-27%/6-8%, and deeper Exp 11-style deviations.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Extended-Precision Anti-Endorsement Integration (EPAI-xp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"EPAI-xp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component
    # concave-then-heavy mixture on [0, 2.4] ----
    # gamma_u ~ U[0, 1]. With probability 0.55 the subject draws gamma
    # uniformly from [0, 1.2] (near-flat validity weighting); with
    # probability 0.45 from [1.2, 2.4] (the steep-gating subpopulation).
    # This places ~30-35% of subjects above gamma = 1.5 while extending
    # support past 2.0 (the 'extended precision' regime).
    u = float(parameters["gamma_u"])
    if u < 0.55:
        gamma = 1.2 * (u / 0.55)
    else:
        gamma = 1.2 + 1.2 * ((u - 0.55) / 0.45)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 2.0, ln 7.0]  =>  beta
    # log-uniform over [2.0, 7.0]. Floor raised above RSD-vgp's 1.8
    # (no subject shows evidence of beta below ~1.8); ceiling extended
    # to 7.0 to steepen probe confidence. Calibrated against the RAW
    # evidence scale below so that every subject's effective normalized
    # slope beta * sum(v^gamma) stays far above the 9.5 threshold
    # (min ~ 2.0 * 7.4 ~ 15 on any plausible validity vector).
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
- log_beta: [0.693, 1.946]
- epsilon: [0.03, 0.10]
- m: [0.78, 0.96]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1580 (var=0.0120) vs this=0.1537 (var=0.0082)
- Experiment 2: real=0.1814 (var=0.0118) vs this=0.1571 (var=0.0041)
- Experiment 3: real=-0.7167 (var=0.0939) vs this=-0.7200 (var=0.0585)
- Experiment 4: real=-0.5230 (var=0.0540) vs this=-0.5635 (var=0.0198)
- Experiment 5: real=0.5944 (var=0.0376) vs this=0.5704 (var=0.0179)
- Experiment 6: real=-0.6758 (var=0.0393) vs this=-0.6883 (var=0.0128)
- Experiment 7: real=0.7044 (var=0.0382) vs this=0.7178 (var=0.0239)
- Experiment 8: real=0.3467 (var=0.0108) vs this=0.3438 (var=0.0047)
- Experiment 9: real=-1.7633 (var=3.8102) vs this=-0.6950 (var=1.2398)
- Experiment 10: real=0.9113 (var=0.0227) vs this=0.8983 (var=0.0297)
- Experiment 11: real=52.5984 (var=6.4849) vs this=67.3084 (var=8.4566)
- Experiment 12: real=-0.4144 (var=0.4154) vs this=-0.3417 (var=0.3534)
- Experiment 13: real=6.0286 (var=0.5984) vs this=-30.9335 (var=3.3944)
- Experiment 14: real=0.2983 (var=0.0006) vs this=0.1396 (var=0.0003)


---

### `pi_6` (overall score: 0.683)

**Description**
Graded Sparse-Preference with Heterogeneous Attenuation (GSP), unnormalized-evidence variant. Subjects integrate ALL cues additively with near-flat validity weighting under an anti-endorsement (sparse-preferring) reading: the evidence for option A over B is E = sum_j v_j^gamma (b_j - a_j), where each endorsement (rating = 1) carried by an option counts AGAINST it, so the sparser option accumulates evidence. The cue weights are used RAW (v_j^gamma, unnormalized) so the evidence scale preserves the ABSOLUTE endorsement-count margin of the trial independent of the experiment's feature count — the cross-experiment scale separation (moderate pooled preference on 5-cue shallow-margin trials vs strong preference on 8-cue margin-2 conflicts) that per-experiment normalization provably erases. The anti-endorsement reading is graded per subject: each subject carries a continuous polarity weight m in [0.75, 1.0] (per-trial probability of responding with the inverted polarity), a log-uniformly dispersed inverse temperature beta, and a lapse epsilon in [0.02, 0.12]: p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with identical rating vectors yielding exactly 50%. The population is a continuum of attenuations concentrated on the inverted side (no empty middle, no mirror cluster), with mean effective attenuation (1-eps)(2m-1) ~ 0.70 matching the observed extremity anchors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Graded Sparse-Preference with Heterogeneous Attenuation (GSP),
    # UNNORMALIZED-evidence variant.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GSP expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    # log-uniform inverse temperature: beta = exp(log_beta),
    # log_beta ~ U[ln 1.1, ln 3.2]  =>  beta log-uniform over [1.1, 3.2],
    # calibrated to the RAW (unnormalized) evidence scale below.
    # Lower bound raised from 0.8 to 1.1 (the interpolation lever);
    # upper bound capped at 3.2 (guardrail for Exp4/Exp6 overshoot).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization by
    # sum(w). This preserves the absolute endorsement-margin scale:
    # the evidence magnitude reflects the raw count margin of the
    # trial (a margin-2 conflict carries twice the evidence of a
    # margin-1 conflict) independent of n_features.
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. The sparser option accumulates
    # evidence. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic of the inverted evidence.
    # (both branches exponentiate a non-positive argument, so no
    # overflow is possible)
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED polarity (sparse-preferring), with probability
    # (1 - m) with the conventional polarity. This is the expectation of
    # a per-trial Bernoulli polarity draw against a continuous,
    # strongly-inverted per-subject weight m in [0.75, 1.0].
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
- gamma: [0, 1.5]
- log_beta: [0.095, 1.163]
- epsilon: [0.02, 0.12]
- m: [0.75, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1580 (var=0.0120) vs this=0.1693 (var=0.0082)
- Experiment 2: real=0.1814 (var=0.0118) vs this=0.1793 (var=0.0080)
- Experiment 3: real=-0.7167 (var=0.0939) vs this=-0.6950 (var=0.0763)
- Experiment 4: real=-0.5230 (var=0.0540) vs this=-0.5237 (var=0.0265)
- Experiment 5: real=0.5944 (var=0.0376) vs this=0.5414 (var=0.0227)
- Experiment 6: real=-0.6758 (var=0.0393) vs this=-0.6900 (var=0.0330)
- Experiment 7: real=0.7044 (var=0.0382) vs this=0.7544 (var=0.0264)
- Experiment 8: real=0.3467 (var=0.0108) vs this=0.3467 (var=0.0078)
- Experiment 9: real=-1.7633 (var=3.8102) vs this=-0.6480 (var=2.0074)
- Experiment 10: real=0.9113 (var=0.0227) vs this=0.7849 (var=0.0519)
- Experiment 11: real=52.5984 (var=6.4849) vs this=-21.1257 (var=0.4647)
- Experiment 12: real=-0.4144 (var=0.4154) vs this=0.2650 (var=0.2313)
- Experiment 13: real=6.0286 (var=0.5984) vs this=15.9085 (var=0.2273)
- Experiment 14: real=0.2983 (var=0.0006) vs this=0.0638 (var=0.0001)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0945 -> ACCEPTED
- iter 2: loss=0.0893 -> ACCEPTED
- iter 3: loss=0.0802 -> ACCEPTED
- iter 4: loss=0.0808 -> REJECTED
- iter 5: loss=0.0984 -> REJECTED
- iter 6: loss=0.0736 -> ACCEPTED
- iter 7: loss=0.0696 -> ACCEPTED
- iter 8: loss=0.0735 -> REJECTED
- iter 9: loss=0.0827 -> REJECTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.0696 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.1743 (var=0.0074) (Δ vs real +0.0163)
  - iter 2: 0.1363 (var=0.0062) (Δ vs real -0.0217)
  - iter 3: 0.1420 (var=0.0048) (Δ vs real -0.0160)
  - iter 4: 0.1297 (var=0.0067) (Δ vs real -0.0283)
  - iter 5: 0.1423 (var=0.0054) (Δ vs real -0.0157)
  - iter 6: 0.1650 (var=0.0047) (Δ vs real +0.0070)
  - iter 7: 0.1323 (var=0.0039) (Δ vs real -0.0257)
  - iter 8: 0.1327 (var=0.0064) (Δ vs real -0.0253)
  - iter 9 (most recent): 0.1273 (var=0.0052) (Δ vs real -0.0307)
**Other theories' values on this metric (for reference):**
- pi_1: 0.3480 (var=0.0087)
- pi_2: -0.2963 (var=0.0068)
- pi_3: 0.2473 (var=0.0026)
- pi_4: 0.1530 (var=0.0069)
- pi_5: 0.2150 (var=0.0067)
- pi_6: 0.1693 (var=0.0082)
- pi_7: 0.1660 (var=0.0056)
- pi_8: 0.1537 (var=0.0082)

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
  - iter 1: 0.1757 (var=0.0103) (Δ vs real -0.0057)
  - iter 2: 0.1550 (var=0.0084) (Δ vs real -0.0264)
  - iter 3: 0.1579 (var=0.0091) (Δ vs real -0.0236)
  - iter 4: 0.1750 (var=0.0079) (Δ vs real -0.0064)
  - iter 5: 0.1471 (var=0.0069) (Δ vs real -0.0343)
  - iter 6: 0.1921 (var=0.0071) (Δ vs real +0.0107)
  - iter 7: 0.1643 (var=0.0050) (Δ vs real -0.0171)
  - iter 8: 0.1857 (var=0.0087) (Δ vs real +0.0043)
  - iter 9 (most recent): 0.1657 (var=0.0067) (Δ vs real -0.0157)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8664 (var=0.0110)
- pi_1: 0.1536 (var=0.0134)
- pi_3: 0.2557 (var=0.0053)
- pi_4: 0.2579 (var=0.0137)
- pi_5: 0.1479 (var=0.0061)
- pi_6: 0.1793 (var=0.0080)
- pi_7: 0.1764 (var=0.0097)
- pi_8: 0.1571 (var=0.0041)

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
  - iter 1: -0.6417 (var=0.0973) (Δ vs real +0.0750)
  - iter 2: -0.6500 (var=0.0639) (Δ vs real +0.0667)
  - iter 3: -0.7183 (var=0.0700) (Δ vs real -0.0017)
  - iter 4: -0.7267 (var=0.0372) (Δ vs real -0.0100)
  - iter 5: -0.7467 (var=0.0564) (Δ vs real -0.0300)
  - iter 6: -0.7867 (var=0.0459) (Δ vs real -0.0700)
  - iter 7: -0.7367 (var=0.0557) (Δ vs real -0.0200)
  - iter 8: -0.8267 (var=0.0527) (Δ vs real -0.1100)
  - iter 9 (most recent): -0.7333 (var=0.0469) (Δ vs real -0.0167)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0033 (var=0.0461)
- pi_3: 1.0483 (var=0.0759)
- pi_2: 0.7083 (var=0.0512)
- pi_4: -0.5050 (var=0.1068)
- pi_5: -0.7033 (var=0.1239)
- pi_6: -0.6950 (var=0.0763)
- pi_7: -0.7067 (var=0.0731)
- pi_8: -0.7200 (var=0.0585)

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
  - iter 1: -0.5176 (var=0.0245) (Δ vs real +0.0055)
  - iter 2: -0.5486 (var=0.0230) (Δ vs real -0.0256)
  - iter 3: -0.5549 (var=0.0288) (Δ vs real -0.0318)
  - iter 4: -0.5053 (var=0.0282) (Δ vs real +0.0178)
  - iter 5: -0.5378 (var=0.0213) (Δ vs real -0.0148)
  - iter 6: -0.5234 (var=0.0238) (Δ vs real -0.0004)
  - iter 7: -0.5347 (var=0.0187) (Δ vs real -0.0117)
  - iter 8: -0.5631 (var=0.0166) (Δ vs real -0.0400)
  - iter 9 (most recent): -0.5504 (var=0.0265) (Δ vs real -0.0273)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5548 (var=0.0148)
- pi_1: 0.0081 (var=0.0061)
- pi_2: 0.5897 (var=0.0315)
- pi_4: -0.4252 (var=0.0349)
- pi_5: -0.5389 (var=0.0255)
- pi_6: -0.5237 (var=0.0265)
- pi_7: -0.5575 (var=0.0274)
- pi_8: -0.5635 (var=0.0198)

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
  - iter 1: 0.5826 (var=0.0166) (Δ vs real -0.0118)
  - iter 2: 0.5759 (var=0.0190) (Δ vs real -0.0185)
  - iter 3: 0.5444 (var=0.0276) (Δ vs real -0.0500)
  - iter 4: 0.5577 (var=0.0199) (Δ vs real -0.0367)
  - iter 5: 0.6233 (var=0.0117) (Δ vs real +0.0289)
  - iter 6: 0.5701 (var=0.0175) (Δ vs real -0.0243)
  - iter 7: 0.5493 (var=0.0192) (Δ vs real -0.0451)
  - iter 8: 0.5978 (var=0.0135) (Δ vs real +0.0034)
  - iter 9 (most recent): 0.5985 (var=0.0172) (Δ vs real +0.0041)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0009 (var=0.0089)
- pi_4: 0.4379 (var=0.0326)
- pi_2: -0.6876 (var=0.0238)
- pi_3: 0.0089 (var=0.0056)
- pi_5: 0.5894 (var=0.0647)
- pi_6: 0.5414 (var=0.0227)
- pi_7: 0.5915 (var=0.0172)
- pi_8: 0.5704 (var=0.0179)

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
  - iter 1: -0.6983 (var=0.0346) (Δ vs real -0.0225)
  - iter 2: -0.6946 (var=0.0207) (Δ vs real -0.0188)
  - iter 3: -0.7200 (var=0.0202) (Δ vs real -0.0442)
  - iter 4: -0.7004 (var=0.0233) (Δ vs real -0.0246)
  - iter 5: -0.7321 (var=0.0243) (Δ vs real -0.0563)
  - iter 6: -0.7163 (var=0.0214) (Δ vs real -0.0404)
  - iter 7: -0.6696 (var=0.0274) (Δ vs real +0.0062)
  - iter 8: -0.6929 (var=0.0262) (Δ vs real -0.0171)
  - iter 9 (most recent): -0.6913 (var=0.0272) (Δ vs real -0.0154)
**Other theories' values on this metric (for reference):**
- pi_4: -0.5012 (var=0.0594)
- pi_1: 0.0138 (var=0.0135)
- pi_2: 0.7354 (var=0.0335)
- pi_3: -0.0125 (var=0.0078)
- pi_5: -0.6912 (var=0.1205)
- pi_6: -0.6900 (var=0.0330)
- pi_7: -0.6850 (var=0.0317)
- pi_8: -0.6883 (var=0.0128)

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
  - iter 1: 0.6444 (var=0.0238) (Δ vs real -0.0600)
  - iter 2: 0.7222 (var=0.0235) (Δ vs real +0.0178)
  - iter 3: 0.6811 (var=0.0253) (Δ vs real -0.0233)
  - iter 4: 0.7089 (var=0.0274) (Δ vs real +0.0044)
  - iter 5: 0.7356 (var=0.0224) (Δ vs real +0.0311)
  - iter 6: 0.6933 (var=0.0239) (Δ vs real -0.0111)
  - iter 7: 0.7000 (var=0.0253) (Δ vs real -0.0044)
  - iter 8: 0.7256 (var=0.0240) (Δ vs real +0.0211)
  - iter 9 (most recent): 0.6889 (var=0.0195) (Δ vs real -0.0156)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7611 (var=0.0118)
- pi_4: 0.5167 (var=0.0365)
- pi_1: 0.6944 (var=0.0505)
- pi_2: 0.7267 (var=0.0418)
- pi_3: 0.6067 (var=0.0143)
- pi_6: 0.7544 (var=0.0264)
- pi_7: 0.7111 (var=0.0259)
- pi_8: 0.7178 (var=0.0239)

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
  - iter 1: 0.3300 (var=0.0047) (Δ vs real -0.0167)
  - iter 2: 0.3308 (var=0.0065) (Δ vs real -0.0158)
  - iter 3: 0.3542 (var=0.0062) (Δ vs real +0.0075)
  - iter 4: 0.3279 (var=0.0068) (Δ vs real -0.0187)
  - iter 5: 0.3442 (var=0.0057) (Δ vs real -0.0025)
  - iter 6: 0.3433 (var=0.0069) (Δ vs real -0.0033)
  - iter 7: 0.3696 (var=0.0031) (Δ vs real +0.0229)
  - iter 8: 0.3625 (var=0.0047) (Δ vs real +0.0158)
  - iter 9 (most recent): 0.3500 (var=0.0054) (Δ vs real +0.0033)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2367 (var=0.0084)
- pi_5: 0.3758 (var=0.0028)
- pi_1: 0.3283 (var=0.0115)
- pi_2: 0.3421 (var=0.0094)
- pi_3: 0.1929 (var=0.0036)
- pi_6: 0.3467 (var=0.0078)
- pi_7: 0.3521 (var=0.0065)
- pi_8: 0.3438 (var=0.0047)

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
  - iter 1: -0.7799 (var=1.9369) (Δ vs real +0.9833)
  - iter 2: -0.5997 (var=1.4966) (Δ vs real +1.1636)
  - iter 3: -0.9729 (var=2.3275) (Δ vs real +0.7903)
  - iter 4: -0.9791 (var=1.8786) (Δ vs real +0.7841)
  - iter 5: -0.8579 (var=1.5113) (Δ vs real +0.9054)
  - iter 6: -0.9790 (var=1.4118) (Δ vs real +0.7843)
  - iter 7: -0.9047 (var=1.7376) (Δ vs real +0.8586)
  - iter 8: -0.6717 (var=1.9883) (Δ vs real +1.0916)
  - iter 9 (most recent): -0.8936 (var=1.3509) (Δ vs real +0.8697)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6498 (var=1.2048)
- pi_6: -0.6480 (var=2.0074)
- pi_1: -2.7859 (var=6.5552)
- pi_2: -1.5565 (var=3.7090)
- pi_3: -1.7169 (var=0.8671)
- pi_4: -1.4521 (var=2.5200)
- pi_7: -1.3031 (var=2.4039)
- pi_8: -0.6950 (var=1.2398)

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
  - iter 1: 0.8920 (var=0.0248) (Δ vs real -0.0193)
  - iter 2: 0.8539 (var=0.0408) (Δ vs real -0.0574)
  - iter 3: 0.8422 (var=0.0362) (Δ vs real -0.0691)
  - iter 4: 0.8139 (var=0.0549) (Δ vs real -0.0974)
  - iter 5: 0.8244 (var=0.0422) (Δ vs real -0.0870)
  - iter 6: 0.8319 (var=0.0485) (Δ vs real -0.0794)
  - iter 7: 0.8657 (var=0.0366) (Δ vs real -0.0457)
  - iter 8: 0.8835 (var=0.0328) (Δ vs real -0.0278)
  - iter 9 (most recent): 0.8448 (var=0.0417) (Δ vs real -0.0665)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7849 (var=0.0519)
- pi_5: 0.5780 (var=0.0892)
- pi_1: 0.9526 (var=0.0112)
- pi_2: 0.9697 (var=0.0075)
- pi_3: 0.5167 (var=0.0397)
- pi_4: 0.6639 (var=0.0486)
- pi_7: 0.9161 (var=0.0235)
- pi_8: 0.8983 (var=0.0297)

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
  - iter 1: 30.7611 (var=3.5302) (Δ vs real -21.8373)
  - iter 2: 34.5122 (var=5.6399) (Δ vs real -18.0862)
  - iter 3: 109.3492 (var=7.6061) (Δ vs real +56.7508)
  - iter 4: 58.8717 (var=7.7056) (Δ vs real +6.2733)
  - iter 5: 154.8748 (var=19.0966) (Δ vs real +102.2763)
  - iter 6: 94.4860 (var=13.2270) (Δ vs real +41.8875)
  - iter 7: 88.3846 (var=7.0698) (Δ vs real +35.7862)
  - iter 8: 101.7443 (var=16.7677) (Δ vs real +49.1459)
  - iter 9 (most recent): 114.6804 (var=19.7859) (Δ vs real +62.0820)
**Other theories' values on this metric (for reference):**
- pi_7: 96.6930 (var=15.9176)
- pi_6: -21.1257 (var=0.4647)
- pi_1: -150.7700 (var=0.2496)
- pi_2: 220.1524 (var=4.1508)
- pi_3: -87.8944 (var=0.2065)
- pi_4: 18.4709 (var=6.0537)
- pi_5: -41.9588 (var=1.0440)
- pi_8: 67.3084 (var=8.4566)

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
  - iter 1: -0.0564 (var=0.3984) (Δ vs real +0.3580)
  - iter 2: -0.2105 (var=0.4711) (Δ vs real +0.2039)
  - iter 3: -0.3105 (var=0.4005) (Δ vs real +0.1040)
  - iter 4: -0.1558 (var=0.4879) (Δ vs real +0.2586)
  - iter 5: -0.2233 (var=0.4208) (Δ vs real +0.1911)
  - iter 6: -0.2154 (var=0.4284) (Δ vs real +0.1990)
  - iter 7: -0.3951 (var=0.4055) (Δ vs real +0.0193)
  - iter 8: -0.3948 (var=0.4687) (Δ vs real +0.0196)
  - iter 9 (most recent): -0.2170 (var=0.4733) (Δ vs real +0.1974)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2650 (var=0.2313)
- pi_7: -0.2520 (var=0.3854)
- pi_1: 0.7899 (var=0.0150)
- pi_2: -0.9161 (var=0.0663)
- pi_3: 0.7753 (var=0.0098)
- pi_4: 0.0191 (var=0.3421)
- pi_5: 0.5030 (var=0.2038)
- pi_8: -0.3417 (var=0.3534)

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
  - iter 1: 4.1711 (var=0.5507) (Δ vs real -1.8576)
  - iter 2: 10.4280 (var=0.3365) (Δ vs real +4.3994)
  - iter 3: 4.6351 (var=0.3723) (Δ vs real -1.3935)
  - iter 4: 4.8981 (var=0.4358) (Δ vs real -1.1305)
  - iter 5: 0.1662 (var=0.6703) (Δ vs real -5.8624)
  - iter 6: -1.8767 (var=0.7093) (Δ vs real -7.9053)
  - iter 7: 8.8091 (var=0.2079) (Δ vs real +2.7805)
  - iter 8: 5.4044 (var=0.4781) (Δ vs real -0.6242)
  - iter 9 (most recent): 1.8610 (var=0.1316) (Δ vs real -4.1677)
**Other theories' values on this metric (for reference):**
- pi_7: 5.3165 (var=0.3027)
- pi_8: -30.9335 (var=3.3944)
- pi_1: 136.1063 (var=0.3030)
- pi_2: -106.1209 (var=4.0607)
- pi_3: 135.2209 (var=0.2267)
- pi_4: 38.8942 (var=0.7056)
- pi_5: 7.2875 (var=0.2574)
- pi_6: 15.9085 (var=0.2273)

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
  - iter 1: 0.1015 (var=0.0001) (Δ vs real -0.1968)
  - iter 2: 0.0983 (var=0.0001) (Δ vs real -0.2000)
  - iter 3: 0.1360 (var=0.0002) (Δ vs real -0.1623)
  - iter 4: 0.1030 (var=0.0001) (Δ vs real -0.1953)
  - iter 5: 0.1509 (var=0.0002) (Δ vs real -0.1475)
  - iter 6: 0.1633 (var=0.0002) (Δ vs real -0.1350)
  - iter 7: 0.1302 (var=0.0002) (Δ vs real -0.1682)
  - iter 8: 0.1719 (var=0.0002) (Δ vs real -0.1265)
  - iter 9 (most recent): 0.1134 (var=0.0001) (Δ vs real -0.1849)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1396 (var=0.0003)
- pi_7: 0.0990 (var=0.0001)
- pi_1: 0.0175 (var=0.0000)
- pi_2: 0.7299 (var=0.0001)
- pi_3: 0.0173 (var=0.0000)
- pi_4: 0.1168 (var=0.0002)
- pi_5: 0.0871 (var=0.0002)
- pi_6: 0.0638 (var=0.0001)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Stochastic-Gate Anti-Endorsement Integration, revision 7 (SGAI-r7). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma_t (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population is bimodal in precision: a dominant high-precision cluster (80%; beta log-uniform on [1.8, 6.0] — the floor restored to the RSD boundary so the majority's beta support no longer spills below BOTH theory boxes of the Exp 13 grid, which is what flipped the RSD-vs-EPAI log Bayes factor negative in r6) whose habitual validity exponent is distributed 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0], and an 8% sliver in (2.0, 2.2], none above 2.2 — plus a 20% low-precision cluster (beta log-uniform on [1.2, 1.8], habitual gamma uniform on [0, 1.0], diffuse lapse 0.15-0.25 restored per two converging data points that the diffuse setting maximizes its GSP-ward Bayes-factor pull). On each trial, with probability delta ~ 0.10, the subject re-samples cue attention, drawing an effective per-trial exponent from a symmetric distribution around their OWN habitual gamma (gamma_t ~ U[max(0, gamma-0.4), min(2.2, gamma+0.4)]). Falsifiable signatures: psychometric mid-slopes shallower than saturated asymptotes imply, supra-binomial trial-to-trial inconsistency on repeated identical trials, a bimodal precision population straddling the RSD/GSP beta boundary at 1.8, a mildly positive RSD-vs-EPAI log Bayes factor, and per-subject gamma posteriors that spill above 2.0 for the steep-gamma bins without any habitual mass above 2.2.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Stochastic-Gate Anti-Endorsement Integration, revision 7 (SGAI-r7).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SGAI-r7 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Population: habitual validity exponent gamma (majority) ----
    # gamma_u ~ U[0, 1]. The confirmed anchor distribution (the only
    # setting that has ever put Exp 11 in band while keeping Exp 13 in
    # band): 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0],
    # and an 8% sliver in (2.0, 2.2]. NO habitual mass above 2.2
    # (guardrail for Exp 13).
    u = float(parameters["gamma_u"])
    if u < 0.20:
        gamma = 0.8 + 0.5 * (u / 0.20)
    elif u < 0.68:
        gamma = 1.3 + 0.4 * ((u - 0.20) / 0.48)
    elif u < 0.92:
        gamma = 1.7 + 0.3 * ((u - 0.68) / 0.24)
    else:
        gamma = 2.0 + 0.2 * ((u - 0.92) / 0.08)

    # ---- Precision subpopulation (bimodal) ----
    # 20% of subjects are low-precision: beta log-uniform on [1.2, 1.8]
    # (deep inside the GSP box, below the RSD floor of 1.8), DIFFUSE
    # lapse epsilon in [0.15, 0.25] (restored: two converging data
    # points -- iter 3 and iter 5/6 -- show the diffuse setting
    # maximizes this cluster's GSP-ward pull on the Exp 12 tanh BF,
    # while every sharpening regressed it), and habitual gamma forced
    # LOW (uniform on [0, 1.0]). The remaining 80% draw beta
    # log-uniformly over [1.8, 6.0]: the r6 floor of 1.6 sat below
    # BOTH theory boxes of the Exp 13 grid (RSD starts at 1.8, EPAI at
    # 2.0), flipping the RSD-vs-EPAI log Bayes factor negative
    # (-1.88 vs observed +6.03). Restoring the floor to the RSD
    # boundary 1.8 (the midpoint between the in-band iter-4 floor 2.0
    # and the sign-flipping iter-6 floor 1.6) returns the majority to
    # RSD-identified territory while still leaving the lower-gamma
    # majority bins (gamma <= 1.5) with beta in [1.8, 3.2] inside the
    # GSP overlap, preserving part of r6's Exp 1/2 gains.
    if float(parameters["prec_u"]) < 0.20:
        beta = float(np.exp(float(parameters["lp_log_beta"])))
        epsilon = float(parameters["lp_epsilon"])
        gamma = 1.0 * u  # low-precision subjects keep gamma <= 1.0
    else:
        beta = float(np.exp(float(parameters["log_beta"])))
        epsilon = float(parameters["epsilon"])

    m = float(parameters["m"])
    delta = float(parameters["delta"])

    # ---- Trial-level stochastic attention gate (RELATIVE draw) ----
    # With probability delta (~0.10) the subject re-samples cue
    # attention for THIS trial, drawing an effective exponent from a
    # SYMMETRIC distribution AROUND THEIR OWN habitual gamma with a
    # +/-0.4 width. The draw is relative, so the low-precision
    # cluster's shallow gamma is never globally overridden, while
    # steep-gamma subjects still jitter across the 2.0 boundary on a
    # minority of trials. Confirmed dead lever for Exp 14; kept only
    # in its gentlest, least harmful form.
    if np.random.random() < delta:
        lo = max(0.0, gamma - 0.4)
        hi = min(2.2, gamma + 0.4)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features.
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
    # (1 - m) with the conventional polarity.
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
- epsilon: [0.03, 0.10]
- m: [0.80, 0.98]
- delta: [0.10, 0.12]
- prec_u: [0, 1]
- lp_log_beta: [0.182, 0.588]
- lp_epsilon: [0.15, 0.25]
- validities: validities

`rationale`: Minimal-diff edit of the ACCEPTED SGAI-r6 base, applying exactly the iter-6 critic's two prescribed corrections and nothing else. The per-trial likelihood core (anti-endorsement reading, raw v^gamma_t weights, m/eps polarity-lapse shell), the confirmed majority-gamma anchor (20/48/24/8, none above 2.2), the relative +/-0.4 gate at delta ~0.10, the 20% LP cluster size, and the m floor of 0.80 are all untouched. EDIT 1 — MAJORITY BETA FLOOR RAISED FROM 1.6 TO 1.8 (log_beta range [0.470, 1.792] -> [0.588, 1.792]): the r6 floor of 1.6 sat below BOTH theory boxes of Exp 13's quadrature grid (RSD's beta prior starts at 1.8, EPAI's at 2.0), so the shallow low-beta majority subjects scored EPAI-ward and flipped the RSD-vs-EPAI log Bayes factor to -1.88 against the observed +6.03. 1.8 is the critic's prescribed midpoint between the in-band iter-4 floor (2.0 -> Exp 13 = +4.90, Exp 11 = 58.9) and the sign-flipping iter-6 floor (1.6 -> -1.88, 94.5); it restores the majority to RSD-identified territory (expected Exp 13 in roughly +2..+5, Exp 11 dropping from 94.5 toward 60-75, observed 52.6) while the lower-gamma majority bins with beta in [1.8, 3.2] remain in the GSP overlap, preserving much of r6's Exp 1/2 accuracy (r6's slight overshoots there — 0.165/0.192 vs 0.158/0.181 — should shrink toward observed, well inside the +/-0.02 guardrail). EDIT 2 — LP EPSILON REVERTED FROM [0.10, 0.18] TO THE DIFFUSE [0.15, 0.25]: two independent data points now agree (iter 3 vs iters 5/6) that the diffuse setting maximizes the LP cluster's GSP-ward pull on Exp 12 (-0.310 at iter 3, the loop best, vs -0.215/-0.223 after each sharpening), while sharpening never helped Exps 9/10/11. Expected: Exp 12 recovering to roughly -0.28..-0.32 (target <= -0.28, observed -0.414), Exp 9 drifting from -0.98 toward -1.1. FLAGGED AS STRUCTURAL, NOT CHASED: Exp 14 (expected ~0.12-0.15 vs observed 0.298 — four iterations confirm that neither trial-level gamma jitter (free A_eff in the measurement model absorbs it as attenuation) nor majority spill under the prescribed gamma <= 2.2 support cap can close a ~30-between-subject-SD gap, and every chase has damaged Exps 11/13 far more than it helped Exp 14) and the Exp 9 residual (-1.0 vs -1.76 — no in-family lever has ever moved it below -0.97; the theories that reach it do so via mechanisms outside this family). Guardrails for this candidate: Exp 13 > 0 (ideally +2..+5), Exp 11 in [50, 80], Exp 12 <= -0.28, Exp 10 >= 0.83 (the floor raise removes r6's shallowest majority subjects, so the slope posterior should hold or improve from 0.832), Exps 1/2 within +/-0.02, Exps 3-8 essentially unchanged, aggregate loss < 0.0736.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: SGAI was accepted by the loss gate, and it preserves the core anti-endorsement/raw-weight fits well: Experiments 1-6, 8, 10 are all essentially on top of the observed values (e.g., Exp 1: 0.174 vs 0.158; Exp 4: -0.518 vs -0.523; Exp 5: 0.583 vs 0.594; Exp 6: -0.698 vs -0.676; Exp 10: 0.892 vs 0.911). Exp 13 also landed in the arbiter's prescribed band (4.17 vs observed 6.03, target +3 to +7), repairing pi_8's -31 catastrophe. However, the candidate fails on exactly the three experiments the SGAI revisions were built to fix, and the failures are diagnostic. (1) Exp 14 (tail mass): 0.1015 vs observed 0.2983 — statistically indistinguishable from pi_7's 0.0990. The trial-level gate as implemented (delta ~0.17, gamma_t redrawn UNIFORMLY over [0, 2.2]) does not inflate the per-subject posterior mass above gamma = 2. The reason is mechanistic: a uniform redraw mostly dilutes the habitual signal toward a shallow, near-linear component, and the Exp 14 measurement model absorbs that dilution by lowering its free A_eff (grid down to 0.35) — not by shifting posterior gamma upward. To make subjects LOOK high-gamma, the gated trials must mimic steep-validity-weighting behavior, i.e., the re-draw must be biased toward the upper end of the support, not uniform. (2) Exp 12 (tanh GSP-vs-RSD BF): -0.056 vs observed -0.414 (pi_8 achieved -0.342). The 13% low-precision subpopulation with beta in [1.6, 2.0] is too small and sits too close to the RSD beta floor (1.8) to pull the bounded BF meaningfully GSP-ward; most of its beta support [1.8, 2.0] is INSIDE the RSD box, so these subjects are scored as weak RSD, not GSP. (3) Exp 11 (RSD-vs-GSP joint log BF): 30.8 vs observed 52.6 — the majority population is not confidently RSD-like enough. Note the resolution of the apparent Exp 11 / Exp 12 tension is bimodality: a dominant cluster firmly inside the RSD box (beta well above 1.8) plus a somewhat larger, DEEPER low-precision cluster (beta down to ~1.3-1.5, gamma <= 1.2) that lands inside the GSP box. The current 13% at [1.6, 2.0] straddles the boundary and captures neither. Secondary misses: Exp 9 (-0.78 vs -1.76; the added noise pulls the per-subject RFI-vs-GSP BF toward 0, whereas enlarging/deepening the GSP-like cluster would push it negative as observed) and Exp 7 (0.644 vs 0.704; mean A_eff ~0.67 is slightly too attenuated — both Exp 7 and Exp 8 are under, so a small increase in effective attenuation helps both).
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps polarity-lapse shell, stochastic gate, low-precision subpopulation) with these in-family recalibrations: (a) RESTRUCTURE THE GATE DRAW, not just its rate: on gated trials draw gamma_t from an upper-biased distribution over the support — e.g., gamma_t ~ U[1.2, 2.2] or a Beta-shaped density concentrated near the top of [0, 2.2] (optionally with a small point mass at gamma_t = 0 for raw cue counting). Uniform redraws are absorbed as A_eff attenuation; upper-biased redraws produce the steep-validity choice patterns that the Exp 14 measurement model attributes to gamma > 2, which is what the observed 0.2983 requires. Consider also raising delta toward [0.25, 0.40] — the sign-invariant contrast metrics (Exps 1-6) and the saturated-extremity metrics (Exps 7/8) are largely insensitive to gamma-jitter on high-margin trials, and Exp 10 has headroom (0.892 vs 0.911). (b) DEEPEN AND SLIGHTLY ENLARGE the low-precision cluster: 15-18% of subjects with beta log-uniform on roughly [1.3, 2.0] (below the RSD floor so they genuinely land in the GSP box) and epsilon 0.15-0.25; keep their gamma low (<= 1.2). This pushes Exp 12 toward -0.35 to -0.45 and Exp 9 toward -1.5 or lower, matching observation. (c) STRENGTHEN THE MAJORITY CLUSTER'S RSD CREDENTIALS to lift Exp 11 from ~31 toward ~50 and protect Exp 13 as the gate draw moves upward: shift the main beta log-uniform range up (e.g., [2.0, 6.0]) and concentrate habitual gamma mass in [0.8, 1.8] with the (2.0, 2.2] sliver reduced to ~5% or kept only via the gate. (d) NUDGE mean A_eff from ~0.67 to ~0.70 (raise the m floor slightly or lower the eps ceiling) to close the small undershoots on Exps 7 and 8. Monitor Exp 13 (keep the log BF positive, ideally 4-7) and Exp 10 (keep the slope posterior ~0.89-0.92) as the guardrails; if the upper-biased gate drags Exp 13 below ~+3, trade off by lowering delta rather than re-flattening the draw. Do not place habitual mass above gamma = 2.2 — that is what flipped pi_8 to -31 on Exp 13.

**Outcome of this advice:** iter 1 candidate loss=0.0945 -> iter 2 candidate loss=0.0893 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: SGAI-r2 was accepted by the loss gate (0.0945 -> 0.0893) and several of the in-family recalibrations worked exactly as intended: Exp 7 improved to 0.722 vs 0.704 (the A_eff nudge), Exp 11 rose from 30.8 to 34.5, and Exp 12 moved from -0.056 to -0.21 — both in the right direction. However, the iteration's PRIMARY objective failed, and two guardrails slipped. (1) EXP 14 DID NOT MOVE: 0.0983 vs observed 0.2983, essentially unchanged from iter 1's 0.1015 and indistinguishable from pi_7's 0.0990. The upper-biased gate draw (gamma_t ~ U[1.2, 2.2] at delta 0.25-0.40) was implemented faithfully and had NO effect on the estimated tail mass. This is now the second confirmation of a mechanistic fact: per-trial gamma jitter — whatever its shape — cannot inflate the Exp 14 posterior mass above gamma = 2, because the measurement model's free A_eff (grid down to 0.35) absorbs mixture noise as attenuation rather than as steepness. The gate is a dead lever for Exp 14. (2) THE GATE IS NOW ACTIVELY COSTLY: Exp 9 went the WRONG way (-0.78 -> -0.60 vs observed -1.76) and Exp 10 dropped (0.892 -> 0.854 vs 0.911) — within-subject jitter flattens exactly the per-subject psychometric identification these two metrics rely on. (3) A STRUCTURAL BUG-LIKE INTERACTION: the gate draw U[1.2, 2.2] applies to LOW-PRECISION subjects too, overriding their habitual gamma <= 1.2 on 25-40% of trials. This makes the supposedly GSP-box cluster behave steeply on a third of its trials, diluting the bimodal population structure — likely why Exp 12 only reached -0.21 instead of the predicted -0.35..-0.45, and why Exp 9 did not go more negative. The arbiter's original spec was a draw 'around their mean gamma', not a global upper-biased draw. (4) Exp 13 overshot the guardrail band (+10.4 vs observed 6.03, target +3..+7): the majority cluster is now slightly TOO RSD-like. (5) Remaining misses: Exp 11 (34.5 vs 52.6), Exp 12 (-0.21 vs -0.414), Exp 9 (-0.60 vs -1.76), Exp 14 (0.098 vs 0.298) — all population-level. Exps 1-6, 8 remain essentially on top of the data.
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, stochastic gate, bimodal precision population), with these recalibrations: (a) MAKE THE GATE DRAW RELATIVE TO THE SUBJECT'S OWN HABITUAL GAMMA, per the arbiter's original spec ('a wide distribution around their mean gamma'): e.g., gamma_t ~ U[gamma, min(gamma + 1.0, 2.2)] or a symmetric wide draw centered on gamma. The current global U[1.2, 2.2] draw overrides the low-precision cluster's shallow gamma on a third of its trials, destroying the bimodality that Exps 9 and 12 measure; a relative draw restores the GSP cluster's per-subject credentials at zero cost elsewhere. (b) LOWER DELTA back toward the arbiter's original [0.10, 0.20]. Two iterations have now shown the gate cannot deliver Exp 14's tail mass (free A_eff absorbs the jitter) while it demonstrably degrades Exp 9 (-0.60 vs -1.76) and Exp 10 (0.854 vs 0.911); a gentler gate recovers both. (c) REDISTRIBUTE HABITUAL GAMMA UPWARD IN THE MAJORITY CLUSTER as the NEW primary lever for Exp 14: concentrate mass in roughly [1.3, 2.0] with mean ~1.6-1.7 (e.g., 15% in [0.8, 1.3], 50% in [1.3, 1.7], 30% in [1.7, 2.0], keep the 5% sliver in (2.0, 2.2], none above 2.2). Subjects with true gamma near 1.8-2.0 have per-subject posteriors that SPILL above 2.0, which is the only in-family route to the observed 0.298 tail mass — and it simultaneously pushes Exp 11 up toward 52.6, pulls Exp 13 back from +10.4 into the +3..+7 band, steepens Exp 10 back toward 0.91, pushes Exp 9 negative, and nudges Exps 1/2 up toward 0.158/0.181 (both currently slightly under). (d) DEEPEN THE LOW-PRECISION CLUSTER modestly (to ~20%, beta log-uniform on [1.2, 1.8], gamma <= 1.0, epsilon 0.15-0.25): with fix (a) in place these subjects now stay genuinely shallow on gated trials, so each one contributes much more GSP pull; this should move Exp 12 from -0.21 toward -0.41 and Exp 9 toward -1.5 without needing a large cluster that would sink Exp 11. Guardrails: keep Exp 13 in +3..+7 (the gamma shift should reduce it — verify it does not undershoot), keep Exp 10 in 0.89-0.92, and do NOT raise mean A_eff further (Exp 7 at 0.722 is already slightly over the observed 0.704). Do not place habitual mass above gamma = 2.2.

**Outcome of this advice:** iter 2 candidate loss=0.0893 -> iter 3 candidate loss=0.0802 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: SGAI-r3 was accepted by the loss gate (0.0893 -> 0.0802) and the recalibrations that worked are clear: Exp 13 is back inside the guardrail band (4.64 vs observed 6.03, target +3..+7), Exp 12 continues its monotone march toward the target (-0.056 -> -0.21 -> -0.31 vs -0.414), Exp 9 improved (-0.60 -> -0.97 vs -1.76), Exp 3 is essentially exact (-0.718 vs -0.717), and Exps 4, 6, 8 remain close. The relative gate draw also fixed the bimodality-destroying interaction diagnosed in iter 2. However, the iteration's PRIMARY lever overshot badly on its main target while barely moving its intended one. (1) EXP 11 OVERSHOT MASSIVELY: 109.3 vs observed 52.6 (var 6.5) — worse than pi_7's 96.7 and roughly triple the iter-2 value of 34.5. The habitual-gamma redistribution (35% of the majority at 1.7-2.2) made the population too strongly RSD-like on the joint log Bayes factor. My iter-2 advice to push gamma up was directionally right but the step size was roughly twice too large; the correct setting sits between the iter-2 and iter-3 distributions. (2) EXP 14 BARELY MOVED: 0.136 vs observed 0.298 (var 0.0006 — the miss is ~40 between-subject SDs). The 'posterior spill' from gamma 1.7-2.0 subjects plus the 5% sliver delivered only +0.038 over iter 2. Combined with iter 2's null result on the gate, this confirms that neither per-trial jitter nor modest spill can close the gap; the remaining in-family mass must come from (a) a larger (2.0, 2.2] sliver (the arbiter explicitly allows up to 10%; currently only 5%), and (b) more low-precision subjects whose diffuse per-subject posteriors spread mass across the gamma grid including above 2.0. (3) EXP 10 SLIPPED BELOW GUARDRAIL: 0.892 -> 0.854 -> 0.842 vs observed 0.911 — the fingerprint of the enlarged low-precision cluster (their per-subject slope posteriors sit well below the majority's). (4) EXP 5 DEGRADED (0.583 -> 0.544 vs 0.594), likely also from the upward gamma shift. (5) Small persistent undershoots on Exps 1/2 (0.142/0.158 vs 0.158/0.181). Net: the candidate is the best base so far, but the population layer is now mis-calibrated in a well-identified direction: too much high-gamma majority mass (Exp 11 +57) and too little tail-generating mass (Exp 14 -0.16).
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, relative stochastic gate, bimodal precision population), with a MIDPOINT correction — explicitly not an oscillation, but settings between the iter-2 and iter-3 values: (a) PULL THE MAJORITY HABITUAL GAMMA BACK HALFWAY: redistribute to roughly 20% [0.8, 1.3], 50% [1.3, 1.7], 25% [1.7, 2.0], 5% (2.0, 2.2]. Iter 2's distribution gave Exp 11 = 34.5, iter 3's gave 109.3; the observed 52.6 sits between, closer to iter 2's side. This should also partially recover Exp 5 (0.544 -> ~0.57+) and stop the Exp 1/2 undershoot from deepening. (b) ENLARGE THE LOW-PRECISION CLUSTER FROM 20% TO ~24-25%, keeping it DEEP (beta log-uniform [1.2, 1.8], habitual gamma <= 1.0, epsilon 0.15-0.25). Each genuinely GSP-box subject contributes a negative per-subject log BF to Exp 11 (pulling 109 toward ~50-70 without touching the majority), pushes Exp 12 from -0.31 toward -0.41, pushes Exp 9 from -0.97 toward -1.5 or lower, and — because their flat psychometrics yield diffuse gamma posteriors — adds tail mass above 2.0 for Exp 14. This is the single lever that moves Exps 9, 11, 12, and 14 in their observed directions simultaneously. (c) RAISE THE (2.0, 2.2] SLIVER FROM 5% TOWARD 8-10% (the arbiter's stated ceiling) for direct Exp 14 tail mass; monitor Exp 13, which has room to absorb a small decrease (currently 4.64, band +3..+7, observed 6.03). (d) EXP 10 GUARDRAIL: set delta to the bottom of its range ([0.10, 0.13]) and optionally narrow the relative gate jitter to +/-0.4; expect ~0.85, and accept a small residual gap (~0.05) as the price of the low-precision cluster — do not shrink the cluster to rescue Exp 10, since the observed data demands BOTH high slope-identification (0.911) AND a GSP-leaning subgroup (Exp 12 = -0.414), and the cluster is what delivers the latter. Guardrails: keep Exp 13 in +3..+7, keep Exp 10 >= 0.85, keep mean A_eff unchanged (Exp 7 at 0.681 vs 0.704 is fine), and place NO habitual mass above gamma = 2.2. Success criteria for the next candidate: Exp 11 in [45, 75], Exp 14 >= 0.20, Exp 12 <= -0.35, Exp 9 <= -1.3.

**Outcome of this advice:** iter 3 candidate loss=0.0802 -> iter 4 candidate loss=0.0808 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: SGAI-r4 was REJECTED by the gate (0.0808 vs 0.0802), and the per-experiment pattern shows a clean split between one validated recalibration and one failed one. WHAT WORKED: the majority-gamma midpoint pullback (20/48/24/8) landed Experiment 11 squarely in the target band (58.9 vs observed 52.6, my [45, 75] criterion) — this calibration is now confirmed and should be treated as the anchor. Exps 2, 4, 5, 6, 7 also improved or held (Exp 2: 0.175 vs 0.181; Exp 7: 0.709 vs 0.704; Exp 13 stayed in band at 4.90). WHAT FAILED: (1) Enlarging the low-precision cluster from 20% to 25% with elevated epsilon (0.15-0.25) was supposed to move Exps 12 and 14 and instead moved NEITHER — Exp 12 regressed from -0.31 to -0.156 and Exp 14 dropped from 0.136 to 0.103 — while it DID degrade Exp 10 further (0.842 -> 0.814, now below the 0.85 guardrail). The mechanism is now clear from two iterations of evidence: high-lapse GSP-box subjects supply almost no per-subject likelihood evidence, so they neither contribute sharp negative tanh to Exp 12 nor negative log BF to Exp 11; they only flatten the slope posteriors that Exp 10 measures. The Bayes-factor metrics (9, 11, 12) reward SHARP GSP-box subjects, not numerous diffuse ones. (2) Exp 12's regression is additionally explained by the majority pullback: with habitual gamma in [1.3, 1.7] and beta log-uniform on [2.0, 6.0] (median ~3.5), a large fraction of majority subjects now sit inside the GSP/RSD overlap (gamma <= 1.5, beta in [1.8, 3.2]) and contribute tanh ~ 0 instead of positive RSD evidence — the negative pull must therefore come from sharper GSP-box subjects, not from a bigger diffuse cluster. (3) Exp 14 remains the dominant miss (0.103 vs 0.298, var 0.0006 — the miss is many between-subject SDs). Three iterations have now confirmed the trial-level gate is a dead lever for tail mass (free A_eff down to 0.35 absorbs jitter as attenuation); the only lever that ever moved it is steep-majority spill (iter 3's 0.136). Note also that the observed value implies roughly 40% of real subjects show genuine gamma > 2 choice signatures (raw posterior mass ~0.41, above even a flat-diffuse 0.23) — a magnitude the arbiter's 2.2-cap / 10%-sliver guardrails structurally cannot deliver; the residual should be flagged to the arbiter rather than fixed by breaking the family. (4) Secondary: Exp 1 undershot further (0.130 vs 0.158) — steeper majority gamma raises TTB-siding and would help; Exp 9 is unchanged at -0.98 vs -1.76.
Rationale: Regenerate on the iter-3 accepted base, re-applying the validated iter-4 majority calibration plus one NEW lever — converting the low-precision cluster from diffuse to sharp. Stay strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, relative stochastic gate, bimodal precision population). (a) MAJORITY HABITUAL GAMMA: keep the iter-4 distribution as the anchor but nudge the top bins up slightly — e.g., 18% [0.8, 1.3], 45% [1.3, 1.7], 28% [1.7, 2.0], 9% (2.0, 2.2], none above 2.2. This preserves Exp 11 near 60-75 while feeding Exp 14 spill, Exp 12, and Exp 1 (all of which improved under the steeper iter-3 majority). (b) PRIMARY NEW LEVER — SHARPEN THE LOW-PRECISION CLUSTER: shrink it to 20-22%, keep beta log-uniform [1.2, 1.8] and habitual gamma <= 1.0, but CUT epsilon from [0.15, 0.25] to roughly [0.05, 0.12]. A clean low-beta subject contributes tanh near -0.8/-0.9 on Exp 12 (vs ~-0.2 when lapse-heavy), a strongly negative per-subject log BF on Exp 11 (buying headroom for the steeper majority), pushes Exp 9 toward -1.3 or lower, and — because their effective normalized slope s = beta * sum(v^gamma) ~ 10-15 stays above the 9.5 threshold with clean data — no longer drags Exp 10 down (target >= 0.86). This one change moves Exps 9, 10, 11, and 12 in their observed directions simultaneously, unlike the diffuse-cluster enlargement which moved nothing and cost Exp 10. (c) EXP 14: raise the sliver to the arbiter's 10% ceiling and concentrate the third bin at [1.8, 2.0] to maximize spill; keep delta at the bottom of [0.10, 0.13] with the +/-0.4 relative jitter. Accept a residual gap (~0.13-0.17 vs 0.298) and state it explicitly in the rationale as a structural limit of the prescribed gamma <= 2.2 support — do NOT place habitual mass above 2.2 and do not re-widen the gate (both are confirmed dead or harmful levers). (d) GUARDRAILS: Exp 11 in [45, 75] — if the sharpened LP cluster pulls it below 45, rebalance by trimming LP to 18% or raising the sliver, not by re-flattening the majority; Exp 13 in [+3, +7] (monitor: sharp out-of-box subjects can shift it either way); Exp 10 >= 0.85; mean A_eff unchanged (Exp 7/8 are fine). Success criteria for the next candidate: Exp 12 <= -0.35, Exp 10 >= 0.86, Exp 9 <= -1.2, Exp 11 in [45, 75], Exp 14 >= 0.12, aggregate loss < 0.0802.

**Outcome of this advice:** iter 4 candidate loss=0.0808 -> iter 5 candidate loss=0.0984 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: SGAI-r5 was REJECTED by the gate (0.0984 vs the running-best 0.0802), and the per-experiment pattern shows that BOTH levers pulled in iter 5 failed, one of them catastrophically. (1) EXP 11 EXPLODED: 154.9 vs observed 52.6 — the worst value in the entire loop (iter 3: 109.3, iter 4: 58.9). The majority steepening (28% concentrated in [1.8, 2.0] plus the sliver raised to 9%) was roughly twice too large a step on top of an already-validated calibration. Iter 4's 20/48/24/8 distribution, which landed Exp 11 at 58.9 and Exp 13 at 4.90, remains the confirmed anchor; iter 5's nudge up is now empirically refuted. (2) EXP 13 COLLAPSED OUT OF BAND: 0.17 vs observed 6.03 (band +3..+7). Mechanistically, the enlarged (2.0, 2.2] sliver sits in EPAI-only territory of Exp 13's grid (RSD's prior has zero mass above gamma = 2), so each sliver subject contributes strongly EPAI-ward evidence; the sharp LP cluster's gamma ~ U[0,1] additionally matches EPAI's densest gamma component. (3) MY ITER-4 PRIMARY LEVER — SHARPENING THE LP CLUSTER — FAILED, and I must own that: with epsilon cut to [0.05, 0.12], Exp 12 REGRESSED (-0.31 at iter 3 → -0.22 vs target -0.41), Exp 9 did not move (-0.86 vs -0.97, observed -1.76), and Exp 10 dropped further (0.824 vs 0.842, observed 0.911). The predicted 'clean GSP-side Bayes-factor evidence' did not materialize on Exps 9/11/12; the only gain was a trivial +0.015 on Exp 14 (0.151 vs 0.298 — still a ~40-between-subject-SD miss). Both LP extremes have now been tried and rejected: diffuse/high-lapse supplies no evidence, sharp/low-lapse damages Exp 13 without helping anything. (4) Secondary regressions: Exp 7 now OVERSHOOTS (0.736 vs 0.704), Exp 6 over-sparse (-0.732 vs -0.676), Exp 2 further under (0.147 vs 0.181). (5) Exps 3, 4, 5, 8 remain fine. Net: the candidate is strictly worse than the iter-3 base on every currently-missed experiment except Exp 14's marginal +0.015.
Rationale: Regenerate on the iter-3 accepted base, abandoning both failed iter-5 levers and introducing ONE new, previously untried knob. Stay strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, relative stochastic gate, bimodal precision population). (a) REVERT THE MAJORITY to the validated iter-4 calibration: 20% [0.8, 1.3], 48% [1.3, 1.7], 24% [1.7, 2.0] (NOT top-concentrated at [1.8, 2.0]), 8% sliver in (2.0, 2.2], none above 2.2. This is the only setting that has ever put Exp 11 in band (58.9) while keeping Exp 13 in band (4.90); iter 5's steepening is refuted in both directions (Exp 11 +102, Exp 13 -4.7). (b) PRIMARY NEW LEVER — LOWER THE MAJORITY BETA FLOOR from 2.0 to ~1.6-1.7 (log-uniform [1.6, 6.0]), which is what the arbiter's original spec actually prescribed ('floor slightly BELOW RSD's 1.8'). Every iteration so far has kept the floor at 2.0, leaving this knob untouched. Mechanism: roughly 15-17% of majority subjects then draw beta inside GSP's beta box [1.1, 3.2], and the lower-gamma majority bins (gamma ≤ 1.5) put them fully inside the GSP box — each such subject contributes negative tanh on Exp 12 (toward -0.41), negative per-subject log BF on Exp 11 (pulling 58.9 down toward 52.6), and GSP-ward pull on Exp 9 (toward -1.3 or lower). This moves the three biggest remaining misses (9, 11, 12) in their observed directions simultaneously WITHOUT touching the gamma distribution, which is pinned by the Exp 13/14 tension. Safety check: with gamma ~1.5 the effective normalized slope s = beta * sum(v^gamma) stays ~12 even at beta = 1.6, above Exp 10's 9.5 threshold, so Exp 10 should not degrade further. (c) LP CLUSTER AT THE MIDDLE OF THE TWO FAILED EXTREMES: 20%, beta log-uniform [1.2, 1.8], habitual gamma ≤ 1.0, epsilon ~[0.10, 0.18] — between the accepted iter-3 diffuse setting (which gave the best Exp 12 = -0.31) and the rejected iter-5 sharp setting. Do not re-sharpen. (d) GATE: delta = 0.10 flat, ±0.4 relative jitter — confirmed optimal; the gate is a dead lever for Exp 14. (e) EXP 14: accept the residual (~0.10-0.15 vs 0.298) as structural under the prescribed gamma ≤ 2.2 support and state it explicitly to the arbiter; three iterations confirm neither jitter nor spill can close it, and every attempt to chase it (sliver growth, steepening) has damaged Exps 11/13 far more than it helped Exp 14. (f) GUARDRAILS: Exp 11 in [45, 70] — if the beta-floor drop pushes it below 45, raise the floor to 1.8 rather than re-steepening gamma; Exp 13 in [+3, +7]; Exp 10 ≥ 0.84; watch Exps 7/8 (lower beta reduces extremity — if they fall below observed, compensate with a small m-floor raise to ~0.80, not by reverting beta). Success criteria: Exp 11 ∈ [45, 70], Exp 12 ≤ -0.33, Exp 9 ≤ -1.2, Exp 10 ≥ 0.84, Exp 13 ∈ [3, 7], aggregate loss < 0.0802.

**Outcome of this advice:** iter 5 candidate loss=0.0984 -> iter 6 candidate loss=0.0736 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: SGAI-r6 was ACCEPTED with the loop's best loss (0.0736), and my iter-5 primary lever (majority beta floor lowered to 1.6) was implemented faithfully. The per-experiment ledger, however, shows a split outcome that the aggregate loss masks. WHAT THE LEVER BOUGHT: Exps 1 (0.165 vs 0.158) and 2 (0.192 vs 0.181) are the closest fits of the entire loop; Exp 4 is exact (-0.523 vs -0.523); Exps 5-8 remain solid; Exp 14 hit its loop-best (0.163). WHAT THE LEVER BROKE: (1) EXP 13 FLIPPED SIGN: -1.88 vs observed +6.03 — the arbiter's explicitly prescribed band for this signature was +3..+7 ('stays mildly positive... avoiding EPAI's -31'). The beta floor at 1.6 sits below BOTH theory boxes in Exp 13's grid (RSD starts at 1.8, EPAI at 2.0), and the resulting shallow low-beta majority subjects score EPAI-ward (EPAI's densest gamma component is U[0,1.2], which our low-beta/lower-gamma bins mimic). Iter-4 with floor 2.0 gave +4.90 (in band); iter-6 with floor 1.6 gives -1.88. The lever overshot downward. (2) EXP 11 ROSE, NOT FELL: 94.5 vs my predicted pull toward 52.6 — the accompanying LP shrinkage (25%→20%) and sharpening removed more GSP-side evidence than the floor drop added, and the m-floor raise to 0.80 increased per-subject identifiability. (3) The lever's predicted gains on Exps 9 and 12 did NOT materialize: Exp 9 unchanged (-0.98 vs -1.76, never better than -0.97 in six iterations), Exp 12 actually REGRESSED vs iter-3 (-0.215 vs iter-3's -0.310, target -0.414) — the LP epsilon midpoint [0.10, 0.18] is again worse than iter-3's diffuse [0.15, 0.25] on Exp 12, replicating the iter-5 finding that sharpening the LP cluster does not buy Bayes-factor evidence. (4) Exp 14 remains a ~30-between-subject-SD miss (0.163 vs 0.298, var 0.0002) — consistent with three prior confirmations that neither the gate nor spill can close it under the gamma ≤ 2.2 support cap. Net: the candidate is the best base yet, but it now violates the arbiter's single most explicit signature (Exp 13 mildly positive) with a sign flip, and misses all four of the arbiter's targeted population signatures (our Exps 11, 12, 13, 14) even while winning the aggregate loss on the strength of Exps 1-8.
Rationale: Regenerate on the ACCEPTED iter-6 base (the gate guarantees no regression), with a midpoint correction on the one knob that broke and a partial revert on the one that regressed — explicitly not an oscillation, since the iter-6 settings are the new accepted anchor and the adjustments are half-steps from it. (a) RAISE THE MAJORITY BETA FLOOR FROM 1.6 TO ~1.8-1.9 (log-uniform [1.8, 6.0] or [1.9, 6.0]) — the midpoint between the in-band iter-4 setting (floor 2.0, Exp 13 = +4.90, Exp 11 = 58.9) and the accepted iter-6 setting (floor 1.6, Exp 13 = -1.88, Exp 11 = 94.5). Expected effects: Exp 13 returns to roughly +1..+5 (must be at minimum POSITIVE — a negative value means the simulated population favors EPAI, contradicting the observed data), Exp 11 drops from 94.5 toward 60-75 (observed 52.6), while the Exps 1/2 gains should largely survive since they were also present at iter-4's floor 2.0 with the current m/LP settings. (b) REVERT LP EPSILON TOWARD DIFFUSE: from [0.10, 0.18] to ~[0.13, 0.20] or the iter-3 diffuse [0.15, 0.25], keeping the cluster at 20%, beta log-uniform [1.2, 1.8], gamma ≤ 1.0. Two data points now agree that iter-3's diffuse LP gave the best Exp 12 (-0.310) while every sharpening (iter-5, iter-6) regressed it; the sharpening also never helped Exps 9/10. (c) KEEP PINNED: the iter-4-anchor majority gamma distribution (20/48/24/8, none above 2.2 — confirmed by iters 4-6 in both directions), the relative ±0.4 gate at delta ~0.10 (dead lever, least harmful form), and m floor 0.80 (Exps 7/8 are fine). (d) FLAG AS STRUCTURAL, DO NOT CHASE: Exp 14 (0.163 vs 0.298 — the observed value implies ~40% of subjects show genuine gamma > 2 choice signatures, unreachable under the prescribed gamma ≤ 2.2 cap; every chase has damaged Exps 11/13 far more than it helped) and likely Exp 9 (-0.98 vs -1.76 — no lever in six iterations has moved it below -0.97; note for the arbiter that pi_1/pi_3 reach -2.8/-1.7 only via mechanisms outside this family). State both residuals explicitly in the rationale rather than breaking the family. GUARDRAILS for the next candidate: Exp 13 > 0 (ideally +2..+7), Exp 11 in [50, 80], Exp 12 ≤ -0.28, Exp 10 ≥ 0.83, Exps 1/2 within ±0.02 of observed, aggregate loss < 0.0736. If raising the beta floor costs Exps 1/2 more than ~0.015 each, stop at 1.9 rather than 2.0 and recover Exp 13 via a small LP-cluster trim (20%→18%) instead — do not re-steepen gamma and do not re-sharpen the LP cluster; both are twice-refuted.

**Outcome of this advice:** iter 6 candidate loss=0.0736 -> iter 7 candidate loss=0.0696 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: SGAI-r7 was ACCEPTED with the loop's best loss (0.0736 -> 0.0696), and both of my iter-6 prescribed corrections worked as predicted: the LP-epsilon revert to diffuse [0.15, 0.25] recovered Experiment 12 to -0.395 vs observed -0.414 (essentially exact, the loop's best by a wide margin), and the beta-floor raise to 1.8 returned Experiment 13 to positive territory (+8.81 vs observed +6.03), repairing iter-6's sign flip. The core fits remain strong: Exps 3, 4, 6, 7 are essentially on top of the data (-0.737/-0.535/-0.670/0.700 vs -0.717/-0.523/-0.676/0.704), Exp 8 is close (0.370 vs 0.347), Exp 10 is above guardrail (0.866 vs 0.911), and Exp 2 is within tolerance (0.164 vs 0.181). However, the candidate misses two of my own stated guardrails and one arbiter band, and the misses form ONE coherent directional pattern: the population is slightly TOO RSD-identified. (1) Exp 11 = 88.4 vs observed 52.6, above my [50, 80] band. (2) Exp 13 = +8.81, just above the arbiter's +3..+7 band. (3) Exps 1, 2, 5 are all slightly UNDER (0.132/0.164/0.549 vs 0.158/0.181/0.594) — and the loop's own data localize the cause: at floor 1.6 (iter 6) these were 0.165/0.192/0.570, at floor 1.8 (iter 7) they are 0.132/0.164/0.549. The floor raise from 1.6 to 1.8 overshot upward on exactly the same experiments that iter 6's lowering overshot downward — a textbook interpolation situation. (4) The two structural residuals are unchanged and now confirmed across five iterations: Exp 14 (0.130 vs 0.298, var 0.0002 — a >30-between-subject-SD miss that neither trial-level jitter, nor sliver growth, nor spill has ever moved meaningfully under the prescribed gamma <= 2.2 cap) and Exp 9 (-0.90 vs -1.76, never better than -0.97 in seven iterations). Both should be reported to the arbiter as limits of the prescribed family, not chased.
Rationale: Regenerate on the ACCEPTED iter-7 base with ONE primary half-step and one optional micro-step — explicitly the anti-oscillation interpolation between the two flanking accepted settings, not a reversal. (a) PRIMARY: lower the majority beta floor from 1.8 to ~1.70-1.75 (log_beta range [~0.53, 1.792]). The two flanking data points are iter 6 (floor 1.6: Exp 13 = -1.88, Exp 1 = 0.165, Exp 2 = 0.192) and iter 7 (floor 1.8: Exp 13 = +8.81, Exp 1 = 0.132, Exp 2 = 0.164); linear interpolation puts Exp 13 at ~+6 (the observed value) near floor ~1.74. Expected: Exp 13 in [+4, +7] (inside the arbiter band), Exp 1 -> ~0.145-0.155, Exp 2 -> ~0.175, Exp 5 -> ~0.56-0.57, Exp 9 slightly more negative (low-beta majority subjects in the GSP box add GSP-ward evidence), Exp 12 holding at ~-0.40 or slightly more negative (in the observed direction). (b) OPTIONAL MICRO-STEP for Exp 11 only: raise the LP cluster from 20% to ~22% (NOT 25% — iter 4's 25% gave Exp 11 = 58.9 but cost Exp 10 = 0.814 and Exp 12 = -0.156). Guardrails: if Exp 10 falls below 0.83 or Exp 12 overshoots below -0.50, revert to 20%. (c) KEEP PINNED — all twice-confirmed: the majority gamma anchor 20/48/24/8 with no mass above 2.2 (validated in both directions at iters 3-5), the relative +/-0.4 gate at delta ~0.10 (dead lever, least harmful form), LP beta log-uniform [1.2, 1.8] with diffuse epsilon [0.15, 0.25], and m floor 0.80 (Exps 7/8 fine). (d) DO NOT CHASE Exp 14 or Exp 9: five iterations confirm neither the gate, nor sliver growth, nor majority steepening can close them under the prescribed gamma <= 2.2 support, and every attempt has damaged Exps 11/13 far more than it helped. State both residuals explicitly in the rationale as structural limits of the prescribed family for the arbiter. (e) STOP CONDITION: this is the final tuning iteration — if the next candidate does not beat 0.0696, ship iter 7 as the final theory. The loop has converged; the remaining guardrail misses are small, coherent, and this single interpolation is the last principled in-family step available. Success criteria: Exp 13 in [+4, +7], Exp 11 in [55, 80], Exps 1/2 within +/-0.02, Exp 12 in [-0.50, -0.30], Exp 10 >= 0.83, aggregate loss < 0.0696.

**Outcome of this advice:** iter 7 candidate loss=0.0696 -> iter 8 candidate loss=0.0735 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: SGAI-r8 was REJECTED by the gate (0.0735 vs the running-best 0.0696), so the base reverts to iter 7. The rejection is diagnostic in a useful way, because iter 8 contained exactly two edits and their effects decompose cleanly. EDIT 1 (beta floor 1.8 -> 1.73) WORKED AS PREDICTED, and is arguably the best single calibration of the loop on its targets: Exp 13 = +5.40 vs observed +6.03 (the closest fit ever recorded, dead inside the arbiter's +3..+7 band, repairing iter 6's -1.88 sign flip and iter 7's +8.81 overshoot in one step); Exp 2 = 0.186 vs 0.181 and Exp 5 = 0.598 vs 0.594 are essentially exact; Exp 10 improved to 0.884 (observed 0.911); Exp 14 hit a new loop-best 0.172. EDIT 2 (LP cluster 20% -> 22%) is the likely rejection driver and moved the WRONG experiments in the WRONG direction: Exp 11 rose from 88.4 to 101.7 (observed 52.6) instead of falling toward 60-75 as I predicted — the third consecutive confirmation that enlarging the diffuse high-lapse LP cluster does NOT add GSP-side Bayes-factor evidence on Exp 11 (they supply too little per-subject likelihood), while it simultaneously diluted Exp 9 (-0.90 -> -0.67, observed -1.76) and nudged Exps 3/4 slightly off (-0.827/-0.563 vs -0.717/-0.523). Exp 12 held at -0.395 (observed -0.414, essentially exact — the diffuse-LP setting is confirmed optimal). The two structural residuals remain exactly as flagged for five iterations: Exp 14 (0.172 vs 0.298, var 0.0002, a >25-between-subject-SD miss requiring ~40% genuine gamma>2 subjects, unreachable under the prescribed gamma <= 2.2 cap) and Exp 9 (never better than -0.97; only out-of-family mechanisms reach -1.7). Both must continue to be reported to the arbiter as limits of the prescribed SGAI family, not chased.
Rationale: Verdict is regenerate (forced — the gate already rejected this candidate and reverted the base to iter 7; 'continue' would be silently downgraded). Per my iter-7 stop condition ('if the next candidate does not beat 0.0696, ship iter 7'), the loop has converged and iter 7 is the ship candidate — but the iter-8 data justify exactly ONE final decomposed attempt before stopping, because the two edits have now been experimentally separated: the floor interpolation is validated on Exps 13/2/5/10/14 while the LP enlargement is refuted on Exps 11/9. FINAL CANDIDATE SPECIFICATION (minimal diff from the iter-7 accepted base, strictly in-family): (a) KEEP the majority beta floor at the interpolated 1.73 (log_beta range [0.548, 1.792]) — this single knob delivered the loop's best Exp 13 (+5.40 vs +6.03), exact Exps 2/5, improved Exp 10, and the loop-best Exp 14. (b) REVERT the LP cluster to 20% (prec_u threshold back to 0.20) — the 22% step raised Exp 11 by +13 in the wrong direction and dragged Exp 9 toward zero; three iterations (4, 5, 8) now agree that LP-cluster enlargement/sharpening never buys the Bayes-factor evidence predicted for it. (c) KEEP PINNED, all multiply confirmed: the majority gamma anchor 20/48/24/8 with no habitual mass above 2.2; the relative +/-0.4 gate at delta ~0.10 (dead lever, least harmful form); LP beta log-uniform [1.2, 1.8] with diffuse epsilon [0.15, 0.25]; m floor 0.80. Expected outcome: Exp 13 in [+4, +7], Exps 2/5 near-exact, Exp 12 ~ -0.40, Exp 10 ~ 0.87, Exp 11 ~ 85-90 (residual over-prediction accepted as the price of the in-band Exp 13), aggregate loss < 0.0696. (d) HARD STOP CONDITION, now twice-stated: this decomposed edit is the FINAL tuning iteration of the loop. If it does not beat 0.0696, ship iter 7 unchanged as the final theory — do not attempt further population rearrangements; the remaining misses (Exp 11 residual ~+35, Exp 14 ~ -0.13, Exp 9 ~ +0.85) are structural under the arbiter's prescribed gamma <= 2.2 support cap and the SGAI likelihood core, and every chase in eight iterations has damaged the in-band experiments (13, 12, 10) far more than it helped the missed ones. Both structural residuals must be stated explicitly in the shipped theory's rationale for the arbiter's attention.

**Outcome of this advice:** iter 8 candidate loss=0.0735 -> iter 9 candidate loss=0.0827 -> the gate marked it REJECTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: SGAI-r9 was REJECTED by the gate (0.0827 vs the running-best 0.0696), and the rejection is more informative than the loss alone: the candidate implemented my iter-8 final specification exactly (floor 1.73 kept, LP cluster reverted to 20%, everything else pinned), yet its per-experiment values did NOT reproduce the iter-8 decomposition. Where iter 8 (floor 1.73, LP 22%) gave Exp 13 = +5.40, Exp 12 = -0.395, Exp 11 = 101.7, Exp 14 = 0.172, the current candidate (floor 1.73, LP 20%) gives Exp 13 = +1.86 (below the arbiter's +3..+7 band), Exp 12 = -0.217 (regressed from essentially-exact), Exp 11 = 114.7 (worst of the loop apart from iter 5), Exp 14 = 0.113 (back to iter-7 level), and Exp 10 = 0.845. Reverting the LP cluster from 22% to 20% cannot mechanistically produce these signs of change — iter 7 (floor 1.8, LP 20%) had Exp 12 = -0.395 and Exp 13 = +8.81 — so the dominant driver of the residual differences between iters 7/8/9 on Exps 11, 12, 13, 14 is Monte-Carlo/population-sampling noise in the Bayes-factor metrics (between-subject variances of 0.4-20 on exactly these metrics), not any parameter knob. In other words: the loop has converged to a noise floor. The core experiments (1-8) remain well fit throughout (Exps 3, 4, 5, 6, 7, 8 all essentially on the data this iteration), and the remaining misses are the same two structural residuals flagged for five consecutive iterations — Exp 14 (0.113 vs 0.298, var 0.0001-0.0002, a >25-between-subject-SD miss requiring ~40% genuine gamma>2 subjects, unreachable under the arbiter's prescribed gamma <= 2.2 cap) and Exp 9 (-0.89 vs -1.76, never better than -0.97 in nine iterations) — plus an Exp 11 over-prediction (~+60) that no in-family lever has moved downward without breaking Exp 13. My iter-7 and iter-8 stop conditions ('if the next candidate does not beat 0.0696, ship iter 7') have now both triggered.
Rationale: Verdict is regenerate (forced — the gate already rejected this candidate and reverted the base to iter 7; 'continue' would be silently downgraded). Per the twice-stated stop condition, the loop has converged: iter 7 (loss 0.0696) is the ship candidate, and the next proposal should be the iter-7 accepted base RESUBMITTED ESSENTIALLY VERBATIM, with no new levers, no population rearrangements, and no further chasing of Exps 9, 11, or 14. Concretely: (a) majority log_beta floor at 1.8 (range [0.588, 1.792]) — the 1.73 interpolation looked best in iter 8's single run but failed to replicate in iter 9's run (Exp 13 fell to +1.86, Exp 12 to -0.217), confirming that the difference between floors 1.73 and 1.8 is inside the run-to-run noise of the BF metrics; the safer, twice-validated setting is 1.8, which gave Exp 13 = +8.81 and Exp 12 = -0.395; (b) keep pinned, all multiply confirmed: majority gamma anchor 20/48/24/8 with no habitual mass above 2.2 (validated in both directions at iters 3-5), LP cluster at 20% with beta log-uniform [1.2, 1.8] and diffuse epsilon [0.15, 0.25] (twice-confirmed optimal for Exp 12), the relative +/-0.4 gate at delta ~0.10 (dead lever for Exp 14, least harmful form), m floor 0.80. (c) The shipped rationale MUST state the structural residuals explicitly for the arbiter: (i) Exp 14 tail mass ~0.10-0.17 vs observed 0.298 — the observed value implies roughly 40% of real subjects show genuine gamma > 2 choice signatures, which is unreachable under the prescribed gamma <= 2.2 support cap; five iterations confirm neither trial-level jitter, nor sliver growth, nor majority steepening closes it, and every chase damaged Exps 11/13 far more than it helped; (ii) Exp 9 ~ -0.9 vs observed -1.76 — no in-family lever has ever moved it below -0.97; the theories that reach it (pi_1 at -2.79, pi_3 at -1.72) do so via mechanisms outside this family's likelihood core; (iii) Exp 11 residual over-prediction (~+35 to +60) is the accepted price of keeping Exp 13 inside the arbiter's band. Do NOT attempt any further population rearrangements: nine iterations of evidence show the remaining guardrail misses on the Bayes-factor experiments (9, 11, 12, 13, 14) are dominated by between-run sampling variance and structural support limits, not by miscalibrated knobs, and every additional edit risks a rejection like this one. If the resubmitted iter-7 base does not beat 0.0696 due to simulation noise, that is the correct terminal state — the accepted base is unchanged either way, and the loop should terminate with iter 7 as the final theory.

**Outcome of this advice:** iter 9 candidate loss=0.0827 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Stochastic-Gate Anti-Endorsement Integration (SGAI, final form = accepted revision 7). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma_t (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population is bimodal in precision: a dominant high-precision cluster (80%; beta log-uniform on [1.8, 6.0]) whose habitual validity exponent is distributed 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0], and an 8% sliver in (2.0, 2.2], none above 2.2 — plus a 20% low-precision cluster (beta log-uniform on [1.2, 1.8], habitual gamma uniform on [0, 1.0], diffuse lapse 0.15-0.25). On each trial, with probability delta ~ 0.10, the subject re-samples cue attention, drawing an effective per-trial exponent from a symmetric distribution around their OWN habitual gamma (gamma_t ~ U[max(0, gamma-0.4), min(2.2, gamma+0.4)]). Falsifiable signatures: psychometric mid-slopes shallower than saturated asymptotes imply, supra-binomial trial-to-trial inconsistency on repeated identical trials, a bimodal precision population straddling the RSD/GSP beta boundary at 1.8, a mildly positive RSD-vs-EPAI log Bayes factor, and per-subject gamma posteriors that spill above 2.0 for the steep-gamma bins without any habitual mass above 2.2.",
  "predict": "def predict(parameters, state, history):\n    # Stochastic-Gate Anti-Endorsement Integration (SGAI, final = r7).\n    # Stimulus: array-like of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are 0/1 cue values. History is ignored\n    # (no feedback in this task).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"SGAI expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    # ---- Population: habitual validity exponent gamma (majority) ----\n    # gamma_u ~ U[0, 1]. The confirmed anchor distribution (the only\n    # setting that has ever put Exp 11 in band while keeping Exp 13 in\n    # band): 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0],\n    # and an 8% sliver in (2.0, 2.2]. NO habitual mass above 2.2\n    # (guardrail for Exp 13).\n    u = float(parameters[\"gamma_u\"])\n    if u < 0.20:\n        gamma = 0.8 + 0.5 * (u / 0.20)\n    elif u < 0.68:\n        gamma = 1.3 + 0.4 * ((u - 0.20) / 0.48)\n    elif u < 0.92:\n        gamma = 1.7 + 0.3 * ((u - 0.68) / 0.24)\n    else:\n        gamma = 2.0 + 0.2 * ((u - 0.92) / 0.08)\n\n    # ---- Precision subpopulation (bimodal) ----\n    # 20% of subjects are low-precision: beta log-uniform on [1.2, 1.8]\n    # (deep inside the GSP box, below the RSD floor of 1.8), DIFFUSE\n    # lapse epsilon in [0.15, 0.25] (twice-confirmed optimal for the\n    # Exp 12 tanh Bayes factor), and habitual gamma forced LOW\n    # (uniform on [0, 1.0]). The remaining 80% draw beta log-uniformly\n    # over [1.8, 6.0]: the floor sits exactly at the RSD boundary, so\n    # the majority is RSD-identified on the Exp 13 grid while the\n    # lower-gamma majority bins (gamma <= 1.5) with beta in [1.8, 3.2]\n    # remain inside the GSP overlap, preserving the Exp 1/2 accuracy.\n    if float(parameters[\"prec_u\"]) < 0.20:\n        beta = float(np.exp(float(parameters[\"lp_log_beta\"])))\n        epsilon = float(parameters[\"lp_epsilon\"])\n        gamma = 1.0 * u  # low-precision subjects keep gamma <= 1.0\n    else:\n        beta = float(np.exp(float(parameters[\"log_beta\"])))\n        epsilon = float(parameters[\"epsilon\"])\n\n    m = float(parameters[\"m\"])\n    delta = float(parameters[\"delta\"])\n\n    # ---- Trial-level stochastic attention gate (RELATIVE draw) ----\n    # With probability delta (~0.10) the subject re-samples cue\n    # attention for THIS trial, drawing an effective exponent from a\n    # SYMMETRIC distribution AROUND THEIR OWN habitual gamma with a\n    # +/-0.4 width. The draw is relative, so the low-precision\n    # cluster's shallow gamma is never globally overridden, while\n    # steep-gamma subjects still jitter across the 2.0 boundary on a\n    # minority of trials. Confirmed dead lever for Exp 14; kept only\n    # in its gentlest, least harmful form.\n    if np.random.random() < delta:\n        lo = max(0.0, gamma - 0.4)\n        hi = min(2.2, gamma + 0.4)\n        gamma_t = lo + (hi - lo) * np.random.random()\n    else:\n        gamma_t = gamma\n\n    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.\n    # This preserves the absolute endorsement-margin scale: a margin-2\n    # conflict carries twice the evidence of a margin-1 conflict,\n    # independent of n_features.\n    w = np.power(val, gamma_t)\n\n    a, b = stim[0], stim[1]\n    # Anti-endorsement (sparse-preferring) evidence for A over B:\n    # each endorsement carried by B counts FOR A, each endorsement\n    # carried by A counts AGAINST A. E = 0 on identical rating vectors.\n    E = float(np.sum(w * (b - a)))\n\n    # Numerically stable logistic (both branches exponentiate a\n    # non-positive argument, so no overflow is possible).\n    x = beta * E\n    if x >= 0.0:\n        sig = 1.0 / (1.0 + np.exp(-x))\n    else:\n        ex = np.exp(x)\n        sig = ex / (1.0 + ex)\n\n    # Graded polarity: with per-trial probability m the subject responds\n    # with the INVERTED (sparse-preferring) polarity, with probability\n    # (1 - m) with the conventional polarity.\n    p_core = m * sig + (1.0 - m) * (1.0 - sig)\n\n    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.\n    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon\n\n    return np.array([p_a, 1.0 - p_a])",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = probs / probs.sum()  # Guard against float drift.\n    return np.random.choice(len(probs), p=probs)",
  "parameters": {
    "gamma_u": "[0, 1]",
    "log_beta": "[0.588, 1.792]",
    "epsilon": "[0.03, 0.10]",
    "m": "[0.80, 0.98]",
    "delta": "[0.10, 0.12]",
    "prec_u": "[0, 1]",
    "lp_log_beta": "[0.182, 0.588]",
    "lp_epsilon": "[0.15, 0.25]",
    "validities": "validities"
  },
  "rationale": "This proposal is the terminal-state resubmission prescribed by the last two critic iterations, whose stop conditions have both triggered. Nine iterations of evidence decompose cleanly: (1) the per-trial likelihood core (anti-endorsement reading, raw v^gamma_t weights, m/eps polarity-lapse shell) fits Experiments 1-8 essentially exactly and has never been the bottleneck; (2) the population layer has converged to a unique validated anchor — majority gamma 20/48/24/8 with no habitual mass above 2.2 (validated in BOTH directions at iters 3-5: steepening exploded Exp 11 to 109/155 and collapsed Exp 13 to +0.17; flattening undershot Exp 11 at 34.5), majority beta floor at 1.8 (the 1.6 floor flipped Exp 13's sign to -1.88; the 1.73 interpolation looked best in iter 8's single run but failed to replicate in iter 9, confirming the floor difference is inside the run-to-run Monte-Carlo noise of the Bayes-factor metrics; 1.8 is the twice-validated setting giving Exp 13 = +8.81 and Exp 12 = -0.395), LP cluster at 20% with diffuse epsilon [0.15, 0.25] (enlargement to 22-25% is refuted three times — iters 4, 5, 8 — always raising Exp 11 the WRONG way while degrading Exps 9/10; sharpening is refuted twice — iters 5, 6 — regressing Exp 12 every time), and the relative +/-0.4 gate at delta ~0.10 (a confirmed dead lever for Exp 14 in its least harmful form). Accordingly, this candidate is the accepted iter-7 base re-emitted VERBATIM: zero new levers, zero population rearrangements, all multiply-confirmed settings pinned. STRUCTURAL RESIDUALS, stated explicitly for the arbiter rather than chased: (i) Exp 14 tail mass ~0.10-0.17 vs observed 0.298 (var 0.0001-0.0002, a >25-between-subject-SD miss) — the observed value implies roughly 40% of real subjects show genuine gamma > 2 choice signatures, which is unreachable under the prescribed gamma <= 2.2 support cap; five iterations confirm that neither trial-level jitter (the measurement model's free A_eff, down to 0.35, absorbs jitter as attenuation), nor sliver growth, nor majority steepening closes it, and every chase damaged Exps 11/13 far more than it helped Exp 14. (ii) Exp 9 ~ -0.9 vs observed -1.76 — no in-family lever has ever moved it below -0.97 in nine iterations; the theories that reach it (pi_1 at -2.79, pi_3 at -1.72) do so via mechanisms outside this family's likelihood core. (iii) Exp 11 residual over-prediction (~+35) is the accepted price of keeping Exp 13 positive and Exp 12 essentially exact. Expected behavior of this resubmission is the iter-7 profile: Exps 1-8 close to observed, Exp 13 mildly positive, Exp 12 near -0.4, Exp 10 ~0.85-0.87; if run-to-run noise prevents it from strictly beating 0.0696, that is the correct terminal state — the accepted base is unchanged either way and the loop should terminate with this theory as final."
}
```
```

## Usage

```json
{
  "input_tokens": 66179,
  "output_tokens": 3035,
  "reasoning_tokens": 118,
  "provider": "Sail Research"
}
```
