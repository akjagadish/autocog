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
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace DCD with a 'Softened Validity-Distrust with Count Mixture' (SVD) theory. Core claims: (1) RETAIN VPS's confirmed backbone — every endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity, with NO dampening near v=0.5 and NO trust refund (this is what Exp 1 decisively confirmed: charge d_j = c0 + c1*v_j^kappa, kappa >= 1, c0, c1 > 0). (2) Add a dual-process mixture: on each trial the subject is, with probability w, a 'skeptic' who integrates the distrust difference D = P(B) - P(A) via sigmoid(beta*D), and with probability (1-w) a 'counter' who simply chooses the option with FEWER total endorsements (pure tallying on counts, indifferent at matched counts). This count branch explains the systematic shrinkage of observed proportions toward 0.5 relative to pure VPS: in Exp 2's coin-flip exchange cells it pulls the coin-flip tolerance down from 0.61 toward the observed 0.51 (count-following favors the larger coin-flip coalitions less than VPS's distrust does); in Exp 9's count-conflict cells it pulls the high-validity-option rate up from 0.28 toward the observed 0.38; in Exp 10 it pulls the low-validity-coalition rate down from 0.83 toward 0.78; and in Exp 1's matched-count cells it is exactly neutral (counts equal), so the strong VPS-signed result (0.37) is preserved with w near 0.8-0.9. (3) Allow a larger lapse epsilon (0.10-0.25) and/or between-subject heterogeneity in beta to further soften extremes (Exps 3, 5). Parameter sketch: c0 in [0.05, 0.25], c1 in [0.5, 1.3], kappa in [1.5, 2.5], beta in [1.0, 2.5], w in [0.6, 0.95], epsilon in [0.08, 0.20]. This theory differs from DCD in that its moderation comes from an explicit count-heuristic mixture and noise, not from a trust term that reverses sign near chance validity — the specific mechanism Exp 1 falsified — and it differs from VPS by predicting systematically less extreme choice proportions, which VPS over- or under-shoots in Exps 2, 9, and 10. It should be competitive with (and empirically distinguishable from) VPS on cells where counts and validity-distrust conflict at unequal endorsement counts, while agreeing with VPS on the matched-count validity gradient where VPS was strongly confirmed.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.745)

**Description**
Dampened Confidence-Distrust Integration (DCD). Every expert endorsement is treated as a liability whose magnitude grows with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. The per-endorsement distrust charge is d_j = c0 + c1*(2*v_j - 1)^kappa, where the (2v-1) rescaling DAMPENS the charge near chance validity (a 55% expert's endorsement costs barely more than a coin flip's, i.e. only the flat baseline c0) while remaining steeply increasing toward confident experts. Two additions repair the incumbents' failures: (i) a large flat charge c0 preserves pure endorsement-count avoidance, so when the low-validity option carries MORE endorsements the count signal can overcome validity-distrust (the count-conflict cells where VPS is wrong); (ii) a small residual validity-weighted trust term alpha * sum_j logit(v_j)*(a_j - b_j) captures residual endorsement-following that pulls matched-count choices back toward and slightly past chance in the direction of the higher-validity endorser (softening VPS's extremes). The decision variable is D = (P(B) - P(A)) + alpha * sum_j logit(v_j)*(a_j - b_j), where P(X) = sum_j x_j*d_j is the accumulated suspicion for option X; the subject chooses via P(A) = sigmoid(beta*D) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so distrust weights are fixed for the whole block. DCD nests VPS as a limiting case (alpha -> 0, rescaling absorbed) and SDI as another (c1 -> 0), while filling the empirically human region between them.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Dampened Confidence-Distrust Integration (DCD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Per-endorser liability (dampened confidence-distrust charge):
    #   d_j = c0 + c1 * (2*v_j - 1)^kappa
    #     c0            -> flat suspicion floor: every endorsement is a
    #                      defect, and a near-chance expert's praise
    #                      costs essentially only this floor (the
    #                      (2v-1) rescaling dampens distrust near
    #                      v = 0.5 relative to a raw v^kappa scaling)
    #     c1*(2v-1)^kap -> steeply INCREASING in claimed validity:
    #                      a confident expert's endorsement is a big
    #                      suspected defect
    # Accumulated suspicion:  P(X) = sum_j x_j * d_j
    # Residual trust term:    E = sum_j logit(v_j) * (a_j - b_j)
    # Decision variable:      D = (P(B) - P(A)) + alpha * E
    #   (positive favors A: B carries more accumulated suspicion,
    #    plus a small validity-weighted endorsement edge for A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so
    # the distrust weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the logit against v = 1 and v < 0.5. The (2v-1) power
    # is safe for v = 0.5 (charge collapses to the flat floor c0).
    v = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])

    # Dampened confidence-distrust charge per endorser: flat
    # suspicion floor plus a steeply validity-scaled distrust term
    # that vanishes toward chance validity.
    d = c0 + c1 * np.power(2.0 * v - 1.0, kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual validity-weighted endorsement (trust) term: a small
    # tendency to follow the more credibly endorsed option that
    # pulls matched-count choices toward (and slightly past)
    # chance in favor of the higher-validity endorser.
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- kappa: [1.85, 2.15]
- c0: [0.26, 0.30]
- c1: [0.70, 0.85]
- alpha: [0.08, 0.12]
- beta: [2.2, 2.6]
- epsilon: [0.08, 0.12]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.6844 (var=0.0026)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.1978 (var=0.0023)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.4858 (var=0.0140)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.1987 (var=0.0006)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.1043 (var=0.0024)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.2387 (var=0.0095)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.3286 (var=0.0033)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.8054 (var=0.0034)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.5269 (var=0.0032)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.4429 (var=0.0037)


---

### `pi_5` (overall score: 0.705)

**Description**
Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- kappa: [1.6, 2.4]
- beta: [1.6, 2.4]
- epsilon: [0.07, 0.13]
- c0: [0.05, 0.15]
- c1: [0.7, 1.3]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.7272 (var=0.0025)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.2456 (var=0.0038)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.5708 (var=0.0133)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.2142 (var=0.0004)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.0852 (var=0.0016)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.1862 (var=0.0087)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.2844 (var=0.0060)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.8263 (var=0.0045)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.3820 (var=0.0036)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.6071 (var=0.0030)


---

### `pi_4` (overall score: 0.524)

**Description**
Skeptical Defect Integration (SDI). Subjects do not treat an expert's positive rating ('1') as an asset; they treat it as a liability — an endorsement from a fallible expert is noise-laden, adverse evidence. On each trial the decision maker computes a penalty for each option, P(X) = sum_j d_j * x_j, where d_j = (1 - v_j)^delta is the skepticism weight attached to expert j (delta = 0 recovers pure unweighted defect counting — every endorsement is one unit of suspicion; larger delta discounts endorsements from low-validity experts more steeply, since a bad expert's praise is especially untrustworthy). The subject chooses the option with the LOWER total penalty, i.e., the option that attracted less endorsement. Choice probability is P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. Exact penalty ties are broken by a weak validity-weighted endorsement term alpha * sum_j logit(v_j) * (a_j - b_j), which captures residual heterogeneity: as alpha grows, SDI smoothly recovers VWEI-like endorsement-following behavior. The empirically human region sits at delta near 0 (near-pure defect counting) with moderate noise. SDI is the only account in this domain that reproduces the SIGN of every observed metric: it follows the TTB winner whenever the tally winner carries a larger endorsing coalition (Exp 1 metric positive), follows the tally loser / top-cue favorite when the tally winner has more endorsements (Exp 2 metric negative), prefers the option with fewer endorsements even when a dominant high-validity cue endorses the rival (Exp 3 metric strongly negative, HIGH minus LOW), and produces a negative signed-evidence slope because P(choose TTB winner) DECREASES as validity-weighted evidence accumulates behind the more-endorsed option (Exp 4). It makes a novel falsifiable prediction: on designs where the high-validity option carries MORE positive ratings than the low-validity option, SDI predicts majority choices OPPOSITE to both TTB and VWEI.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Skeptical Defect Integration (SDI).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Penalty for option X:  P(X) = sum_j d_j * x_j,
    #   d_j = (1 - v_j)^delta  (skepticism weight; delta = 0 -> pure
    #   unweighted defect counting).
    # Decision variable:      D = (P(B) - P(A)) + alpha * E,
    #   E = sum_j logit(v_j) * (a_j - b_j)  (weak validity-weighted
    #   endorsement tie-break).
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so skepticism weights are fixed for the block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SDI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard against v = 1 (infinite logit) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)

    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])

    # Skepticism weights: an endorsement from expert j is a liability,
    # scaled by how fallible the expert is.
    d = (1.0 - v) ** delta

    # Total penalty (accumulated suspicion) for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Weak validity-weighted endorsement term (tie-break only).
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A (A carries fewer / more-heavily-discounted
    # liabilities than B, plus a small endorsement edge).
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties and endorsement) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- delta: [0.0, 0.05]
- beta: [0.9, 1.1]
- epsilon: [0.09, 0.14]
- alpha: [0.0, 0.05]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.7792 (var=0.0018)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.2684 (var=0.0040)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.4146 (var=0.0116)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.2020 (var=0.0005)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.1348 (var=0.0038)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.3037 (var=0.0089)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.6227 (var=0.0042)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.4600 (var=0.0059)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.4895 (var=0.0044)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.3810 (var=0.0022)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree
```

**Observed (real) value:** 0.7117 (var=0.0205)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0125)
- pi_2: 0.1203 (var=0.0072)
- pi_3: 0.6869 (var=0.0033)
- pi_4: 0.7792 (var=0.0018)
- pi_5: 0.7272 (var=0.0025)
- pi_6: 0.6844 (var=0.0026)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5
```

**Observed (real) value:** -0.2562 (var=0.0128)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3394 (var=0.0118)
- pi_1: -0.3231 (var=0.0164)
- pi_3: -0.2566 (var=0.0023)
- pi_4: -0.2684 (var=0.0040)
- pi_5: -0.2456 (var=0.0038)
- pi_6: -0.1978 (var=0.0023)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Validity-sensitivity contrast for the cue-2 cascade design.

    For every trial we compute (a) the Take-The-Best winner (first
    discriminating cue in validity order) and (b) the VWEI evidence
    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed
    relative to the TTB winner (Erel = E * (+1 if TTB winner is A
    else -1)).

    Two trial classes:
      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides
             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).
      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)
             decides but the weighted evidence of the opposing
             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).

    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).

    TTB predicts the SAME choice probability on every non-tied pair
    (its score is binary), so this difference is 0 in expectation.
    VWEI predicts a large positive difference (~ +0.42). The contrast
    cancels each subject's overall response rate (beta / epsilon),
    leaving only the stimulus-driven validity sensitivity.
    """
    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])
    lo = np.log(v / (1.0 - v))
    w = lo ** 3.0
    order = np.argsort(-v, kind="stable")

    highs = []
    lows = []

    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.shape[0] != 9 or b.shape[0] != 9:
            continue

        # VWEI-style evidence for A over B (gamma = 3).
        d = np.sign(a - b)
        E = float(np.dot(w, d))

        # TTB winner: first discriminating cue in validity order.
        win = None
        for j in order:
            if a[j] > b[j]:
                win = 0
                break
            if b[j] > a[j]:
                win = 1
                break
        if win is None:
            continue  # no discriminating cue (does not occur here)

        # Evidence relative to the TTB winner.
        s = 1.0 if win == 0 else -1.0
        Erel = E * s

        r_val = int(r)
        chose_winner = 1.0 if r_val == win else 0.0

        if Erel > 3.0:
            highs.append(chose_winner)
        elif Erel < -0.5:
            lows.append(chose_winner)

    if not highs or not lows:
        return 0.0

    return float(np.mean(highs) - np.mean(lows))
```

**Observed (real) value:** -0.5467 (var=0.0421)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0108 (var=0.0045)
- pi_3: 0.4433 (var=0.0115)
- pi_2: 0.5433 (var=0.0293)
- pi_4: -0.4146 (var=0.0116)
- pi_5: -0.5708 (var=0.0133)
- pi_6: -0.4858 (var=0.0140)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Signed-evidence slope.

    For every trial we compute the VWEI evidence
        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3
    and the TTB winner (first discriminating cue in validity order;
    validities are strictly descending so cue order = feature order).

    Define the signed regressor  x = +E if TTB's winner is A, x = -E if
    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's
    winner.  Return the OLS slope of y on x.

    TTB's probability of choosing its own winner is a single constant
    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected
    slope is exactly 0.  VWEI's probability of choosing TTB's winner is
    sigmoid(beta * x) on every trial, giving a clearly positive slope
    (~0.05 at the design center, sign-stable over the whole parameter
    box).  The two theories are maximally separated on this statistic.
    """
    if data is None or len(data) == 0:
        return 0.0

    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64], dtype=float)
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)
    w = np.log(v / (1.0 - v)) ** 3.0

    def _to_vec(cell):
        if isinstance(cell, str):
            return [float(ch) for ch in cell if ch in '01']
        return [float(val) for val in cell]

    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):
        return 0.0

    resp = data['response'].to_numpy(dtype=float)

    d = np.sign(a - b)                     # +1 favors A, -1 favors B
    E = d @ w                              # VWEI evidence for A

    disc = d != 0
    keep = disc.any(axis=1)                # drop degenerate all-tie rows
    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]
    if d.shape[0] == 0:
        return 0.0

    # First discriminating cue (validities strictly descending -> order 0..8)
    first_idx = disc.argmax(axis=1)
    rows = np.arange(d.shape[0])
    d_first = d[rows, first_idx]
    winner_a = d_first > 0                  # TTB's winner is option A

    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner
    y = np.where(winner_a, 1.0 - resp, resp)
    x = np.where(winner_a, E, -E)

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    yc = y - y.mean()
    slope = float(np.dot(xc, yc)) / denom
    return slope
```

**Observed (real) value:** -0.1938 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0475 (var=0.0011)
- pi_1: 0.0002 (var=0.0008)
- pi_2: 0.1895 (var=0.0024)
- pi_4: -0.2020 (var=0.0005)
- pi_5: -0.2142 (var=0.0004)
- pi_6: -0.1987 (var=0.0006)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Top-cue allegiance under endorsement-majority conflict.

    Selected trials: the 95%-validity expert (feature index 0) endorses
    exactly one option, AND that option carries strictly MORE total
    endorsements (sum of ratings) than its rival.  On these trials the
    metric is the empirical rate of choosing the option endorsed by the
    top expert.

    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98
    here: the amplified log-odds weight of v=.95 dominates every
    possible opposing coalition of the remaining cues, and 50%-expert
    endorsements contribute exactly zero weight.
    SDI must AVOID the top-cue option here (~0.15-0.25): every
    endorsement is a liability, and the top-cue option is also the
    endorsement-majority option.
    """
    df = data.reset_index(drop=True)

    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)
    resp = df['response'].to_numpy(dtype=float)

    if a.ndim != 2 or a.shape[0] == 0:
        return 0.5

    chose_a = resp < 0.5

    # Which option does the top (95%) expert endorse?
    top_a = a[:, 0] > 0.5
    top_b = b[:, 0] > 0.5
    # Keep only trials where exactly one option gets the top expert's
    # endorsement (guards against degenerate rows).
    xor_top = np.logical_xor(top_a, top_b)

    # Total endorsement counts per option.
    cnt_a = a.sum(axis=1)
    cnt_b = b.sum(axis=1)

    # Is the top-cue-endorsed option also the endorsement-majority
    # option (strictly more endorsements)?
    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)

    sel = xor_top & top_option_more
    if not np.any(sel):
        return 0.5

    # Did the subject choose the option endorsed by the top expert?
    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])
    return float(np.mean(followed))
```

**Observed (real) value:** 0.1143 (var=0.0077)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1348 (var=0.0038)
- pi_3: 0.9867 (var=0.0003)
- pi_1: 0.8433 (var=0.0110)
- pi_2: 0.9057 (var=0.0054)
- pi_5: 0.0852 (var=0.0016)
- pi_6: 0.1043 (var=0.0024)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-dissociation composite built on the two strongest contrasts
    # engineered into this design:
    #
    # COMPONENT 1 ("big-gap single endorsers", pairs 6/18, 8 trials/subject):
    #   both options carry exactly ONE endorsement; the endorsers' validities
    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the
    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);
    #   SDI is structurally near-blind at matched endorsement counts
    #   (P ~ 0.50-0.55 across its whole box).
    #
    # COMPONENT 2 ("ridge-crest coalition", pairs 10/22, 8 trials/subject):
    #   one option carries a single 0.80-validity endorsement, the other a
    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing
    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly
    #   favors the single endorser (coalition P ~ 0.10).
    #
    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.
    # Both components use only 8 trials each but their gaps point the SAME
    # way after sign alignment, so the composite has a large mean gap and
    # modest binomial-only between-subject variance.

    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    scores_gap = []    # 1 = chose the higher-validity single endorser
    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser

    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != V.shape or b.shape != V.shape:
            continue
        ca = int(round(a.sum()))
        cb = int(round(b.sum()))
        chose_a = (int(r) == 0)

        if ca == 1 and cb == 1:
            # Single endorsement on each side: who endorsed, and how good
            # is that expert?
            ia = int(np.argmax(a))
            ib = int(np.argmax(b))
            va = V[ia]
            vb = V[ib]
            if abs(va - vb) >= 0.25:
                hi_is_a = (va > vb)
                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)

        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):
            # Ridge-crest conflict: verify it is exactly the designed cell
            # (single 0.80 expert vs four 0.75 experts) and score the
            # coalition choice.
            if ca == 1:
                single_vec, coal_vec = a, b
                single_is_a = True
            else:
                single_vec, coal_vec = b, a
                single_is_a = False
            i_single = int(np.argmax(single_vec))
            coal_idx = np.where(coal_vec > 0.5)[0]
            if (
                abs(V[i_single] - 0.80) < 1e-9
                and len(coal_idx) == 4
                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))
            ):
                # +1 when the subject chose the coalition side
                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)

    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5
    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5
    return 0.5 * (r_gap + r_coal)
```

**Observed (real) value:** 0.1925 (var=0.0159)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7625 (var=0.0125)
- pi_4: 0.3037 (var=0.0089)
- pi_1: 0.4863 (var=0.0079)
- pi_2: 0.6925 (var=0.0122)
- pi_5: 0.1862 (var=0.0087)
- pi_6: 0.2387 (var=0.0095)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Validity-following rate in theory-conflict cells"
    #
    # The 15 experts split into a HIGH-validity group (features 0-6:
    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a
    # LOW-validity group (features 7-14: validity 0.60). Every trial in
    # this design pits a "pure high" option against a "pure low"
    # option. Two families of cells separate SDI from VPS:
    #
    # Family 1 (matched endorsement counts, validity gap): SDI's
    #   penalty is a near-flat function of validity (delta ~ 0), so it
    #   is pinned at ~0.51 for the high-validity option (its weak
    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's
    #   penalty rises steeply in validity, so it systematically picks
    #   the LOW-validity option (~0.28).
    #
    # Family 2 (count conflict where the FEWER-endorsed option is the
    #   high-validity one): SDI follows the endorsement count and
    #   picks the high-validity option (~0.72). VPS's validity-scaled
    #   distrust makes the low-validity majority cheaper, so it picks
    #   the high-validity option only ~0.30 of the time.
    #
    # The metric is the average of P(choose the high-validity-group
    #   option) over these two families. SDI predicts ~0.62; VPS
    #   predicts ~0.29.

    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,
                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

    f1_hits = 0.0
    f1_n = 0
    f2_hits = 0.0
    f2_n = 0

    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        m = int(min(a.size, b.size, V.size))
        if m == 0:
            continue
        a = a[:m]
        b = b[:m]

        top = V[:m] >= 0.65  # high-validity expert group
        a_top = float(np.sum(a[top]))
        a_bot = float(np.sum(a[~top]))
        b_top = float(np.sum(b[top]))
        b_bot = float(np.sum(b[~top]))

        # Require each option to be endorsed purely by one group
        # (all trials in this design satisfy this; guards oddities).
        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):
            continue
        ca = a_top + a_bot
        cb = b_top + b_bot
        if ca == 0 or cb == 0:
            continue

        a_is_top = a_top > 0
        b_is_top = b_top > 0
        if a_is_top == b_is_top:
            continue

        chose_a = (int(round(float(r))) == 0)
        chose_top = chose_a if a_is_top else (not chose_a)

        if ca == cb:
            # Family 1: matched counts, validity gap.
            f1_n += 1
            f2_unused = None
            f1_hits += 1.0 if chose_top else 0.0
        else:
            # Family 2: count conflict; keep only cells where the
            # FEWER-endorsed option is the high-validity one
            # (the cells where SDI and VPS point in opposite
            # directions). Cells where the fewer-endorsed option is
            # the low-validity one are agreed upon by both theories
            # and are excluded.
            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)
            if fewer_is_top:
                f2_n += 1
                f2_hits += 1.0 if chose_top else 0.0

    p1 = f1_hits / f1_n if f1_n > 0 else np.nan
    p2 = f2_hits / f2_n if f2_n > 0 else np.nan
    vals = [p for p in (p1, p2) if not np.isnan(p)]
    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.3815 (var=0.0238)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6227 (var=0.0042)
- pi_5: 0.2844 (var=0.0060)
- pi_1: 0.8411 (var=0.0121)
- pi_2: 0.3319 (var=0.0035)
- pi_3: 0.9332 (var=0.0008)
- pi_6: 0.3286 (var=0.0033)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Matched-count validity-reversal index.
    #
    # On trials where A and B carry the SAME number of endorsements (k vs k),
    # the two theories make opposite-sign predictions:
    #   VPS: the coalition of LOWER-validity endorsers is preferred
    #        (a high-validity endorsement is a bigger suspected defect),
    #        with preference strengthening in k.
    #   SDI: at matched count its validity sensitivity has the wrong sign
    #        and is bounded, so it stays at or below chance for the
    #        low-validity coalition on every such trial.
    #
    # The metric is the empirical proportion of choices of the
    # LOW-validity coalition across all matched-count trials
    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96
    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.
    # It is well-defined on a single subject's slice (48 Bernoulli
    # trials) and on the pooled data.

    v = np.array([
        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,
        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,
    ], dtype=float)

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:
            continue

        ca = float(a.sum())
        cb = float(b.sum())
        # Keep only matched-count trials with at least one endorsement
        # on each side (the validity-reversal assay cells).
        if ca == 0.0 or cb == 0.0 or ca != cb:
            continue

        # Mean validity of the endorsing coalition on each side.
        va = float(np.mean(v[a > 0]))
        vb = float(np.mean(v[b > 0]))
        if va == vb:
            continue

        low_is_a = va < vb
        chose_a = (int(row["response"]) == 0)
        hits.append(1.0 if (chose_a == low_is_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7808 (var=0.0201)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8263 (var=0.0045)
- pi_4: 0.4600 (var=0.0059)
- pi_1: 0.1496 (var=0.0101)
- pi_2: 0.5108 (var=0.0054)
- pi_3: 0.0317 (var=0.0006)
- pi_6: 0.8054 (var=0.0034)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Weighted proportion of choosing the side whose endorsements come from
    # MORE VALID experts, on matched-endorsement-count trials.
    #
    # Design validities (n_features=20, fixed): indices 0-5 are 0.50,
    # indices 6-19 are 0.60. On every trial where both options carry the
    # SAME number of endorsements, the flat count terms (c0) of both
    # DCD and VPS cancel exactly, so the choice is driven purely by the
    # validity composition of the endorsers:
    #   - VPS: penalty strictly increasing in v  -> always avoids the
    #     0.60 experts -> weighted P(hi-validity side) well below 0.5
    #     (mid ~0.38, box <= ~0.45).
    #   - DCD: dampened distrust (2v-1)^kappa collapses near chance and
    #     is opposed by the residual trust term alpha*logit(v) ->
    #     essentially at/above chance (mid ~0.51, box >= ~0.49).
    # Trials are weighted by m = |#0.60-endorsements difference|, which
    # emphasizes the cells (m=2,3,4) where the two theories diverge most.
    V = np.array([0.5] * 6 + [0.6] * 14, dtype=float)
    hi = (V > 0.55).astype(float)  # indicator of the 0.60-validity experts

    if data is None or len(data) == 0:
        return 0.5

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    num = 0.0
    den = 0.0
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)
        if a.shape[0] != V.shape[0] or b.shape[0] != V.shape[0]:
            continue
        tA = float(np.sum(a))
        tB = float(np.sum(b))
        # Only matched-count trials: the flat c0 terms cancel and the
        # choice isolates the validity-composition sensitivity.
        if tA != tB:
            continue
        hA = float(np.dot(hi, a))  # number of 0.60-expert endorsements on A
        hB = float(np.dot(hi, b))
        if hA == hB:
            continue  # pure tie in validity composition
        m = abs(hA - hB)
        r = int(resp[i])
        chose_hi = (r == 0) if (hA > hB) else (r == 1)
        den += m
        if chose_hi:
            num += m

    if den <= 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.3729 (var=0.0159)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5269 (var=0.0032)
- pi_5: 0.3820 (var=0.0036)
- pi_1: 0.7308 (var=0.0072)
- pi_2: 0.5032 (var=0.0040)
- pi_3: 0.5099 (var=0.0041)
- pi_4: 0.4895 (var=0.0044)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # COIN-FLIP ENDORSEMENT TOLERANCE.
    #
    # For every trial, count how many endorsements each option receives
    # from the five chance-validity (v = 0.50) experts (feature indices
    # 15-19 in this experiment's fixed validity vector). Every one of the
    # 24 unique pairs pits an option carrying coin-flip endorsements
    # against an option carrying none, so the trial is scored:
    #   1.0  if the subject chose the option with MORE 0.50-expert
    #        endorsements (the 'coin-flip tolerant' choice),
    #   0.0  otherwise.
    # The metric is the mean of this score over all trials.
    #
    # This is a direct assay of the one structural difference between the
    # two confidence-distrust theories: the geometry of the per-endorsement
    # distrust charge. VPS charges c0 + c1*v^kappa, so a 0.50 endorsement
    # still costs a substantial c1*0.25-0.33 distrust units and the
    # internal exchange rate between mid-validity (0.75) endorsements and
    # coin-flips is ~1.9 coin-flips per 0.75. DCD charges
    # c0 + c1*(2v-1)^kappa, which collapses to the bare floor c0 at
    # v = 0.50 (and refunds part of the 0.75 charge via the residual trust
    # term), yielding an exchange rate of only ~1.3. Hence VPS subjects
    # are systematically MORE willing to accept coin-flip endorsements,
    # across every cell family in the design:
    #   composition cells (mixed {v_hi, 0.50} vs homogeneous mid): VPS
    #     picks the mixed side (0.53-0.59), DCD avoids it (0.33-0.45);
    #   exchange cells (0.75-coalition vs k coin-flips): VPS keeps
    #     choosing the coin-flip side through k=3-4, DCD abandons it;
    #   calibration cells ({0.90} vs {0.50}, {0.50,0.50} vs {0.90,0.90}):
    #     VPS 0.73/0.86 vs DCD 0.65/0.76.
    # Mid-parameter predictions: VPS ~0.59, DCD ~0.45 (disjoint across
    # their full parameter boxes), with per-subject SD ~0.05-0.06 from
    # 96 Bernoulli trials.

    validities = np.array([
        0.92, 0.90, 0.90, 0.90, 0.85, 0.77, 0.77,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72,
        0.50, 0.50, 0.50, 0.50, 0.50,
    ])
    coin_mask = np.isclose(validities, 0.5)

    responses = data["response"].to_numpy()
    ratings_a = data["option_a_ratings"].tolist()
    ratings_b = data["option_b_ratings"].tolist()

    scores = []
    for i in range(len(ratings_a)):
        a = np.asarray(ratings_a[i], dtype=float).ravel()
        b = np.asarray(ratings_b[i], dtype=float).ravel()
        n = min(a.shape[0], b.shape[0], validities.shape[0])
        mask = coin_mask[:n]
        ca = float(np.sum(a[:n][mask]))
        cb = float(np.sum(b[:n][mask]))
        if ca == cb:
            # No coin-flip asymmetry on this trial; skip defensively.
            continue
        chose_a = (int(responses[i]) == 0)
        c_chosen = ca if chose_a else cb
        c_other = cb if chose_a else ca
        scores.append(1.0 if c_chosen > c_other else 0.0)

    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5096 (var=0.0121)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6071 (var=0.0030)
- pi_6: 0.4429 (var=0.0037)
- pi_1: 0.4433 (var=0.0012)
- pi_2: 0.6129 (var=0.0030)
- pi_3: 0.5190 (var=0.0010)
- pi_4: 0.3810 (var=0.0022)


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
```json
{
  "theory": "Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is treated as a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: the per-endorsement distrust charge is d_j = c0 + c1*v_j^kappa (kappa >= 1.5), with NO dampening near v = 0.5 and NO trust refund — the raw-validity geometry that Experiment 1 decisively confirmed. Decision making is a dual-process mixture at the trial level: with probability w the subject acts as a 'skeptic' who integrates the validity-distrust difference D = P(B) - P(A) (accumulated suspicion, option with LOWER suspicion preferred) through a sigmoid; with probability (1-w) the subject acts as a 'counter' who simply picks the option with FEWER total endorsements (pure tallying, indifferent at matched counts). A uniform lapse epsilon (0.08-0.20) further softens extremes. The count branch explains the systematic shrinkage of observed proportions toward 0.5 relative to pure validity-distrust: it pulls coin-flip tolerance down (Exp 2/10), lifts the high-validity option's rate in count-conflict cells (Exp 9), lowers the low-validity-coalition rate (Exp 10), and is exactly neutral on matched-count validity-gradient cells (Exps 1, 8), preserving the strongly confirmed VPS-signed results there. Moderation comes from an explicit count-heuristic mixture and noise, not from a trust term that reverses sign near chance validity.",
  "predict": "def predict(parameters, state, history):\n    # Softened Validity-Distrust with Count Mixture (SVD).\n    # Stimulus: array-like of shape (2, n_features), row 0 = option A,\n    # row 1 = option B; entries are binary expert ratings.\n    #\n    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose\n    # magnitude grows with the endorser's RAW claimed validity:\n    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near\n    #                                v = 0.5, NO trust refund)\n    #   accumulated suspicion: P(X) = sum_j x_j * d_j\n    #   D = P(B) - P(A)  (positive favors A)\n    #   P_skeptic(A) = sigmoid(beta * D)\n    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;\n    #   choose the option with FEWER total endorsements; exactly\n    #   indifferent (0.5) at matched counts.\n    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),\n    # mixed with a uniform lapse epsilon.\n    # History is ignored: validities are in the instructions, so the\n    # distrust weights are fixed for the whole block.\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"SVD expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n\n    v = np.asarray(parameters[\"validities\"], dtype=float)\n    if v.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {v.shape[0]} != n_features {n_features}.\"\n        )\n    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.\n\n    kappa = float(parameters[\"kappa\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    c0 = float(parameters[\"c0\"])\n    c1 = float(parameters[\"c1\"])\n    w_mix = float(parameters[\"w\"])\n\n    # --- Skeptic branch: raw validity-proportional distrust charge ---\n    d = c0 + c1 * np.power(v, kappa)\n    pen_a = float(np.dot(d, a))\n    pen_b = float(np.dot(d, b))\n\n    # D > 0 favors A (B carries more accumulated suspicion).\n    D = pen_b - pen_a\n    x = beta * D\n    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid\n\n    # --- Counter branch: pure endorsement-count tallying ---\n    cnt_a = float(np.sum(a))\n    cnt_b = float(np.sum(b))\n    if cnt_a < cnt_b:\n        p_count_a = 1.0      # A has fewer endorsements -> prefer A\n    elif cnt_a > cnt_b:\n        p_count_a = 0.0\n    else:\n        p_count_a = 0.5      # matched counts -> indifferent\n\n    # --- Dual-process mixture ---\n    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a\n    p_core = np.array([p_a, 1.0 - p_a], dtype=float)\n\n    # Independent lapse: with probability epsilon pick uniformly.\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "c0": "[0.05, 0.25]",
    "c1": "[0.5, 1.3]",
    "kappa": "[1.5, 2.5]",
    "beta": "[1.0, 2.5]",
    "w": "[0.6, 0.95]",
    "epsilon": "[0.08, 0.20]",
    "validities": "validities"
  },
  "rationale": "The arbiter's diagnosis is that DCD's moderation mechanism — the (2v-1)^kappa dampening plus a residual trust term — was falsified by Experiment 1 (which confirmed the RAW-validity distrust geometry of VPS), while pure VPS systematically over- or under-shoots in Exps 2, 9, and 10 where endorsement counts and validity-distrust conflict at unequal counts. SVD keeps the confirmed backbone exactly (d_j = c0 + c1*v_j^kappa, kappa >= 1.5, no dampening, no trust refund) and adds moderation through an explicit dual-process mixture instead: with probability w the subject is a validity-distrust skeptic, with probability (1-w) a pure endorsement counter. This single mechanism simultaneously repairs the residual misfits: (i) Exp 7's count-conflict cells (fewer-endorsed high-validity option) — the counter branch deterministically picks the fewer-endorsed option, lifting the high-validity rate from VPS's 0.284 toward the observed 0.382 (mixture at w~0.8 gives ~0.38); (ii) Exp 10's coin-flip tolerance — the counter avoids the larger coin-flip coalitions, pulling VPS's 0.607 down toward the observed 0.510; (iii) Exp 2's conflict cells — the counter opposes the endorsement majority, deepening VPS's -0.246 toward the observed -0.256; (iv) Exp 9's count-conflict cells pull the high-validity rate up from 0.28 toward 0.37. Crucially, on matched-count assays (Exp 1's validity gradient, Exp 8's validity-reversal index) the counter is exactly neutral (0.5), so the strongly confirmed VPS-signed results there are preserved through the w-weighted mixture with only the lapse-induced shrinkage. The larger lapse range (0.08-0.20) provides additional softening for extreme cells (Exps 3, 5) without ever reversing the sign of the validity-distrust sensitivity — unlike DCD's trust term, which is the specific mechanism Experiment 1 falsified. SVD is empirically distinguishable from both VPS (which it nests at w=1) and pure tallying (w=0), and its mixture structure is a standard, cognitively motivated dual-process architecture (heuristic cue-based integration vs. simple count-based tallying)."
}
```
```

## Usage

```json
{
  "input_tokens": 36657,
  "output_tokens": 4232,
  "reasoning_tokens": 1699,
  "provider": "AkashML"
}
```
