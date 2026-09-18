# generation_iter_07_attempt_00

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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 with an uncertainty-shrunk reliability voting theory. The new theory should propose that, in a no-feedback task involving fictitious experts, participants distrust the precision of communicated validity values and shrink expert-specific reliabilities strongly toward a common above-chance credibility level. Each discriminating endorsement therefore contributes approximately one vote, while only a limited residual weight reflects communicated reliability. Unlike pi_4, this residual reliability signal should be accumulated cue by cue rather than represented by softmax summaries of the two endorsing groups, and confidence should arise from a saturating transformation of total normalized evidence rather than a separate consensus-amplification module. A useful evidence form is a dominant signed tally plus a bounded, divisively normalized sum of reliability deviations, followed by a saturating choice function and a small lapse component. Subject variation should include the degree of reliability shrinkage, sensitivity, and lapse rate. Strong shrinkage predicts near indifference in Experiment 1's equal-tally conditions, high tally-winner choice in Experiments 2 and 6, and strong margin-dependent evidence in Experiment 4. A modest residual reliability term should preserve the minority/strongest-cue choices seen in Experiments 3 and 5. This creates a theoretically meaningful competitor to pi_4: approximately equal weighting emerges from uncertainty about communicated reliabilities, not from categorical votes plus bounded group summaries and explicit consensus amplification.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.926)

**Description**
Compressed-Reliability Consensus theory holds that people represent every discriminating expert primarily as one categorical vote and encode communicated reliability through a separate, bounded comparison of the two endorsing groups. For each option, attention is biased moderately toward its more reliable endorsers, but their reliabilities are summarized as a softmax-weighted mean rather than accumulated expert by expert. Net evidence is therefore the full signed tally plus an identity-sensitive contrast between the two bounded group summaries. Agreement additionally amplifies confidence as the absolute tally margin grows, with amplification saturating after a decisive coalition has formed. This representation permits expert identity to determine equal-tally choices, preserves substantial influence for near-chance experts, and allows sufficiently numerous weak experts to override the strongest expert. Reliability sensitivity, consensus amplification, choice sensitivity, and lapse rate vary across subjects. Because the task provides no correctness feedback, choices are reversal-symmetric and independent of history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Compressed-Reliability Consensus model. History is intentionally ignored
    # because the task supplies no outcome feedback from which to recalibrate.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # Every discriminating expert contributes a full baseline vote through the
    # signed tally. Reliability is encoded separately as a bounded summary of
    # each endorsing side, preventing identity increments from accumulating
    # without bound as a coalition grows.
    v = np.clip(validities, 0.5, 1.0)
    reliability_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)

    difference = stimulus[0] - stimulus[1]
    a_endorsers = difference > 0.0
    b_endorsers = difference < 0.0
    tally_margin = float(np.sum(a_endorsers) - np.sum(b_endorsers))

    def side_reliability(mask):
        values = reliability_code[mask]
        if values.size == 0:
            return 0.0
        attention_logits = 4.0 * values
        attention_logits = attention_logits - np.max(attention_logits)
        attention = np.exp(attention_logits)
        attention /= attention.sum()
        return float(np.dot(attention, values))

    reliability_contrast = (
        side_reliability(a_endorsers) - side_reliability(b_endorsers)
    )
    alpha = float(parameters["reliability_sensitivity"])
    weighted_balance = tally_margin + alpha * reliability_contrast

    # Agreement is represented separately by the unweighted vote margin.
    # It changes confidence but does not erase which experts supplied the
    # votes. Leaving margin-one and tied tallies unamplified preserves
    # uncertainty in sparse conflicts and identity effects in equal tallies.
    # Amplification saturates at the level reached by a margin-five coalition.
    excess_consensus = max(abs(tally_margin) - 1.0, 0.0)
    consensus_gain = float(parameters["consensus_amplification"])
    confidence_multiplier = 1.0 + consensus_gain * min(excess_consensus ** 1.5, 8.0)
    evidence_for_a = weighted_balance * confidence_multiplier

    sensitivity = float(parameters["choice_sensitivity"])
    logits = np.array(
        [0.5 * sensitivity * evidence_for_a,
         -0.5 * sensitivity * evidence_for_a],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(probabilities.size, 1.0 / probabilities.size)
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- reliability_sensitivity: [0.65, 1.05]
- consensus_amplification: [0.03, 0.07]
- choice_sensitivity: [0.13, 0.30]
- lapse_rate: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3392 (var=0.0040)
- Experiment 2: real=89.1422 (var=7.5896) vs this=100.3141 (var=2.8389)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.4698 (var=0.0021)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.7775 (var=0.0083)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=0.0071 (var=0.0478)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8037 (var=0.0033)


---

### `pi_2` (overall score: 0.299)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.1350 (var=0.0104)
- Experiment 2: real=89.1422 (var=7.5896) vs this=1.9009 (var=0.0916)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.1190 (var=0.0098)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8606 (var=0.0060)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.0256 (var=0.0579)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8612 (var=0.0065)


---

### `pi_3` (overall score: 0.262)

**Description**
Communicated-Reliability Accumulation theory proposes that people integrate every discriminating expert rating after translating communicated validity into subjective diagnostic weight. The weight is a mildly superlinear log-odds transformation plus a modest common reliability floor: w_j = logit(v_j)^gamma + c. The log-odds component privileges more reliable experts, while the positive floor represents a minimal contribution assigned to any expert judged better than chance. Net evidence favoring A is E = sum_j w_j(A_j-B_j). Consequently, sufficiently numerous weaker experts can overturn a stronger cue, and concordant cue-number margins generate graded confidence. Choice sensitivity varies moderately across people, while only very rare lapses produce validity-independent guessing. Choice history is ignored because the task provides no outcome feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Communicated-Reliability Accumulation.
    # state has shape (2, n_features), with rows for options A and B.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Clipping only protects the log-odds transform at its mathematical
    # endpoints. A modest common floor strengthens accumulation by cue number,
    # while mild superlinearity preserves the leverage of highly valid cues.
    v = np.clip(validities, 0.5, 1.0 - 1e-12)
    log_odds = np.log(v) - np.log1p(-v)
    gamma = float(parameters["gamma"])
    reliability_floor = float(parameters["reliability_floor"])
    reliability_weights = np.power(log_odds, gamma) + reliability_floor

    a = stimulus[0]
    b = stimulus[1]
    evidence_for_a = float(np.dot(reliability_weights, a - b))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Symmetric logits make their difference beta * evidence_for_a, so the
    # core probability of A is sigmoid(beta * evidence_for_a).
    logits = np.array(
        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities /= total
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.24, 0.58]
- epsilon: [0.0, 0.005]
- gamma: [1.02, 1.05]
- reliability_floor: [0.10, 0.14]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3775 (var=0.0056)
- Experiment 2: real=89.1422 (var=7.5896) vs this=49.6494 (var=1.7369)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.5394 (var=0.0025)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.3869 (var=0.0089)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=0.5887 (var=0.0712)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.5665 (var=0.0017)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2463 -> ACCEPTED
- iter 2: loss=0.2965 -> REJECTED
- iter 3: loss=0.1428 -> ACCEPTED
- iter 4: loss=0.1191 -> ACCEPTED
- iter 5: loss=0.1095 -> ACCEPTED
- iter 6: loss=0.1031 -> ACCEPTED
- iter 7: loss=0.1036 -> REJECTED
Running-best (last ACCEPTED) base: iter 6 at loss=0.1031 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.2050 (var=0.0023) (Δ vs real -0.1983)
  - iter 2: 0.2767 (var=0.0060) (Δ vs real -0.1267)
  - iter 3: 0.2367 (var=0.0033) (Δ vs real -0.1667)
  - iter 4: 0.2471 (var=0.0051) (Δ vs real -0.1562)
  - iter 5: 0.2579 (var=0.0036) (Δ vs real -0.1454)
  - iter 6: 0.2629 (var=0.0045) (Δ vs real -0.1404)
  - iter 7 (most recent): 0.2742 (var=0.0035) (Δ vs real -0.1292)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Previous candidate values (this loop):**
  - iter 1: 26.8735 (var=0.3073) (Δ vs real -62.2688)
  - iter 2: 51.2353 (var=1.1080) (Δ vs real -37.9069)
  - iter 3: 76.3061 (var=1.7568) (Δ vs real -12.8362)
  - iter 4: 51.3945 (var=0.8966) (Δ vs real -37.7478)
  - iter 5: 60.7144 (var=1.3470) (Δ vs real -28.4278)
  - iter 6: 66.3434 (var=2.7327) (Δ vs real -22.7989)
  - iter 7 (most recent): 60.6359 (var=1.5863) (Δ vs real -28.5063)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with fewer positive ratings."""
    if data is None or len(data) == 0:
        return float("nan")

    minority_choices = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        tally_margin_a = float(np.sum(a > b) - np.sum(b > a))
        r = int(response)

        if tally_margin_a < 0:       # A has fewer cue wins
            minority_choices.append(float(r == 0))
        elif tally_margin_a > 0:     # B has fewer cue wins
            minority_choices.append(float(r == 1))
        # Tied tallies have no minority option and are omitted.

    if len(minority_choices) == 0:
        return float("nan")
    return float(np.mean(minority_choices))
```

**Observed (real) value:** 0.4508 (var=0.0031)
**Previous candidate values (this loop):**
  - iter 1: 0.3792 (var=0.0030) (Δ vs real -0.0717)
  - iter 2: 0.4619 (var=0.0040) (Δ vs real +0.0110)
  - iter 3: 0.4175 (var=0.0025) (Δ vs real -0.0333)
  - iter 4: 0.4462 (var=0.0034) (Δ vs real -0.0046)
  - iter 5: 0.4375 (var=0.0026) (Δ vs real -0.0133)
  - iter 6: 0.4331 (var=0.0021) (Δ vs real -0.0177)
  - iter 7 (most recent): 0.4188 (var=0.0032) (Δ vs real -0.0321)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    a = np.vstack([np.asarray(x, dtype=float) for x in data["option_a_ratings"]])
    b = np.vstack([np.asarray(x, dtype=float) for x in data["option_b_ratings"]])
    responses = np.asarray(data["response"], dtype=int)

    # Positive margin means that A wins more binary feature comparisons;
    # negative margin means that B does.
    delta = a - b
    tally_margin = np.sum(delta > 0, axis=1) - np.sum(delta < 0, axis=1)

    # Select the two orientations of the diagnostic 6-versus-1 conflict:
    # the tally winner has a five-cue margin but is opposed by expert 1.
    strongest_cue = delta[:, 0]
    critical = (np.abs(tally_margin) == 5) & (strongest_cue * tally_margin < 0)
    if not np.any(critical):
        return float("nan")

    chose_a = responses == 0
    tally_winner_is_a = tally_margin > 0

    # Rate at which the numerous weak experts override the strongest expert.
    return float(np.mean(chose_a[critical] == tally_winner_is_a[critical]))
```

**Observed (real) value:** 0.8213 (var=0.0169)
**Previous candidate values (this loop):**
  - iter 1: 0.8606 (var=0.0036) (Δ vs real +0.0394)
  - iter 2: 0.7125 (var=0.0222) (Δ vs real -0.1088)
  - iter 3: 0.8700 (var=0.0029) (Δ vs real +0.0487)
  - iter 4: 0.8575 (var=0.0043) (Δ vs real +0.0363)
  - iter 5: 0.8456 (var=0.0035) (Δ vs real +0.0244)
  - iter 6: 0.8431 (var=0.0040) (Δ vs real +0.0219)
  - iter 7 (most recent): 0.8381 (var=0.0051) (Δ vs real +0.0169)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as choosing the option endorsed by the 95%-valid
    # expert (feature 0), irrespective of whether that option is A or B.
    by_size = {1: [], 2: [], 4: []}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        # In this design, coalition size is the number of experts endorsing A
        # over B (equal to the number endorsing B over A).
        coalition_size = int(np.sum(difference > 0.0))
        if coalition_size not in by_size:
            continue

        high_expert_option = 0 if difference[0] > 0.0 else 1
        chose_high_expert = float(int(row['response']) == high_expert_option)
        by_size[coalition_size].append(chose_high_expert)

    if any(len(by_size[k]) == 0 for k in (1, 2, 4)):
        return float('nan')

    p1 = float(np.mean(by_size[1]))
    p2 = float(np.mean(by_size[2]))
    p4 = float(np.mean(by_size[4]))

    # Approximate inverse-noise weighting of the conditions with the largest
    # predicted between-theory probability differences. The size-3 condition
    # is omitted because both theories predict it close to chance.
    return float(1.8 * (p1 - 0.5) + 0.9 * (p2 - 0.5) - (p4 - 0.5))
```

**Observed (real) value:** -0.0382 (var=0.0352)
**Previous candidate values (this loop):**
  - iter 1: 0.2035 (var=0.0594) (Δ vs real +0.2417)
  - iter 2: 0.3202 (var=0.0643) (Δ vs real +0.3584)
  - iter 3: 0.1493 (var=0.0619) (Δ vs real +0.1875)
  - iter 4: -0.0170 (var=0.0441) (Δ vs real +0.0212)
  - iter 5: -0.0014 (var=0.0472) (Δ vs real +0.0367)
  - iter 6: 0.0142 (var=0.0399) (Δ vs real +0.0523)
  - iter 7 (most recent): -0.0011 (var=0.0495) (Δ vs real +0.0371)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with the larger raw vote tally."""
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=np.float64)
        b = np.asarray(b_cell, dtype=np.float64)
        margin = float(np.sum(a - b))
        r = int(response)

        if margin > 0.0:
            scores.append(1.0 if r == 0 else 0.0)
        elif margin < 0.0:
            scores.append(1.0 if r == 1 else 0.0)
        else:
            scores.append(0.5)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8462 (var=0.0122)
**Previous candidate values (this loop):**
  - iter 1: 0.8527 (var=0.0017) (Δ vs real +0.0065)
  - iter 2: 0.7042 (var=0.0121) (Δ vs real -0.1421)
  - iter 3: 0.8692 (var=0.0015) (Δ vs real +0.0229)
  - iter 4: 0.8538 (var=0.0014) (Δ vs real +0.0075)
  - iter 5: 0.8623 (var=0.0013) (Δ vs real +0.0160)
  - iter 6: 0.8612 (var=0.0017) (Δ vs real +0.0150)
  - iter 7 (most recent): 0.8525 (var=0.0014) (Δ vs real +0.0063)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Uncertain-Credibility Normalized Voting theory proposes that people do not treat instructed expert validities as precisely known reliabilities. In the absence of feedback, each communicated validity is shrunk toward a shared, moderately credible prototype. Consequently, every discriminating expert supplies one dominant categorical vote, while the incompletely trusted deviation of that expert's validity from the prototype supplies only a residual adjustment. Extreme communicated deviations are compressed smoothly because participants regard unusually high or low precision claims as especially uncertain, while still retaining limited distinctions among the most credible experts. These residual adjustments are accumulated cue by cue, preserving expert identity, but their total is divisively normalized and increasingly bounded once a clear tally coalition exists. Residual reliability is also contextually gated: it is strongly discounted in exact tally ties, where conflicting reliability claims provide no categorical direction, while otherwise retaining its ordinary residual weight without a special margin-one amplification. The resulting total evidence is passed through a saturating response transformation, so additional agreement increases confidence strongly at first but has diminishing effects once evidence is decisive. Between-person differences arise in reliability shrinkage, response sensitivity, and lapse rate. Because choices reveal no correctness information, the representation is not updated from choice history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Uncertain-Credibility Normalized Voting. History is intentionally ignored:
    # without outcome feedback, previous choices cannot identify expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # A difference of +1 is an endorsement of A and -1 an endorsement of B.
    # Nondiscriminating experts contribute neither a vote nor reliability evidence.
    signed_endorsements = np.sign(stimulus[0] - stimulus[1])
    discriminating = signed_endorsements != 0.0
    tally_evidence = float(np.sum(signed_endorsements))

    # Map validity onto above-chance diagnostic strength. The reference value
    # zero corresponds to a common validity of .75. Lower-tail deviations are
    # winsorized, while upper-tail deviations are smoothly compressed so that
    # very credible experts remain distinguishable without receiving full weight.
    v = np.clip(validities, 0.5, 1.0)
    communicated_strength = 2.0 * (v - 0.5)
    raw_deviation = communicated_strength - 0.5
    lower_bounded = np.maximum(raw_deviation, -0.25)
    upper_excess = np.maximum(lower_bounded - 0.25, 0.0)
    reliability_deviation = (
        np.minimum(lower_bounded, 0.25)
        + 0.13 * np.tanh(upper_excess / 0.10)
    )

    # Strong shrinkage leaves only a small fraction of communicated deviations.
    # Deviations are nevertheless summed expert by expert, unlike a summary of
    # the two endorsing groups. Divisive normalization bounds their joint impact.
    shrinkage = float(parameters["reliability_shrinkage"])
    retained_precision = 1.0 - shrinkage
    active_deviations = reliability_deviation[discriminating]
    active_signs = signed_endorsements[discriminating]

    if active_deviations.size == 0:
        reliability_adjustment = 0.0
    else:
        signed_sum = float(np.dot(active_signs, active_deviations))
        deviation_mass = float(np.sum(np.abs(active_deviations)))
        coalition_protection = 0.12 * max(abs(tally_evidence) - 1.0, 0.0) ** 2
        reliability_adjustment = (
            4.0 * retained_precision * signed_sum
            / (1.0 + retained_precision * deviation_mass + coalition_protection)
        )

        # Reliability is least trusted when the categorical tally is exactly
        # tied. Outside exact ties it receives its ordinary residual weight;
        # there is no separate amplification at a one-vote margin.
        absolute_tally = abs(tally_evidence)
        tie_gate = 1.0 - 0.94 * np.exp(-((absolute_tally / 0.35) ** 2))
        reliability_adjustment *= tie_gate

    total_evidence_for_a = tally_evidence + reliability_adjustment

    # Confidence is generated by one saturating transformation of total
    # evidence, rather than by a separate consensus-amplification mechanism.
    normalized_evidence = np.tanh(total_evidence_for_a / 4.0)
    sensitivity = float(parameters["choice_sensitivity"])
    decision_variable = sensitivity * normalized_evidence

    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * p_core + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- reliability_shrinkage: [0.58, 0.84]
- choice_sensitivity: [2.20, 2.70]
- lapse_rate: [0.0, 0.12]
- validities: validities

`rationale`: This conservatively extends the accepted candidate with only three targeted changes. First, the remaining 1.20 margin-one multiplier is removed, because its earlier reduction improved Experiment 2 and its special amplification flattened the required tally-margin profile. Second, the bounded upper-tail contribution is increased modestly from 0.10 to 0.13, preserving more information about exceptionally credible experts and compensating for the removed boost on Experiment 1 conflicts without globally lowering reliability shrinkage. Exact-tie attenuation is strengthened from 0.90 to 0.94 so this upper-tail relaxation does not restore the strongest-cue distortion in Experiment 5. Third, only lapse heterogeneity is widened, from [0.0, 0.08] to [0.0, 0.12]. This should increase between-subject dispersion while gently reducing the slightly excessive tally-choice means in Experiments 4 and 6 and moving sub-chance reliability-favored rates in Experiment 1 toward indifference. Cue-wise accumulation, divisive normalization, coalition protection, saturation geometry, shrinkage, and sensitivity are otherwise unchanged.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, and it succeeds where a dominant tally should win: Experiment 4 is close (0.861 versus 0.821) and Experiment 6 is essentially exact (0.853 versus 0.846). However, it over-shrinks residual reliability information. It chooses the TTB winner too rarely in Experiment 1 (0.205 versus 0.403) and the reliability-favored minority too rarely in Experiment 3 (0.379 versus 0.451). It also produces much too little margin-dependent evidence in Experiment 2 (log-BF 26.87 versus 89.14). Experiment 5 remains substantially wrong (0.204 versus -0.038), indicating that the current deviation-mass normalization disproportionately preserves the strongest cue's influence in small equal-tally coalitions while suppressing cumulative reliability differences in larger coalitions. Finally, simulated between-subject variance is markedly too low in Experiments 2, 4, and 6, so the stated heterogeneity ranges are too narrow.
Rationale: Retain the recommended mechanism, but recalibrate its residual and response geometry. First, allow moderately more reliability information through—via somewhat lower shrinkage or a fitted residual-gain parameter—to raise TTB/minority choices in Experiments 1 and 3 and slightly reduce the excessive tally rate in Experiment 4. Do not simply amplify the current adjustment unchanged, because that would likely worsen Experiment 5. Instead, weaken or rescale the deviation-mass term in the divisive denominator so cue-wise reliability differences can accumulate more across larger equal-sized coalitions; the present denominator suppresses large-coalition residual evidence too aggressively relative to one- and two-cue conflicts. Second, the tanh scale of 2.5 saturates too early. Use a broader evidence scale together with higher sensitivity, preserving choice rates at large margins while creating a steeper distinction between small and large tally margins; this directly targets Experiment 2 without sacrificing the Experiment 6 mean. Finally, widen subject-level sensitivity and lapse ranges, and possibly shrinkage variation, to address the severe underdispersion. These are in-family adjustments to the residual normalization, saturation, and parameter ranges rather than a change of theory.

**Outcome of this advice:** iter 1 candidate loss=0.2463 -> iter 2 candidate loss=0.2965 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, but this iteration is not experiment-invariant and was correctly rejected by the accept gate. Strengthening the residual reliability channel improves Experiment 1 (0.277 versus 0.403), brings Experiment 3 close to target (0.462 versus 0.451), and raises Experiment 2's margin evidence (51.24 versus 89.14). Those gains come at a larger cost elsewhere: tally-winner choice falls too far in Experiment 4 (0.713 versus 0.821) and Experiment 6 (0.704 versus 0.846), while Experiment 5 worsens from 0.204 to 0.320 against an observed -0.038. Thus the revised residual is no longer merely a limited reliability correction; it can substantially counteract margins of three to five votes. The weaker deviation-mass denominator and large residual-gain range also amplify the strongest expert in Experiment 5's small equal-tally coalitions rather than reproducing the observed coalition-size pattern. The broader heterogeneity does improve variance in Experiments 4 and 6, but only while moving their means far from the data; Experiment 2 remains both underfit in its mean and underdispersed.
Rationale: Because this iteration was rejected, work from the unchanged iteration-1 accepted base. My previous recommendation to jointly weaken the denominator, increase residual gain, broaden saturation, and widen all parameter ranges did not improve aggregate loss, so do not repeat that package. Try a different in-family correction: keep the accepted model's stronger normalization and response geometry initially, but explicitly bound residual reliability relative to the tally margin—for example, add an absolute-tally term to the divisive denominator or pass the residual through a low-amplitude tanh before adding it to the tally. Residual influence should remain appreciable for one-vote conflicts, which can improve Experiments 1 and 3, but should be sharply limited at margins three and five so Experiments 4 and 6 retain their already accurate accepted-base means. For Experiment 5, compress or winsorize each cue's reliability deviation before cue-wise summation rather than globally increasing residual gain. This reduces disproportionate leverage from the single 95%-valid expert while still allowing several moderate cue deviations to accumulate, and it stays within the prescribed cue-wise uncertainty-shrinkage mechanism. Tune these residual-shape changes before altering saturation. If Experiment 2 still needs more margin dependence, make only a small, one-at-a-time response adjustment—such as an intermediate saturation scale between 2.5 and 5.0 with sensitivity recalibrated to preserve the margin-five choice rate—rather than another simultaneous broadening. Address subject variance only after the cross-experiment means are restored.

**Outcome of this advice:** iter 2 candidate loss=0.2965 -> iter 3 candidate loss=0.1428 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, and the accepted changes substantially improve experiment invariance: aggregate loss falls from 0.2463 to 0.1428. The broader response scale sharply improves Experiment 2's margin-dependent evidence (76.31 versus 89.14), while tally dominance remains close in Experiments 4 and 6 (0.870 versus 0.821 and 0.869 versus 0.846). Experiment 3 is also reasonably close (0.418 versus 0.451). Nevertheless, two systematic failures remain. In Experiment 1, reliability-favored TTB choices are still much too rare (0.237 versus 0.403), indicating insufficient residual reliability influence in nonzero, especially small-margin, tally conflicts. In Experiment 5, the strongest-cue coalition statistic remains too positive (0.149 versus -0.038), indicating excessive residual influence in exact tally ties. Thus a single residual treatment is still conflating two contexts: it is too weak when reliability must counter a small tally majority, yet too strong when there is no tally direction. Simulated between-subject variance also remains substantially too low in Experiments 2, 4, and 6, although variance tuning should remain secondary to correcting these means.
Rationale: The previous recommendation was accepted and should be extended rather than reversed: retain cue-wise winsorization, tally-dependent coalition protection, and the current broad saturation geometry. Make a localized in-family change to the residual normalization based on tally context. Attenuate the reliability adjustment specifically when the raw tally is exactly tied, which should reduce Experiment 5 without suppressing cue-wise accumulation generally. Conversely, permit a modestly larger residual adjustment when the absolute tally margin is one, raising the TTB/minority rates in Experiments 1 and 3. Keep the strong protection for margins above one so Experiments 4 and 6 do not repeat iteration 2's collapse. This could be implemented as a smooth bounded multiplier or denominator term indexed by absolute tally magnitude, rather than a larger global residual gain or broadly lower shrinkage. Avoid another wholesale sensitivity or saturation change: Experiment 2 is now much closer, and large-margin means are already slightly high. After the mean pattern is improved, selectively widen subject-level shrinkage or sensitivity variation while preserving their central values; do not repeat the previously rejected package of widening all ranges together.

**Outcome of this advice:** iter 3 candidate loss=0.1428 -> iter 4 candidate loss=0.1191 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, and the accepted contextual gate improves aggregate loss from 0.1428 to 0.1191. Exact-tie attenuation is especially successful: Experiment 5 moves from 0.149 to -0.017, close to the observed -0.038. Experiment 3 is now essentially exact (0.446 versus 0.451), Experiment 6 remains excellent (0.854 versus 0.846), and Experiment 4 is reasonably close though still somewhat too tally-dominant (0.858 versus 0.821). However, experiment invariance is not yet adequate. Experiment 1 still substantially underpredicts TTB choices in tally conflicts (0.247 versus 0.403), despite the narrow-margin boost. More importantly, Experiment 2 regresses sharply from the previously accepted 76.31 to 51.39 against an observed 89.14, indicating that the margin-one multiplier has disrupted the desired margin-dependent choice profile. Between-subject variability also remains much too low in Experiments 2, 4, and 6.
Rationale: Preserve the successful exact-tie attenuation and the strong margin-five coalition protection; those changes explain the gains in Experiments 5 and 6. Do not globally increase residual gain or weaken normalization, since that previously collapsed Experiments 4 and 6. Instead, reduce or narrow the current 1.65 margin-one multiplier, whose introduction coincided with the large Experiment 2 regression and yielded only a small Experiment 1 improvement. Recover reliability-sensitive choices through the cue-wise reliability representation itself: the current clipping maps all validities above .875 to the same deviation, eliminating distinctions among the strongest experts. Modestly relax or smooth only this upper-tail compression while retaining strong shrinkage, divisive normalization, and tally-dependent protection. This should increase strongest-cue/minority choices in Experiments 1 and 3, while Experiment 5 remains protected by the tie gate and margin-five conflicts in Experiments 4 and 6 remain protected by the denominator. It may also restore Experiment 2 if its high-validity cues generally align with the tally winner, rather than injecting generic margin-one noise. After restoring the Experiment 2 mean, selectively widen sensitivity around its current center to address underdispersion; avoid the previously rejected package of simultaneously widening all parameter ranges.

**Outcome of this advice:** iter 4 candidate loss=0.1191 -> iter 5 candidate loss=0.1095 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, and the latest accepted revision improves aggregate loss from 0.1191 to 0.1095. The smooth upper reliability tail raises Experiment 1 in the intended direction (0.258 versus 0.403), and reducing the margin-one boost partly restores Experiment 2 (60.71 versus 89.14). Experiments 3–6 are broadly well captured: Experiment 3 is close (0.438 versus 0.451), Experiment 4 is only modestly too tally-dominant (0.846 versus 0.821), Experiment 5 remains near its unusual negative target (-0.001 versus -0.038), and Experiment 6 is close (0.862 versus 0.846). Nevertheless, the remaining failures are systematic. Reliability-favored choices in Experiment 1 remain substantially too rare, while Experiment 2 still shows too little dependence on tally margin. The latter is not just a scale mismatch: its simulated between-subject variance is also far too low (1.35 versus 7.59). Variance is similarly compressed in Experiments 4 and 6, despite their accurate means.
Rationale: The previous recommendation was accepted, so extend it conservatively rather than changing mechanism families or reviving the rejected global increase in residual gain. First, remove or further attenuate the remaining 1.20 margin-one multiplier. Its reduction already improved Experiment 2, suggesting that even the residual boost continues to flatten the margin profile; returning it close to 1.0 should recover more of the strong margin dependence previously obtained at iteration 3. Compensate for the resulting loss in Experiment 1 through a small additional relaxation of only the smooth upper-tail reliability compression—for example, modestly increase its bounded asymptote or slope—rather than lowering shrinkage globally. This targets the highly credible opposing cues relevant to Experiment 1 and should also move the slightly excessive tally rates in Experiments 4 and 6 downward. To prevent that adjustment from undoing Experiment 5, strengthen the exact-tie attenuation correspondingly, while retaining divisive normalization and margin-five coalition protection. Finally, widen sensitivity or lapse heterogeneity one parameter at a time around the current central values. The means in Experiments 4 and 6 are slightly high, so modestly broader lapse variation is a plausible first variance correction; avoid the previously rejected package of simultaneously widening all ranges.

**Outcome of this advice:** iter 5 candidate loss=0.1095 -> iter 6 candidate loss=0.1031 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, and the latest changes were again accepted, reducing aggregate loss from 0.1095 to 0.1031. Experiments 3, 4, 5, and 6 are reasonably captured at the pooled level: the respective errors are -0.018, +0.022, +0.052, and +0.015. However, the two principal failures remain substantial. Experiment 1 still underpredicts reliability-favored TTB choices by 0.140 (0.263 versus 0.403), while Experiment 2 still produces too little margin-dependent evidence (66.34 versus 89.14). The model also remains markedly underdispersed in Experiments 2, 4, and 6. The latest upper-tail and lapse adjustments helped aggregate fit only modestly and Experiment 5 has drifted away from its best iteration-4 fit, so continuing to relax the upper reliability tail or strengthen the tie gate is unlikely to resolve the central problem.
Rationale: Retain the mechanism family, exact-tie attenuation, cue-wise residual summation, saturation, and strong protection at five-vote margins. The next adjustment should target the shape of coalition protection rather than again increasing the upper-tail asymptote or globally lowering shrinkage. The current quadratic term begins suppressing reliability immediately above a one-vote margin: it is 0.12 at margin two and 0.48 at margin three. Delay that protection through intermediate margins and make it rise more sharply only near margins four to five. This provides a useful experiment-invariant correction: in Experiment 1, reliability typically opposes the tally at margins two and three, so greater residual retention should raise TTB choices; in Experiment 2, the more reliable cues generally support the tally winner, so the same change should strengthen larger-margin choices relative to margin-one choices and increase the tally-over-TTB Bayes factor. Keeping strong protection at margin five should preserve Experiments 4 and 6, avoiding the collapse caused by the rejected global residual amplification. Do not further widen lapse at this stage: it cannot generate the missing margin profile and may erode Experiment 2. After recalibrating coalition protection, address underdispersion with a modest widening of choice-sensitivity heterogeneity around the same center, one parameter at a time. Preserve the current reliability-tail mapping initially, because further relaxation has shown diminishing gains in Experiment 1 while moving Experiment 5 in the wrong direction.

**Outcome of this advice:** iter 6 candidate loss=0.1031 -> iter 7 candidate loss=0.1036 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed uncertainty-shrunk, cue-wise reliability-voting family, but the delayed-quartic coalition protection did not improve experiment invariance and was rejected by the gate. Relative to the accepted iteration-6 base, it raises Experiment 1's TTB-conflict rate from 0.263 to 0.274 and slightly improves the already close large-margin results in Experiments 4 and 6. Those gains are outweighed by deterioration in Experiment 2's tally-margin evidence from 66.34 to 60.64, and in Experiment 3's minority-choice rate from 0.433 to 0.419. Experiment 1 also remains far below its observed 0.403, while Experiment 2 remains far below 89.14. Thus releasing residual reliability broadly at margins two and three creates the wrong profile: it provides only a small increase in reliability-favored conflict choices while flattening, rather than strengthening, the margin dependence needed in Experiment 2. Across-subject variance remains severely compressed, especially in Experiments 2, 4, and 6.
Rationale: Because the delayed-quartic change was rejected, do not repeat the recommendation to broadly postpone coalition protection; the next proposal should build on the unchanged iteration-6 accepted base. Retain its quadratic protection, exact-tie gate, reliability mapping, cue-wise accumulation, and saturation form. A better next in-family knob is modestly wider choice-sensitivity heterogeneity around the same center, tested without simultaneous changes to shrinkage or lapse—for example, expand the current [2.20, 2.70] interval toward roughly [1.9, 3.0]. This directly addresses the persistent underdispersion. It also has a potentially favorable mean effect through the nonlinear choice function: sensitivity mixtures should pull the overly high tally rates in Experiments 4 and 6 downward, while increasing average probabilities of currently rare TTB/minority responses in Experiments 1 and 3. Unlike additional lapse, sensitivity variation preserves margin dependence and may increase the subject-level spread of Experiment 2's Bayes factors. This is distinct from the previously rejected iteration-1 package, which changed residual gain, normalization, saturation, and several heterogeneity ranges simultaneously. If coalition protection is revisited later, use only a very small localized reduction at margin two—between the accepted quadratic and rejected quartic forms—rather than removing protection across both margins two and three.

**Outcome of this advice:** iter 7 candidate loss=0.1036 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Uncertain-Credibility Normalized Voting theory proposes that people do not treat instructed expert validities as precisely known reliabilities. In the absence of feedback, each communicated validity is shrunk toward a shared, moderately credible prototype. Consequently, every discriminating expert supplies one dominant categorical vote, while the incompletely trusted deviation of that expert's validity from the prototype supplies only a residual adjustment. Extreme communicated deviations are compressed smoothly because participants regard unusually high or low precision claims as especially uncertain, while still retaining limited distinctions among the most credible experts. These residual adjustments are accumulated cue by cue, preserving expert identity, but their total is divisively normalized and increasingly bounded once a clear tally coalition exists. Residual reliability is also contextually gated: it is strongly discounted in exact tally ties, where conflicting reliability claims provide no categorical direction, while otherwise retaining its ordinary residual weight without a special margin-one amplification. The resulting total evidence is passed through a saturating response transformation, so additional agreement increases confidence strongly at first but has diminishing effects once evidence is decisive. Between-person differences arise in reliability shrinkage, response sensitivity, and lapse rate. Because choices reveal no correctness information, the representation is not updated from choice history.",
  "predict": "def predict(parameters, state, history):\n    # Uncertain-Credibility Normalized Voting. History is intentionally ignored:\n    # without outcome feedback, previous choices cannot identify expert accuracy.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    n_features = stimulus.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match \"\n            f\"n_features {n_features}.\"\n        )\n\n    # A difference of +1 is an endorsement of A and -1 an endorsement of B.\n    # Nondiscriminating experts contribute neither a vote nor reliability evidence.\n    signed_endorsements = np.sign(stimulus[0] - stimulus[1])\n    discriminating = signed_endorsements != 0.0\n    tally_evidence = float(np.sum(signed_endorsements))\n\n    # Map validity onto above-chance diagnostic strength. The reference value\n    # zero corresponds to a common validity of .75. Lower-tail deviations are\n    # winsorized, while upper-tail deviations are smoothly compressed so that\n    # very credible experts remain distinguishable without receiving full weight.\n    v = np.clip(validities, 0.5, 1.0)\n    communicated_strength = 2.0 * (v - 0.5)\n    raw_deviation = communicated_strength - 0.5\n    lower_bounded = np.maximum(raw_deviation, -0.25)\n    upper_excess = np.maximum(lower_bounded - 0.25, 0.0)\n    reliability_deviation = (\n        np.minimum(lower_bounded, 0.25)\n        + 0.13 * np.tanh(upper_excess / 0.10)\n    )\n\n    # Strong shrinkage leaves only a small fraction of communicated deviations.\n    # Deviations are nevertheless summed expert by expert, unlike a summary of\n    # the two endorsing groups. Divisive normalization bounds their joint impact.\n    shrinkage = float(parameters[\"reliability_shrinkage\"])\n    retained_precision = 1.0 - shrinkage\n    active_deviations = reliability_deviation[discriminating]\n    active_signs = signed_endorsements[discriminating]\n\n    if active_deviations.size == 0:\n        reliability_adjustment = 0.0\n    else:\n        signed_sum = float(np.dot(active_signs, active_deviations))\n        deviation_mass = float(np.sum(np.abs(active_deviations)))\n        coalition_protection = 0.12 * max(abs(tally_evidence) - 1.0, 0.0) ** 2\n        reliability_adjustment = (\n            4.0 * retained_precision * signed_sum\n            / (1.0 + retained_precision * deviation_mass + coalition_protection)\n        )\n\n        # Reliability is least trusted when the categorical tally is exactly\n        # tied. Outside exact ties it receives its ordinary residual weight;\n        # there is no separate amplification at a one-vote margin.\n        absolute_tally = abs(tally_evidence)\n        tie_gate = 1.0 - 0.94 * np.exp(-((absolute_tally / 0.35) ** 2))\n        reliability_adjustment *= tie_gate\n\n    total_evidence_for_a = tally_evidence + reliability_adjustment\n\n    # Confidence is generated by one saturating transformation of total\n    # evidence, rather than by a separate consensus-amplification mechanism.\n    normalized_evidence = np.tanh(total_evidence_for_a / 4.0)\n    sensitivity = float(parameters[\"choice_sensitivity\"])\n    decision_variable = sensitivity * normalized_evidence\n\n    logits = np.array(\n        [0.5 * decision_variable, -0.5 * decision_variable],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    p_core = np.exp(logits)\n    p_core /= p_core.sum()\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probabilities = (1.0 - lapse) * p_core + lapse * 0.5\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= probabilities.sum()\n    return probabilities",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = float(np.sum(probabilities))\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.full(\n            probabilities.size, 1.0 / probabilities.size, dtype=np.float64\n        )\n    else:\n        probabilities /= total\n    return int(np.random.choice(probabilities.size, p=probabilities))",
  "parameters": {
    "reliability_shrinkage": "[0.58, 0.84]",
    "choice_sensitivity": "[1.90, 3.00]",
    "lapse_rate": "[0.0, 0.12]",
    "validities": "validities"
  },
  "rationale": "This is a single-line, minimal-diff extension of the accepted iteration-6 model: only the choice-sensitivity range is widened from [2.20, 2.70] to [1.90, 3.00], while preserving its midpoint at 2.45. The rejected delayed-quartic coalition protection is not retained; the accepted quadratic protection, cue-wise residual accumulation, reliability mapping, exact-tie gate, saturation scale, shrinkage range, and lapse range remain unchanged. Greater sensitivity heterogeneity directly targets the persistent underdispersion in Experiments 2, 4, and 6. Because the logistic response is nonlinear, mixing lower- and higher-sensitivity participants should also soften the slightly excessive majority-choice rates in Experiments 4 and 6 and move rare TTB or minority responses in Experiments 1 and 3 toward indifference, without using extra lapse noise that would indiscriminately erase tally-margin dependence. The edit therefore tests the critic's proposed heterogeneity correction in isolation rather than repeating previously rejected simultaneous changes to evidence representation and multiple parameter ranges."
}
```

## Usage

```json
{
  "input_tokens": 20028,
  "output_tokens": 1967
}
```
