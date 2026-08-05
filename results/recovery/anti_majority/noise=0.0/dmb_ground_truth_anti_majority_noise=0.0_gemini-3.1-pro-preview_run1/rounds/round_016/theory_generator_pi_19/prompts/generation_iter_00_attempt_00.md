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
- THEORY 1 = `pi_18`
- THEORY 2 = `pi_14`
- The recommendation below acts on THEORY 1 (= `pi_18`).

Propose a brand-new theory to replace THEORY 1. The new theory should build on the conceptual successes of THEORY 2 (which correctly identifies that subjects penalize variance/spread in unique features) but introduce a mechanism for 'Feature Coherence' or 'Density'. For instance, decision-makers might evaluate the density or clustering of active features, strongly preferring options where positive features are contiguous or similar in validity (low variance), and heavily discounting isolated high-validity features if they are disconnected from the rest of the option's profile. This would explain both the general spread-penalty effects seen across the benchmark and the specific preference for clustered features observed in Experiment 16.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_14` (overall score: 0.674)

**Description**
Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7675 (var=0.0218) vs this=0.7602 (var=0.0175)
- Experiment 2: real=0.2552 (var=0.0312) vs this=0.3014 (var=0.0318)
- Experiment 3: real=0.6317 (var=0.0095) vs this=0.7217 (var=0.0250)
- Experiment 4: real=0.2888 (var=0.0207) vs this=0.1997 (var=0.0245)
- Experiment 5: real=0.3309 (var=0.0110) vs this=0.2511 (var=0.0107)
- Experiment 6: real=-0.1823 (var=0.0238) vs this=0.1129 (var=0.0217)
- Experiment 7: real=0.8678 (var=0.0153) vs this=0.7411 (var=0.0548)
- Experiment 8: real=-0.1200 (var=0.0258) vs this=-0.0116 (var=0.0155)
- Experiment 9: real=0.1572 (var=0.0102) vs this=0.1542 (var=0.0224)
- Experiment 10: real=0.1454 (var=0.0162) vs this=0.1571 (var=0.0082)
- Experiment 11: real=0.7428 (var=0.0066) vs this=0.7394 (var=0.0075)
- Experiment 12: real=0.1758 (var=0.0096) vs this=0.1758 (var=0.0269)
- Experiment 13: real=0.3307 (var=0.0208) vs this=0.5510 (var=0.0046)
- Experiment 14: real=0.8456 (var=0.0113) vs this=0.1606 (var=0.0128)
- Experiment 15: real=0.2095 (var=0.0222) vs this=0.2295 (var=0.0326)
- Experiment 16: real=0.8400 (var=0.0141) vs this=0.7908 (var=0.0552)
- Experiment 17: real=0.0567 (var=0.0378) vs this=0.0600 (var=0.0527)
- Experiment 18: real=0.2232 (var=0.0305) vs this=0.1311 (var=0.0189)
- Experiment 19: real=0.0008 (var=0.0064) vs this=-0.0175 (var=0.0052)
- Experiment 20: real=0.0767 (var=0.0438) vs this=0.0183 (var=0.0434)
- Experiment 21: real=0.8175 (var=0.0099) vs this=0.8400 (var=0.0165)
- Experiment 22: real=0.1744 (var=0.0135) vs this=0.1892 (var=0.0363)
- Experiment 23: real=0.1375 (var=0.0066) vs this=0.2529 (var=0.0642)
- Experiment 24: real=0.8830 (var=0.0105) vs this=0.7363 (var=0.0482)
- Experiment 25: real=0.6378 (var=0.0198) vs this=0.7378 (var=0.0098)
- Experiment 26: real=0.8733 (var=0.0127) vs this=0.6850 (var=0.0631)
- Experiment 27: real=0.3052 (var=0.0260) vs this=0.1356 (var=0.0139)
- Experiment 28: real=0.8579 (var=0.0059) vs this=0.8568 (var=0.0117)
- Experiment 29: real=0.1675 (var=0.0112) vs this=0.2125 (var=0.0431)
- Experiment 30: real=0.8526 (var=0.0133) vs this=0.8189 (var=0.0197)
- Experiment 31: real=0.8500 (var=0.0094) vs this=0.8250 (var=0.0273)
- Experiment 32: real=0.7100 (var=0.0395) vs this=0.7025 (var=0.0632)
- Experiment 33: real=0.3067 (var=0.0344) vs this=0.1767 (var=0.0260)
- Experiment 34: real=0.1278 (var=0.0107) vs this=0.1867 (var=0.0206)


---

### `pi_16` (overall score: 0.583)

**Description**
Variance Aversion and Feature Consistency

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        base_score = np.sum(active_w)
        
        # Apply variance penalty for multiple unique features
        if len(active_w) > 1:
            # Use standard deviation scaled by the number of active features
            # so the penalty competes symmetrically with the additive base score.
            std_w = np.std(active_w)
            return base_score - lambda_var * std_w * len(active_w)
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_var: [0.0, 50.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7675 (var=0.0218) vs this=0.7569 (var=0.0178)
- Experiment 2: real=0.2552 (var=0.0312) vs this=0.2876 (var=0.0055)
- Experiment 3: real=0.6317 (var=0.0095) vs this=0.6648 (var=0.0097)
- Experiment 4: real=0.2888 (var=0.0207) vs this=0.1944 (var=0.0125)
- Experiment 5: real=0.3309 (var=0.0110) vs this=0.3234 (var=0.0060)
- Experiment 6: real=-0.1823 (var=0.0238) vs this=0.1379 (var=0.0413)
- Experiment 7: real=0.8678 (var=0.0153) vs this=0.8339 (var=0.0092)
- Experiment 8: real=-0.1200 (var=0.0258) vs this=0.0237 (var=0.0282)
- Experiment 9: real=0.1572 (var=0.0102) vs this=0.1142 (var=0.0051)
- Experiment 10: real=0.1454 (var=0.0162) vs this=0.1171 (var=0.0053)
- Experiment 11: real=0.7428 (var=0.0066) vs this=0.7206 (var=0.0050)
- Experiment 12: real=0.1758 (var=0.0096) vs this=0.1333 (var=0.0115)
- Experiment 13: real=0.3307 (var=0.0208) vs this=0.5820 (var=0.0018)
- Experiment 14: real=0.8456 (var=0.0113) vs this=0.1794 (var=0.0119)
- Experiment 15: real=0.2095 (var=0.0222) vs this=0.2221 (var=0.0345)
- Experiment 16: real=0.8400 (var=0.0141) vs this=0.8675 (var=0.0228)
- Experiment 17: real=0.0567 (var=0.0378) vs this=0.0767 (var=0.0438)
- Experiment 18: real=0.2232 (var=0.0305) vs this=0.1363 (var=0.0142)
- Experiment 19: real=0.0008 (var=0.0064) vs this=-0.0042 (var=0.0033)
- Experiment 20: real=0.0767 (var=0.0438) vs this=0.0233 (var=0.0378)
- Experiment 21: real=0.8175 (var=0.0099) vs this=0.8856 (var=0.0097)
- Experiment 22: real=0.1744 (var=0.0135) vs this=0.1323 (var=0.0212)
- Experiment 23: real=0.1375 (var=0.0066) vs this=0.2562 (var=0.0543)
- Experiment 24: real=0.8830 (var=0.0105) vs this=0.8052 (var=0.0356)
- Experiment 25: real=0.6378 (var=0.0198) vs this=0.7731 (var=0.0074)
- Experiment 26: real=0.8733 (var=0.0127) vs this=0.7625 (var=0.0536)
- Experiment 27: real=0.3052 (var=0.0260) vs this=0.1504 (var=0.0107)
- Experiment 28: real=0.8579 (var=0.0059) vs this=0.8789 (var=0.0160)
- Experiment 29: real=0.1675 (var=0.0112) vs this=0.8638 (var=0.0229)
- Experiment 30: real=0.8526 (var=0.0133) vs this=0.5932 (var=0.1372)
- Experiment 31: real=0.8500 (var=0.0094) vs this=0.8600 (var=0.0151)
- Experiment 32: real=0.7100 (var=0.0395) vs this=0.7396 (var=0.0358)
- Experiment 33: real=0.3067 (var=0.0344) vs this=0.1100 (var=0.0093)
- Experiment 34: real=0.1278 (var=0.0107) vs this=0.1433 (var=0.0126)


---

### `pi_15` (overall score: 0.559)

**Description**
Strict Diminishing Returns on Absolute Subjective Utility: Decision-makers evaluate options by first cancelling out shared features. They then assess the unique features relative to a subjective validity threshold, where features below the threshold act as negative evidence (penalties). To reflect limited attention capacity, these features are ranked by their absolute utility (magnitude of impact, whether positive or negative), and their contribution is heavily discounted by a strict diminishing returns multiplier based on their rank.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Rank active features by absolute thresholded utility descending
        order = np.argsort(np.abs(active_w))[::-1]
        sorted_w = active_w[order]
        
        # Apply rank-dependent scaling (delta <= 1 for strict diminishing returns)
        ranks = np.arange(len(sorted_w))
        discounted_w = sorted_w * (delta ** ranks)
        
        return np.sum(discounted_w)
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.1, 20.0]
- rho: [0.0, 2.0]
- delta: [0.0, 1.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7675 (var=0.0218) vs this=0.8325 (var=0.0114)
- Experiment 2: real=0.2552 (var=0.0312) vs this=0.1814 (var=0.0185)
- Experiment 3: real=0.6317 (var=0.0095) vs this=0.8160 (var=0.0217)
- Experiment 4: real=0.2888 (var=0.0207) vs this=0.1675 (var=0.0114)
- Experiment 5: real=0.3309 (var=0.0110) vs this=0.1517 (var=0.0057)
- Experiment 6: real=-0.1823 (var=0.0238) vs this=0.0040 (var=0.0069)
- Experiment 7: real=0.8678 (var=0.0153) vs this=0.8144 (var=0.0452)
- Experiment 8: real=-0.1200 (var=0.0258) vs this=-0.0394 (var=0.0044)
- Experiment 9: real=0.1572 (var=0.0102) vs this=0.2556 (var=0.0521)
- Experiment 10: real=0.1454 (var=0.0162) vs this=0.3267 (var=0.0686)
- Experiment 11: real=0.7428 (var=0.0066) vs this=0.5022 (var=0.0202)
- Experiment 12: real=0.1758 (var=0.0096) vs this=0.5194 (var=0.0414)
- Experiment 13: real=0.3307 (var=0.0208) vs this=0.7113 (var=0.0090)
- Experiment 14: real=0.8456 (var=0.0113) vs this=0.1947 (var=0.0111)
- Experiment 15: real=0.2095 (var=0.0222) vs this=0.2984 (var=0.0530)
- Experiment 16: real=0.8400 (var=0.0141) vs this=0.7258 (var=0.0626)
- Experiment 17: real=0.0567 (var=0.0378) vs this=0.0383 (var=0.0691)
- Experiment 18: real=0.2232 (var=0.0305) vs this=0.1289 (var=0.0057)
- Experiment 19: real=0.0008 (var=0.0064) vs this=-0.0083 (var=0.0048)
- Experiment 20: real=0.0767 (var=0.0438) vs this=0.0083 (var=0.0478)
- Experiment 21: real=0.8175 (var=0.0099) vs this=0.6937 (var=0.0612)
- Experiment 22: real=0.1744 (var=0.0135) vs this=0.3103 (var=0.0916)
- Experiment 23: real=0.1375 (var=0.0066) vs this=0.2083 (var=0.0464)
- Experiment 24: real=0.8830 (var=0.0105) vs this=0.7681 (var=0.0507)
- Experiment 25: real=0.6378 (var=0.0198) vs this=0.7358 (var=0.0049)
- Experiment 26: real=0.8733 (var=0.0127) vs this=0.8108 (var=0.0390)
- Experiment 27: real=0.3052 (var=0.0260) vs this=0.4274 (var=0.0828)
- Experiment 28: real=0.8579 (var=0.0059) vs this=0.5374 (var=0.0475)
- Experiment 29: real=0.1675 (var=0.0112) vs this=0.3613 (var=0.0574)
- Experiment 30: real=0.8526 (var=0.0133) vs this=0.5026 (var=0.0170)
- Experiment 31: real=0.8500 (var=0.0094) vs this=0.3375 (var=0.0328)
- Experiment 32: real=0.7100 (var=0.0395) vs this=0.1796 (var=0.0668)
- Experiment 33: real=0.3067 (var=0.0344) vs this=0.5283 (var=0.0559)
- Experiment 34: real=0.1278 (var=0.0107) vs this=0.4122 (var=0.0484)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    ttb_winner = np.zeros(len(data))
    
    for i in range(len(data)):
        winner = -1
        for j in range(5):
            if diff[i, j] > 0:
                winner = 0
                break
            elif diff[i, j] < 0:
                winner = 1
                break
        ttb_winner[i] = winner
        
    return float(np.mean(data['response'].values == ttb_winner))
```

**Observed (real) value:** 0.7675 (var=0.0218)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8335 (var=0.0163)
- pi_2: 0.1446 (var=0.0073)
- pi_3: 0.7538 (var=0.0288)
- pi_4: 0.5567 (var=0.0496)
- pi_5: 0.7315 (var=0.0170)
- pi_6: 0.7883 (var=0.0147)
- pi_7: 0.7963 (var=0.0147)
- pi_8: 0.7269 (var=0.0090)
- pi_9: 0.7027 (var=0.0111)
- pi_10: 0.6975 (var=0.0853)
- pi_11: 0.7431 (var=0.0078)
- pi_12: 0.7773 (var=0.0132)
- pi_13: 0.6219 (var=0.0295)
- pi_14: 0.7602 (var=0.0175)
- pi_15: 0.8325 (var=0.0114)
- pi_16: 0.7569 (var=0.0178)
- pi_17: 0.8217 (var=0.0245)
- pi_18: 0.7804 (var=0.0452)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    strict_mask = a_wins != b_wins
    if not np.any(strict_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = np.array(data['response'].tolist())
    
    match = (tally_preds[strict_mask] == responses[strict_mask])
    return float(np.mean(match))
```

**Observed (real) value:** 0.2552 (var=0.0312)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8726 (var=0.0061)
- pi_1: 0.1450 (var=0.0142)
- pi_3: 0.3233 (var=0.0094)
- pi_4: 0.4183 (var=0.0418)
- pi_5: 0.2874 (var=0.0114)
- pi_6: 0.2312 (var=0.0107)
- pi_7: 0.2562 (var=0.0079)
- pi_8: 0.3031 (var=0.0102)
- pi_9: 0.3010 (var=0.0098)
- pi_10: 0.3383 (var=0.0782)
- pi_11: 0.3002 (var=0.0074)
- pi_12: 0.2433 (var=0.0123)
- pi_13: 0.3755 (var=0.0301)
- pi_14: 0.3014 (var=0.0318)
- pi_15: 0.1814 (var=0.0185)
- pi_16: 0.2876 (var=0.0055)
- pi_17: 0.1814 (var=0.0227)
- pi_18: 0.1776 (var=0.0305)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_choice = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            matches.append(1 if resp == ttb_choice else 0)
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0095)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8517 (var=0.0107)
- pi_3: 0.7373 (var=0.0120)
- pi_2: 0.2046 (var=0.0068)
- pi_4: 0.5800 (var=0.0319)
- pi_5: 0.7281 (var=0.0104)
- pi_6: 0.7925 (var=0.0073)
- pi_7: 0.7623 (var=0.0122)
- pi_8: 0.6606 (var=0.0122)
- pi_9: 0.6531 (var=0.0082)
- pi_10: 0.6865 (var=0.0546)
- pi_11: 0.6800 (var=0.0148)
- pi_12: 0.7779 (var=0.0157)
- pi_13: 0.6554 (var=0.0374)
- pi_14: 0.7217 (var=0.0250)
- pi_15: 0.8160 (var=0.0217)
- pi_16: 0.6648 (var=0.0097)
- pi_17: 0.7854 (var=0.0384)
- pi_18: 0.7629 (var=0.0437)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Focus on diagnostic trials where one option has more positive features than the other.
    # In this specific design, these are exactly the trials where the single highest-validity
    # discriminating cue favors the option with FEWER total features.
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filt = sum_a[mask]
    sum_b_filt = sum_b[mask]
    
    # Determine which option has the greater number of positive features
    # 0 for A, 1 for B
    compensatory_choice = (sum_b_filt > sum_a_filt).astype(int)
    
    # Calculate the proportion of choices aligning with the compensatory (WADD-like) option
    match = (filtered_data['response'] == compensatory_choice).mean()
    return float(match)

```

**Observed (real) value:** 0.2888 (var=0.0207)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3234 (var=0.0244)
- pi_1: 0.1822 (var=0.0195)
- pi_2: 0.8612 (var=0.0089)
- pi_4: 0.4116 (var=0.0598)
- pi_5: 0.2784 (var=0.0094)
- pi_6: 0.2397 (var=0.0150)
- pi_7: 0.2288 (var=0.0145)
- pi_8: 0.2447 (var=0.0135)
- pi_9: 0.2772 (var=0.0096)
- pi_10: 0.2844 (var=0.0777)
- pi_11: 0.2238 (var=0.0163)
- pi_12: 0.2734 (var=0.0177)
- pi_13: 0.3828 (var=0.0341)
- pi_14: 0.1997 (var=0.0245)
- pi_15: 0.1675 (var=0.0114)
- pi_16: 0.1944 (var=0.0125)
- pi_17: 0.1706 (var=0.0244)
- pi_18: 0.2425 (var=0.0670)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # The validities are strictly decreasing from cue 0 to cue 3.
    # So the Take-The-Best (TTB) choice is simply determined by the first cue 
    # (from index 0 to 3) where the two options differ.
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(4):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0
        
    data['ttb_choice'] = data.apply(get_ttb_choice, axis=1)
    data['is_ttb'] = (data['response'] == data['ttb_choice']).astype(float)
    
    # Create a hashable string representation of the trial pair to group by
    data['trial_str'] = data.apply(lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])), axis=1)
    
    # Calculate the proportion of TTB-consistent choices for each unique trial type
    trial_means = data.groupby('trial_str')['is_ttb'].mean()
    
    # Return the standard deviation of these proportions across the 10 trial types
    return float(trial_means.std())
```

**Observed (real) value:** 0.3309 (var=0.0110)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1383 (var=0.0053)
- pi_3: 0.0645 (var=0.0053)
- pi_1: 0.0228 (var=0.0013)
- pi_2: 0.2610 (var=0.0036)
- pi_5: 0.0660 (var=0.0014)
- pi_6: 0.0459 (var=0.0022)
- pi_7: 0.1707 (var=0.0032)
- pi_8: 0.2154 (var=0.0127)
- pi_9: 0.3350 (var=0.0072)
- pi_10: 0.1249 (var=0.0190)
- pi_11: 0.3516 (var=0.0081)
- pi_12: 0.0636 (var=0.0028)
- pi_13: 0.0441 (var=0.0036)
- pi_14: 0.2511 (var=0.0107)
- pi_15: 0.1517 (var=0.0057)
- pi_16: 0.3234 (var=0.0060)
- pi_17: 0.0808 (var=0.0092)
- pi_18: 0.0668 (var=0.0105)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_cues'] = data['option_a_ratings'].apply(sum)
    data['b_cues'] = data['option_b_ratings'].apply(sum)
    
    def ttb_favors_a(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]: return True
            if b[i] > a[i]: return False
        return False

    data['ttb_a'] = data.apply(ttb_favors_a, axis=1)
    
    df_conflict = data[data['ttb_a']].copy()
    
    high_mask = df_conflict['b_cues'] >= 3
    low_mask = df_conflict['b_cues'] <= 2
    
    p_high = df_conflict.loc[high_mask, 'response'].mean()
    p_low = df_conflict.loc[low_mask, 'response'].mean()
    
    if pd.isna(p_high): p_high = 0.5
    if pd.isna(p_low): p_low = 0.5
    
    return float(p_high - p_low)
```

**Observed (real) value:** -0.1823 (var=0.0238)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0536 (var=0.0101)
- pi_4: 0.0481 (var=0.0115)
- pi_1: 0.0061 (var=0.0074)
- pi_2: 0.1070 (var=0.0081)
- pi_5: -0.0304 (var=0.0102)
- pi_6: -0.0290 (var=0.0102)
- pi_7: -0.0130 (var=0.0073)
- pi_8: 0.0471 (var=0.0153)
- pi_9: 0.0951 (var=0.0299)
- pi_10: 0.0050 (var=0.0111)
- pi_11: 0.1365 (var=0.0288)
- pi_12: -0.0521 (var=0.0082)
- pi_13: -0.0434 (var=0.0137)
- pi_14: 0.1129 (var=0.0217)
- pi_15: 0.0040 (var=0.0069)
- pi_16: 0.1379 (var=0.0413)
- pi_17: 0.0030 (var=0.0063)
- pi_18: -0.0184 (var=0.0107)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Option A's features are a strict subset of Option B's features
    is_subset = np.all(a_ratings <= b_ratings, axis=1) & (np.sum(a_ratings, axis=1) < np.sum(b_ratings, axis=1))
    
    subset_data = data[is_subset]
    if len(subset_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen
    return float(np.mean(subset_data['response'] == 0))
```

**Observed (real) value:** 0.8678 (var=0.0153)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7136 (var=0.0101)
- pi_3: 0.4372 (var=0.0121)
- pi_1: 0.1681 (var=0.0082)
- pi_2: 0.1347 (var=0.0095)
- pi_4: 0.1244 (var=0.0083)
- pi_6: 0.6236 (var=0.0290)
- pi_7: 0.7853 (var=0.0078)
- pi_8: 0.8183 (var=0.0062)
- pi_9: 0.8403 (var=0.0098)
- pi_10: 0.5033 (var=0.1115)
- pi_11: 0.4553 (var=0.0389)
- pi_12: 0.7244 (var=0.0134)
- pi_13: 0.5989 (var=0.0317)
- pi_14: 0.7411 (var=0.0548)
- pi_15: 0.8144 (var=0.0452)
- pi_16: 0.8339 (var=0.0092)
- pi_17: 0.7181 (var=0.0599)
- pi_18: 0.6678 (var=0.0840)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    p_b_3 = data.loc[b_sum == 3, 'response'].mean()
    p_b_2 = data.loc[b_sum == 2, 'response'].mean()
    return float(p_b_3 - p_b_2)
```

**Observed (real) value:** -0.1200 (var=0.0258)
**Other theories' values on this metric (for reference):**
- pi_3: -0.1687 (var=0.0254)
- pi_5: -0.0631 (var=0.0065)
- pi_1: 0.0034 (var=0.0056)
- pi_2: 0.0947 (var=0.0064)
- pi_4: 0.0206 (var=0.0090)
- pi_6: -0.0616 (var=0.0098)
- pi_7: -0.0331 (var=0.0051)
- pi_8: 0.0303 (var=0.0130)
- pi_9: 0.0600 (var=0.0183)
- pi_10: -0.0559 (var=0.0080)
- pi_11: 0.0881 (var=0.0173)
- pi_12: -0.0663 (var=0.0065)
- pi_13: -0.0622 (var=0.0179)
- pi_14: -0.0116 (var=0.0155)
- pi_15: -0.0394 (var=0.0044)
- pi_16: 0.0237 (var=0.0282)
- pi_17: -0.0025 (var=0.0065)
- pi_18: -0.0234 (var=0.0063)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    # Determine TTB choice (first discriminating cue)
    diff = A - B
    idx = np.argmax(np.abs(diff), axis=1)
    ttb_choice = np.where(diff[np.arange(len(diff)), idx] == 1, 0, 1)
    
    # Determine Averaging choice
    sum_a = np.sum(A, axis=1)
    sum_b = np.sum(B, axis=1)
    
    avg_a = np.zeros(len(A))
    mask_a = sum_a > 0
    avg_a[mask_a] = np.sum(A[mask_a] * val, axis=1) / sum_a[mask_a]
    
    avg_b = np.zeros(len(B))
    mask_b = sum_b > 0
    avg_b[mask_b] = np.sum(B[mask_b] * val, axis=1) / sum_b[mask_b]
    
    avg_choice = np.where(avg_a > avg_b, 0, np.where(avg_b > avg_a, 1, -1))
    
    # Filter trials where TTB and Averaging disagree
    disagree_mask = (avg_choice != -1) & (ttb_choice != avg_choice)
    
    if not np.any(disagree_mask):
        return 0.5
        
    responses = data['response'].values[disagree_mask]
    ttb_choices = ttb_choice[disagree_mask]
    
    return float(np.mean(responses == ttb_choices))

```

**Observed (real) value:** 0.1572 (var=0.0102)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5006 (var=0.0132)
- pi_6: 0.6317 (var=0.0193)
- pi_1: 0.8331 (var=0.0125)
- pi_2: 0.8672 (var=0.0102)
- pi_3: 0.8347 (var=0.0083)
- pi_4: 0.8706 (var=0.0075)
- pi_7: 0.2328 (var=0.0172)
- pi_8: 0.1875 (var=0.0125)
- pi_9: 0.1233 (var=0.0071)
- pi_10: 0.6258 (var=0.0826)
- pi_11: 0.1367 (var=0.0067)
- pi_12: 0.4917 (var=0.0161)
- pi_13: 0.5025 (var=0.0376)
- pi_14: 0.1542 (var=0.0224)
- pi_15: 0.2556 (var=0.0521)
- pi_16: 0.1142 (var=0.0051)
- pi_17: 0.4278 (var=0.1115)
- pi_18: 0.4892 (var=0.0835)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.1454 (var=0.0162)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5204 (var=0.0401)
- pi_5: 0.3127 (var=0.0095)
- pi_1: 0.8612 (var=0.0076)
- pi_2: 0.8573 (var=0.0120)
- pi_3: 0.8037 (var=0.0106)
- pi_4: 0.8221 (var=0.0105)
- pi_7: 0.1398 (var=0.0117)
- pi_8: 0.1600 (var=0.0111)
- pi_9: 0.1227 (var=0.0068)
- pi_10: 0.4888 (var=0.1252)
- pi_11: 0.1165 (var=0.0071)
- pi_12: 0.2948 (var=0.0132)
- pi_13: 0.4173 (var=0.0365)
- pi_14: 0.1571 (var=0.0082)
- pi_15: 0.3267 (var=0.0686)
- pi_16: 0.1171 (var=0.0053)
- pi_17: 0.2773 (var=0.0632)
- pi_18: 0.3444 (var=0.0878)

### Experiment 11
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has the highest-validity feature (index 0) and Option B does not.
    a_has_best = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_best = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    mask = a_has_best & ~b_has_best
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times the subject chose Option B (response == 1)
    # despite Option A having the best possible feature.
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.7428 (var=0.0066)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4372 (var=0.0109)
- pi_7: 0.6978 (var=0.0229)
- pi_1: 0.1544 (var=0.0086)
- pi_2: 0.3906 (var=0.0041)
- pi_3: 0.1889 (var=0.0123)
- pi_4: 0.2581 (var=0.0137)
- pi_6: 0.3039 (var=0.0201)
- pi_8: 0.7783 (var=0.0258)
- pi_9: 0.8689 (var=0.0100)
- pi_10: 0.4103 (var=0.0240)
- pi_11: 0.7500 (var=0.0037)
- pi_12: 0.4753 (var=0.0129)
- pi_13: 0.5272 (var=0.0357)
- pi_14: 0.7394 (var=0.0075)
- pi_15: 0.5022 (var=0.0202)
- pi_16: 0.7206 (var=0.0050)
- pi_17: 0.4983 (var=0.0590)
- pi_18: 0.4739 (var=0.0168)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    extreme_chosen = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        idx_a = a.index(1) if 1 in a else len(a)
        idx_b = b.index(1) if 1 in b else len(b)
        
        if idx_a < idx_b:
            if resp == 0:
                extreme_chosen += 1
            total += 1
        elif idx_b < idx_a:
            if resp == 1:
                extreme_chosen += 1
            total += 1
            
    return float(extreme_chosen / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1758 (var=0.0096)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2781 (var=0.0334)
- pi_5: 0.6356 (var=0.0073)
- pi_1: 0.8475 (var=0.0096)
- pi_2: 0.7135 (var=0.0043)
- pi_3: 0.8131 (var=0.0118)
- pi_4: 0.7931 (var=0.0106)
- pi_6: 0.7129 (var=0.0139)
- pi_8: 0.1379 (var=0.0131)
- pi_9: 0.1619 (var=0.0184)
- pi_10: 0.7131 (var=0.0298)
- pi_11: 0.1454 (var=0.0176)
- pi_12: 0.7104 (var=0.0169)
- pi_13: 0.5365 (var=0.0342)
- pi_14: 0.1758 (var=0.0269)
- pi_15: 0.5194 (var=0.0414)
- pi_16: 0.1333 (var=0.0115)
- pi_17: 0.4571 (var=0.0685)
- pi_18: 0.6500 (var=0.0383)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has the highest validity feature (index 0)
    # but also has at least one other feature, creating high variance.
    # Option B in the design always lacks the highest validity feature in these trials.
    is_target_trial = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) > 1)
    target_data = data[is_target_trial]
    
    if len(target_data) == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return (1.0 - target_data['response']).mean()
```

**Observed (real) value:** 0.3307 (var=0.0208)
**Other theories' values on this metric (for reference):**
- pi_8: 0.2930 (var=0.0358)
- pi_7: 0.4693 (var=0.0140)
- pi_1: 0.8307 (var=0.0127)
- pi_2: 0.4317 (var=0.0029)
- pi_3: 0.8063 (var=0.0222)
- pi_4: 0.7087 (var=0.0165)
- pi_5: 0.6553 (var=0.0116)
- pi_6: 0.7267 (var=0.0169)
- pi_9: 0.1910 (var=0.0142)
- pi_10: 0.6543 (var=0.0196)
- pi_11: 0.5660 (var=0.0031)
- pi_12: 0.6900 (var=0.0164)
- pi_13: 0.5777 (var=0.0231)
- pi_14: 0.5510 (var=0.0046)
- pi_15: 0.7113 (var=0.0090)
- pi_16: 0.5820 (var=0.0018)
- pi_17: 0.6480 (var=0.0144)
- pi_18: 0.6877 (var=0.0161)

### Experiment 14
**Design**
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Filter for trials where both options have the same number of positive features
    # In the design, these are trials 1-6. In all of these, Option A has a higher 
    # maximum validity (anchor) but exactly the same remaining features as Option B.
    # Therefore, Option B's features are more clustered (lower variance).
    is_matched = data['option_a_ratings'].apply(sum) == data['option_b_ratings'].apply(sum)
    matched_data = data[is_matched]
    if len(matched_data) == 0:
        return 0.0
    # Return the proportion of times Option B was chosen
    return float(matched_data['response'].mean())
```

**Observed (real) value:** 0.8456 (var=0.0113)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4275 (var=0.0161)
- pi_8: 0.7581 (var=0.0347)
- pi_1: 0.1561 (var=0.0079)
- pi_2: 0.5000 (var=0.0038)
- pi_3: 0.2294 (var=0.0118)
- pi_4: 0.2636 (var=0.0161)
- pi_5: 0.3556 (var=0.0111)
- pi_6: 0.2753 (var=0.0141)
- pi_9: 0.8089 (var=0.0259)
- pi_10: 0.1856 (var=0.0123)
- pi_11: 0.1847 (var=0.0131)
- pi_12: 0.3000 (var=0.0102)
- pi_13: 0.3742 (var=0.0325)
- pi_14: 0.1606 (var=0.0128)
- pi_15: 0.1947 (var=0.0111)
- pi_16: 0.1794 (var=0.0119)
- pi_17: 0.2036 (var=0.0204)
- pi_18: 0.1697 (var=0.0079)

### Experiment 15
**Design**
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (0, 1, 0, 0, 1) and b == (1, 0, 1, 1, 1):
            return 1
        if a == (0, 1, 1, 0, 1) and b == (0, 1, 0, 0, 1):
            return 3
        return 0
    
    trial_types = data.apply(is_target_trial, axis=1)
    
    t1_mask = (trial_types == 1)
    t3_mask = (trial_types == 3)
    
    p_b_t1 = data[t1_mask]['response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t3 = 1.0 - (data[t3_mask]['response'].mean() if t3_mask.sum() > 0 else 0.5)
    
    return float((p_b_t1 + p_a_t3) / 2.0)
```

**Observed (real) value:** 0.2095 (var=0.0222)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4800 (var=0.0203)
- pi_9: 0.2421 (var=0.0098)
- pi_1: 0.8705 (var=0.0109)
- pi_2: 0.8274 (var=0.0168)
- pi_3: 0.6916 (var=0.0103)
- pi_4: 0.8516 (var=0.0146)
- pi_5: 0.5042 (var=0.0076)
- pi_6: 0.5821 (var=0.0111)
- pi_7: 0.6626 (var=0.0195)
- pi_10: 0.5889 (var=0.1116)
- pi_11: 0.4658 (var=0.0058)
- pi_12: 0.4653 (var=0.0088)
- pi_13: 0.4842 (var=0.0528)
- pi_14: 0.2295 (var=0.0326)
- pi_15: 0.2984 (var=0.0530)
- pi_16: 0.2221 (var=0.0345)
- pi_17: 0.4189 (var=0.1029)
- pi_18: 0.4163 (var=0.1014)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    mask = (a_str == '10111101') & (b_str == '01000010')
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8400 (var=0.0141)
**Other theories' values on this metric (for reference):**
- pi_9: 0.8467 (var=0.0177)
- pi_8: 0.4325 (var=0.0853)
- pi_1: 0.1725 (var=0.0158)
- pi_2: 0.1258 (var=0.0100)
- pi_3: 0.1675 (var=0.0139)
- pi_4: 0.0783 (var=0.0074)
- pi_5: 0.5800 (var=0.0192)
- pi_6: 0.3817 (var=0.0249)
- pi_7: 0.2617 (var=0.0362)
- pi_10: 0.3842 (var=0.1181)
- pi_11: 0.8783 (var=0.0112)
- pi_12: 0.6325 (var=0.0189)
- pi_13: 0.5258 (var=0.0581)
- pi_14: 0.7908 (var=0.0552)
- pi_15: 0.7258 (var=0.0626)
- pi_16: 0.8675 (var=0.0228)
- pi_17: 0.6233 (var=0.1174)
- pi_18: 0.5708 (var=0.1384)

### Experiment 17
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Project lists to strings for hashable grouping
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # response=0 means A was chosen, response=1 means B was chosen.
    # So 1 - mean(response) is the empirical probability of choosing A.
    p_A = 1.0 - data.groupby(['A_str', 'B_str'])['response'].mean()
    
    def get_p(a, b):
        if (a, b) in p_A.index:
            return p_A.loc[(a, b)]
        return 0.5
        
    # Trial pairs that differ only by a shared feature added to both options
    # Advocated theory predicts 0 difference; Weakest Link predicts large shifts.
    
    # Trial 2 vs Trial 1 (Shared feature 1 added)
    d1 = get_p('11000', '10110') - get_p('01000', '00110')
    
    # Trial 3 vs Trial 1 (Shared feature 5 added)
    d2 = get_p('01001', '00111') - get_p('01000', '00110')
    
    # Trial 5 vs Trial 4 (Shared feature 1 added)
    d3 = get_p('11100', '10011') - get_p('01100', '00011')
    
    # Trial 8 vs Trial 7 (Shared feature 1 added)
    d4 = get_p('10100', '10010') - get_p('00100', '00010')
    
    return float(np.abs(d1) + np.abs(d2) + np.abs(d3) + np.abs(d4))

```

**Observed (real) value:** 0.0567 (var=0.0378)
**Other theories' values on this metric (for reference):**
- pi_10: 0.0767 (var=0.0611)
- pi_9: 1.2867 (var=0.2767)
- pi_1: 0.0517 (var=0.0604)
- pi_2: 0.0950 (var=0.0526)
- pi_3: 0.0700 (var=0.0438)
- pi_4: 0.0867 (var=0.0533)
- pi_5: 0.1917 (var=0.0581)
- pi_6: 0.1617 (var=0.0796)
- pi_7: 0.5433 (var=0.0683)
- pi_8: 0.6000 (var=0.1150)
- pi_11: 0.1117 (var=0.0651)
- pi_12: 0.2717 (var=0.0608)
- pi_13: 0.0533 (var=0.0561)
- pi_14: 0.0600 (var=0.0527)
- pi_15: 0.0383 (var=0.0691)
- pi_16: 0.0767 (var=0.0438)
- pi_17: 0.0233 (var=0.0549)
- pi_18: 0.0433 (var=0.0339)

### Experiment 18
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_a_single = data['option_a_ratings'].apply(sum) == 1
    is_b_single = data['option_b_ratings'].apply(sum) == 1
    
    target_trials = is_a_single | is_b_single
    if not target_trials.any():
        return 0.0
        
    df_target = data[target_trials]
    
    chose_multi = ((df_target['option_b_ratings'].apply(sum) == 1) & (df_target['response'] == 0)) | \
                  ((df_target['option_a_ratings'].apply(sum) == 1) & (df_target['response'] == 1))
                  
    return float(chose_multi.mean())

```

**Observed (real) value:** 0.2232 (var=0.0305)
**Other theories' values on this metric (for reference):**
- pi_9: 0.1295 (var=0.0109)
- pi_10: 0.4163 (var=0.1142)
- pi_1: 0.1532 (var=0.0111)
- pi_2: 0.8858 (var=0.0073)
- pi_3: 0.1879 (var=0.0273)
- pi_4: 0.4537 (var=0.0623)
- pi_5: 0.2168 (var=0.0176)
- pi_6: 0.1763 (var=0.0129)
- pi_7: 0.1568 (var=0.0122)
- pi_8: 0.1489 (var=0.0100)
- pi_11: 0.1016 (var=0.0080)
- pi_12: 0.1700 (var=0.0154)
- pi_13: 0.3821 (var=0.0369)
- pi_14: 0.1311 (var=0.0189)
- pi_15: 0.1289 (var=0.0057)
- pi_16: 0.1363 (var=0.0142)
- pi_17: 0.2326 (var=0.0460)
- pi_18: 0.2947 (var=0.0857)

### Experiment 19
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Response == 0 means Option A was chosen
    chose_a = (data['response'] == 0).astype(float)
    
    # Identify base trials (last feature is 0 for both options)
    # and derivative trials (last feature is 1 for both options)
    is_derivative = data['option_a_ratings'].apply(lambda x: x[-1] == 1) & data['option_b_ratings'].apply(lambda x: x[-1] == 1)
    is_base = data['option_a_ratings'].apply(lambda x: x[-1] == 0) & data['option_b_ratings'].apply(lambda x: x[-1] == 0)
    
    # Calculate the proportion of choosing Option A in both trial types
    p_a_base = chose_a[is_base].mean()
    p_a_deriv = chose_a[is_derivative].mean()
    
    # The metric is the difference in preference for A
    # Advocated theory: ~0 (shared feature is cancelled)
    # Competing theory: non-zero (shared weak feature increases weakest-link penalty asymmetrically)
    return float(p_a_base - p_a_deriv)
```

**Observed (real) value:** 0.0008 (var=0.0064)
**Other theories' values on this metric (for reference):**
- pi_11: -0.0037 (var=0.0080)
- pi_9: 0.4012 (var=0.0261)
- pi_1: -0.0100 (var=0.0051)
- pi_2: 0.0000 (var=0.0087)
- pi_3: -0.0129 (var=0.0057)
- pi_4: -0.0188 (var=0.0036)
- pi_5: 0.0538 (var=0.0093)
- pi_6: 0.0246 (var=0.0061)
- pi_7: 0.1233 (var=0.0123)
- pi_8: 0.4367 (var=0.0793)
- pi_10: -0.0033 (var=0.0050)
- pi_12: 0.0350 (var=0.0110)
- pi_13: 0.0296 (var=0.0093)
- pi_14: -0.0175 (var=0.0052)
- pi_15: -0.0083 (var=0.0048)
- pi_16: -0.0042 (var=0.0033)
- pi_17: 0.0050 (var=0.0062)
- pi_18: 0.0154 (var=0.0047)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials by their option_a_ratings string representation
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 0, 0]
    # Trial 2: A=[1, 0, 0, 0, 1]
    # Trial 5: A=[0, 0, 1, 0, 0]
    # Trial 6: A=[0, 0, 1, 0, 1]
    t1 = data[data['a_str'] == '10000']
    t2 = data[data['a_str'] == '10001']
    t5 = data[data['a_str'] == '00100']
    t6 = data[data['a_str'] == '00101']
    
    # Response is 0 for A, 1 for B, so (1 - mean(response)) is the proportion of A choices
    p_A_t1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p_A_t2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    p_A_t5 = 1.0 - t5['response'].mean() if len(t5) > 0 else 0.5
    p_A_t6 = 1.0 - t6['response'].mean() if len(t6) > 0 else 0.5
    
    # Advocated theory predicts P(A) drops from T1 to T2, and P(A) rises from T5 to T6.
    # Competing theory predicts no change in either pair since the shared features cancel out.
    return (p_A_t1 - p_A_t2) + (p_A_t6 - p_A_t5)
```

**Observed (real) value:** 0.0767 (var=0.0438)
**Other theories' values on this metric (for reference):**
- pi_9: 1.3000 (var=0.1831)
- pi_11: -0.0100 (var=0.0418)
- pi_1: -0.0350 (var=0.0447)
- pi_2: 0.0417 (var=0.0653)
- pi_3: -0.0150 (var=0.0532)
- pi_4: -0.0433 (var=0.0562)
- pi_5: 0.1517 (var=0.0721)
- pi_6: 0.0017 (var=0.0476)
- pi_7: 0.5200 (var=0.1085)
- pi_8: 0.9733 (var=0.2568)
- pi_10: 0.0167 (var=0.0492)
- pi_12: 0.1500 (var=0.0681)
- pi_13: -0.0450 (var=0.0823)
- pi_14: 0.0183 (var=0.0434)
- pi_15: 0.0083 (var=0.0478)
- pi_16: 0.0233 (var=0.0378)
- pi_17: 0.0300 (var=0.0691)
- pi_18: -0.0500 (var=0.0547)

### Experiment 21
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    t2_a, t2_b = (1, 0, 0, 1, 1), (0, 0, 1, 1, 0)
    t5_a, t5_b = (0, 1, 0, 1, 1), (0, 0, 1, 0, 1)
    
    b_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t2_a and b == t2_b:
            b_chosen.append(resp == 1)
        elif a == t2_b and b == t2_a:
            b_chosen.append(resp == 0)
        elif a == t5_a and b == t5_b:
            b_chosen.append(resp == 1)
        elif a == t5_b and b == t5_a:
            b_chosen.append(resp == 0)
            
    if not b_chosen:
        return 0.5
    return sum(b_chosen) / len(b_chosen)
```

**Observed (real) value:** 0.8175 (var=0.0099)
**Other theories' values on this metric (for reference):**
- pi_11: 0.8562 (var=0.0161)
- pi_12: 0.3369 (var=0.0193)
- pi_1: 0.1469 (var=0.0131)
- pi_2: 0.1406 (var=0.0130)
- pi_3: 0.2362 (var=0.0150)
- pi_4: 0.1356 (var=0.0136)
- pi_5: 0.3569 (var=0.0114)
- pi_6: 0.2800 (var=0.0136)
- pi_7: 0.4612 (var=0.0074)
- pi_8: 0.7137 (var=0.0397)
- pi_9: 0.8287 (var=0.0214)
- pi_10: 0.3175 (var=0.0679)
- pi_13: 0.4700 (var=0.0594)
- pi_14: 0.8400 (var=0.0165)
- pi_15: 0.6937 (var=0.0612)
- pi_16: 0.8856 (var=0.0097)
- pi_17: 0.5525 (var=0.0899)
- pi_18: 0.4906 (var=0.0646)

### Experiment 22
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Identify trials where A has feature 2, B has features 1 and 3
        # and they share at least one low-validity feature (features 4 or 5)
        return (a[0] == 0 and a[1] == 1 and a[2] == 0 and
                b[0] == 1 and b[1] == 0 and b[2] == 1 and
                sum(a[3:]) > 0)
                
    mask = data.apply(is_target, axis=1)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1744 (var=0.0135)
**Other theories' values on this metric (for reference):**
- pi_12: 0.6749 (var=0.0174)
- pi_11: 0.1313 (var=0.0077)
- pi_1: 0.8472 (var=0.0121)
- pi_2: 0.8467 (var=0.0156)
- pi_3: 0.8426 (var=0.0099)
- pi_4: 0.8554 (var=0.0165)
- pi_5: 0.6138 (var=0.0121)
- pi_6: 0.7221 (var=0.0151)
- pi_7: 0.7041 (var=0.0179)
- pi_8: 0.2369 (var=0.0339)
- pi_9: 0.1585 (var=0.0192)
- pi_10: 0.7015 (var=0.0757)
- pi_13: 0.5549 (var=0.0478)
- pi_14: 0.1892 (var=0.0363)
- pi_15: 0.3103 (var=0.0916)
- pi_16: 0.1323 (var=0.0212)
- pi_17: 0.5231 (var=0.1020)
- pi_18: 0.5810 (var=0.1133)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def is_target_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[4] == 0 and b[4] == 1 and a[:4] == b[:4]

    mask = data.apply(is_target_trial, axis=1)
    target_trials = data[mask]
    
    if len(target_trials) == 0:
        return 0.5
        
    return float(target_trials['response'].mean())
```

**Observed (real) value:** 0.1375 (var=0.0066)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6700 (var=0.0271)
- pi_13: 0.3858 (var=0.0318)
- pi_1: 0.8592 (var=0.0080)
- pi_2: 0.8492 (var=0.0097)
- pi_3: 0.5246 (var=0.0078)
- pi_4: 0.8396 (var=0.0158)
- pi_5: 0.2971 (var=0.0157)
- pi_6: 0.2696 (var=0.0131)
- pi_7: 0.1542 (var=0.0096)
- pi_8: 0.1771 (var=0.0196)
- pi_9: 0.1383 (var=0.0056)
- pi_10: 0.4338 (var=0.1169)
- pi_12: 0.2221 (var=0.0130)
- pi_14: 0.2529 (var=0.0642)
- pi_15: 0.2083 (var=0.0464)
- pi_16: 0.2562 (var=0.0543)
- pi_17: 0.2108 (var=0.0349)
- pi_18: 0.2767 (var=0.0455)

### Experiment 24
**Design**
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify trials where B is an empty option and A contains only low-validity features
    mask = (b_str == '00000') & (a_str.isin(['00001', '00010', '00011']))
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times the subject chose B (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8830 (var=0.0105)
**Other theories' values on this metric (for reference):**
- pi_13: 0.6067 (var=0.0509)
- pi_11: 0.4156 (var=0.0214)
- pi_1: 0.1407 (var=0.0121)
- pi_2: 0.1356 (var=0.0141)
- pi_3: 0.4156 (var=0.0202)
- pi_4: 0.1519 (var=0.0126)
- pi_5: 0.3837 (var=0.0191)
- pi_6: 0.3141 (var=0.0250)
- pi_7: 0.2763 (var=0.0237)
- pi_8: 0.3081 (var=0.0176)
- pi_9: 0.4022 (var=0.0293)
- pi_10: 0.5526 (var=0.1180)
- pi_12: 0.3704 (var=0.0337)
- pi_14: 0.7363 (var=0.0482)
- pi_15: 0.7681 (var=0.0507)
- pi_16: 0.8052 (var=0.0356)
- pi_17: 0.7911 (var=0.0517)
- pi_18: 0.6993 (var=0.0811)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_trials_a_chosen = 0
    target_trials_total = 0
    
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        
        unique_a = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if a == 1 and b == 0]
        unique_b = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if b == 1 and a == 0]
        
        if len(unique_a) == 1:
            if len(unique_b) == 0 or unique_a[0] < min(unique_b):
                target_trials_total += 1
                if row['response'] == 0:
                    target_trials_a_chosen += 1
                    
    if target_trials_total == 0:
        return 0.5
    return target_trials_a_chosen / target_trials_total
```

**Observed (real) value:** 0.6378 (var=0.0198)
**Other theories' values on this metric (for reference):**
- pi_11: 0.8253 (var=0.0077)
- pi_14: 0.7378 (var=0.0098)
- pi_1: 0.8647 (var=0.0089)
- pi_2: 0.2606 (var=0.0040)
- pi_3: 0.6872 (var=0.0128)
- pi_4: 0.5839 (var=0.0391)
- pi_5: 0.7419 (var=0.0130)
- pi_6: 0.7669 (var=0.0118)
- pi_7: 0.7950 (var=0.0116)
- pi_8: 0.8008 (var=0.0128)
- pi_9: 0.8097 (var=0.0077)
- pi_10: 0.6236 (var=0.0321)
- pi_12: 0.8033 (var=0.0103)
- pi_13: 0.5864 (var=0.0159)
- pi_15: 0.7358 (var=0.0049)
- pi_16: 0.7731 (var=0.0074)
- pi_17: 0.7256 (var=0.0091)
- pi_18: 0.6997 (var=0.0207)

### Experiment 26
**Design**
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = ((data['A_str'] == '00000') & (data['B_str'] == '00001')) | \
           ((data['A_str'] == '10000') & (data['B_str'] == '10001'))
    
    if mask.sum() == 0:
        return 0.0
    
    return float((data.loc[mask, 'response'] == 0).mean())

```

**Observed (real) value:** 0.8733 (var=0.0127)
**Other theories' values on this metric (for reference):**
- pi_14: 0.6850 (var=0.0631)
- pi_11: 0.3067 (var=0.0370)
- pi_1: 0.1175 (var=0.0129)
- pi_2: 0.1317 (var=0.0091)
- pi_3: 0.4492 (var=0.0203)
- pi_4: 0.1767 (var=0.0141)
- pi_5: 0.5867 (var=0.0140)
- pi_6: 0.5458 (var=0.0214)
- pi_7: 0.5800 (var=0.0160)
- pi_8: 0.5792 (var=0.0104)
- pi_9: 0.6108 (var=0.0103)
- pi_10: 0.6525 (var=0.0900)
- pi_12: 0.5833 (var=0.0140)
- pi_13: 0.6333 (var=0.0506)
- pi_15: 0.8108 (var=0.0390)
- pi_16: 0.7625 (var=0.0536)
- pi_17: 0.6992 (var=0.0852)
- pi_18: 0.7117 (var=0.0729)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
    mask_4 = (a_keys == (0, 1, 1, 0, 0, 0)) & (b_keys == (1, 0, 0, 0, 1, 0))
    # Trial 5: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
    mask_5 = (a_keys == (1, 0, 0, 0, 1, 0)) & (b_keys == (0, 1, 1, 0, 0, 0))
    # Trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
    mask_8 = (a_keys == (1, 0, 0, 0, 0, 1)) & (b_keys == (0, 1, 1, 0, 0, 0))
    
    choices_high_max = 0
    total = 0
    
    if mask_4.any():
        choices_high_max += (data.loc[mask_4, 'response'] == 1).sum()
        total += mask_4.sum()
        
    if mask_5.any():
        choices_high_max += (data.loc[mask_5, 'response'] == 0).sum()
        total += mask_5.sum()
        
    if mask_8.any():
        choices_high_max += (data.loc[mask_8, 'response'] == 0).sum()
        total += mask_8.sum()
        
    if total == 0:
        return 0.5
        
    return float(choices_high_max / total)
```

**Observed (real) value:** 0.3052 (var=0.0260)
**Other theories' values on this metric (for reference):**
- pi_15: 0.4274 (var=0.0828)
- pi_14: 0.1356 (var=0.0139)
- pi_1: 0.8563 (var=0.0154)
- pi_2: 0.5259 (var=0.0105)
- pi_3: 0.7770 (var=0.0483)
- pi_4: 0.6615 (var=0.0304)
- pi_5: 0.4400 (var=0.0146)
- pi_6: 0.6170 (var=0.0308)
- pi_7: 0.1452 (var=0.0157)
- pi_8: 0.1622 (var=0.0134)
- pi_9: 0.1267 (var=0.0097)
- pi_10: 0.2178 (var=0.0270)
- pi_11: 0.1444 (var=0.0113)
- pi_12: 0.4237 (var=0.0182)
- pi_13: 0.4222 (var=0.0276)
- pi_16: 0.1504 (var=0.0107)
- pi_17: 0.2326 (var=0.0480)
- pi_18: 0.3237 (var=0.0482)

### Experiment 28
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has a large spread of validities (features 1 and 5)
    mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 1])
    if mask.sum() == 0:
        return 0.0
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.8579 (var=0.0059)
**Other theories' values on this metric (for reference):**
- pi_14: 0.8568 (var=0.0117)
- pi_15: 0.5374 (var=0.0475)
- pi_1: 0.1595 (var=0.0103)
- pi_2: 0.3342 (var=0.0067)
- pi_3: 0.1779 (var=0.0156)
- pi_4: 0.2400 (var=0.0165)
- pi_5: 0.4653 (var=0.0216)
- pi_6: 0.3505 (var=0.0221)
- pi_7: 0.7879 (var=0.0280)
- pi_8: 0.8342 (var=0.0147)
- pi_9: 0.8621 (var=0.0102)
- pi_10: 0.3826 (var=0.0390)
- pi_11: 0.8684 (var=0.0096)
- pi_12: 0.4574 (var=0.0399)
- pi_13: 0.5737 (var=0.0452)
- pi_16: 0.8789 (var=0.0160)
- pi_17: 0.6211 (var=0.0754)
- pi_18: 0.4005 (var=0.0460)

### Experiment 29
**Design**
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify Trial 1 where Option A and Option B have the exact same range of validities
    # but different standard deviations.
    # Trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0], B=[0, 1, 1, 0, 0, 0, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 1, 1, 0, 0)
    
    # Calculate the proportion of times Option A was chosen in Trial 1
    trial_1_data = data[is_trial_1]
    if len(trial_1_data) == 0:
        return 0.5
        
    return float((trial_1_data['response'] == 0).mean())
```

**Observed (real) value:** 0.1675 (var=0.0112)
**Other theories' values on this metric (for reference):**
- pi_16: 0.8638 (var=0.0229)
- pi_14: 0.2125 (var=0.0431)
- pi_1: 0.8363 (var=0.0123)
- pi_2: 0.4775 (var=0.0153)
- pi_3: 0.1688 (var=0.0144)
- pi_4: 0.7338 (var=0.0228)
- pi_5: 0.3787 (var=0.0157)
- pi_6: 0.6050 (var=0.0357)
- pi_7: 0.2750 (var=0.0308)
- pi_8: 0.8100 (var=0.0300)
- pi_9: 0.2587 (var=0.0184)
- pi_10: 0.2213 (var=0.0371)
- pi_11: 0.2800 (var=0.0374)
- pi_12: 0.3787 (var=0.0283)
- pi_13: 0.5150 (var=0.0337)
- pi_15: 0.3613 (var=0.0574)
- pi_17: 0.3900 (var=0.0949)
- pi_18: 0.2250 (var=0.0397)

### Experiment 30
**Design**
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t2_a = (1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0)
    t2_b = (0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0)
    
    t4_a = (1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0)
    t4_b = (0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0)
    
    mask = ((a_keys == t2_a) & (b_keys == t2_b)) | ((a_keys == t4_a) & (b_keys == t4_b))
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.8526 (var=0.0133)
**Other theories' values on this metric (for reference):**
- pi_14: 0.8189 (var=0.0197)
- pi_16: 0.5932 (var=0.1372)
- pi_1: 0.1511 (var=0.0101)
- pi_2: 0.4826 (var=0.0062)
- pi_3: 0.2574 (var=0.0338)
- pi_4: 0.3216 (var=0.0197)
- pi_5: 0.5005 (var=0.0058)
- pi_6: 0.3542 (var=0.0201)
- pi_7: 0.5816 (var=0.0076)
- pi_8: 0.6005 (var=0.0377)
- pi_9: 0.6884 (var=0.0054)
- pi_10: 0.5826 (var=0.0263)
- pi_11: 0.8411 (var=0.0104)
- pi_12: 0.5021 (var=0.0059)
- pi_13: 0.5205 (var=0.0235)
- pi_15: 0.5026 (var=0.0170)
- pi_17: 0.5316 (var=0.0414)
- pi_18: 0.5689 (var=0.0109)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5 where A=[0, 1, 0, 0, 1] and B=[0, 0, 1, 1, 0]
    a_seq = data['option_a_ratings'].apply(tuple)
    t5_mask = (a_seq == (0, 1, 0, 0, 1))
    
    # Response is 1 if B is chosen, 0 if A is chosen.
    # We return the proportion of times B is chosen in this trial.
    return float(data.loc[t5_mask, 'response'].mean())
```

**Observed (real) value:** 0.8500 (var=0.0094)
**Other theories' values on this metric (for reference):**
- pi_17: 0.4113 (var=0.0724)
- pi_14: 0.8250 (var=0.0273)
- pi_1: 0.1575 (var=0.0204)
- pi_2: 0.5038 (var=0.0101)
- pi_3: 0.3325 (var=0.0365)
- pi_4: 0.2875 (var=0.0336)
- pi_5: 0.3962 (var=0.0145)
- pi_6: 0.3287 (var=0.0272)
- pi_7: 0.6438 (var=0.0260)
- pi_8: 0.7175 (var=0.0483)
- pi_9: 0.8387 (var=0.0139)
- pi_10: 0.2712 (var=0.0326)
- pi_11: 0.7900 (var=0.0278)
- pi_12: 0.4125 (var=0.0173)
- pi_13: 0.5050 (var=0.0272)
- pi_15: 0.3375 (var=0.0328)
- pi_16: 0.8600 (var=0.0151)
- pi_18: 0.2787 (var=0.0578)

### Experiment 32
**Design**
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert option A ratings to string to uniquely identify trial types
    a_str = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    chose_A = (data['response'] == 0).astype(float)
    df = pd.DataFrame({'A_str': a_str, 'chose_A': chose_A})
    
    # Get mean P(Choose A) for each trial type
    p_A = df.groupby('A_str')['chose_A'].mean().to_dict()
    
    # Trials where Advocated Theory predicts high P(A) (no spread penalty for A, high for B)
    # and Competing Theory predicts low P(A) (A has fewer features, B's sum of features dominates)
    p_T5 = p_A.get("10000", 0.5)
    p_T6 = p_A.get("10001", 0.5)
    
    # Trials where Advocated Theory predicts low P(A) (high spread penalty for A, low for B)
    # and Competing Theory predicts high P(A) (A's top feature is stronger, dominating due to diminishing marginal utility)
    p_T1 = p_A.get("01001", 0.5)
    p_T2 = p_A.get("11001", 0.5)
    p_T3 = p_A.get("10010", 0.5)
    p_T4 = p_A.get("10011", 0.5)
    
    # Contrast the two sets of trials
    high_adv = (p_T5 + p_T6) / 2.0
    low_adv = (p_T1 + p_T2 + p_T3 + p_T4) / 4.0
    
    return float(high_adv - low_adv)
```

**Observed (real) value:** 0.7100 (var=0.0395)
**Other theories' values on this metric (for reference):**
- pi_14: 0.7025 (var=0.0632)
- pi_17: 0.1375 (var=0.1269)
- pi_1: -0.0183 (var=0.0089)
- pi_2: -0.3254 (var=0.0243)
- pi_3: 0.0017 (var=0.0528)
- pi_4: -0.1342 (var=0.0275)
- pi_5: 0.1396 (var=0.0163)
- pi_6: 0.1146 (var=0.0146)
- pi_7: 0.2137 (var=0.0145)
- pi_8: 0.2775 (var=0.0139)
- pi_9: 0.3421 (var=0.0234)
- pi_10: -0.2012 (var=0.1098)
- pi_11: 0.7146 (var=0.0459)
- pi_12: 0.1192 (var=0.0142)
- pi_13: 0.1096 (var=0.0741)
- pi_15: 0.1796 (var=0.0668)
- pi_16: 0.7396 (var=0.0358)
- pi_18: -0.0242 (var=0.0518)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert rating lists to string representations for easy comparison
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # We track how often the option with ratings [1, 0, 0, 0, 1] is chosen
    is_A_10001 = (data['A_str'] == '10001')
    is_B_10001 = (data['B_str'] == '10001')
    
    # Only consider trials where the *other* option is either [0, 1, 0, 0, 0] or [0, 1, 1, 0, 0]
    # These correspond to Trial 1 and Trial 7 from the design.
    valid_other = ['01000', '01100']
    
    mask_A = is_A_10001 & data['B_str'].isin(valid_other)
    mask_B = is_B_10001 & data['A_str'].isin(valid_other)
    
    choices_10001 = 0
    total_trials = 0
    
    # If A is [1, 0, 0, 0, 1], response == 0 means it was chosen
    choices_10001 += (data.loc[mask_A, 'response'] == 0).sum()
    total_trials += mask_A.sum()
    
    # If B is [1, 0, 0, 0, 1], response == 1 means it was chosen
    choices_10001 += (data.loc[mask_B, 'response'] == 1).sum()
    total_trials += mask_B.sum()
    
    if total_trials == 0:
        return 0.5
        
    return float(choices_10001 / total_trials)
```

**Observed (real) value:** 0.3067 (var=0.0344)
**Other theories' values on this metric (for reference):**
- pi_18: 0.6033 (var=0.0598)
- pi_14: 0.1767 (var=0.0260)
- pi_1: 0.8517 (var=0.0127)
- pi_2: 0.6708 (var=0.0086)
- pi_3: 0.8158 (var=0.0173)
- pi_4: 0.7383 (var=0.0181)
- pi_5: 0.5233 (var=0.0204)
- pi_6: 0.6800 (var=0.0236)
- pi_7: 0.2558 (var=0.0359)
- pi_8: 0.1408 (var=0.0110)
- pi_9: 0.1400 (var=0.0110)
- pi_10: 0.5717 (var=0.0339)
- pi_11: 0.1200 (var=0.0112)
- pi_12: 0.5725 (var=0.0266)
- pi_13: 0.4550 (var=0.0586)
- pi_15: 0.5283 (var=0.0559)
- pi_16: 0.1100 (var=0.0093)
- pi_17: 0.4142 (var=0.0632)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    mask1 = (a_str == '10001') & (b_str == '01000')
    mask4 = (a_str == '10001') & (b_str == '01100')
    mask7 = (a_str == '10011') & (b_str == '01100')
    
    target_mask = mask1 | mask4 | mask7
    if not target_mask.any():
        return 0.5
        
    return float((data.loc[target_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1278 (var=0.0107)
**Other theories' values on this metric (for reference):**
- pi_14: 0.1867 (var=0.0206)
- pi_18: 0.5894 (var=0.0449)
- pi_1: 0.8317 (var=0.0111)
- pi_2: 0.7472 (var=0.0074)
- pi_3: 0.8483 (var=0.0095)
- pi_4: 0.8050 (var=0.0146)
- pi_5: 0.5322 (var=0.0142)
- pi_6: 0.6883 (var=0.0198)
- pi_7: 0.2589 (var=0.0364)
- pi_8: 0.1528 (var=0.0129)
- pi_9: 0.1167 (var=0.0100)
- pi_10: 0.5289 (var=0.0529)
- pi_11: 0.1511 (var=0.0106)
- pi_12: 0.4972 (var=0.0290)
- pi_13: 0.4517 (var=0.0359)
- pi_15: 0.4122 (var=0.0484)
- pi_16: 0.1433 (var=0.0126)
- pi_17: 0.4106 (var=0.0669)


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
  "theory": "Feature Coherence and Spread Penalty: Decision-makers evaluate options by first cancelling out shared features to simplify the choice. They then assess the remaining unique features relative to a subjective validity threshold. Crucially, they evaluate the 'coherence' or 'density' of the option's full profile: isolated high-validity features that are disconnected from other active features suffer a heavy isolation penalty. After adjusting for isolation, options with multiple unique features suffer a conflict penalty proportional to their spread (max - min) in adjusted validity. This mechanism strongly prefers options with contiguous, clustered features over those with sparse, disconnected profiles, while still penalizing variance among the unique features.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Model expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    rho = float(parameters[\"rho\"])\n    lambda_iso = float(parameters[\"lambda_iso\"])\n    lambda_penalty = float(parameters[\"lambda_penalty\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Transform validities\n    w = (val ** gamma) - rho\n    \n    # Identify shared and unique features\n    shared = (a > 0) & (b > 0)\n    unique_a = np.where((a > 0) & ~shared)[0]\n    unique_b = np.where((b > 0) & ~shared)[0]\n    \n    # Identify all active features for coherence calculation\n    active_a = np.where(a > 0)[0]\n    active_b = np.where(b > 0)[0]\n    \n    def get_score(unique_idx, active_idx):\n        if len(unique_idx) == 0:\n            return 0.0\n            \n        scores = []\n        for i in unique_idx:\n            # Calculate isolation: distance to the nearest other active feature in the full profile\n            if len(active_idx) > 1:\n                iso = min(abs(i - j) for j in active_idx if j != i)\n            else:\n                iso = 1\n            \n            # Apply isolation penalty (iso - 1 means contiguous features have 0 penalty)\n            w_i = w[i] - lambda_iso * (iso - 1)\n            scores.append(w_i)\n            \n        scores = np.array(scores)\n        base_score = np.sum(scores)\n        \n        # Apply variance/spread penalty on the adjusted unique features\n        if len(scores) > 1:\n            conflict = lambda_penalty * (np.max(scores) - np.min(scores))\n            return base_score - conflict\n            \n        return base_score\n        \n    score_a = get_score(unique_a, active_a)\n    score_b = get_score(unique_b, active_b)\n    \n    scores = np.array([score_a, score_b])\n    \n    # Softmax choice rule\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "gamma": "[0.1, 10.0]",
    "rho": "[0.0, 1.0]",
    "lambda_iso": "[0.0, 5.0]",
    "lambda_penalty": "[0.0, 10.0]",
    "beta": "[0.1, 50.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "This theory directly instantiates the arbiter's suggestion to incorporate 'Feature Coherence' or 'Density' while preserving the successful spread-penalty mechanism of pi_14. It operates by evaluating the isolation of each unique feature relative to the option's full profile. Isolation is computed as the index distance to the nearest other active feature. Features that are contiguous (distance 1) receive no penalty, whereas isolated features (distance > 1) suffer a linear penalty proportional to their isolation. This elegantly captures the strong preference for clustered features seen in Experiment 16 and reverses the failure in Experiment 14 (where the previous model failed to penalize a gap in an otherwise strong profile). After adjusting for isolation, the model applies the standard variance/spread penalty to the unique features, ensuring it maintains the broad predictive power of pi_14 across the rest of the benchmark."
}
```
```

## Usage

```json
{
  "prompt_token_count": 44497,
  "candidates_token_count": 1280,
  "total_token_count": 59836
}
```
