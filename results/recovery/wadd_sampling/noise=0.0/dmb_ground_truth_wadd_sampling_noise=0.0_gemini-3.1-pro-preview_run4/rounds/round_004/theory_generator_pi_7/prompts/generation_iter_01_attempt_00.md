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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a 'Probabilistic Take-The-Best' or 'Cue-Skipping Lexicographic' theory. Instead of assigning static weights based on rank (as in Theory 2) or applying a threshold for tallying (as in Theory 1), this theory would posit that decision-makers inspect cues sequentially in order of validity. However, the execution is probabilistic: each cue has a probability of being successfully processed or 'trusted' (perhaps proportional to its validity or a free parameter). If a cue discriminates and is trusted, it determines the choice; if skipped or non-discriminating, the decision-maker moves to the next cue. If all cues are exhausted, they guess. This introduces a natural, validity-dependent noise mechanism that could better capture the variance and specific deviations from strict lexicographic behavior without reverting to tallying.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.865)

**Description**
Validity Threshold Tallying with Sub-Threshold Influence: Decision-makers simplify the choice environment by setting a subjective threshold on cue validity. Cues with a validity above the threshold are considered 'important' and are tallied with unit weight. To account for weak compensatory behavior and tie-breaking, cues below the threshold are not completely ignored but receive a small, uniform sub-threshold weight.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity Threshold Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    alpha = float(parameters["alpha"])
    
    # Cues with validity >= threshold are given unit weight, others receive alpha weight
    w = np.where(val >= threshold, 1.0, alpha)
    
    # Compute tallied scores for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 1.0]
- alpha: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2988 (var=0.0249) vs this=0.2848 (var=0.0277)
- Experiment 2: real=0.7037 (var=0.0287) vs this=0.7700 (var=0.0339)
- Experiment 3: real=0.6050 (var=0.0828) vs this=0.5392 (var=0.0239)
- Experiment 4: real=0.3713 (var=0.0726) vs this=0.5331 (var=0.1027)
- Experiment 5: real=0.4379 (var=0.0545) vs this=0.4247 (var=0.0259)
- Experiment 6: real=0.4083 (var=0.0157) vs this=0.2904 (var=0.0223)
- Experiment 7: real=0.5012 (var=0.0602) vs this=0.5319 (var=0.0343)
- Experiment 8: real=-0.0363 (var=0.0444) vs this=0.0100 (var=0.0143)
- Experiment 9: real=0.6250 (var=0.0810) vs this=0.7058 (var=0.0697)
- Experiment 10: real=0.3458 (var=0.0760) vs this=0.4167 (var=0.0360)


---

### `pi_4` (overall score: 0.792)

**Description**
Decision-makers assign importance to features based on their ordinal rank in validity rather than their exact cardinal values. This Rank-Based Weighting heuristic avoids the extreme sensitivity to numerical validity differences seen in purely compensatory models, while still acknowledging that some cues are more diagnostic than others. Feature weights are computed as a power transformation of their inverse rank (e.g., 1 / rank^gamma). By restricting gamma to lower values, the model maintains a strong compensatory nature, ensuring that multiple lower-ranked cues can outweigh a single higher-ranked cue. Combined with a lower softmax temperature upper bound, it prevents overly deterministic choices and captures the noisier human behavior in conflicting trade-offs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Based Weighting expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Determine the ordinal rank of each feature's validity (1 = highest validity)
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Weights are a power transformation of the inverse rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.6]
- gamma: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2988 (var=0.0249) vs this=0.3685 (var=0.0144)
- Experiment 2: real=0.7037 (var=0.0287) vs this=0.6711 (var=0.0160)
- Experiment 3: real=0.6050 (var=0.0828) vs this=0.6425 (var=0.0201)
- Experiment 4: real=0.3713 (var=0.0726) vs this=0.5062 (var=0.0350)
- Experiment 5: real=0.4379 (var=0.0545) vs this=0.5574 (var=0.0120)
- Experiment 6: real=0.4083 (var=0.0157) vs this=0.4867 (var=0.0072)
- Experiment 7: real=0.5012 (var=0.0602) vs this=0.4475 (var=0.0036)
- Experiment 8: real=-0.0363 (var=0.0444) vs this=0.0737 (var=0.0203)
- Experiment 9: real=0.6250 (var=0.0810) vs this=0.4958 (var=0.0443)
- Experiment 10: real=0.3458 (var=0.0760) vs this=0.5787 (var=0.0109)


---

### `pi_5` (overall score: 0.720)

**Description**
Strategy Selection (WADD and TTB Mixture) with Compensatory Bias

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Selection expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    
    # Strategy 1: Compensatory WADD
    w_wadd = val ** gamma
    score_a = np.sum(w_wadd * a)
    score_b = np.sum(w_wadd * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Strategy 2: Non-compensatory Take-The-Best (TTB)
    order = np.argsort(-val, kind="stable")
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Mixture of strategies
    p_core = alpha * p_wadd + (1.0 - alpha) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- alpha: [0.5, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2988 (var=0.0249) vs this=0.3904 (var=0.0104)
- Experiment 2: real=0.7037 (var=0.0287) vs this=0.6795 (var=0.0180)
- Experiment 3: real=0.6050 (var=0.0828) vs this=0.6867 (var=0.0154)
- Experiment 4: real=0.3713 (var=0.0726) vs this=0.3281 (var=0.0599)
- Experiment 5: real=0.4379 (var=0.0545) vs this=0.3811 (var=0.0209)
- Experiment 6: real=0.4083 (var=0.0157) vs this=0.3162 (var=0.0160)
- Experiment 7: real=0.5012 (var=0.0602) vs this=0.7454 (var=0.0130)
- Experiment 8: real=-0.0363 (var=0.0444) vs this=0.2006 (var=0.0213)
- Experiment 9: real=0.6250 (var=0.0810) vs this=0.6475 (var=0.0256)
- Experiment 10: real=0.3458 (var=0.0760) vs this=0.4371 (var=0.0192)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.6620 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.6620 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # The cues are ordered by validity (descending) from left to right.
    # Find the first cue where the two options differ.
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    
    # Get the difference value at that first discriminating cue.
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Calculate the proportion of responses matching the TTB prediction
    matches = (data['response'].values == ttb_pred)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2988 (var=0.0249)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8233 (var=0.0261) (Δ vs real +0.5246)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8744 (var=0.0084)
- pi_2: 0.1846 (var=0.0061)
- pi_3: 0.3183 (var=0.0105)
- pi_4: 0.3685 (var=0.0144)
- pi_5: 0.3904 (var=0.0104)
- pi_6: 0.2848 (var=0.0277)

### Experiment 2
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Extract options matrices
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying predictions: count wins for each option
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    tally_preds = np.zeros(len(data)) - 1
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Take The Best (TTB) predictions: first discriminating cue
    ttb_preds = np.zeros(len(data)) - 1
    for i in range(len(data)):
        for j in range(a_mat.shape[1]):
            if a_mat[i, j] > b_mat[i, j]:
                ttb_preds[i] = 0
                break
            elif b_mat[i, j] > a_mat[i, j]:
                ttb_preds[i] = 1
                break
                
    # Identify trials where Tallying and TTB make strict, opposing predictions
    valid = (tally_preds != -1) & (ttb_preds != -1) & (tally_preds != ttb_preds)
    
    if not np.any(valid):
        return 0.5
        
    # Calculate the proportion of responses that align with Tallying
    responses = data['response'].values[valid]
    tally_preds_valid = tally_preds[valid]
    
    return float(np.mean(responses == tally_preds_valid))
```

**Observed (real) value:** 0.7037 (var=0.0287)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2261 (var=0.0304) (Δ vs real -0.4776)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8808 (var=0.0067)
- pi_1: 0.1534 (var=0.0064)
- pi_3: 0.8434 (var=0.0130)
- pi_4: 0.6711 (var=0.0160)
- pi_5: 0.6795 (var=0.0180)
- pi_6: 0.7700 (var=0.0339)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 0, 1, 0, 0] and [0, 1, 0, 1, 0]
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    is_B_target = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    
    # Count how many times the target option [1, 0, 1, 0, 0] was chosen
    chose_target_when_A = (data.loc[is_A_target, 'response'] == 0).sum()
    chose_target_when_B = (data.loc[is_B_target, 'response'] == 1).sum()
    
    total_trials = is_A_target.sum() + is_B_target.sum()
    if total_trials == 0:
        return 0.5
        
    return float((chose_target_when_A + chose_target_when_B) / total_trials)
```

**Observed (real) value:** 0.6050 (var=0.0828)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8250 (var=0.0193) (Δ vs real +0.2200)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6775 (var=0.0220)
- pi_2: 0.4742 (var=0.0081)
- pi_1: 0.8642 (var=0.0098)
- pi_4: 0.6425 (var=0.0201)
- pi_5: 0.6867 (var=0.0154)
- pi_6: 0.5392 (var=0.0239)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the unweighted sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Extract the first (highest validity) feature for each option
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    b_first = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where one option has more positive features (higher sum)
    # but lacks the most important feature (first feature is 0).
    # This corresponds exactly to trials 1 and 3 in the design.
    mask_a_higher_but_lacks_first = (sum_a > sum_b) & (a_first == 0)
    mask_b_higher_but_lacks_first = (sum_b > sum_a) & (b_first == 0)
    
    mask = mask_a_higher_but_lacks_first | mask_b_higher_but_lacks_first
    
    if not mask.any():
        return 0.5
        
    filtered = data[mask]
    chose_a = filtered['response'] == 0
    
    a_is_higher = sum_a[mask] > sum_b[mask]
    
    # Calculate the proportion of times the subject chose the option with the higher unweighted sum
    chose_higher = (chose_a & a_is_higher) | (~chose_a & ~a_is_higher)
    
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.3713 (var=0.0726)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1519 (var=0.0128) (Δ vs real -0.2194)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8556 (var=0.0073)
- pi_3: 0.3719 (var=0.0884)
- pi_1: 0.1663 (var=0.0101)
- pi_4: 0.5062 (var=0.0350)
- pi_5: 0.3281 (var=0.0599)
- pi_6: 0.5331 (var=0.1027)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 3: A=[1, 0, 0, 0, 1]. WADD favors B, Rank-based favors A.
    is_t3 = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[4] == 1)
    
    # Trial 5: A=[0, 0, 1, 1, 0]. WADD favors A, Rank-based favors B.
    is_t5 = data['option_a_ratings'].apply(lambda x: x[2] == 1 and x[3] == 1)
    
    t3_data = data[is_t3]
    t5_data = data[is_t5]
    
    score = 0.0
    n = 0
    
    if len(t3_data) > 0:
        # Rank-based predicts choosing A (response == 0)
        score += (t3_data['response'] == 0).sum()
        n += len(t3_data)
        
    if len(t5_data) > 0:
        # Rank-based predicts choosing B (response == 1)
        score += (t5_data['response'] == 1).sum()
        n += len(t5_data)
        
    if n == 0:
        return 0.5
        
    return float(score / n)
```

**Observed (real) value:** 0.4379 (var=0.0545)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8321 (var=0.0244) (Δ vs real +0.3942)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2258 (var=0.0261)
- pi_4: 0.5574 (var=0.0120)
- pi_1: 0.8447 (var=0.0103)
- pi_2: 0.4905 (var=0.0053)
- pi_5: 0.3811 (var=0.0209)
- pi_6: 0.4247 (var=0.0259)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        diff = a - b
        
        # The features are ordered by validity (highest to lowest)
        # The first non-zero difference indicates which option has the highest validity feature
        idx = np.nonzero(diff)[0][0]
        
        # diff[idx] == 1 means A has it, diff[idx] == -1 means B has it
        highest_validity_is_b = 1 if diff[idx] == -1 else 0
        
        # Check if the subject's response (0 for A, 1 for B) aligns with the highest valid feature
        if row['response'] == highest_validity_is_b:
            matches.append(1.0)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4083 (var=0.0157)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8075 (var=0.0212) (Δ vs real +0.3992)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4867 (var=0.0072)
- pi_3: 0.1765 (var=0.0096)
- pi_1: 0.8660 (var=0.0075)
- pi_2: 0.3187 (var=0.0041)
- pi_5: 0.3162 (var=0.0160)
- pi_6: 0.2904 (var=0.0223)

### Experiment 7
**Design**
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5012 (var=0.0602)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8046 (var=0.0162) (Δ vs real +0.3033)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7454 (var=0.0130)
- pi_4: 0.4475 (var=0.0036)
- pi_1: 0.8267 (var=0.0129)
- pi_2: 0.4946 (var=0.0024)
- pi_3: 0.7171 (var=0.0173)
- pi_6: 0.5319 (var=0.0343)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Trial 2: Option A has features 1 and 3 (indices 0 and 2)
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 0 and x[2] == 1)
    
    # Trial 3: Option A has features 2 and 3 (indices 1 and 2)
    t3_mask = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 1 and x[2] == 1)
    
    t2_data = data[t2_mask]
    t3_data = data[t3_mask]
    
    if len(t2_data) == 0 or len(t3_data) == 0:
        return 0.0
        
    p_a_t2 = np.mean(t2_data['response'] == 0)
    p_a_t3 = np.mean(t3_data['response'] == 0)
    
    return float(p_a_t2 - p_a_t3)
```

**Observed (real) value:** -0.0363 (var=0.0444)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6156 (var=0.0759) (Δ vs real +0.6519)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0737 (var=0.0203)
- pi_5: 0.2006 (var=0.0213)
- pi_1: 0.6825 (var=0.0526)
- pi_2: 0.0069 (var=0.0060)
- pi_3: 0.0144 (var=0.0081)
- pi_6: 0.0100 (var=0.0143)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = (data['a_str'] == '11000') & (data['b_str'] == '00111')
    if not mask.any():
        return 0.5
    
    return float((data.loc[mask, 'response'] == 1).mean())
```

**Observed (real) value:** 0.6250 (var=0.0810)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1583 (var=0.0132) (Δ vs real -0.4667)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7058 (var=0.0697)
- pi_4: 0.4958 (var=0.0443)
- pi_1: 0.1533 (var=0.0182)
- pi_2: 0.8642 (var=0.0153)
- pi_3: 0.7342 (var=0.0258)
- pi_5: 0.6475 (var=0.0256)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trials 2 and 3 where Option A has exactly two positive features including the 1st one
    is_target = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 2)
    target_data = data[is_target]
    if len(target_data) == 0:
        return 0.5
    # Return the proportion of times Option A was chosen (response == 0)
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.3458 (var=0.0760)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8142 (var=0.0244) (Δ vs real +0.4683)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5787 (var=0.0109)
- pi_6: 0.4167 (var=0.0360)
- pi_1: 0.8350 (var=0.0136)
- pi_2: 0.4979 (var=0.0043)
- pi_3: 0.2083 (var=0.0173)
- pi_5: 0.4371 (var=0.0192)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Probabilistic Take-The-Best (Cue-Skipping Lexicographic): Decision-makers inspect cues sequentially in descending order of validity. However, execution is probabilistic: each discriminating cue has a probability of being successfully 'trusted', which is proportional to a power transformation of its validity. If trusted, it determines the choice; if skipped, search continues to the next cue. If all cues are exhausted, the decision-maker guesses. This creates a validity-dependent noise mechanism that captures deviations from strict lexicographic behavior without using compensatory tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Probabilistic TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity in descending order
    order = np.argsort(-val, kind="stable")
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of trusting and acting on this discriminating cue
            p_trust = np.clip(alpha * (val[idx] ** gamma), 0.0, 1.0)
            
            if a[idx] > b[idx]:
                p_a += p_reach * p_trust
            else:
                p_b += p_reach * p_trust
                
            # Update the probability of reaching the next cue
            p_reach *= (1.0 - p_trust)
            
    # If all discriminating cues are skipped, guess
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Apply epsilon-greedy lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- alpha: [0.0, 5.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, the Probabilistic Take-The-Best (pTTB) theory models decision-makers as sequentially inspecting cues but probabilistically skipping them. Instead of deterministic stopping or compensatory weighting, the probability of 'trusting' a discriminating cue is a function of its validity (parameterized by alpha and gamma). This naturally produces validity-dependent noise, allowing lower-ranked cues to sometimes determine the choice when higher-ranked ones are skipped, capturing the empirical variance and deviations from strict lexicographic behavior without reverting to tallying.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Probabilistic Take-The-Best candidate was accepted by the gate, but an inspection of the experiment-level metrics reveals that it is behaving almost identically to deterministic Take-The-Best (pi_1). For instance, in Experiment 1, the candidate predicts 0.8233 (pi_1 = 0.8744) while the observed data is 0.2988. In Experiment 2, it predicts 0.2261 (pi_1 = 0.1534) against an observed 0.7037. This occurs because the trust probability `p_trust = np.clip(alpha * (val[idx] ** gamma), 0.0, 1.0)` easily saturates to 1.0 when `alpha` is allowed to range up to 5.0, effectively eliminating the cue-skipping mechanism and reverting the model to deterministic lexicographic search.
Rationale: To prevent the model from collapsing into deterministic Take-The-Best, the probability of trusting a cue must be recalibrated so that cue-skipping happens frequently enough to match human data. Restrict the `alpha` parameter range to [0.0, 1.0] so it acts as a true baseline probability rather than a multiplier that forces saturation. Alternatively, simplify `p_trust` to just `val[idx] ** gamma` or use a logistic function of validity. Maintain the core probabilistic sequential search mechanism, but ensure the parameterization naturally keeps `p_trust` strictly fractional for typical validity values.

**Outcome of this advice:** iter 1 candidate loss=0.6620 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Probabilistic Take-The-Best (Cue-Skipping Lexicographic): Decision-makers inspect cues sequentially in descending order of validity. However, execution is probabilistic: each discriminating cue has a probability of being successfully 'trusted', which is proportional to a power transformation of its validity. If trusted, it determines the choice; if skipped, search continues to the next cue. If all cues are exhausted, the decision-maker guesses. This creates a validity-dependent noise mechanism that captures deviations from strict lexicographic behavior without using compensatory tallying.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Probabilistic TTB expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    alpha = float(parameters[\"alpha\"])\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Sort cues by validity in descending order\n    order = np.argsort(-val, kind=\"stable\")\n    \n    p_a = 0.0\n    p_b = 0.0\n    p_reach = 1.0\n    \n    for idx in order:\n        if a[idx] != b[idx]:\n            # Probability of trusting and acting on this discriminating cue\n            p_trust = np.clip(alpha * (val[idx] ** gamma), 0.0, 1.0)\n            \n            if a[idx] > b[idx]:\n                p_a += p_reach * p_trust\n            else:\n                p_b += p_reach * p_trust\n                \n            # Update the probability of reaching the next cue\n            p_reach *= (1.0 - p_trust)\n            \n    # If all discriminating cues are skipped, guess\n    p_a += p_reach * 0.5\n    p_b += p_reach * 0.5\n    \n    p_core = np.array([p_a, p_b])\n    \n    # Apply epsilon-greedy lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "gamma": "[0.0, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, the `alpha` parameter is now restricted to [0.0, 1.0]. In the previous iteration, allowing `alpha` to reach 5.0 caused the trust probability (`p_trust`) to easily saturate at 1.0, effectively eliminating the cue-skipping mechanism and reverting the model to deterministic Take-The-Best. By constraining `alpha` to a maximum of 1.0, `p_trust` remains strictly fractional for typical validity values, ensuring that cue-skipping occurs frequently enough to match the empirical variance and deviations from strict lexicographic behavior observed in the human data."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14493,
  "candidates_token_count": 886,
  "total_token_count": 15850
}
```
