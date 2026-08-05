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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a 'Validity Threshold Tallying' theory. In this theory, decision-makers simplify the choice environment by setting a threshold on cue validity. Cues with a validity above the threshold are given equal weight (unit weight / tallying), while cues below the threshold are ignored completely. This explains why small differences in high validities don't lead to strict preferences (unlike WADD) and why low-validity cues fail to compensate for fewer high-validity cues (unlike Rank-Based Weighting). The model should fit a threshold parameter and a softmax temperature to account for choice noise.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.854)

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


---

### `pi_5` (overall score: 0.687)

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


---

### `pi_3` (overall score: 0.652)

**Description**
Decision-makers evaluate options by computing a weighted sum of their features, where the weights are a subjective power transformation of the cue validities. This Weighted Additive (WADD) model is fully compensatory, allowing the combined evidence of multiple lower-validity cues to potentially outweigh a single higher-validity cue, while still remaining sensitive to the varying diagnosticity of different features. The parameter governing the power transformation is constrained to prevent extreme exponentiation, ensuring the model remains compensatory rather than collapsing into a non-compensatory heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities into weights
    w = val ** gamma
    
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 3.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2988 (var=0.0249) vs this=0.3183 (var=0.0105)
- Experiment 2: real=0.7037 (var=0.0287) vs this=0.8434 (var=0.0130)
- Experiment 3: real=0.6050 (var=0.0828) vs this=0.6775 (var=0.0220)
- Experiment 4: real=0.3713 (var=0.0726) vs this=0.3719 (var=0.0884)
- Experiment 5: real=0.4379 (var=0.0545) vs this=0.2258 (var=0.0261)
- Experiment 6: real=0.4083 (var=0.0157) vs this=0.1765 (var=0.0096)
- Experiment 7: real=0.5012 (var=0.0602) vs this=0.7171 (var=0.0173)
- Experiment 8: real=-0.0363 (var=0.0444) vs this=0.0144 (var=0.0081)


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
**Other theories' values on this metric (for reference):**
- pi_1: 0.8744 (var=0.0084)
- pi_2: 0.1846 (var=0.0061)
- pi_3: 0.3183 (var=0.0105)
- pi_4: 0.3685 (var=0.0144)
- pi_5: 0.3904 (var=0.0104)

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
**Other theories' values on this metric (for reference):**
- pi_2: 0.8808 (var=0.0067)
- pi_1: 0.1534 (var=0.0064)
- pi_3: 0.8434 (var=0.0130)
- pi_4: 0.6711 (var=0.0160)
- pi_5: 0.6795 (var=0.0180)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.6775 (var=0.0220)
- pi_2: 0.4742 (var=0.0081)
- pi_1: 0.8642 (var=0.0098)
- pi_4: 0.6425 (var=0.0201)
- pi_5: 0.6867 (var=0.0154)

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
**Other theories' values on this metric (for reference):**
- pi_2: 0.8556 (var=0.0073)
- pi_3: 0.3719 (var=0.0884)
- pi_1: 0.1663 (var=0.0101)
- pi_4: 0.5062 (var=0.0350)
- pi_5: 0.3281 (var=0.0599)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.2258 (var=0.0261)
- pi_4: 0.5574 (var=0.0120)
- pi_1: 0.8447 (var=0.0103)
- pi_2: 0.4905 (var=0.0053)
- pi_5: 0.3811 (var=0.0209)

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
**Other theories' values on this metric (for reference):**
- pi_4: 0.4867 (var=0.0072)
- pi_3: 0.1765 (var=0.0096)
- pi_1: 0.8660 (var=0.0075)
- pi_2: 0.3187 (var=0.0041)
- pi_5: 0.3162 (var=0.0160)

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
**Other theories' values on this metric (for reference):**
- pi_5: 0.7454 (var=0.0130)
- pi_4: 0.4475 (var=0.0036)
- pi_1: 0.8267 (var=0.0129)
- pi_2: 0.4946 (var=0.0024)
- pi_3: 0.7171 (var=0.0173)

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
**Other theories' values on this metric (for reference):**
- pi_4: 0.0737 (var=0.0203)
- pi_5: 0.2006 (var=0.0213)
- pi_1: 0.6825 (var=0.0526)
- pi_2: 0.0069 (var=0.0060)
- pi_3: 0.0144 (var=0.0081)


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
  "theory": "Validity Threshold Tallying: Decision-makers simplify the choice environment by setting a subjective threshold on cue validity. Cues with a validity above the threshold are considered 'important' and are given equal weight (tallying), while cues below the threshold are ignored completely. This heuristic bridges the gap between fully compensatory models and strict non-compensatory models by allowing multiple important cues to compensate for each other, while preventing unimportant cues from influencing the decision.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Validity Threshold Tallying expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    threshold = float(parameters[\"threshold\"])\n    \n    # Cues with validity >= threshold are given unit weight, others are ignored\n    w = (val >= threshold).astype(float)\n    \n    # Compute tallied scores for both options\n    score_a = np.sum(w * a)\n    score_b = np.sum(w * b)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax over the scores to compute choice probabilities\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "threshold": "[0.5, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, the Validity Threshold Tallying theory categorizes cues into 'important' and 'unimportant' based on a validity threshold. Important cues are tallied with equal (unit) weight, while unimportant cues are completely ignored. This explains why small differences among high-validity cues do not lead to strict preferences (unlike WADD) and why low-validity cues fail to compensate for fewer high-validity cues (unlike Rank-Based Weighting). By fitting a threshold parameter along with a softmax temperature, the model captures the boundedly rational simplification that humans apply to multi-attribute choices."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10740,
  "candidates_token_count": 726,
  "total_token_count": 12502
}
```
