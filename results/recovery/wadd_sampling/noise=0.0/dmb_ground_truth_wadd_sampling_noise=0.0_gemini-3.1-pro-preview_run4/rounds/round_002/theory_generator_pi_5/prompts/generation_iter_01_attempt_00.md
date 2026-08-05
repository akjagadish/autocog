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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a new 'Hybrid Cardinal-Rank Weighting' theory or a 'Strategy Selection' theory. Decision-makers might not rely purely on cardinal validities or pure ordinal ranks. Instead, they might use weights that are a function of both the cardinal validity and its ordinal rank (e.g., scaling cardinal validities by a rank-based discount factor), or they may engage in strategy selection where a mixture of compensatory WADD and non-compensatory/rank-based heuristics are used across the population. This new theory should capture the strict drop-offs seen in Exps 1, 2, and 4, while preserving the compensatory thresholding behavior that perfectly matched Exp 6.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.847)

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


---

### `pi_3` (overall score: 0.710)

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


---

### `pi_2` (overall score: 0.502)

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
- Experiment 1: real=0.2988 (var=0.0249) vs this=0.1846 (var=0.0061)
- Experiment 2: real=0.7037 (var=0.0287) vs this=0.8808 (var=0.0067)
- Experiment 3: real=0.6050 (var=0.0828) vs this=0.4742 (var=0.0081)
- Experiment 4: real=0.3713 (var=0.0726) vs this=0.8556 (var=0.0073)
- Experiment 5: real=0.4379 (var=0.0545) vs this=0.4905 (var=0.0053)
- Experiment 6: real=0.4083 (var=0.0157) vs this=0.3187 (var=0.0041)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3968 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.3968 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1 (most recent): 0.6529 (var=0.0260) (Δ vs real +0.3542)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8744 (var=0.0084)
- pi_2: 0.1846 (var=0.0061)
- pi_3: 0.3183 (var=0.0105)
- pi_4: 0.3685 (var=0.0144)

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
  - iter 1 (most recent): 0.3847 (var=0.0430) (Δ vs real -0.3189)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8808 (var=0.0067)
- pi_1: 0.1534 (var=0.0064)
- pi_3: 0.8434 (var=0.0130)
- pi_4: 0.6711 (var=0.0160)

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
  - iter 1 (most recent): 0.8042 (var=0.0202) (Δ vs real +0.1992)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6775 (var=0.0220)
- pi_2: 0.4742 (var=0.0081)
- pi_1: 0.8642 (var=0.0098)
- pi_4: 0.6425 (var=0.0201)

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
  - iter 1 (most recent): 0.1656 (var=0.0193) (Δ vs real -0.2056)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8556 (var=0.0073)
- pi_3: 0.3719 (var=0.0884)
- pi_1: 0.1663 (var=0.0101)
- pi_4: 0.5062 (var=0.0350)

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
  - iter 1 (most recent): 0.6021 (var=0.0248) (Δ vs real +0.1642)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2258 (var=0.0261)
- pi_4: 0.5574 (var=0.0120)
- pi_1: 0.8447 (var=0.0103)
- pi_2: 0.4905 (var=0.0053)

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
  - iter 1 (most recent): 0.6531 (var=0.0311) (Δ vs real +0.2448)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4867 (var=0.0072)
- pi_3: 0.1765 (var=0.0096)
- pi_1: 0.8660 (var=0.0075)
- pi_2: 0.3187 (var=0.0041)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Decision-makers do not rely purely on cardinal validities or pure ordinal ranks when evaluating options. Instead, they employ a Hybrid Cardinal-Rank Weighting strategy, where the subjective importance of a feature is determined by scaling its cardinal validity by a rank-based discount factor. This allows individuals to remain sensitive to the exact magnitudes of cue validities (enabling compensatory trade-offs where multiple lower-validity cues can outweigh a higher-validity one) while simultaneously exhibiting strict drop-offs in attention or trust for lower-ranked cues. By varying the sensitivity to both cardinal values and ordinal ranks across the population, this theory captures both the compensatory thresholding behavior and the sharper non-compensatory-like drop-offs observed in human choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid Weighting expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    
    # Determine the ordinal rank of each feature's validity (1 = highest validity)
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Weights combine cardinal validity and ordinal rank
    w = (val ** gamma) / (ranks ** delta)
    
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
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 3.0]
- delta: [0.0, 3.0]
- validities: validities

`rationale`: Based on the arbiter's feedback, this proposal instantiates a Hybrid Cardinal-Rank Weighting theory. The previous WADD model (pi_3) relied exclusively on cardinal validities, while the Rank-Based model (pi_4) relied exclusively on ordinal ranks. By defining the subjective feature weights as w = (validity^gamma) / (rank^delta), the model seamlessly bridges both mechanisms. The gamma parameter controls sensitivity to cardinal validity magnitudes (preserving the compensatory thresholding behavior seen in Exp 6), while the delta parameter introduces a rank-based discount factor that captures the strict drop-offs in reliance on lower-ranked cues (better matching Exps 1, 2, and 4). This hybrid approach offers the flexibility needed to explain the varied human behavior across the experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate implements a Hybrid Cardinal-Rank Weighting theory, which was accepted by the gate (improving over the previous base). However, the simulated values still diverge significantly from the observed data across multiple experiments. For example, in Exp 1 (Obs=0.2988, Cand=0.6529) and Exp 2 (Obs=0.7037, Cand=0.3847), the model over- or under-predicts the reliance on specific cues. The single-strategy hybrid weighting approach struggles to capture the bimodal nature of human decision-making, where some subjects might be purely compensatory while others are strictly non-compensatory.
Rationale: While the hybrid weighting improved the overall loss, it fails to capture the specific empirical values for Exps 1, 2, and 4. Since the arbiter explicitly allowed a 'Strategy Selection' theory, revise the candidate to implement a mixture model. Instead of computing a single hybrid weight for all decisions, model the population as a mixture of two distinct strategies: a compensatory WADD strategy (using pure cardinal validities) and a non-compensatory strategy (e.g., Take-The-Best or pure rank-based). Introduce a mixture parameter (e.g., `alpha` in [0, 1]) that represents the probability of a subject using the compensatory strategy vs the non-compensatory strategy on any given trial, and compute the final choice probabilities as a weighted sum of the probabilities from each strategy.

**Outcome of this advice:** iter 1 candidate loss=0.3968 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Strategy Selection (WADD and TTB Mixture)",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Strategy Selection expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    alpha = float(parameters[\"alpha\"])\n    \n    # Strategy 1: Compensatory WADD\n    w_wadd = val ** gamma\n    score_a = np.sum(w_wadd * a)\n    score_b = np.sum(w_wadd * b)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_wadd = e / np.sum(e)\n    \n    # Strategy 2: Non-compensatory Take-The-Best (TTB)\n    order = np.argsort(-val, kind=\"stable\")\n    p_ttb = np.array([0.5, 0.5])\n    for idx in order:\n        if a[idx] > b[idx]:\n            p_ttb = np.array([1.0, 0.0])\n            break\n        elif b[idx] > a[idx]:\n            p_ttb = np.array([0.0, 1.0])\n            break\n            \n    # Mixture of strategies\n    p_core = alpha * p_wadd + (1.0 - alpha) * p_ttb\n    \n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "gamma": "[0.0, 5.0]",
    "alpha": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, this proposal replaces the single-strategy hybrid weighting with a Strategy Selection (mixture) model. It posits that human decision-makers probabilistically employ either a compensatory Weighted Additive (WADD) strategy based on cardinal validities or a strictly non-compensatory Take-The-Best (TTB) strategy. The parameter `alpha` represents the probability of adopting the WADD strategy on a given evaluation. By computing the final choice probabilities as a weighted sum of the two strategies, the model captures the bimodal nature of the population—explaining both the compensatory thresholding behavior in Exp 6 and the strict drop-offs seen in Exps 1, 2, and 4."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11461,
  "candidates_token_count": 838,
  "total_token_count": 14099
}
```
