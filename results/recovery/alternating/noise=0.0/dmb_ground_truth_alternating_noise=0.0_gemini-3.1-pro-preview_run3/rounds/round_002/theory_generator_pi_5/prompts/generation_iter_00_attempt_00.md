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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Propose a 'Response Heuristic / Spatial Bias' theory. Since the data shows subjects completely ignore feature values (evidenced by 50% accuracy on pure dominance trials), this new theory should model decision-making as a function of the options' presentation (e.g., a persistent bias toward Option A or Option B, or a tendency to repeat the previous choice/alternate) rather than their feature vectors. This provides a mechanistic alternative to pure uniform random guessing while maintaining zero sensitivity to cue validities.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.971)

**Description**
Random Guessing: The data indicates that subjects in these specific experimental paradigms completely ignore all feature information and validities. Likely due to a lack of trial-by-trial feedback or low motivation, subjects simply choose between the two options with equal probability on every trial, resulting in chance-level performance across all metrics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # Reference the dummy parameter to satisfy the parameter usage constraint
    _ = float(parameters["dummy"])
    
    # Pure random guessing: uniform 50/50 probability regardless of the stimulus
    return np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- dummy: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4921 (var=0.0028)
- Experiment 2: real=-0.0237 (var=0.0386) vs this=0.0304 (var=0.0186)
- Experiment 3: real=0.2667 (var=0.0067) vs this=0.3167 (var=0.0076)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4978 (var=0.0028)
- Experiment 5: real=0.0700 (var=0.0576) vs this=-0.0175 (var=0.0149)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.4944 (var=0.0119)


---

### `pi_3` (overall score: 0.851)

**Description**
Weighted Additive (WADD) model with non-linear validity scaling and extended lapse rate: Decision makers evaluate options by computing a weighted sum of their feature values. The weights are formed by centering the cue validities (val - 0.5) and raising their absolute values to a power gamma (preserving sign), which allows the model to interpolate between equal-weighting, proportional weighting, and non-compensatory behavior without extreme log-odds scaling. To account for the high degree of noise or chance-level behavior observed in specific paradigms, the lapse rate (epsilon) can range up to 1.0, and beta down to 0.0, allowing the model to capture completely random guessing natively.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Center validities so 0.5 gives 0 weight, avoiding discrete parameter
    base_w = val - 0.5
        
    # Signed exponentiation to handle negative base_w with fractional gamma safely
    w = np.sign(base_w) * (np.abs(base_w) ** gamma)
        
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5010 (var=0.0037)
- Experiment 2: real=-0.0237 (var=0.0386) vs this=0.0627 (var=0.0377)
- Experiment 3: real=0.2667 (var=0.0067) vs this=0.3333 (var=0.0071)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.5184 (var=0.0034)
- Experiment 5: real=0.0700 (var=0.0576) vs this=0.1392 (var=0.0667)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.6089 (var=0.0249)


---

### `pi_2` (overall score: 0.112)

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
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.2267 (var=0.0038)
- Experiment 2: real=-0.0237 (var=0.0386) vs this=0.6340 (var=0.0248)
- Experiment 3: real=0.2667 (var=0.0067) vs this=0.0658 (var=0.0065)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.5009 (var=0.0015)
- Experiment 5: real=0.0700 (var=0.0576) vs this=0.7250 (var=0.0319)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.8478 (var=0.0117)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0093)
- pi_2: 0.2267 (var=0.0038)
- pi_3: 0.5010 (var=0.0037)
- pi_4: 0.4921 (var=0.0028)

### Experiment 2
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Determine which option is favored by the highest validity cue (cue 0)
    cue_0_a = a_ratings[:, 0]
    cue_0_b = b_ratings[:, 0]
    cue_0_winner = (cue_0_b > cue_0_a).astype(int)
    
    # Did the subject choose the option favored by cue 0?
    chose_cue_0 = (data['response'] == cue_0_winner).astype(float)
    
    # Calculate the tally difference in favor of the cue 0 winner
    tally_a = (a_ratings > b_ratings).sum(axis=1)
    tally_b = (b_ratings > a_ratings).sum(axis=1)
    tally_diff = np.where(cue_0_winner == 0, tally_a - tally_b, tally_b - tally_a)
    
    # Compare choice probabilities when tally strongly agrees vs. disagrees/ties with cue 0
    high_diff_mask = tally_diff >= 3
    low_diff_mask = tally_diff <= 0
    
    if not np.any(high_diff_mask) or not np.any(low_diff_mask):
        return 0.0
        
    high_diff_rate = chose_cue_0[high_diff_mask].mean()
    low_diff_rate = chose_cue_0[low_diff_mask].mean()
    
    return float(high_diff_rate - low_diff_rate)

```

**Observed (real) value:** -0.0237 (var=0.0386)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6340 (var=0.0248)
- pi_1: -0.0025 (var=0.0080)
- pi_3: 0.0627 (var=0.0377)
- pi_4: 0.0304 (var=0.0186)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    subject_mins = []
    for subj, subj_df in data.groupby('subject_id'):
        pair_adherence = {}
        for _, row in subj_df.iterrows():
            a = tuple(row['option_a_ratings'])
            b = tuple(row['option_b_ratings'])
            # Use frozenset to group identical trial pairs regardless of A/B presentation order
            pair_key = frozenset({a, b})
            
            # Determine Take The Best (TTB) prediction
            ttb_choice = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            
            if ttb_choice is None:
                continue
                
            adherence = 1 if row['response'] == ttb_choice else 0
            
            if pair_key not in pair_adherence:
                pair_adherence[pair_key] = []
            pair_adherence[pair_key].append(adherence)
            
        if not pair_adherence:
            continue
            
        # Compute the proportion of TTB adherence for each unique trial pair
        proportions = [sum(advs)/len(advs) for advs in pair_adherence.values()]
        # Take the minimum adherence across all pairs for this subject
        subject_mins.append(min(proportions))
        
    if not subject_mins:
        return 0.0
        
    # Return the average of these minimums across all subjects
    return float(sum(subject_mins) / len(subject_mins))
```

**Observed (real) value:** 0.2667 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7283 (var=0.0191)
- pi_3: 0.3333 (var=0.0071)
- pi_2: 0.0658 (var=0.0065)
- pi_4: 0.3167 (var=0.0076)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5184 (var=0.0034)
- pi_1: 0.8229 (var=0.0163)
- pi_2: 0.5009 (var=0.0015)
- pi_4: 0.4978 (var=0.0028)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option B dominates on the top 3 most valid features (Trials 7 & 8)
    b_top3 = data['option_b_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 1)
    
    # Identify trials where Option A dominates on the top 3 most valid features (Trials 1 & 2)
    a_top3 = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 1)
    
    chose_b = (data['response'] == 1).astype(float)
    
    # Difference in probability of choosing B when B has the best features vs when A has them
    mean_b_when_b_best = chose_b[b_top3].mean()
    mean_b_when_a_best = chose_b[a_top3].mean()
    
    # Fallback to 0.0 if for some reason a subject is missing these trials
    if pd.isna(mean_b_when_b_best) or pd.isna(mean_b_when_a_best):
        return 0.0
        
    return float(mean_b_when_b_best - mean_b_when_a_best)
```

**Observed (real) value:** 0.0700 (var=0.0576)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0175 (var=0.0149)
- pi_3: 0.1392 (var=0.0667)
- pi_1: 0.6633 (var=0.0483)
- pi_2: 0.7250 (var=0.0319)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    correct = 0
    total = 0
    for _, row in data.iterrows():
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        if sum_a == 5 and sum_b == 0:
            if row['response'] == 0:
                correct += 1
            total += 1
        elif sum_a == 0 and sum_b == 5:
            if row['response'] == 1:
                correct += 1
            total += 1
    return float(correct / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6089 (var=0.0249)
- pi_4: 0.4944 (var=0.0119)
- pi_1: 0.8433 (var=0.0139)
- pi_2: 0.8478 (var=0.0117)


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
  "theory": "Response Heuristic / Spatial Bias: Subjects in these paradigms often ignore feature information entirely and instead rely on presentation-based response heuristics. Decision-making is driven by a persistent spatial bias (e.g., a preference for Option A over Option B) combined with a sequential dependence, such as a tendency to repeat the previous choice or alternate between options. This produces behavior that appears random with respect to cue validities but contains structured autocorrelation and spatial preference.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    beta_bias = float(parameters[\"beta_bias\"])\n    beta_repeat = float(parameters[\"beta_repeat\"])\n    \n    # Determine the indicator for the previous choice\n    # 1.0 if Option A was chosen, -1.0 if Option B was chosen, 0.0 if first trial\n    if len(history[\"response\"]) == 0:\n        last_a = 0.0\n    else:\n        last_a = 1.0 if history[\"response\"][-1] == 0 else -1.0\n        \n    # Logit for choosing Option A\n    z = beta_bias + beta_repeat * last_a\n    \n    # Convert to probability using sigmoid\n    p_a = 1.0 / (1.0 + np.exp(-z))\n    \n    return np.array([p_a, 1.0 - p_a])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta_bias": "[-3.0, 3.0]",
    "beta_repeat": "[-3.0, 3.0]"
  },
  "rationale": "Following the arbiter's feedback, this model instantiates a 'Response Heuristic / Spatial Bias' theory. Because subjects exhibit chance-level accuracy on pure dominance trials, they are likely ignoring feature values completely. This model captures their choices purely as a function of presentation: a baseline spatial bias toward Option A or B (`beta_bias`) and a sequential dependence to either repeat or alternate their previous choice (`beta_repeat`). This provides a mechanistic alternative to pure uniform random guessing, capturing subject-level autocorrelations and side preferences while remaining entirely insensitive to cue validities."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9330,
  "candidates_token_count": 581,
  "total_token_count": 11722
}
```
