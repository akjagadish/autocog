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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'Noisy Tallying' (or Equal Weights) theory. Instead of sorting features by validity (TTB) or computing weighted sums (WADD), decision-makers simply count the total number of positive features (1s) for each option and choose the one with the higher count. To capture the extreme smoothing seen in the data, this Tallying model must be coupled with a high lapse rate and a low inverse temperature, predicting choices that are only slightly biased toward the option with more positive features.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.970)

**Description**
Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4863 (var=0.0089) vs this=0.5012 (var=0.0086)
- Experiment 2: real=0.5067 (var=0.0118) vs this=0.4758 (var=0.0103)
- Experiment 3: real=0.4775 (var=0.0047) vs this=0.5021 (var=0.0064)
- Experiment 4: real=0.5100 (var=0.0127) vs this=0.4725 (var=0.0091)
- Experiment 5: real=-0.0063 (var=0.0098) vs this=0.0179 (var=0.0218)
- Experiment 6: real=0.0442 (var=0.0130) vs this=0.0083 (var=0.0095)
- Experiment 7: real=-0.0316 (var=0.0199) vs this=0.0168 (var=0.0232)
- Experiment 8: real=0.0979 (var=0.0009) vs this=0.0939 (var=0.0008)


---

### `pi_3` (overall score: 0.744)

**Description**
Take-The-Best (TTB) heuristic with subjective validity noise: Decision-makers search through features in descending order of their subjective validity, but their assessment of validity is noisy. On any given evaluation, they sample validities with some variance, sort features by these sampled validities, and stop at the first feature that discriminates between the two options. Response noise is modeled via a softmax over the resulting binary preference scores, plus an independent lapse rate. Averaging over multiple samples naturally smooths the aggregate predictions when validities are close.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma = float(parameters["sigma"])
    
    n_samples = 50
    avg_p_core = np.zeros(2)
    
    for _ in range(n_samples):
        noisy_v = validities + np.random.normal(0, sigma, size=validities.shape)
        order = np.argsort(noisy_v)[::-1]
        
        scores = np.zeros(2)
        # Search for the first discriminating feature
        for idx in order:
            if stim[0, idx] > stim[1, idx]:
                scores[0] = 1.0
                break
            elif stim[1, idx] > stim[0, idx]:
                scores[1] = 1.0
                break
                
        # Softmax over the scores
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        avg_p_core += p_core
        
    avg_p_core /= n_samples
    
    n_opts = avg_p_core.shape[0]
    return (1.0 - epsilon) * avg_p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- sigma: [0.0, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4863 (var=0.0089) vs this=0.4437 (var=0.0223)
- Experiment 2: real=0.5067 (var=0.0118) vs this=0.5017 (var=0.0285)
- Experiment 3: real=0.4775 (var=0.0047) vs this=0.5908 (var=0.0120)
- Experiment 4: real=0.5100 (var=0.0127) vs this=0.3725 (var=0.0154)
- Experiment 5: real=-0.0063 (var=0.0098) vs this=0.0853 (var=0.0269)
- Experiment 6: real=0.0442 (var=0.0130) vs this=-0.0196 (var=0.0170)
- Experiment 7: real=-0.0316 (var=0.0199) vs this=0.1842 (var=0.0255)
- Experiment 8: real=0.0979 (var=0.0009) vs this=0.1417 (var=0.0051)


---

### `pi_4` (overall score: 0.597)

**Description**
Strategy Mixture: Decision-makers are heterogeneous and use a mixture of non-compensatory and compensatory heuristics. On any given trial, a decision is made either via Take-The-Best (relying solely on the most valid discriminating feature) or Tallying (counting the total number of winning features), governed by a mixing parameter. When these strategies conflict, their opposing choices average out across the population, naturally capturing the ~0.5 aggregate choice proportions observed across experiments without relying on massive uniform noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Take-The-Best (TTB) predictions
    order = np.argsort(validities)[::-1]
    ttb_scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            ttb_scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying predictions
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    tally_scores = np.array([a_wins, b_wins])
    
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture of the two strategies
    p_core = p_ttb * p_ttb_dist + (1.0 - p_ttb) * p_tally_dist
    
    # Uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4863 (var=0.0089) vs this=0.4519 (var=0.0649)
- Experiment 2: real=0.5067 (var=0.0118) vs this=0.4117 (var=0.0496)
- Experiment 3: real=0.4775 (var=0.0047) vs this=0.5200 (var=0.0393)
- Experiment 4: real=0.5100 (var=0.0127) vs this=0.4858 (var=0.0608)
- Experiment 5: real=-0.0063 (var=0.0098) vs this=0.2305 (var=0.0377)
- Experiment 6: real=0.0442 (var=0.0130) vs this=-0.1308 (var=0.0125)
- Experiment 7: real=-0.0316 (var=0.0199) vs this=0.1621 (var=0.0331)
- Experiment 8: real=0.0979 (var=0.0009) vs this=0.2179 (var=0.0080)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    t1_mask = a_tuples == (0, 0, 1, 1, 1)
    t2_mask = a_tuples == (1, 1, 0, 0, 0)
    
    t1_tally_match = (data.loc[t1_mask, 'response'] == 0).sum()
    t2_tally_match = (data.loc[t2_mask, 'response'] == 1).sum()
    
    total = t1_mask.sum() + t2_mask.sum()
    if total == 0:
        return 0.5
        
    return float((t1_tally_match + t2_tally_match) / total)
```

**Observed (real) value:** 0.4863 (var=0.0089)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8300 (var=0.0116)
- pi_2: 0.4738 (var=0.0772)
- pi_3: 0.4437 (var=0.0223)
- pi_4: 0.4519 (var=0.0649)
- pi_5: 0.5012 (var=0.0086)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 pits an option with 3 low-validity features (A) against an option with 2 high-validity features (B).
    # Tallying strictly prefers A (3 wins vs 2 wins), leading to a response near 0.
    # WADD tends to prefer B, because the sum of the top 2 validities (0.9 + 0.8 = 1.7) 
    # is greater than the sum of the bottom 3 (0.6 + 0.5 + 0.5 = 1.6), leading to a higher rate of response 1.
    mask = data['option_a_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1, 1])
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5067 (var=0.0118)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4108 (var=0.0539)
- pi_1: 0.1617 (var=0.0119)
- pi_3: 0.5017 (var=0.0285)
- pi_4: 0.4117 (var=0.0496)
- pi_5: 0.4758 (var=0.0103)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def is_wadd_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):
            return resp == 1
        elif a == (0, 1, 1, 1, 1) and b == (1, 0, 0, 0, 0):
            return resp == 0
            
        # Trial 2
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 1):
            return resp == 1
        elif a == (0, 1, 0, 1, 1) and b == (1, 0, 1, 0, 0):
            return resp == 0
            
        # Trial 4
        elif a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return resp == 1
        elif a == (0, 0, 1, 1, 1) and b == (0, 1, 0, 0, 0):
            return resp == 0
            
        return np.nan

    wadd_choices = data.apply(is_wadd_choice, axis=1)
    return float(wadd_choices.dropna().mean())
```

**Observed (real) value:** 0.4775 (var=0.0047)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5908 (var=0.0120)
- pi_2: 0.7438 (var=0.0146)
- pi_1: 0.8596 (var=0.0084)
- pi_4: 0.5200 (var=0.0393)
- pi_5: 0.5021 (var=0.0064)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_chosen = 0
    total = 0
    
    for idx, row in data.iterrows():
        a_str = ''.join(map(str, row['option_a_ratings']))
        b_str = ''.join(map(str, row['option_b_ratings']))
        resp = row['response']
        
        # Trial 1: 10000 vs 01111. TTB prefers 10000.
        if a_str == '10000' and b_str == '01111':
            ttb_chosen += 1 if resp == 0 else 0
            total += 1
        elif a_str == '01111' and b_str == '10000':
            ttb_chosen += 1 if resp == 1 else 0
            total += 1
            
        # Trial 3: 01000 vs 00111. TTB prefers 01000.
        elif a_str == '01000' and b_str == '00111':
            ttb_chosen += 1 if resp == 0 else 0
            total += 1
        elif a_str == '00111' and b_str == '01000':
            ttb_chosen += 1 if resp == 1 else 0
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_chosen / total)
```

**Observed (real) value:** 0.5100 (var=0.0127)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2142 (var=0.0235)
- pi_3: 0.3725 (var=0.0154)
- pi_1: 0.1292 (var=0.0095)
- pi_4: 0.4858 (var=0.0608)
- pi_5: 0.4725 (var=0.0091)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuple = data['option_a_ratings'].apply(tuple)
    t1_a = (0, 1, 0, 0, 0)
    t2_a = (0, 1, 1, 1, 0)
    
    t1_data = data[a_tuple == t1_a]
    t2_data = data[a_tuple == t2_a]
    
    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.0
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.0
    
    return float(p_a_t2 - p_a_t1)
```

**Observed (real) value:** -0.0063 (var=0.0098)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0853 (var=0.0269)
- pi_4: 0.2305 (var=0.0377)
- pi_1: 0.3358 (var=0.0277)
- pi_2: 0.1600 (var=0.0831)
- pi_5: 0.0179 (var=0.0218)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert ratings to a 2D numpy array
    a_ratings = np.stack(data['option_a_ratings'].values)
    
    # Trials 1, 3, 5, 7 have the top feature (index 0) active for Option A
    # Trials 2, 4, 6, 8 have the second-best feature (index 1) active for Option A, and index 0 is tied
    is_a0_1 = a_ratings[:, 0] == 1
    
    chose_a = (data['response'] == 0).values
    
    # Calculate the proportion of times Option A was chosen in each condition
    p_a_when_top_feature = np.mean(chose_a[is_a0_1])
    p_a_when_second_feature = np.mean(chose_a[~is_a0_1])
    
    return float(p_a_when_top_feature - p_a_when_second_feature)
```

**Observed (real) value:** 0.0442 (var=0.0130)
**Other theories' values on this metric (for reference):**
- pi_4: -0.1308 (var=0.0125)
- pi_3: -0.0196 (var=0.0170)
- pi_1: -0.2383 (var=0.0114)
- pi_2: -0.1171 (var=0.0609)
- pi_5: 0.0083 (var=0.0095)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_t1(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 1, 1, 1, 1)
    
    def is_t5(row):
        return tuple(row['option_a_ratings']) == (0, 1, 1, 0, 0) and tuple(row['option_b_ratings']) == (0, 0, 0, 1, 1)
        
    t1_mask = data.apply(is_t1, axis=1)
    t5_mask = data.apply(is_t5, axis=1)
    
    p_a_t1 = np.mean(data[t1_mask]['response'] == 0) if t1_mask.sum() > 0 else 0.5
    p_a_t5 = np.mean(data[t5_mask]['response'] == 0) if t5_mask.sum() > 0 else 0.5
    
    return float(p_a_t5 - p_a_t1)
```

**Observed (real) value:** -0.0316 (var=0.0199)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1842 (var=0.0255)
- pi_5: 0.0168 (var=0.0232)
- pi_1: 0.3432 (var=0.0215)
- pi_2: 0.3295 (var=0.0857)
- pi_4: 0.1621 (var=0.0331)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subj_metric(df):
        keys = df['option_a_ratings'].apply(tuple)
        p_A = 1.0 - df.groupby(keys)['response'].mean()
        return float((p_A - 0.5).abs().mean())
    
    return float(data.groupby('subject_id').apply(subj_metric).mean())
```

**Observed (real) value:** 0.0979 (var=0.0009)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0939 (var=0.0008)
- pi_3: 0.1417 (var=0.0051)
- pi_1: 0.3291 (var=0.0094)
- pi_2: 0.2916 (var=0.0067)
- pi_4: 0.2179 (var=0.0080)


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
  "theory": "Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Tallying: Sum of active features for each option (ignoring validities)\n    scores = np.sum(stim, axis=1)\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Blend with uniform lapse rate\n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.0, 0.5]",
    "epsilon": "[0.5, 1.0]"
  },
  "rationale": "Following the arbiter's feedback, this theory implements a Noisy Tallying heuristic. Instead of sorting features by validity (TTB) or computing weighted sums (WADD), the model simply counts the total number of positive features for each option. To capture the extreme smoothing seen in the data (where choices hover around 0.5 and differences between conditions are minimal), the model uses a high lapse rate (epsilon in [0.5, 1.0]) and a low inverse temperature (beta in [0.0, 0.5]). This ensures that the predictions are only slightly biased toward the option with more positive features, matching the aggregate empirical patterns."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10895,
  "candidates_token_count": 614,
  "total_token_count": 13095
}
```
