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

Propose a 'Heuristic Mixture: Take-The-Best and Tallying' theory. Instead of a single complex integration mechanism, this theory posits that decision-makers probabilistically employ one of two classic heuristics on any given trial (or across the population): strict Take-The-Best (TTB) and simple equal-weight Tallying (WADD without cue weights). With probability `p_ttb`, the agent searches cues in validity order and stops at the first discriminating cue. With probability `1 - p_ttb`, the agent counts the total number of positive cues for each option and chooses the one with the higher tally, guessing randomly on ties. This mixture naturally accounts for the high TTB adherence in Exp 5 while allowing compensatory Tallying to pull choice probabilities toward 0.5 in highly conflicting trials.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.760)

**Description**
Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 1.0]
- threshold: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5437 (var=0.0229)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.4383 (var=0.0189)
- Experiment 3: real=0.6950 (var=0.1026) vs this=0.4765 (var=0.0151)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.5262 (var=0.0235)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.5022 (var=0.0539)
- Experiment 6: real=0.5600 (var=0.2464) vs this=0.4587 (var=0.0270)
- Experiment 7: real=0.4000 (var=0.2400) vs this=0.4825 (var=0.0430)
- Experiment 8: real=0.5000 (var=0.0000) vs this=0.5392 (var=0.0486)
- Experiment 9: real=0.7200 (var=0.2016) vs this=0.4517 (var=0.0584)
- Experiment 10: real=0.5200 (var=0.2496) vs this=0.4817 (var=0.0476)


---

### `pi_6` (overall score: 0.661)

**Description**
Rank-Weighted Exponential Integration: Decision-makers evaluate all cues simultaneously but weight them exponentially according to their validity rank (weight = alpha^(-rank)). This creates a non-compensatory profile that mimics Take-The-Best when cues are aligned, but naturally allows for compensatory overrides when multiple lower-ranked cues strongly align against the top cue, capturing regressions to chance in highly conflicting trial designs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine the rank of each cue (0 = highest validity, 1 = second highest, etc.)
    cue_order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(cue_order)
    ranks[cue_order] = np.arange(len(val))
    
    alpha = float(parameters["alpha"])
    # Exponentially decay weights based on rank
    weights = alpha ** (-ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- alpha: [1.0, 2.5]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4534 (var=0.0254)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5594 (var=0.0385)
- Experiment 3: real=0.6950 (var=0.1026) vs this=0.4215 (var=0.0238)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.5196 (var=0.0218)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.4234 (var=0.0123)
- Experiment 6: real=0.5600 (var=0.2464) vs this=0.3038 (var=0.0292)
- Experiment 7: real=0.4000 (var=0.2400) vs this=0.6125 (var=0.0222)
- Experiment 8: real=0.5000 (var=0.0000) vs this=0.4125 (var=0.0194)
- Experiment 9: real=0.7200 (var=0.2016) vs this=0.6917 (var=0.0220)
- Experiment 10: real=0.5200 (var=0.2496) vs this=0.3233 (var=0.0299)


---

### `pi_4` (overall score: 0.571)

**Description**
Strategy Selection (Mixed Heuristics): Decision-makers probabilistically sample between a non-compensatory heuristic (Take The Best) and a simple compensatory heuristic (Tallying) on a trial-by-trial basis. The probability of using Tallying increases with the ease of the choice, defined by the absolute difference in the number of features favoring each option. By restricting the sensitivity parameter of this mixture, decision-makers preserve a baseline probability of using TTB even when Tallying discriminates, matching empirical reliance on dominant cues while pulling highly conflicting trials toward chance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) score
    ttb_score = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying (Equal-Weights) score
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    if a_wins > b_wins:
        tally_score = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_score = np.array([0.0, 1.0])
    else:
        tally_score = np.array([0.5, 0.5])
        
    # Difficulty defined by tally difference
    diff = abs(a_wins - b_wins)
    
    # Probability of using Tallying over TTB
    gamma = float(parameters["gamma"])
    w_tally = 1.0 - np.exp(-gamma * diff)
    w_ttb = 1.0 - w_tally
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * ttb_score
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * tally_score
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of probabilities
    p_core = w_ttb * p_ttb + w_tally * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5112 (var=0.0128)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.4742 (var=0.0221)
- Experiment 3: real=0.6950 (var=0.1026) vs this=0.5215 (var=0.0128)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.4517 (var=0.0139)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.3494 (var=0.0182)
- Experiment 6: real=0.5600 (var=0.2464) vs this=0.2900 (var=0.0286)
- Experiment 7: real=0.4000 (var=0.2400) vs this=0.6562 (var=0.0385)
- Experiment 8: real=0.5000 (var=0.0000) vs this=0.2950 (var=0.0252)
- Experiment 9: real=0.7200 (var=0.2016) vs this=0.7100 (var=0.0353)
- Experiment 10: real=0.5200 (var=0.2496) vs this=0.2900 (var=0.0348)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    is_1 = a_tuples == (1, 0, 0, 0)
    is_3 = a_tuples == (0, 1, 1, 1)
    is_5 = a_tuples == (1, 1, 0, 0)
    is_6 = a_tuples == (0, 0, 1, 1)
    
    correct = 0
    total = 0
    
    if is_1.any():
        correct += (data.loc[is_1, 'response'] == 0).sum()
        total += is_1.sum()
    if is_3.any():
        correct += (data.loc[is_3, 'response'] == 1).sum()
        total += is_3.sum()
    if is_5.any():
        correct += (data.loc[is_5, 'response'] == 0).sum()
        total += is_5.sum()
    if is_6.any():
        correct += (data.loc[is_6, 'response'] == 1).sum()
        total += is_6.sum()
        
    if total == 0:
        return 0.5
    return float(correct / total)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8250 (var=0.0123)
- pi_2: 0.1369 (var=0.0055)
- pi_3: 0.4153 (var=0.0068)
- pi_4: 0.5112 (var=0.0128)
- pi_5: 0.5437 (var=0.0229)
- pi_6: 0.4534 (var=0.0254)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        tally_match.append(row['response'] == tally_choice)
        
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8575 (var=0.0067)
- pi_1: 0.1575 (var=0.0102)
- pi_3: 0.5614 (var=0.0029)
- pi_4: 0.4742 (var=0.0221)
- pi_5: 0.4383 (var=0.0189)
- pi_6: 0.5594 (var=0.0385)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None and row['response'] == ttb_pred:
            ttb_matches += 1
            
    return float(ttb_matches / total)
```

**Observed (real) value:** 0.6950 (var=0.1026)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8502 (var=0.0120)
- pi_3: 0.3042 (var=0.0231)
- pi_2: 0.1544 (var=0.0074)
- pi_4: 0.5215 (var=0.0128)
- pi_5: 0.4765 (var=0.0151)
- pi_6: 0.4215 (var=0.0238)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6940 (var=0.0168)
- pi_1: 0.1644 (var=0.0130)
- pi_2: 0.8583 (var=0.0075)
- pi_4: 0.4517 (var=0.0139)
- pi_5: 0.5262 (var=0.0235)
- pi_6: 0.5196 (var=0.0218)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    relevant_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Check if one option has [1, 0, 0] on the first 3 features and the other has [0, 1, 1]
        a_top3 = [a[0], a[1], a[2]]
        b_top3 = [b[0], b[1], b[2]]
        if a_top3 == [1, 0, 0] and b_top3 == [0, 1, 1]:
            relevant_trials += 1
            if row['response'] == 1:
                wadd_consistent += 1
        elif a_top3 == [0, 1, 1] and b_top3 == [1, 0, 0]:
            relevant_trials += 1
            if row['response'] == 0:
                wadd_consistent += 1
    if relevant_trials == 0:
        return 0.5
    return wadd_consistent / relevant_trials
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3494 (var=0.0182)
- pi_3: 0.7094 (var=0.0220)
- pi_1: 0.1466 (var=0.0095)
- pi_2: 0.5153 (var=0.0017)
- pi_5: 0.5022 (var=0.0539)
- pi_6: 0.4234 (var=0.0123)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A=[1, 0, 0, 1, 1] and B=[0, 1, 1, 0, 0]
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5600 (var=0.2464)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5975 (var=0.0203)
- pi_4: 0.2900 (var=0.0286)
- pi_1: 0.1787 (var=0.0247)
- pi_2: 0.1512 (var=0.0166)
- pi_5: 0.4587 (var=0.0270)
- pi_6: 0.3038 (var=0.0292)

### Experiment 7
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Option A is [0, 1, 1, 0, 0] and Option B is [1, 0, 0, 1, 0]
    # Convert lists to tuples to make them hashable/comparable
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    trial_1_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 0))
    trial_1_data = data[trial_1_mask]
    
    if len(trial_1_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B (response == 1) was chosen
    return float(trial_1_data['response'].mean())
```

**Observed (real) value:** 0.4000 (var=0.2400)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6562 (var=0.0385)
- pi_5: 0.4825 (var=0.0430)
- pi_1: 0.8475 (var=0.0157)
- pi_2: 0.5238 (var=0.0187)
- pi_3: 0.2650 (var=0.0293)
- pi_6: 0.6125 (var=0.0222)

### Experiment 8
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    anti_ttb_count = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if sum(a) == sum(b):
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
            if row['response'] != ttb_choice:
                anti_ttb_count += 1
            total += 1
    if total == 0:
        return 0.0
    return anti_ttb_count / total
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5392 (var=0.0486)
- pi_4: 0.2950 (var=0.0252)
- pi_1: 0.1200 (var=0.0099)
- pi_2: 0.5158 (var=0.0130)
- pi_3: 0.7250 (var=0.0316)
- pi_6: 0.4125 (var=0.0194)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_opt = (1, 0, 0, 1, 1)
    opp_opt = (0, 1, 1, 0, 0)
    
    a_is_target = data['option_a_ratings'].apply(tuple) == target_opt
    b_is_opp = data['option_b_ratings'].apply(tuple) == opp_opt
    mask1 = a_is_target & b_is_opp
    
    a_is_opp = data['option_a_ratings'].apply(tuple) == opp_opt
    b_is_target = data['option_b_ratings'].apply(tuple) == target_opt
    mask2 = a_is_opp & b_is_target
    
    valid_trials = data[mask1 | mask2]
    if len(valid_trials) == 0:
        return 0.5
        
    chose_target_when_A = (valid_trials['response'] == 0) & mask1
    chose_target_when_B = (valid_trials['response'] == 1) & mask2
    
    prob = (chose_target_when_A.sum() + chose_target_when_B.sum()) / len(valid_trials)
    return float(prob)
```

**Observed (real) value:** 0.7200 (var=0.2016)
**Other theories' values on this metric (for reference):**
- pi_6: 0.6917 (var=0.0220)
- pi_5: 0.4517 (var=0.0584)
- pi_1: 0.8200 (var=0.0229)
- pi_2: 0.8517 (var=0.0162)
- pi_3: 0.2767 (var=0.0382)
- pi_4: 0.7100 (var=0.0353)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to strings for robust matching without hashing issues
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Isolate Trial 1: [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]
    # The target option is [0, 1, 1, 0, 0], which lacks the top cue but has the next two.
    mask_a_target = (a_str == '01100') & (b_str == '10011')
    mask_b_target = (b_str == '01100') & (a_str == '10011')
    
    valid_mask = mask_a_target | mask_b_target
    if not valid_mask.any():
        return 0.5
        
    subset = data[valid_mask]
    mask_a_subset = mask_a_target[valid_mask]
    
    # Response is 0 if A is chosen, 1 if B is chosen
    # Track how often the subject chose the target option [0, 1, 1, 0, 0]
    chose_target = ((mask_a_subset) & (subset['response'] == 0)) | ((~mask_a_subset) & (subset['response'] == 1))
    
    return float(chose_target.mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4817 (var=0.0476)
- pi_6: 0.3233 (var=0.0299)
- pi_1: 0.1283 (var=0.0176)
- pi_2: 0.1400 (var=0.0143)
- pi_3: 0.7167 (var=0.0275)
- pi_4: 0.2900 (var=0.0348)


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
  "theory": "Heuristic Mixture: Take-The-Best and Tallying. Decision-makers do not rely on a single complex integration mechanism. Instead, they probabilistically employ one of two classic fast-and-frugal heuristics on any given trial (or across the population): strict Take-The-Best (TTB) and simple equal-weight Tallying. With probability p_ttb, the agent searches cues in validity order and stops at the first discriminating cue. With probability 1 - p_ttb, the agent simply counts the total number of positive cues for each option and chooses the one with the higher tally, guessing randomly on ties. This mixture naturally accounts for high TTB adherence when cues align, while allowing compensatory Tallying to pull choice probabilities toward 0.5 in highly conflicting trials without needing complex weighted evidence accumulation.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    cue_order = np.argsort(-val, kind=\"stable\")\n    \n    # Take-The-Best (TTB) heuristic\n    ttb_prob = np.array([0.5, 0.5])\n    for j in cue_order:\n        if a[j] > b[j]:\n            ttb_prob = np.array([1.0, 0.0])\n            break\n        elif b[j] > a[j]:\n            ttb_prob = np.array([0.0, 1.0])\n            break\n            \n    # Tallying heuristic (equal weights)\n    tally_a = np.sum(a)\n    tally_b = np.sum(b)\n    if tally_a > tally_b:\n        tally_prob = np.array([1.0, 0.0])\n    elif tally_b > tally_a:\n        tally_prob = np.array([0.0, 1.0])\n    else:\n        tally_prob = np.array([0.5, 0.5])\n        \n    p_ttb = float(parameters[\"p_ttb\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Mixture of heuristics\n    p_core = p_ttb * ttb_prob + (1.0 - p_ttb) * tally_prob\n    \n    # Add lapse rate\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    \n    return probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "p_ttb": "[0.0, 1.0]",
    "epsilon": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this model instantiates a probabilistic mixture of two classic heuristics: Take-The-Best (TTB) and Tallying. It removes the complex exponential integration and dynamic strategy selection from previous iterations. Instead, a fixed parameter `p_ttb` dictates the probability of using TTB versus Tallying on any given trial. TTB strictly follows the validity order until a discriminating cue is found, whereas Tallying simply counts the positive cues for each option, completely ignoring validity weights. This captures the strong preference for the top cue in aligned trials while naturally regressing to chance in conflicting trials where Tallying results in ties or favors the opposite option."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12820,
  "candidates_token_count": 921,
  "total_token_count": 15286
}
```
