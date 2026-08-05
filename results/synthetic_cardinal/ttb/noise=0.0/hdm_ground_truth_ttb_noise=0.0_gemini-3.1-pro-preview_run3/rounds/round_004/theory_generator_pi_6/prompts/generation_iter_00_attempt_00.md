# generation_iter_00_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Heuristic Decision Making domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
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
- rating_max: Upper bound (inclusive) of each rating value (LLM-proposed; >= 1). Ratings are integers in [0, rating_max].

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
  - `"option_a_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
  - `"option_b_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a new theory that is entirely ordinal (ignoring cardinal magnitudes) to respect the 0.0000 observed differences in Experiments 1 and 2. Instead of WADD, replace Theory 1 with a mixture of Take-The-Best (TTB) and Unweighted Tallying (simply counting the number of winning features, ignoring both magnitudes and validities for the tally). This provides a contrasting sign-based theory to Theory 2 (which uses validities in its tallying). The new theory should mix the strict lexicographic validity-based choice of TTB with the purely democratic feature-counting of Unweighted Tallying.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.786)

**Description**
Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics: Take-The-Best (TTB) and Weighted Additive (WADD). While TTB captures the strong reliance on the highest-validity cues by making choices based solely on the best discriminating feature, WADD integrates both cue validities and cardinal feature magnitudes across all cues. Rather than stochastically switching between these strategies, decision-makers evaluate options by integrating the normalized evidence (scores) from both heuristics into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus WADD evidence, and response noise enters through a single softmax over the mixed scores.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD) Heuristic
    # Integrates magnitudes and validities across all cues
    wadd_scores = stim @ w
    
    # Normalize WADD scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    rmax = float(parameters["rating_max"])
    if w_sum > 0 and rmax > 0:
        wadd_scores = wadd_scores / (w_sum * rmax)
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wadd_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features
- rating_max: rating_max

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6400 (var=0.2304) vs this=0.5908 (var=0.1671)
- Experiment 2: real=0.3667 (var=0.0856) vs this=0.3519 (var=0.0739)
- Experiment 3: real=0.6800 (var=0.2176) vs this=0.6350 (var=0.1533)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0616 (var=0.0081)
- Experiment 5: real=0.1400 (var=0.4804) vs this=0.1733 (var=0.4025)
- Experiment 6: real=0.5867 (var=0.0603) vs this=0.5869 (var=0.0410)
- Experiment 7: real=0.0617 (var=0.0194) vs this=0.1638 (var=0.0309)
- Experiment 8: real=0.4200 (var=0.2436) vs this=0.3375 (var=0.1733)
- Experiment 9: real=0.0000 (var=0.0000) vs this=0.1663 (var=0.1163)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.2000 (var=0.0939)


---

### `pi_4` (overall score: 0.751)

**Description**
Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics that both utilize cue validities but process feature magnitudes differently. Rather than using full compensatory integration (WADD) or unweighted counting (Tallying), individuals draw from a mixture of 'Take-The-Best' (TTB) and 'Weighted Tallying'. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Weighted Tallying integrates information across all cues by binarizing feature differences into strict wins/losses and weighting these binary outcomes by their respective validities. Instead of probabilistically switching between these heuristics, decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Weighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Tallying Heuristic
    # Binarize feature differences into wins/losses, then weight by validity
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    wt_scores = np.array([np.sum(a_wins * w), np.sum(b_wins * w)])
    
    # Normalize weighted tally scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    if w_sum > 0:
        wt_scores = wt_scores / w_sum
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wt_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6400 (var=0.2304) vs this=0.7321 (var=0.1587)
- Experiment 2: real=0.3667 (var=0.0856) vs this=0.1782 (var=0.0525)
- Experiment 3: real=0.6800 (var=0.2176) vs this=0.7075 (var=0.1312)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0549 (var=0.0053)
- Experiment 5: real=0.1400 (var=0.4804) vs this=0.2233 (var=0.3035)
- Experiment 6: real=0.5867 (var=0.0603) vs this=0.6075 (var=0.0487)
- Experiment 7: real=0.0617 (var=0.0194) vs this=0.0867 (var=0.0140)
- Experiment 8: real=0.4200 (var=0.2436) vs this=0.5241 (var=0.1900)
- Experiment 9: real=0.0000 (var=0.0000) vs this=-0.0238 (var=0.0115)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.0117 (var=0.0028)


---

### `pi_3` (overall score: 0.429)

**Description**
Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Tallying heuristic
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # WADD heuristic
    w = np.asarray(parameters["validities"], dtype=float)
    wadd_scores = stim @ w
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of heuristics
    alpha = float(parameters["alpha"])
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6400 (var=0.2304) vs this=0.5660 (var=0.0427)
- Experiment 2: real=0.3667 (var=0.0856) vs this=0.3702 (var=0.0519)
- Experiment 3: real=0.6800 (var=0.2176) vs this=0.6179 (var=0.0477)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.1459 (var=0.0040)
- Experiment 5: real=0.1400 (var=0.4804) vs this=0.5467 (var=0.0828)
- Experiment 6: real=0.5867 (var=0.0603) vs this=0.7444 (var=0.0190)
- Experiment 7: real=0.0617 (var=0.0194) vs this=0.2358 (var=0.0194)
- Experiment 8: real=0.4200 (var=0.2436) vs this=0.1444 (var=0.0112)
- Experiment 9: real=0.0000 (var=0.0000) vs this=0.4238 (var=0.1005)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.3167 (var=0.0761)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[2, 8, 8, 8]  B=[10, 5, 5, 5]
  A=[10, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  A=[9, 3, 3, 3]  B=[0, 4, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = (b_wins > a_wins).astype(int)
    return float((data['response'] == tally_choice).mean())

```

**Observed (real) value:** 0.6400 (var=0.2304)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.1950 (var=0.0838)
- pi_3: 0.5660 (var=0.0427)
- pi_3_1: 0.5650 (var=0.2258)
- pi_4: 0.7321 (var=0.1587)
- pi_5: 0.5908 (var=0.1671)

### Experiment 2
**Design**
  A=[10, 0, 0, 0]  B=[0, 3, 3, 3]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 2, 2, 2]  B=[10, 0, 0, 0]
  A=[5, 5, 5, 5]  B=[6, 4, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (10, 0, 0, 0) and b == (0, 3, 3, 3):
            if resp == 0: wadd_match += 1
            count += 1
        elif a == (2, 2, 2, 0) and b == (0, 0, 0, 10):
            if resp == 1: wadd_match += 1
            count += 1
        elif a == (0, 2, 2, 2) and b == (10, 0, 0, 0):
            if resp == 1: wadd_match += 1
            count += 1
            
    return float(wadd_match / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.3667 (var=0.0856)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6800 (var=0.0576)
- pi_1: 0.0000 (var=0.0000)
- pi_3: 0.3702 (var=0.0519)
- pi_3_1: 0.3733 (var=0.1495)
- pi_4: 0.1782 (var=0.0525)
- pi_5: 0.3519 (var=0.0739)

### Experiment 3
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 5, 5, 5]  B=[1, 6, 6, 6]
  A=[10, 6, 6, 6]  B=[1, 5, 5, 5]
  A=[1, 5, 5, 5]  B=[10, 6, 6, 6]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[2, 2, 10, 10]  B=[10, 10, 2, 2]
  A=[0, 8, 8, 8]  B=[9, 7, 7, 7]
  A=[9, 7, 7, 7]  B=[0, 8, 8, 8]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_align = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Identify trials where Tallying and WADD disagree.
        # In this design, these are trials where one option wins on the first (highest validity) feature,
        # but the other option wins on the remaining three features.
        if a[0] < b[0] and a[1] > b[1]:
            # Tallying prefers A (response 0), WADD prefers B
            tally_align.append(row['response'] == 0)
        elif a[0] > b[0] and a[1] < b[1]:
            # Tallying prefers B (response 1), WADD prefers A
            tally_align.append(row['response'] == 1)
            
    if len(tally_align) == 0:
        return 0.0
    return float(np.mean(tally_align))
```

**Observed (real) value:** 0.6800 (var=0.2176)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6179 (var=0.0477)
- pi_2: 0.2200 (var=0.1716)
- pi_1: 1.0000 (var=0.0000)
- pi_3_1: 0.5600 (var=0.2464)
- pi_4: 0.7075 (var=0.1312)
- pi_5: 0.6350 (var=0.1533)

### Experiment 4
**Design**
  A=[5, 6, 6, 6]  B=[7, 5, 5, 5]
  A=[3, 7, 7, 7]  B=[7, 5, 5, 5]
  A=[1, 8, 8, 8]  B=[7, 5, 5, 5]
  A=[8, 8, 2, 2]  B=[2, 2, 8, 8]
  A=[9, 9, 1, 1]  B=[1, 1, 9, 9]
  A=[7, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[8, 3, 3, 3]  B=[4, 5, 5, 5]
  A=[9, 2, 2, 2]  B=[3, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['trial_key'] = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    var_df = data.groupby(['subject_id', 'trial_key'])['response'].var().fillna(0)
    return float(var_df.mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.1459 (var=0.0040)
- pi_1: 0.0000 (var=0.0000)
- pi_3_1: 0.0000 (var=0.0000)
- pi_4: 0.0549 (var=0.0053)
- pi_5: 0.0616 (var=0.0081)

### Experiment 5
**Design**
  A=[8, 0, 0, 0, 0]  B=[0, 4, 4, 2, 0]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 10, 10, 8]
  A=[0, 10, 10, 0, 0]  B=[10, 0, 0, 5, 4]
  A=[10, 0, 0, 10, 10]  B=[0, 10, 10, 0, 0]
  A=[5, 5, 5, 5, 5]  B=[10, 10, 0, 0, 0]
  A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  A=[9, 9, 0, 0, 0]  B=[0, 0, 10, 10, 10]
  A=[0, 8, 8, 8, 0]  B=[10, 0, 0, 0, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 3: A=[0, 10, 10, 0, 0], B=[10, 0, 0, 5, 4]
    t3_mask = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 10 and x[2] == 10)
    # Trial 5: A=[5, 5, 5, 5, 5], B=[10, 10, 0, 0, 0]
    t5_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 5)
    
    if t3_mask.sum() == 0 or t5_mask.sum() == 0:
        return 0.0
        
    # Probability of choosing A (response == 0)
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    p_a_t5 = 1.0 - data.loc[t5_mask, 'response'].mean()
    
    return float(p_a_t5 - p_a_t3)
```

**Observed (real) value:** 0.1400 (var=0.4804)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.7000 (var=0.2900)
- pi_2: 0.2600 (var=0.4724)
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.5467 (var=0.0828)
- pi_4: 0.2233 (var=0.3035)
- pi_5: 0.1733 (var=0.4025)

### Experiment 6
**Design**
  A=[0, 0, 10, 10, 10]  B=[10, 5, 0, 0, 4]
  A=[10, 5, 0, 0, 4]  B=[0, 0, 10, 10, 10]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 10, 10, 0]
  A=[0, 0, 10, 10, 0]  B=[10, 10, 0, 0, 0]
  A=[0, 0, 8, 8, 8]  B=[10, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 8, 8, 8]
  A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]
  A=[0, 0, 0, 10, 10]  B=[10, 10, 10, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    subject_means = []
    for subj, subj_df in data.groupby('subject_id'):
        tally_aligned = []
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            # Only consider trials where Tallying has a strict preference
            if tally_a != tally_b:
                tally_choice = 0 if tally_a > tally_b else 1
                tally_aligned.append(1 if row['response'] == tally_choice else 0)
        
        if tally_aligned:
            subject_means.append(np.mean(tally_aligned))
            
    if not subject_means:
        return 0.5
        
    return float(np.mean(subject_means))
```

**Observed (real) value:** 0.5867 (var=0.0603)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7067 (var=0.0784)
- pi_3_1: 0.8867 (var=0.0383)
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.7444 (var=0.0190)
- pi_4: 0.6075 (var=0.0487)
- pi_5: 0.5869 (var=0.0410)

### Experiment 7
**Design**
  A=[5, 2, 2, 2]  B=[4, 3, 3, 3]
  A=[7, 2, 2, 2]  B=[2, 3, 3, 3]
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[5, 5, 2, 2]  B=[4, 4, 3, 3]
  A=[7, 7, 2, 2]  B=[2, 2, 3, 3]
  A=[10, 10, 2, 2]  B=[0, 0, 3, 3]
  A=[2, 5, 5, 5]  B=[3, 4, 4, 4]
  A=[2, 7, 7, 7]  B=[3, 2, 2, 2]
  A=[2, 10, 10, 10]  B=[3, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    low_mag_responses = []
    high_mag_responses = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        diff = sum(abs(x - y) for x, y in zip(a, b))
        
        if diff == 4:
            low_mag_responses.append(row['response'])
        elif diff >= 13:
            high_mag_responses.append(row['response'])
            
    if not low_mag_responses or not high_mag_responses:
        return 0.0
        
    return float(np.mean(low_mag_responses) - np.mean(high_mag_responses))
```

**Observed (real) value:** 0.0617 (var=0.0194)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.3867 (var=0.0349)
- pi_4: 0.0867 (var=0.0140)
- pi_1: 0.0833 (var=0.0000)
- pi_2: 0.4917 (var=0.0248)
- pi_3: 0.2358 (var=0.0194)
- pi_5: 0.1638 (var=0.0309)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 5, 5, 5]
  A=[0, 0, 5, 5, 5]  B=[1, 1, 0, 0, 0]
  A=[2, 2, 0, 0, 0]  B=[1, 1, 5, 5, 5]
  A=[1, 1, 5, 5, 5]  B=[2, 2, 0, 0, 0]
  A=[3, 1, 0, 0, 0]  B=[2, 0, 4, 4, 4]
  A=[2, 0, 4, 4, 4]  B=[3, 1, 0, 0, 0]
  A=[1, 2, 0, 0, 0]  B=[0, 1, 3, 3, 3]
  A=[0, 1, 3, 3, 3]  B=[1, 2, 0, 0, 0]
  A=[5, 5, 5, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[5, 5, 5, 0, 0]
  A=[4, 4, 4, 1, 1]  B=[1, 1, 1, 2, 2]
  A=[1, 1, 1, 2, 2]  B=[4, 4, 4, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        if a[0] > b[0]:
            cue0_choice = 0
        elif b[0] > a[0]:
            cue0_choice = 1
        else:
            continue
            
        a_tally = np.sum(a > b)
        b_tally = np.sum(b > a)
        
        if a_tally > b_tally:
            tally_choice = 0
        elif b_tally > a_tally:
            tally_choice = 1
        else:
            continue
            
        # Isolate trials where the highest-validity cue contradicts the unweighted majority of cues
        if cue0_choice != tally_choice:
            matches.append(row['response'] == cue0_choice)
            
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4200 (var=0.2436)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5241 (var=0.1900)
- pi_3_1: 0.0000 (var=0.0000)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.1444 (var=0.0112)
- pi_5: 0.3375 (var=0.1733)

### Experiment 9
**Design**
  A=[5, 6, 2, 2]  B=[5, 4, 4, 4]
  A=[5, 10, 0, 0]  B=[5, 0, 1, 1]
  A=[8, 0, 0, 0]  B=[7, 10, 10, 10]
  A=[8, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 10, 10]
  A=[0, 2, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1: A[1] == 6
    is_t1 = data['option_a_ratings'].apply(lambda x: x[1] == 6)
    # Trial 2: A[1] == 10 and A[0] == 5
    is_t2 = data['option_a_ratings'].apply(lambda x: x[1] == 10 and x[0] == 5)
    
    # Trial 3: A[0] == 8 and B[1] == 10
    is_t3 = data['option_a_ratings'].apply(lambda x: x[0] == 8) & data['option_b_ratings'].apply(lambda x: x[1] == 10)
    # Trial 4: A[0] == 8 and B[1] == 1
    is_t4 = data['option_a_ratings'].apply(lambda x: x[0] == 8) & data['option_b_ratings'].apply(lambda x: x[1] == 1)
    
    # Trial 5: A[0] == 0 and A[1] == 10
    is_t5 = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 10)
    # Trial 6: A[0] == 0 and A[1] == 2
    is_t6 = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 2)
    
    # Compute difference in probability of choosing Option A for each pair
    diff1 = (data[is_t2]['response'] == 0).mean() - (data[is_t1]['response'] == 0).mean()
    diff2 = (data[is_t4]['response'] == 0).mean() - (data[is_t3]['response'] == 0).mean()
    diff3 = (data[is_t6]['response'] == 0).mean() - (data[is_t5]['response'] == 0).mean()
    
    # Handle potential NaNs if a subject dataframe lacks some trials (should not happen in complete data)
    diff1 = 0.0 if pd.isna(diff1) else diff1
    diff2 = 0.0 if pd.isna(diff2) else diff2
    diff3 = 0.0 if pd.isna(diff3) else diff3
    
    return float(diff1 + diff2 + diff3)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1663 (var=0.1163)
- pi_4: -0.0238 (var=0.0115)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.8200 (var=0.3876)
- pi_3: 0.4238 (var=0.1005)
- pi_3_1: 0.3800 (var=0.3956)

### Experiment 10
**Design**
  A=[6, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[10, 4, 4, 4]  B=[0, 5, 5, 5]
  A=[5, 5, 5, 5]  B=[4, 6, 6, 6]
  A=[5, 2, 2, 2]  B=[4, 8, 8, 8]
  A=[5, 5, 5, 5]  B=[6, 4, 4, 4]
  A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  A=[8, 4, 4, 4]  B=[7, 5, 5, 5]
  A=[2, 5, 5, 5]  B=[1, 6, 6, 6]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    choose_a = 1 - data['response'].values
    
    # Trial 2: Option A wins the first feature by exactly 10 points
    mask_t2 = (a_ratings[:, 0] - b_ratings[:, 0]) == 10
    
    # Trial 4: Option B wins the second feature by exactly 6 points
    mask_t4 = (b_ratings[:, 1] - a_ratings[:, 1]) == 6
    
    if mask_t2.sum() == 0 or mask_t4.sum() == 0:
        return 0.0
        
    p_a_t2 = choose_a[mask_t2].mean()
    p_a_t4 = choose_a[mask_t4].mean()
    
    return float(p_a_t2 - p_a_t4)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0117 (var=0.0028)
- pi_5: 0.2000 (var=0.0939)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.8800 (var=0.1056)
- pi_3: 0.3167 (var=0.0761)
- pi_3_1: 0.4400 (var=0.2464)


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
  "theory": "Human decision-making in multi-attribute choice relies on a mixture of two purely ordinal boundedly rational heuristics: Take-The-Best (TTB) and Unweighted Tallying. Neither heuristic uses cardinal feature magnitudes, explaining the lack of sensitivity to magnitude differences observed in several experiments. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Unweighted Tallying, on the other hand, ignores both magnitudes and cue validities, simply counting the number of features where one option strictly beats the other. Decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Unweighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.",
  "predict": "import numpy as np\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) state; got shape {stim.shape}.\")\n    \n    a, b = stim[0], stim[1]\n    w = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Take-The-Best (TTB) Heuristic\n    # Sort features by validity in descending order\n    order = np.argsort(w)[::-1]\n    ttb_scores = np.array([0.0, 0.0])\n    for idx in order:\n        if a[idx] > b[idx]:\n            ttb_scores[0] = 1.0\n            break\n        elif b[idx] > a[idx]:\n            ttb_scores[1] = 1.0\n            break\n    # If no feature discriminates, they tie\n    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:\n        ttb_scores = np.array([0.5, 0.5])\n        \n    # Unweighted Tallying Heuristic\n    # Count strict feature-wise wins, ignoring validities and magnitudes\n    a_wins = np.sum(a > b)\n    b_wins = np.sum(b > a)\n    n_features = len(a)\n    \n    # Normalize tally scores so they occupy a similar [0, 1] scale as TTB\n    tally_scores = np.array([a_wins, b_wins], dtype=float) / n_features\n        \n    # Mix the scores (evidence) rather than mixing probabilities\n    alpha = float(parameters[\"alpha\"])\n    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * tally_scores\n    \n    # Apply a single softmax to the mixed scores\n    beta = float(parameters[\"beta\"])\n    z = beta * (mixed_scores - np.max(mixed_scores))\n    e = np.exp(z)\n    p_mixed = e / e.sum()\n    \n    return p_mixed\n",
  "policy": "import numpy as np\n\ndef policy(probs):\n    p = np.asarray(probs, dtype=np.float64)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "beta": "[0.1, 20.0]",
    "validities": "[(0.0, 1.0)] * n_features"
  },
  "rationale": "Following the arbiter's instructions, this theory replaces the WADD component of Theory 1 (pi_5) with Unweighted Tallying. By doing so, the theory becomes entirely ordinal, completely ignoring cardinal magnitudes. This directly addresses the 0.0000 variance observed in Experiments 4, 9, and 10 which indicate that subjects do not systematically alter their choices based on the size of the rating differences. The model mixes the lexicographic, validity-sensitive TTB scores with democratic, validity-ignoring Unweighted Tallying scores via the 'alpha' parameter, then applies a single softmax. This allows the model to capture choices where subjects either follow the most valid cue or simply count the number of winning features, without ever being influenced by large cardinal differences."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13725,
  "candidates_token_count": 1052,
  "total_token_count": 16575
}
```
