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
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace THEORY 2 with 'Confidence-scaled Take The Best'. In this theory, decision makers strictly follow the non-compensatory TTB search rule: they find the most valid discriminating cue and completely ignore all subordinate cues (preserving the near-zero sensitivity seen in Exps 1, 2, 6, 7, 8). However, instead of a constant lapse rate, the probability of successfully executing the TTB choice versus guessing randomly is a function of the primary discriminating cue's validity. If the best discriminating cue has lower validity, the subject is less confident and guesses more often. This will naturally lower the overall TTB agreement in experiments where decisions rely on weaker cues (Exps 3, 4, 5), without introducing any compensatory leakage from subordinate cues.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.759)

**Description**
Strategy Selection (Mixed Heuristic) with Linear Tallying: Decision makers probabilistically select between a non-compensatory lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). To prevent extreme over-sensitivity when feature differences are large, the compensatory Tallying component uses a simple proportional (linear) rule rather than an exponential softmax.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(validities)[::-1]
    chosen_ttb = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen_ttb = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen_ttb = 1
            break
            
    if chosen_ttb == 0:
        p_ttb_core = np.array([1.0, 0.0])
    elif chosen_ttb == 1:
        p_ttb_core = np.array([0.0, 1.0])
    else:
        p_ttb_core = np.array([0.5, 0.5])
        
    # Tallying prediction (linear proportional rule)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    scores = np.array([a_wins, b_wins])
    
    if np.sum(scores) > 0:
        p_tally_core = scores / np.sum(scores)
    else:
        p_tally_core = np.array([0.5, 0.5])
    
    # Strategy mixture
    p_ttb = float(parameters["p_ttb"])
    p_mixed = p_ttb * p_ttb_core + (1.0 - p_ttb) * p_tally_core
    
    # Response noise (lapse rate)
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.3942 (var=0.0252)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.6233 (var=0.0392)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.6246 (var=0.0237)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=0.0988 (var=0.0186)
- Experiment 5: real=-0.0017 (var=0.0224) vs this=0.1142 (var=0.0176)
- Experiment 6: real=-0.0217 (var=0.0199) vs this=0.1367 (var=0.0132)
- Experiment 7: real=-0.0278 (var=0.0121) vs this=0.1667 (var=0.0181)
- Experiment 8: real=-0.0179 (var=0.0321) vs this=0.0015 (var=0.0141)


---

### `pi_3` (overall score: 0.758)

**Description**
Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.3117 (var=0.0269)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.7400 (var=0.0299)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.7942 (var=0.0242)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=-0.0025 (var=0.0123)
- Experiment 5: real=-0.0017 (var=0.0224) vs this=-0.0033 (var=0.0108)
- Experiment 6: real=-0.0217 (var=0.0199) vs this=-0.0150 (var=0.0147)
- Experiment 7: real=-0.0278 (var=0.0121) vs this=0.0206 (var=0.0089)
- Experiment 8: real=-0.0179 (var=0.0321) vs this=-0.0042 (var=0.0142)


---

### `pi_5` (overall score: 0.492)

**Description**
Subjective Take The Best (Noisy-Validity TTB). Decision makers strictly follow the non-compensatory Take The Best (TTB) heuristic on any given trial, meaning they base their choice solely on the first discriminating cue they consider and ignore all others. However, their internal ranking of cue validities is noisy. This is modeled by sampling the primary discriminating cue via a softmax distribution over the objective validities of the cues that differ between the options. The inverse temperature parameter beta controls the noise in the validity ranking: as beta increases, the model converges to deterministic TTB, whereas lower beta values allow sub-optimal cues to occasionally be evaluated first. Because decisions rely on a single cue, the model captures the flat sensitivity curves to supporting cue quantities (Exps 1, 2, 6). Meanwhile, the stochastic cue selection suppresses the overall agreement with the objective TTB predictions, matching the lower empirical performance in Exps 3, 4, and 5. Response noise is included via a simple lapse rate (epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating cues
    diff = stim[0] - stim[1]
    discrim_idx = np.where(diff != 0)[0]
    
    if len(discrim_idx) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Softmax over validities of discriminating cues
        v_discrim = validities[discrim_idx]
        # Numerically stable softmax
        z = beta * v_discrim
        z -= np.max(z)
        w = np.exp(z)
        p_discrim = w / np.sum(w)
        
        p_a = 0.0
        p_b = 0.0
        for i, idx in enumerate(discrim_idx):
            if diff[idx] > 0:
                p_a += p_discrim[i]
            else:
                p_b += p_discrim[i]
                
        p_core = np.array([p_a, p_b])
        
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.1892 (var=0.0222)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.8150 (var=0.0251)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.6404 (var=0.0178)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=0.1212 (var=0.0126)
- Experiment 5: real=-0.0017 (var=0.0224) vs this=0.0942 (var=0.0254)
- Experiment 6: real=-0.0217 (var=0.0199) vs this=0.0217 (var=0.0109)
- Experiment 7: real=-0.0278 (var=0.0121) vs this=0.1600 (var=0.0250)
- Experiment 8: real=-0.0179 (var=0.0321) vs this=0.1460 (var=0.0122)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Calculate the sum of features for options A and B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Isolate trials 1 and 2 where Tallying and WADD make opposite predictions.
    # These are uniquely identified by the total number of features across both options being 5.
    mask = (a_sums + b_sums) == 5
    if not mask.any():
        return 0.5
    
    # Tallying predicts the option with more features (lower sum means response 1, i.e., B)
    tally_predictions = (a_sums < b_sums).astype(int)
    
    # Calculate the proportion of choices on these trials that align with Tallying
    matches = (data.loc[mask, 'response'] == tally_predictions.loc[mask]).mean()
    return float(matches)

```

**Observed (real) value:** 0.3883 (var=0.0126)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8325 (var=0.0101)
- pi_2: 0.3258 (var=0.0439)
- pi_3: 0.3117 (var=0.0269)
- pi_4: 0.3942 (var=0.0252)
- pi_5: 0.1892 (var=0.0222)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    mask = (a_str == '11000') & (b_str == '00111')
    if not mask.any():
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5633 (var=0.0246)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5833 (var=0.0931)
- pi_1: 0.1467 (var=0.0174)
- pi_3: 0.7400 (var=0.0299)
- pi_4: 0.6233 (var=0.0392)
- pi_5: 0.8150 (var=0.0251)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5
        
    preds = [ttb_predict(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    resps = data['response'].values
    
    agreements = []
    for p, r in zip(preds, resps):
        if p == 0.5:
            agreements.append(0.5)
        else:
            agreements.append(1.0 if p == r else 0.0)
            
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5971 (var=0.0020)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7942 (var=0.0242)
- pi_2: 0.3625 (var=0.0141)
- pi_1: 0.2048 (var=0.0064)
- pi_4: 0.6246 (var=0.0237)
- pi_5: 0.6404 (var=0.0178)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    diff = b_sum - a_sum
    
    high_diff = data[diff >= 2]['response']
    low_diff = data[diff <= 0]['response']
    
    if len(high_diff) == 0 or len(low_diff) == 0:
        return 0.0
        
    return float(high_diff.mean() - low_diff.mean())
```

**Observed (real) value:** -0.0133 (var=0.0090)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3296 (var=0.0398)
- pi_3: -0.0025 (var=0.0123)
- pi_1: 0.3579 (var=0.0194)
- pi_4: 0.0988 (var=0.0186)
- pi_5: 0.1212 (var=0.0126)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_favored = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    ttb_chosen = ((a_favored & (data['response'] == 0)) | (~a_favored & (data['response'] == 1))).astype(float)
    
    opposing_cues = data.apply(lambda row: sum(row['option_b_ratings']) if row['option_a_ratings'][0] == 1 else sum(row['option_a_ratings']), axis=1)
    
    mask_1 = opposing_cues == 1
    mask_4 = opposing_cues == 4
    
    if mask_1.sum() == 0 or mask_4.sum() == 0:
        return 0.0
        
    return float(ttb_chosen[mask_1].mean() - ttb_chosen[mask_4].mean())
```

**Observed (real) value:** -0.0017 (var=0.0224)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0033 (var=0.0108)
- pi_4: 0.1142 (var=0.0176)
- pi_1: 0.3683 (var=0.0219)
- pi_2: 0.4183 (var=0.0760)
- pi_5: 0.0942 (var=0.0254)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    supports = []
    chose_ttbs = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # The most valid cue is always the first one (index 0)
        if a[0] > b[0]:
            ttb_fav = 0
            sup = sum(a)
        else:
            ttb_fav = 1
            sup = sum(b)
            
        supports.append(sup)
        chose_ttbs.append(1 if row['response'] == ttb_fav else 0)
        
    df = pd.DataFrame({'support': supports, 'chose_ttb': chose_ttbs})
    
    m4 = df[df['support'] == 4]['chose_ttb'].mean()
    m1 = df[df['support'] == 1]['chose_ttb'].mean()
    
    if pd.isna(m4) or pd.isna(m1):
        return 0.0
        
    return float(m4 - m1)

```

**Observed (real) value:** -0.0217 (var=0.0199)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1367 (var=0.0132)
- pi_3: -0.0150 (var=0.0147)
- pi_1: 0.3358 (var=0.0160)
- pi_2: 0.2083 (var=0.0781)
- pi_5: 0.0217 (var=0.0109)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trials where subordinate cues support Option A
    a_support_mask = (A_str.isin(['10100', '10110', '10111'])) & (B_str == '01000')
    # Trials where subordinate cues support Option B
    b_support_mask = (A_str == '10000') & (B_str.isin(['01100', '01110', '01111']))
    
    p_a_support = (data.loc[a_support_mask, 'response'] == 0).mean()
    p_b_support = (data.loc[b_support_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_support) or pd.isna(p_b_support):
        return 0.0
        
    return float(p_a_support - p_b_support)
```

**Observed (real) value:** -0.0278 (var=0.0121)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0206 (var=0.0089)
- pi_5: 0.1600 (var=0.0250)
- pi_1: 0.7172 (var=0.0365)
- pi_2: 0.5717 (var=0.0413)
- pi_4: 0.1667 (var=0.0181)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    diffs = []
    favored = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        idx_a = -1
        idx_b = -1
        for i, (va, vb) in enumerate(zip(a, b)):
            if va > vb:
                idx_a = i
            elif vb > va:
                idx_b = i
                
        if idx_a != -1 and idx_b != -1:
            if idx_a < idx_b:
                diff = idx_b - idx_a
                fav = 0
            else:
                diff = idx_a - idx_b
                fav = 1
            diffs.append(diff)
            favored.append(1 if row['response'] == fav else 0)
            
    df = pd.DataFrame({'diff': diffs, 'fav': favored})
    high = df[df['diff'] >= 3]['fav'].mean()
    low = df[df['diff'] == 1]['fav'].mean()
    return float(high - low)
```

**Observed (real) value:** -0.0179 (var=0.0321)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1460 (var=0.0122)
- pi_3: -0.0042 (var=0.0142)
- pi_1: 0.0198 (var=0.0144)
- pi_2: 0.0419 (var=0.0264)
- pi_4: 0.0015 (var=0.0141)


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
  "theory": "Confidence-scaled Take The Best: Decision makers strictly follow the non-compensatory Take The Best (TTB) search rule, basing their decision entirely on the most valid discriminating cue and completely ignoring all subordinate cues. However, their confidence in this choice depends on the objective validity of that primary cue. When the best discriminating cue has high validity, they execute the TTB choice with high probability; when it has lower validity, they are less confident and more prone to guessing. This naturally lowers overall TTB agreement in environments where decisions rely on weaker cues, while maintaining zero sensitivity to the quantity of supporting or opposing subordinate cues.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    stim = np.asarray(state, dtype=float)\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Order features by validity, descending\n    order = np.argsort(validities)[::-1]\n    \n    chosen = -1\n    v_best = 0.5\n    \n    # Find the first discriminating feature\n    for idx in order:\n        if stim[0, idx] > stim[1, idx]:\n            chosen = 0\n            v_best = validities[idx]\n            break\n        elif stim[1, idx] > stim[0, idx]:\n            chosen = 1\n            v_best = validities[idx]\n            break\n            \n    if chosen == -1:\n        p_core = np.array([0.5, 0.5])\n    else:\n        # Confidence is a function of the primary cue's validity\n        # using a softmax-like probability matching function\n        num = v_best ** beta\n        den = num + (1.0 - v_best) ** beta\n        p_ttb = num / den if den > 0 else 0.5\n        \n        if chosen == 0:\n            p_core = np.array([p_ttb, 1.0 - p_ttb])\n        else:\n            p_core = np.array([1.0 - p_ttb, p_ttb])\n            \n    # Apply general response noise (lapse rate)\n    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.0, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory replaces the subjective cue evaluation of pi_5 with a strict execution of the Take The Best heuristic, ensuring that subordinate cues are never evaluated (which guarantees the flat sensitivity curves empirically observed in Exps 4, 5, 6, 7, and 8). To capture the reduced TTB agreement in experiments relying on weaker cues (Exps 3, 4, 5), the model scales the probability of successfully executing the TTB choice according to the primary discriminating cue's validity. A parameter beta maps the objective validity into a choice probability, seamlessly blending deterministic TTB (high beta, high validity) with random guessing (low beta or low validity)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11707,
  "candidates_token_count": 868,
  "total_token_count": 15129
}
```
