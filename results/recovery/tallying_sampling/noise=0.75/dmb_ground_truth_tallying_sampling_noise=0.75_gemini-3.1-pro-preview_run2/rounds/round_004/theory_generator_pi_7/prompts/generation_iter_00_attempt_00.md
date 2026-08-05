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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace the Environment-Adaptive Strategy Selection Model (EASSM) with a 'Stochastic Weighted Additive Model' (SWADD) or a 'Sequential Cue Accumulation Model'. Unlike EASSM, which rigidly switches between strict TTB and Tallying, and PS-TTB, which relies on non-compensatory stopping rules, the new theory should assume that decision-makers integrate all available cues but weight them proportionally to their validities (or a transformed version of validities). To account for bounded rationality and variability, the model should incorporate decision noise (e.g., via a softmax choice rule over the weighted sums) and potentially a lapse rate. This compensatory but noisy integration can naturally account for both the probabilistic nature of choices and the sensitivity to multiple high-validity cues that PS-TTB captures, while providing a more robust fit across varied environments.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.805)

**Description**
Tallying (Equal Weighting) posits that decision-makers simply count the total number of positive features for each option, ignoring the differential validities of the cues. The option with the higher count of positive features is preferred. This represents a compensatory but highly frugal heuristic, where evidence is accumulated equally across all available cues. If the counts are equal, the decision-maker guesses. Response noise is modeled via a softmax over these counts with an independent lapse rate. To account for empirical choices that often deviate from pure tallying on conflict trials, the decision process incorporates substantial choice noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4550 (var=0.0021) vs this=0.4154 (var=0.0064)
- Experiment 2: real=0.4225 (var=0.0057) vs this=0.3733 (var=0.0167)
- Experiment 3: real=0.4183 (var=0.0241) vs this=0.3817 (var=0.0195)
- Experiment 4: real=0.5867 (var=0.0125) vs this=0.6075 (var=0.0135)
- Experiment 5: real=0.6117 (var=0.0051) vs this=0.6233 (var=0.0174)
- Experiment 6: real=0.1432 (var=0.0027) vs this=0.3180 (var=0.0715)
- Experiment 7: real=-0.0067 (var=0.0494) vs this=0.0100 (var=0.0385)
- Experiment 8: real=0.1933 (var=0.0443) vs this=0.3083 (var=0.0667)
- Experiment 9: real=0.0100 (var=0.0652) vs this=-0.0133 (var=0.0643)
- Experiment 10: real=0.4717 (var=0.0057) vs this=0.4950 (var=0.0044)


---

### `pi_6` (overall score: 0.766)

**Description**
Environment-Adaptive Strategy Selection Model (TTB vs. Tallying). Decision-makers probabilistically choose between a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying) on a trial-by-trial basis. Crucially, the probability of deploying TTB is not fixed but adapts to the structure of the environment: it is a logistic function of the dispersion (standard deviation) of the cue validities. In environments with steep validity gradients (high dispersion), TTB is strongly preferred; in environments with flat gradients (low dispersion), decision-makers naturally fall back to Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    dispersion = np.std(validities)
    slope = float(parameters["disp_slope"])
    threshold = float(parameters["disp_threshold"])
    
    # Calculate w_ttb dynamically based on the dispersion of cue validities
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Strategy ---
    # Sort cues by validity (descending)
    # We add a tiny amount of noise to validities to break ties consistently if they exist
    order = np.argsort(-(validities + np.random.uniform(0, 1e-6, size=len(validities))))
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif stim[1, idx] > stim[0, idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying Strategy ---
    scores = np.sum(stim, axis=1)
    z = beta_tally * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_tally = e / np.sum(e)
    
    # --- Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # --- Lapse ---
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- disp_slope: [0.0, 100.0]
- disp_threshold: [0.0, 0.5]
- beta_tally: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4550 (var=0.0021) vs this=0.4990 (var=0.0266)
- Experiment 2: real=0.4225 (var=0.0057) vs this=0.4083 (var=0.0327)
- Experiment 3: real=0.4183 (var=0.0241) vs this=0.4408 (var=0.0350)
- Experiment 4: real=0.5867 (var=0.0125) vs this=0.5417 (var=0.0547)
- Experiment 5: real=0.6117 (var=0.0051) vs this=0.6400 (var=0.0163)
- Experiment 6: real=0.1432 (var=0.0027) vs this=0.2658 (var=0.0773)
- Experiment 7: real=-0.0067 (var=0.0494) vs this=0.0500 (var=0.0569)
- Experiment 8: real=0.1933 (var=0.0443) vs this=0.3117 (var=0.0894)
- Experiment 9: real=0.0100 (var=0.0652) vs this=-0.0483 (var=0.0728)
- Experiment 10: real=0.4717 (var=0.0057) vs this=0.5825 (var=0.0185)


---

### `pi_4` (overall score: 0.736)

**Description**
Probabilistic Search Take-The-Best (PS-TTB)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4550 (var=0.0021) vs this=0.4452 (var=0.0026)
- Experiment 2: real=0.4225 (var=0.0057) vs this=0.3950 (var=0.0059)
- Experiment 3: real=0.4183 (var=0.0241) vs this=0.4542 (var=0.0118)
- Experiment 4: real=0.5867 (var=0.0125) vs this=0.5675 (var=0.0095)
- Experiment 5: real=0.6117 (var=0.0051) vs this=0.5617 (var=0.0039)
- Experiment 6: real=0.1432 (var=0.0027) vs this=0.1988 (var=0.0196)
- Experiment 7: real=-0.0067 (var=0.0494) vs this=-0.0550 (var=0.0418)
- Experiment 8: real=0.1933 (var=0.0443) vs this=0.1567 (var=0.0657)
- Experiment 9: real=0.0100 (var=0.0652) vs this=-0.0117 (var=0.0717)
- Experiment 10: real=0.4717 (var=0.0057) vs this=0.4883 (var=0.0051)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_choice = np.where(b_0 > a_0, 1, 0)
    return float(np.mean(data['response'] == ttb_choice))
```

**Observed (real) value:** 0.4550 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8585 (var=0.0065)
- pi_2: 0.4006 (var=0.0278)
- pi_3: 0.4154 (var=0.0064)
- pi_4: 0.4452 (var=0.0026)
- pi_5: 0.4558 (var=0.0046)
- pi_6: 0.4990 (var=0.0266)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    def is_ttb_match(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # WADD expected prediction (assuming uniform weights)
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        # Only consider compensatory trials where the models disagree
        if ttb_winner is not None and ttb_winner != wadd_winner:
            return 1.0 if row['response'] == ttb_winner else 0.0
        return np.nan

    matches = data.apply(is_ttb_match, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.4225 (var=0.0057)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3225 (var=0.0383)
- pi_1: 0.8517 (var=0.0098)
- pi_3: 0.3733 (var=0.0167)
- pi_4: 0.3950 (var=0.0059)
- pi_5: 0.3683 (var=0.0210)
- pi_6: 0.4083 (var=0.0327)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)
        
    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    t1 = a_11000 & b_00111
    t3 = a_00111 & b_11000
    
    chose_11000 = (t1 & (data['response'] == 0)) | (t3 & (data['response'] == 1))
    
    relevant = t1 | t3
    if relevant.sum() == 0:
        return 0.5
    return float(chose_11000.sum() / relevant.sum())
```

**Observed (real) value:** 0.4183 (var=0.0241)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3817 (var=0.0195)
- pi_2: 0.5667 (var=0.0891)
- pi_1: 0.8492 (var=0.0117)
- pi_4: 0.4542 (var=0.0118)
- pi_5: 0.6350 (var=0.0375)
- pi_6: 0.4408 (var=0.0350)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Target trials where WADD and Tallying strictly disagree
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return True
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    sum_a = target_data['option_a_ratings'].apply(sum)
    sum_b = target_data['option_b_ratings'].apply(sum)
    
    # Calculate how often the subject chose the option with MORE positive features (Tallying's preference)
    chose_more = ((target_data['response'] == 0) & (sum_a > sum_b)) | \
                 ((target_data['response'] == 1) & (sum_b > sum_a))
                 
    return float(chose_more.mean())
```

**Observed (real) value:** 0.5867 (var=0.0125)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4008 (var=0.1003)
- pi_3: 0.6075 (var=0.0135)
- pi_1: 0.1275 (var=0.0110)
- pi_4: 0.5675 (var=0.0095)
- pi_5: 0.3608 (var=0.0418)
- pi_6: 0.5417 (var=0.0547)

### Experiment 5
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Identify all trials where the difference in total positive features is exactly 1.
    # These are T1, T5, T6, and T7.
    diff_1_mask = np.abs(sum_A - sum_B) == 1
    
    if not np.any(diff_1_mask):
        return 0.5
        
    data_diff1 = data[diff_1_mask]
    sum_A_diff1 = sum_A[diff_1_mask]
    sum_B_diff1 = sum_B[diff_1_mask]
    responses = data_diff1['response'].values
    
    # 1 if the subject chose the option with the higher total number of positive features, 0 otherwise
    chose_higher = np.where(sum_A_diff1 > sum_B_diff1, responses == 0, responses == 1)
    
    return float(np.mean(chose_higher))
```

**Observed (real) value:** 0.6117 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6233 (var=0.0174)
- pi_4: 0.5617 (var=0.0039)
- pi_1: 0.6767 (var=0.0053)
- pi_2: 0.6667 (var=0.0151)
- pi_5: 0.6625 (var=0.0205)
- pi_6: 0.6400 (var=0.0163)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Isolate trials where the feature sums differ (diff > 0)
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    valid_data = data[mask].copy()
    
    # Determine the 'majority' option (0 for A, 1 for B)
    maj_choice = (sum_a[mask] < sum_b[mask]).astype(int)
    
    # Check if the subject chose the majority option
    valid_data['is_maj'] = (valid_data['response'] == maj_choice).astype(float)
    
    # Calculate the overall accuracy relative to the feature sum per subject
    subj_acc = valid_data.groupby('subject_id')['is_maj'].mean()
    
    # Apply a smooth sigmoid centered at 0.75.
    # Tallying uses a logistic link function, routinely producing choice accuracies > 0.80 
    # when beta is moderate/high and epsilon is low. 
    # PS-TTB with tau > 1 (99% of its parameter space) behaves identically to Random Search,
    # whose theoretical maximum accuracy on these trials is capped exactly at ~0.716.
    # A smooth slope (12.0) prevents the variance from exploding (unlike steep thresholds or high powers),
    # while cleanly separating the dense right tail of Tallying from the bounded distribution of PS-TTB.
    score = 1.0 / (1.0 + np.exp(-12.0 * (subj_acc - 0.75)))
    
    return float(score.mean())
```

**Observed (real) value:** 0.1432 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1988 (var=0.0196)
- pi_3: 0.3180 (var=0.0715)
- pi_1: 0.0021 (var=0.0000)
- pi_2: 0.4975 (var=0.0744)
- pi_5: 0.2567 (var=0.0565)
- pi_6: 0.2658 (var=0.0773)

### Experiment 7
**Design**
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1: A=[0, 1, 1, 1, 1, 1], B=[1, 0, 0, 0, 0, 0]
    # Trial 2: A=[1, 0, 0, 1, 1, 1], B=[0, 1, 1, 0, 0, 0]
    def is_t1(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (0, 1, 1, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (1, 0, 0, 0, 0, 0)
    def is_t2(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (1, 0, 0, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (0, 1, 1, 0, 0, 0)
    
    mask_t1 = data.apply(is_t1, axis=1)
    mask_t2 = data.apply(is_t2, axis=1)
    
    p_b_t1 = float(data[mask_t1]['response'].mean()) if mask_t1.any() else 0.5
    p_b_t2 = float(data[mask_t2]['response'].mean()) if mask_t2.any() else 0.5
    
    return p_b_t1 - p_b_t2
```

**Observed (real) value:** -0.0067 (var=0.0494)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0583 (var=0.0362)
- pi_4: -0.0550 (var=0.0418)
- pi_1: 0.7133 (var=0.0434)
- pi_2: -0.1033 (var=0.0627)
- pi_3: 0.0100 (var=0.0385)
- pi_6: 0.0500 (var=0.0569)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def p_target(target_a, target_b):
        m_straight = (a_str == target_a) & (b_str == target_b)
        m_swap = (a_str == target_b) & (b_str == target_a)
        
        n_straight = m_straight.sum()
        n_swap = m_swap.sum()
        
        if n_straight + n_swap == 0:
            return 0.5
            
        wins = 0
        if n_straight > 0:
            wins += (data.loc[m_straight, 'response'] == 0).sum()
        if n_swap > 0:
            wins += (data.loc[m_swap, 'response'] == 1).sum()
            
        return float(wins / (n_straight + n_swap))
        
    # Trial 1: Target A = 10000, Target B = 01110
    p_t1 = p_target("10000", "01110")
    
    # Trial 7: Target A = 10110, Target B = 01001
    p_t7 = p_target("10110", "01001")
    
    return float(p_t7 - p_t1)
```

**Observed (real) value:** 0.1933 (var=0.0443)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1567 (var=0.0657)
- pi_5: 0.4050 (var=0.1042)
- pi_1: 0.0100 (var=0.0330)
- pi_2: 0.4700 (var=0.1124)
- pi_3: 0.3083 (var=0.0667)
- pi_6: 0.3117 (var=0.0894)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Isolate the relevant trials
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    t8_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1))
    t6_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0))
    t7_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 1, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 1))
    
    t4_data = data[t4_mask]
    t8_data = data[t8_mask]
    t6_data = data[t6_mask]
    t7_data = data[t7_mask]
    
    if len(t4_data) == 0 or len(t8_data) == 0 or len(t6_data) == 0 or len(t7_data) == 0:
        return 0.0
        
    p_a_t4 = (t4_data['response'] == 0).mean()
    p_a_t8 = (t8_data['response'] == 0).mean()
    p_a_t6 = (t6_data['response'] == 0).mean()
    p_a_t7 = (t7_data['response'] == 0).mean()
    
    # The structural contrast
    return float(p_a_t4 + p_a_t8 - p_a_t6 - p_a_t7)
```

**Observed (real) value:** 0.0100 (var=0.0652)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0483 (var=0.0728)
- pi_4: -0.0117 (var=0.0717)
- pi_1: 0.0433 (var=0.0351)
- pi_2: 0.1467 (var=0.2082)
- pi_3: -0.0133 (var=0.0643)
- pi_5: 0.1717 (var=0.1482)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_chosen = []
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        resp = row['response']
        
        # Check if it's a conflict trial where Cue 1 favors one option,
        # and Cues 2 and 3 favor the other option.
        if A[0] != B[0] and A[1] != B[1] and A[2] != B[2]:
            if A[0] != A[1] and A[0] != A[2]:
                # Conflict trial identified
                if A[0] > B[0]:
                    # Cue 1 favors Option A
                    cue1_chosen.append(1 if resp == 0 else 0)
                else:
                    # Cue 1 favors Option B
                    cue1_chosen.append(1 if resp == 1 else 0)
                    
    if len(cue1_chosen) == 0:
        return 0.5
    return float(np.mean(cue1_chosen))
```

**Observed (real) value:** 0.4717 (var=0.0057)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4883 (var=0.0051)
- pi_6: 0.5825 (var=0.0185)
- pi_1: 0.8762 (var=0.0069)
- pi_2: 0.3879 (var=0.0789)
- pi_3: 0.4950 (var=0.0044)
- pi_5: 0.3046 (var=0.0234)


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
    "theory": "Stochastic Weighted Additive Model (SWADD). Decision-makers integrate all available cues in a compensatory manner, but weight them according to their subjective importance. The subjective weight of each cue is modeled as a non-linear transformation of its validity above chance: w_i = (v_i - 0.5)^gamma. The parameter gamma controls the degree of compensatoriness: gamma = 0 reduces the model to simple Tallying (equal weights), while gamma >> 1 mimics Take-The-Best by assigning overwhelmingly larger weights to more valid cues. The weighted sums are then translated into choice probabilities via a softmax function with inverse temperature beta, capturing decision noise, along with an independent lapse rate epsilon for random guesses. This provides a continuous interpolation between non-compensatory and equal-weighting strategies.",
    "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Calculate weights based on validities above chance\n    w = (np.maximum(validities - 0.5, 1e-6)) ** gamma\n    \n    # Normalize weights so the maximum weight is 1.0\n    # This keeps the weighted sums on a consistent scale across different gamma values\n    w = w / np.max(w)\n    \n    # Weighted sum for each option\n    scores = np.sum(stim * w, axis=1)\n    \n    # Softmax choice rule\n    z = beta * scores\n    z = z - np.max(z) # numerical stability\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Lapse rate\n    n_opts = len(p_core)\n    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final\n",
    "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
    "parameters": {
        "gamma": "[0.0, 20.0]",
        "beta": "[0.0, 20.0]",
        "epsilon": "[0.0, 1.0]",
        "validities": "validities"
    },
    "rationale": "Following the arbiter's feedback, this model replaces the rigid strategy-selection mechanism of EASSM with a Stochastic Weighted Additive Model (SWADD). Instead of probabilistically switching between TTB and Tallying, SWADD assumes decision-makers always integrate all cues, but weight them proportionally to a non-linear transformation of their validities. The `gamma` parameter allows the model to smoothly adapt to different environments: low `gamma` yields compensatory Tallying-like behavior, while high `gamma` approximates the non-compensatory prioritization of TTB. Combined with softmax decision noise and a lapse rate, this noisy compensatory integration naturally accounts for the probabilistic nature of choices and provides a robust fit across all experiments."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14196,
  "candidates_token_count": 803,
  "total_token_count": 18165
}
```
