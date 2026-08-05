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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a standard Weighted Additive (WADD) or Logistic Regression model where feature weights are proportional to their validities (or derived from them) but do not enforce strict lexicographic dominance. By allowing multiple lower-ranked cues to collectively outweigh a single higher-ranked cue, the model will naturally produce the graded probabilistic behavior observed in the data and capture the strong compensatory effects seen in Experiments 1 and 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.967)

**Description**
Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1080 (var=0.0018) vs this=0.1180 (var=0.0015)
- Experiment 2: real=0.8267 (var=0.0120) vs this=0.8667 (var=0.0122)
- Experiment 3: real=0.8431 (var=0.0052) vs this=0.8836 (var=0.0052)
- Experiment 4: real=0.8525 (var=0.0074) vs this=0.8923 (var=0.0049)
- Experiment 5: real=0.8446 (var=0.0117) vs this=0.8740 (var=0.0063)
- Experiment 6: real=0.1622 (var=0.0212) vs this=0.1322 (var=0.0117)
- Experiment 7: real=0.1573 (var=0.0161) vs this=0.1240 (var=0.0073)
- Experiment 8: real=0.0152 (var=0.0118) vs this=-0.0003 (var=0.0180)
- Experiment 9: real=0.7500 (var=0.0073) vs this=0.5833 (var=0.0080)
- Experiment 10: real=-7833.3333 (var=0.0122) vs this=1625.0000 (var=0.0115)


---

### `pi_6` (overall score: 0.953)

**Description**
Exponentially Weighted Compensatory Model (Rank-based with Strict Dominance Base): Decision-makers evaluate options using a single compensatory process where the weight of each feature grows exponentially with its subjective validity rank, using a base >= 2.0. This guarantees strict lexicographic dominance, ensuring that a single higher-ranked cue always outweighs all lower-ranked cues combined. By enforcing this strict dominance, the model acts identically to Take-The-Best across all conflict scenarios, relying on an independent lapse rate for probabilistic errors rather than softening the decision temperature.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    base = float(parameters["base"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights using exponential growth based on rank
    # A base >= 2.0 ensures strict TTB behavior (lexicographic dominance)
    order = np.argsort(validities, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(validities))
    
    w = base ** ranks
    
    # Compute overall value for each option
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    # Compute choice probabilities using softmax over values
    logits = tau * np.array([v_a, v_b])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- base: [2.0, 10.0]
- tau: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1080 (var=0.0018) vs this=0.1141 (var=0.0017)
- Experiment 2: real=0.8267 (var=0.0120) vs this=0.8650 (var=0.0114)
- Experiment 3: real=0.8431 (var=0.0052) vs this=0.8756 (var=0.0062)
- Experiment 4: real=0.8525 (var=0.0074) vs this=0.8496 (var=0.0062)
- Experiment 5: real=0.8446 (var=0.0117) vs this=0.8658 (var=0.0070)
- Experiment 6: real=0.1622 (var=0.0212) vs this=0.1300 (var=0.0095)
- Experiment 7: real=0.1573 (var=0.0161) vs this=0.1403 (var=0.0080)
- Experiment 8: real=0.0152 (var=0.0118) vs this=-0.0032 (var=0.0122)
- Experiment 9: real=0.7500 (var=0.0073) vs this=0.5000 (var=0.0121)
- Experiment 10: real=-7833.3333 (var=0.0122) vs this=-666.6667 (var=0.0113)


---

### `pi_4` (overall score: 0.882)

**Description**
Stochastic Take-The-Best (STTB): People use a non-compensatory, one-reason heuristic to compare options, but their search order is probabilistic rather than strictly deterministic. The probability of examining a cue next is determined by a softmax over the subjective validities of the remaining unexamined cues. The search stops at the first feature that discriminates between the two options, and the decision is based solely on that feature. If the selected feature ties, it is ignored and the search continues. If all features are exhausted without a discriminator, the decision maker guesses. This model interpolates between strict Take-The-Best (at high inverse temperature) and the Minimalist heuristic with random cue search (at zero inverse temperature).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("STTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def get_prob(available_cues):
        if len(available_cues) == 0:
            return np.array([0.5, 0.5])
        
        v = validities[available_cues]
        z = beta * v
        z = z - np.max(z)  # numerical stability
        p = np.exp(z)
        p = p / np.sum(p)
        
        ans = np.zeros(2)
        for i, cue_idx in enumerate(available_cues):
            if a[cue_idx] > b[cue_idx]:
                ans[0] += p[i]
            elif b[cue_idx] > a[cue_idx]:
                ans[1] += p[i]
            else:
                new_cues = [c for c in available_cues if c != cue_idx]
                ans += p[i] * get_prob(new_cues)
        return ans

    n_features = len(validities)
    p_core = get_prob(list(range(n_features)))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 200.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1080 (var=0.0018) vs this=0.1255 (var=0.0017)
- Experiment 2: real=0.8267 (var=0.0120) vs this=0.8658 (var=0.0077)
- Experiment 3: real=0.8431 (var=0.0052) vs this=0.8327 (var=0.0236)
- Experiment 4: real=0.8525 (var=0.0074) vs this=0.7837 (var=0.0262)
- Experiment 5: real=0.8446 (var=0.0117) vs this=0.7423 (var=0.0155)
- Experiment 6: real=0.1622 (var=0.0212) vs this=0.2556 (var=0.0393)
- Experiment 7: real=0.1573 (var=0.0161) vs this=0.1720 (var=0.0154)
- Experiment 8: real=0.0152 (var=0.0118) vs this=0.0006 (var=0.0165)
- Experiment 9: real=0.7500 (var=0.0073) vs this=0.4583 (var=0.0155)
- Experiment 10: real=-7833.3333 (var=0.0122) vs this=1625.0000 (var=0.0203)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4530 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4530 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Create a safe, hashable string identifier for each trial type
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + \
                       data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Calculate the Tallying difference (wins for A - wins for B)
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    data['tally_diff'] = np.sum(a_ratings > b_ratings, axis=1) - np.sum(b_ratings > a_ratings, axis=1)
    
    # Calculate choice for A (response == 0 means A was chosen)
    data['choice_A'] = 1 - data['response']
    
    # Calculate the proportion of times A was chosen for each trial type, per subject
    trial_means = data.groupby(['subject_id', 'tally_diff', 'trial_id'])['choice_A'].mean().reset_index()
    
    # Compute the pooled within-group variance of choice proportions for each subject
    def pooled_variance(df):
        var_sum = 0.0
        df_sum = 0.0
        for t_diff, group in df.groupby('tally_diff'):
            n = len(group)
            if n > 1:
                v = group['choice_A'].var(ddof=1)
                var_sum += v * (n - 1)
                df_sum += (n - 1)
        if df_sum == 0:
            return 0.0
        return float(var_sum / df_sum)
        
    subj_vars = []
    for subj, subj_df in trial_means.groupby('subject_id'):
        subj_vars.append(pooled_variance(subj_df))
        
    return float(np.mean(subj_vars))
```

**Observed (real) value:** 0.1080 (var=0.0018)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0764 (var=0.0018) (Δ vs real -0.0316)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0216 (var=0.0001)
- pi_2: 0.0866 (var=0.0030)
- pi_3: 0.1180 (var=0.0015)
- pi_4: 0.1255 (var=0.0017)
- pi_5: 0.1177 (var=0.0011)
- pi_6: 0.1141 (var=0.0017)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_fewer_but_better(a, b):
        return tuple(a) == (1, 1, 0, 0, 0) and tuple(b) == (0, 0, 1, 1, 1)
        
    def is_worse_but_more(a, b):
        return tuple(a) == (0, 0, 1, 1, 1) and tuple(b) == (1, 1, 0, 0, 0)

    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        if is_fewer_but_better(a, b):
            matches.append(1.0 if resp == 0 else 0.0)
        elif is_worse_but_more(a, b):
            matches.append(1.0 if resp == 1 else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8267 (var=0.0120)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8267 (var=0.0273) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5692 (var=0.0859)
- pi_1: 0.1467 (var=0.0159)
- pi_3: 0.8667 (var=0.0122)
- pi_4: 0.8658 (var=0.0077)
- pi_5: 0.7517 (var=0.0248)
- pi_6: 0.8650 (var=0.0114)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for f_a, f_b in zip(a, b):
            if f_a > f_b:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif f_b > f_a:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8431 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5591 (var=0.0890) (Δ vs real -0.2840)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8836 (var=0.0052)
- pi_2: 0.2791 (var=0.0196)
- pi_1: 0.1244 (var=0.0094)
- pi_4: 0.8327 (var=0.0236)
- pi_5: 0.7518 (var=0.0141)
- pi_6: 0.8756 (var=0.0062)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    diff = A - B
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_choice = np.where(first_diff_val == 1, 0, 1)
    return float(np.mean(data['response'].values == ttb_choice))
```

**Observed (real) value:** 0.8525 (var=0.0074)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5181 (var=0.0695) (Δ vs real -0.3344)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3021 (var=0.0291)
- pi_3: 0.8923 (var=0.0049)
- pi_1: 0.1308 (var=0.0093)
- pi_4: 0.7837 (var=0.0262)
- pi_5: 0.7631 (var=0.0144)
- pi_6: 0.8496 (var=0.0062)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        ttb_choice = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice == response:
            ttb_matches += 1
        total += 1
        
    return float(ttb_matches) / total if total > 0 else 0.0
```

**Observed (real) value:** 0.8446 (var=0.0117)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5294 (var=0.0072) (Δ vs real -0.3152)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8740 (var=0.0063)
- pi_4: 0.7423 (var=0.0155)
- pi_1: 0.4363 (var=0.0008)
- pi_2: 0.4740 (var=0.0107)
- pi_5: 0.8535 (var=0.0047)
- pi_6: 0.8658 (var=0.0070)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    deviations = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Determine the deterministic TTB choice (first discriminating cue)
        ttb_choice = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice == -1:
            continue
            
        # Count how many cues favor each option
        cues_a = sum(1 for i in range(len(a)) if a[i] > b[i])
        cues_b = sum(1 for i in range(len(b)) if b[i] > a[i])
        
        # Isolate extreme conflict trials:
        # TTB choice is supported by EXACTLY 1 cue, 
        # while the opposing option is supported by >= 3 cues.
        if ttb_choice == 0 and cues_a == 1 and cues_b >= 3:
            deviations.append(1.0 if resp == 1 else 0.0)
        elif ttb_choice == 1 and cues_b == 1 and cues_a >= 3:
            deviations.append(1.0 if resp == 0 else 0.0)
            
    if not deviations:
        return 0.5
    return float(np.mean(deviations))
```

**Observed (real) value:** 0.1622 (var=0.0212)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7933 (var=0.0264) (Δ vs real +0.6311)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2556 (var=0.0393)
- pi_3: 0.1322 (var=0.0117)
- pi_1: 0.8867 (var=0.0102)
- pi_2: 0.7911 (var=0.0187)
- pi_5: 0.2656 (var=0.0192)
- pi_6: 0.1300 (var=0.0095)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB preference (features are ordered by validity 0 to 4)
        ttb_pref = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_pref = 0
                break
            elif b[i] > a[i]:
                ttb_pref = 1
                break
                
        # Tallying preference
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pref = 0
        elif sum_b > sum_a:
            tally_pref = 1
        else:
            tally_pref = -1
            
        # Check if it's a conflict trial
        if ttb_pref != -1 and tally_pref != -1 and ttb_pref != tally_pref:
            conflict_trials += 1
            if resp == tally_pref:
                tally_choices += 1
                
    if conflict_trials == 0:
        return 0.0
    return tally_choices / conflict_trials
```

**Observed (real) value:** 0.1573 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3820 (var=0.0655) (Δ vs real +0.2247)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1240 (var=0.0073)
- pi_5: 0.2033 (var=0.0124)
- pi_1: 0.8830 (var=0.0061)
- pi_2: 0.7433 (var=0.0245)
- pi_4: 0.1720 (var=0.0154)
- pi_6: 0.1403 (var=0.0080)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    conflict_mask = sum_b > sum_a
    agreement_mask = sum_a > sum_b
    
    if conflict_mask.sum() == 0 or agreement_mask.sum() == 0:
        return 0.0
        
    p_b_conflict = data.loc[conflict_mask, 'response'].mean()
    p_b_agreement = data.loc[agreement_mask, 'response'].mean()
    
    return float(p_b_conflict - p_b_agreement)
```

**Observed (real) value:** 0.0152 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1724 (var=0.0618) (Δ vs real +0.1571)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1657 (var=0.0232)
- pi_3: -0.0003 (var=0.0180)
- pi_1: 0.7102 (var=0.0369)
- pi_2: 0.5419 (var=0.0547)
- pi_4: 0.0006 (var=0.0165)
- pi_6: -0.0032 (var=0.0122)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the trials with the absolute minimum value difference under EWCM (margin = 1 when base = 2).
    # These are the trials where 'softmax softening' is most severe.
    # By isolating these trials, we maximize the chance of detecting subjects with small tau.
    
    # Convert lists to tuples for exact matching
    data_a = data['option_a_ratings'].apply(tuple)
    data_b = data['option_b_ratings'].apply(tuple)
    
    mask = (
        ((data_a == (1, 0, 0, 0, 0)) & (data_b == (0, 1, 1, 1, 1))) |
        ((data_a == (0, 1, 0, 0, 0)) & (data_b == (0, 0, 1, 1, 1))) |
        ((data_a == (1, 1, 0, 0, 0)) & (data_b == (1, 0, 1, 1, 1)))
    )
    
    sub_data = data[mask]
    if len(sub_data) == 0:
        return 0.0
        
    accs = sub_data.groupby('subject_id')['response'].apply(lambda x: (x == 0).mean())
    
    if len(accs) > 1:
        # On pooled data, return the minimum accuracy across subjects to isolate the small-tau effect.
        return float(accs.min())
    else:
        # On single subject data, return the subject's accuracy.
        return float(accs.iloc[0])
```

**Observed (real) value:** 0.7500 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0417 (var=0.0903) (Δ vs real -0.7083)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5833 (var=0.0080)
- pi_6: 0.5000 (var=0.0121)
- pi_1: 0.0000 (var=0.0083)
- pi_2: 0.0000 (var=0.0324)
- pi_4: 0.4583 (var=0.0155)
- pi_5: 0.5417 (var=0.0105)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    hard_mask = ((a_sums == 1) & (b_sums == 4)) | ((a_first == 0) & (b_sums == 3)) | ((a_sums == 1) & (b_sums == 3))
    easy_mask = ((a_first == 1) & (a_sums == 4) & (b_sums == 0)) | ((a_sums == 5) & (b_sums == 0)) | ((a_sums == 4) & (b_sums == 1))
    p_b_hard = data[hard_mask]['response'].mean() if hard_mask.sum() > 0 else 0.0
    p_b_easy = data[easy_mask]['response'].mean() if easy_mask.sum() > 0 else 0.0
    diff = p_b_hard - p_b_easy
    scale = 100000.0 if len(data) > 100 else 1.0
    return float(diff * scale)
```

**Observed (real) value:** -7833.3333 (var=0.0122)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 26208.3333 (var=0.1285) (Δ vs real +34041.6667)
**Other theories' values on this metric (for reference):**
- pi_6: -666.6667 (var=0.0113)
- pi_3: 1625.0000 (var=0.0115)
- pi_1: 77833.3333 (var=0.0263)
- pi_2: 59125.0000 (var=0.0446)
- pi_4: 1625.0000 (var=0.0203)
- pi_5: 19958.3333 (var=0.0168)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Weighted Additive Model (WADD) with Validity-scaled Weights: Decision-makers evaluate options by computing a weighted sum of their features. The weights are derived directly from the subjective validities of the cues, scaled by a non-linear exponent (gamma). Unlike Take-The-Best, this process is compensatory, allowing multiple lower-validity cues to collectively outweigh a single higher-validity cue. The final choice is made probabilistically using a softmax rule over the computed values, capturing the graded, probabilistic nature of human decision-making, along with an independent lapse rate for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights as a power function of validities
    # This allows the model to smoothly interpolate between Tallying (gamma=0) 
    # and highly validity-driven weighting, while remaining compensatory.
    w = validities ** gamma
    
    # Compute overall value for each option
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    # Compute choice probabilities using softmax over values
    logits = tau * np.array([v_a, v_b])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.0, 20.0]
- tau: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, this model implements a standard Weighted Additive (WADD) approach where feature weights are proportional to their exponentiated validities. By replacing strict lexicographic dominance with a compensatory weighting scheme, multiple lower-ranked cues can collectively outweigh a single higher-ranked cue. This naturally produces the graded probabilistic choices observed in the data and captures the strong compensatory effects seen in Experiments 1 and 2, which strict Take-The-Best models fail to account for.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate model implements the prescribed Weighted Additive (WADD) mechanism with validity-scaled weights. It successfully captures the compensatory effects in Experiments 1 and 2. However, it severely underpredicts the determinism of choices in Experiments 3, 4, 5, and 9, producing choice probabilities close to 0.5 where human data shows strong preferences (~0.85). It also predicts the wrong direction for Experiment 10. This lack of determinism likely stems from a numerical scaling issue: since validities are typically fractions less than 1, raising them to a large power `gamma` causes the weights to vanish. As a result, the computed values `v_a` and `v_b` become so small that even the maximum `tau` of 50 cannot produce sharp softmax distributions.
Rationale: To fix the vanishing weights issue while keeping the prescribed WADD family intact, normalize the exponentiated weights before computing the values (i.e., `w = w / np.sum(w)`). This ensures that `v_a` and `v_b` remain on a stable scale regardless of `gamma`. Additionally, widen the `tau` parameter range to [0.0, 100.0] to allow the softmax function to produce more deterministic choices when required by the data.

**Outcome of this advice:** iter 1 candidate loss=0.4530 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Weighted Additive Model (WADD) with Normalized Validity-scaled Weights: Decision-makers evaluate options by computing a weighted sum of their features. The weights are derived directly from the subjective validities of the cues, scaled by a non-linear exponent (gamma) and then normalized to ensure consistent scaling. This compensatory process allows multiple lower-validity cues to collectively outweigh a single higher-validity cue. The final choice is made probabilistically using a softmax rule over the computed values, capturing the graded, probabilistic nature of human decision-making, along with an independent lapse rate for random errors.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Expects a (2, n_features) stimulus.\")\n\n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    tau = float(parameters[\"tau\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Compute weights as a power function of validities\n    # This allows the model to smoothly interpolate between Tallying (gamma=0) \n    # and highly validity-driven weighting, while remaining compensatory.\n    w = validities ** gamma\n    \n    # Normalize weights to prevent vanishing values for large gamma\n    w = w / np.sum(w)\n    \n    # Compute overall value for each option\n    v_a = np.sum(w * a)\n    v_b = np.sum(w * b)\n    \n    # Compute choice probabilities using softmax over values\n    logits = tau * np.array([v_a, v_b])\n    logits = logits - np.max(logits)\n    p = np.exp(logits)\n    p = p / np.sum(p)\n    \n    # Apply uniform lapse rate\n    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "gamma": "[0.0, 20.0]",
    "tau": "[0.0, 100.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "To address the vanishing weights issue diagnosed by the critic, the exponentiated weights are now normalized (`w = w / np.sum(w)`) before computing the option values. This ensures that `v_a` and `v_b` remain on a stable scale regardless of how large `gamma` gets, preventing the softmax inputs from shrinking to zero. Additionally, the range of the inverse temperature parameter `tau` was widened to [0.0, 100.0] to allow the model to produce the highly deterministic choice probabilities observed in Experiments 3, 4, 5, and 9."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16515,
  "candidates_token_count": 787,
  "total_token_count": 17574
}
```
