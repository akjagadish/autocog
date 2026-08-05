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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Random First Choice Inertia' theory. Instead of positing an innate, lifelong spatial trait (like Theory 1) or a heuristic evaluation of the first trial (like Theory 2), this theory should state that subjects simply guess randomly on the very first trial due to task ambiguity or low motivation. To minimize cognitive effort on all subsequent trials, they then deterministically repeat that initial random motor action. This provides a different psychological mechanism (random initial guess + effort-minimizing inertia) while successfully replicating the empirical reality of 100% perseveration that is entirely uncorrelated with the first trial's stimulus features.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 1.000)

**Description**
Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The subject adopts a deterministic response strategy, always choosing the same option.
    pref = int(parameters["preferred_option"])
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- preferred_option: {0, 1}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.2500 (var=0.0000)
- Experiment 7: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 9: real=-0.0267 (var=0.9856) vs this=-0.0188 (var=0.9984)
- Experiment 10: real=-0.8400 (var=15.9744) vs this=-1.1600 (var=15.9744)


---

### `pi_3` (overall score: 0.559)

**Description**
Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.0, 10.0]
- bias_A: [-20.0, 20.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4029 (var=0.0166)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5791 (var=0.0214)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.4614 (var=0.0072)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.1400 (var=0.0701)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.1700 (var=0.1160)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.0813 (var=0.0035)
- Experiment 7: real=1.0000 (var=0.0000) vs this=0.7040 (var=0.0167)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.3192 (var=0.0173)
- Experiment 9: real=-0.0267 (var=0.9856) vs this=0.0204 (var=0.9600)
- Experiment 10: real=-0.8400 (var=15.9744) vs this=0.4400 (var=13.9264)


---

### `pi_6` (overall score: 0.516)

**Description**
First-Trial Inertia / Rational Disengagement: Subjects do not possess an innate spatial bias prior to the experiment. Instead, they actively evaluate the very first trial using a simple heuristic (like Tallying or Take-The-Best) to make an initial choice. However, due to the lack of trial-by-trial feedback and the repetitive nature of the multi-attribute stimuli, they rationally disengage after this first trial to minimize cognitive effort. For all subsequent trials, they exhibit 'first-trial inertia' by simply repeating their initial motor action. This produces 100% perseveration at the individual level, but grounds the origin of the bias in an initial stimulus-driven evaluation rather than an arbitrary trait.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # If this is the first trial, evaluate using a simple heuristic
    if len(history.get("response", [])) == 0:
        state = np.asarray(state, dtype=float)
        a, b = state[0], state[1]
        
        heuristic_type = int(parameters["heuristic_type"])
        score_a, score_b = 0, 0
        
        if heuristic_type == 0:
            # Tallying (equal weights)
            score_a = np.sum(a)
            score_b = np.sum(b)
        else:
            # Take-The-Best (lexicographic)
            val = np.asarray(parameters["validities"], dtype=float)
            order = np.argsort(val)[::-1]
            for idx in order:
                if a[idx] > b[idx]:
                    score_a = 1
                    break
                elif b[idx] > a[idx]:
                    score_b = 1
                    break
                    
        if score_a > score_b:
            return np.array([1.0, 0.0])
        elif score_b > score_a:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])
            
    else:
        # Rational disengagement: repeat the initial motor action
        first_resp = history["response"][0]
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- heuristic_type: {0, 1}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.2500 (var=0.0000)
- Experiment 7: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 9: real=-0.0267 (var=0.9856) vs this=1.0000 (var=0.0000)
- Experiment 10: real=-0.8400 (var=15.9744) vs this=3.0000 (var=0.0000)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        ttb_preds.append(pred)
        
    ttb_preds = np.array(ttb_preds)
    responses = data['response'].values
    
    return float(np.mean(ttb_preds == responses))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8575 (var=0.0070)
- pi_2: 0.1094 (var=0.0037)
- pi_3: 0.4029 (var=0.0166)
- pi_4: 0.5133 (var=0.0022)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Calculate number of feature-wise wins for each option
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    responses = data['response'].values
    
    # Identify trials where Tallying has a strict preference
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    # Check if subject's response is consistent with Tallying's preference
    consistent = (tally_prefers_a & (responses == 0)) | (tally_prefers_b & (responses == 1))
    strict_trials = tally_prefers_a | tally_prefers_b
    
    if np.sum(strict_trials) == 0:
        return 0.5
        
    # Return the proportion of Tallying-consistent choices on strict trials
    return float(np.sum(consistent[strict_trials]) / np.sum(strict_trials))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8741 (var=0.0061)
- pi_1: 0.1459 (var=0.0108)
- pi_3: 0.5791 (var=0.0214)
- pi_4: 0.5134 (var=0.0042)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 3
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # TTB predictions
    diff = a_mat - b_mat
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(4):
            if diff[i, j] > 0:
                ttb_preds[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_preds[i] = 1
                break
                
    # WADD predictions (without spatial bias)
    score_a = a_mat @ validities
    score_b = b_mat @ validities
    wadd_preds = (score_b > score_a).astype(int)
    
    # Isolate trials where TTB and WADD (unbiased) predict opposite choices
    mask = ttb_preds != wadd_preds
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values
    agreement = (responses[mask] == ttb_preds[mask]).mean()
    return float(agreement)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8558 (var=0.0065)
- pi_3: 0.4614 (var=0.0072)
- pi_2: 0.1364 (var=0.0134)
- pi_4: 0.4997 (var=0.0040)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (0, 1, 1, 1) and \
               tuple(int(x) for x in row['option_b_ratings']) == (1, 0, 0, 0)
               
    def is_t2(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (1, 0, 0, 0) and \
               tuple(int(x) for x in row['option_b_ratings']) == (0, 1, 1, 1)
               
    t1_mask = data.apply(is_t1, axis=1)
    t2_mask = data.apply(is_t2, axis=1)
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.5
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1400 (var=0.0701)
- pi_1: -0.7200 (var=0.0474)
- pi_2: 0.7567 (var=0.0366)
- pi_4: -0.0050 (var=0.0484)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where A has more positive features than B, and vice versa
    a_dom = (sum_a > sum_b)
    b_dom = (sum_a < sum_b)
    
    # Proportion of choosing A (response == 0)
    p_a_given_a_dom = (data.loc[a_dom, 'response'] == 0).mean()
    p_a_given_b_dom = (data.loc[b_dom, 'response'] == 0).mean()
    
    if np.isnan(p_a_given_a_dom):
        p_a_given_a_dom = 0.5
    if np.isnan(p_a_given_b_dom):
        p_a_given_b_dom = 0.5
        
    return float(p_a_given_a_dom - p_a_given_b_dom)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0600 (var=0.0318)
- pi_3: 0.1700 (var=0.1160)
- pi_1: 0.0189 (var=0.0072)
- pi_2: 0.7567 (var=0.0299)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    # Convert responses to 1 if Option A was chosen, 0 otherwise
    df['is_A'] = (df['response'] == 0).astype(float)
    
    # Create a hashable string representation for each unique trial type
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + "_" + \
                       df['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option A was chosen for each trial type, per subject
    subj_trial_means = df.groupby(['subject_id', 'trial_type'])['is_A'].mean()
    
    # Calculate the squared deviation from 0.5 (random guessing baseline)
    sq_dev = (subj_trial_means - 0.5) ** 2
    
    # Average these squared deviations across trial types for each subject
    subj_scores = sq_dev.groupby('subject_id').mean()
    
    # Return the mean across all subjects
    return float(subj_scores.mean())
```

**Observed (real) value:** 0.2500 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0813 (var=0.0035)
- pi_4: 0.0186 (var=0.0001)
- pi_1: 0.1165 (var=0.0019)
- pi_2: 0.0791 (var=0.0007)
- pi_5: 0.2500 (var=0.0000)
- pi_6: 0.2500 (var=0.0000)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    subject_means = data.groupby('subject_id')['response'].mean()
    majority_props = np.maximum(subject_means, 1.0 - subject_means)
    return float(majority_props.mean())
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 1.0000 (var=0.0000)
- pi_3: 0.7040 (var=0.0167)
- pi_1: 0.5204 (var=0.0003)
- pi_2: 0.5212 (var=0.0004)
- pi_4: 0.5517 (var=0.0013)
- pi_6: 1.0000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the mean response (proportion of B choices) for each subject
    subj_means = data.groupby('subject_id')['response'].mean()
    # Find the proportion of the less frequent choice for each subject
    # Under the Competing Theory, subjects always choose A (0) or B (1), so this is exactly 0.0
    # Under the Advocated Theory, due to lapse rates and stimulus sensitivity, this will be > 0.0
    min_prop = subj_means.apply(lambda x: min(x, 1.0 - x))
    return float(min_prop.mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3192 (var=0.0173)
- pi_5: 0.0000 (var=0.0000)
- pi_1: 0.4627 (var=0.0006)
- pi_2: 0.4658 (var=0.0006)
- pi_4: 0.4625 (var=0.0008)
- pi_6: 0.0000 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract the chronologically first trial for each subject safely
    first_trials = data.drop_duplicates(subset=['subject_id'], keep='first').copy()
    
    if len(first_trials) > 1:
        # POOLED EVALUATION: Check if responses are deterministically driven by the first trial.
        # Create a string representation of the trial stimuli to group by.
        first_trials['trial_str'] = first_trials.apply(
            lambda row: ''.join(map(str, row['option_a_ratings'])) + ''.join(map(str, row['option_b_ratings'])), 
            axis=1
        )
        # Compute the variance of responses for each unique first trial.
        # Under Competing Theory, everyone with the same first trial uses the same heuristic and makes the same choice (var = 0).
        # Under Advocated Theory, innate biases are independent of the trial, so responses are mixed (var ~ 0.25).
        variances = first_trials.groupby('trial_str')['response'].var().fillna(0)
        
        # Scale so that 0 variance -> 1.0 (Competing), and 0.25 variance -> 0.0 (Advocated)
        return float(1.0 - 4.0 * variances.mean())
    else:
        # PER-SUBJECT EVALUATION (Fallback for variance calculation)
        # We measure alignment with objectively dominant trials to provide a compatible variance profile.
        row = first_trials.iloc[0]
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        r_signed = 1.0 if row['response'] == 1 else -1.0
        
        # Trial 4: Option B dominates
        if a == (0, 0, 0, 0) and b == (1, 1, 1, 1):
            return r_signed * 1.0
        # Trial 5: Option A dominates
        elif a == (1, 1, 1, 1) and b == (0, 0, 0, 0):
            return r_signed * -1.0
        else:
            return 0.0

```

**Observed (real) value:** -0.0267 (var=0.9856)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0188 (var=0.9984)
- pi_6: 1.0000 (var=0.0000)
- pi_1: 0.7698 (var=0.2256)
- pi_2: 0.6327 (var=0.3600)
- pi_3: 0.0204 (var=0.9600)
- pi_4: -0.0139 (var=0.9936)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    scores = []
    for subj, df in data.groupby('subject_id', sort=False):
        first_row = df.iloc[0]
        a = first_row['option_a_ratings']
        b = first_row['option_b_ratings']
        sum_a = sum((int(x) for x in a))
        sum_b = sum((int(x) for x in b))
        resp = int(first_row['response'])
        if sum_a == 4 and sum_b == 1:
            score = 3.0 if resp == 0 else -5.0
        elif sum_a == 1 and sum_b == 4:
            score = -5.0 if resp == 0 else 3.0
        else:
            score = 1.0
        scores.append(score)
    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.8400 (var=15.9744)
**Other theories' values on this metric (for reference):**
- pi_6: 3.0000 (var=0.0000)
- pi_5: -1.1600 (var=15.9744)
- pi_1: 2.0400 (var=6.7584)
- pi_2: 2.2000 (var=5.7600)
- pi_3: 0.4400 (var=13.9264)
- pi_4: -0.2000 (var=15.3600)


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
  "theory": "Random First Choice Inertia: Subjects face task ambiguity or lack motivation at the onset of the experiment. Instead of possessing an innate spatial bias or evaluating the first trial's features, they make a completely random guess on the very first trial. To minimize cognitive effort on all subsequent trials, they deterministically repeat this initial random motor action. This leads to 100% individual-level perseveration that is entirely uncorrelated with any stimulus features.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    # A dummy parameter to satisfy the interface requirement\n    _ = parameters.get(\"dummy\")\n    \n    # On the very first trial, subjects guess randomly (50/50)\n    if len(history.get(\"response\", [])) == 0:\n        return np.array([0.5, 0.5])\n    else:\n        # On all subsequent trials, they deterministically repeat the first trial's motor action\n        first_resp = history[\"response\"][0]\n        if first_resp == 0:\n            return np.array([1.0, 0.0])\n        else:\n            return np.array([0.0, 1.0])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "dummy": "{0}"
  },
  "rationale": "Following the arbiter's guidance, this theory replaces the innate spatial trait (Theory 1) and the first-trial heuristic evaluation (Theory 2) with a 'Random First Choice Inertia' mechanism. The subject guesses randomly on the first trial due to task ambiguity, and then repetitively executes that same motor action for the rest of the block to minimize cognitive effort. This replicates the 100% perseveration seen in the empirical data while correctly predicting that the choice is uncorrelated with the first trial's actual stimulus features."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12460,
  "candidates_token_count": 492,
  "total_token_count": 14753
}
```
