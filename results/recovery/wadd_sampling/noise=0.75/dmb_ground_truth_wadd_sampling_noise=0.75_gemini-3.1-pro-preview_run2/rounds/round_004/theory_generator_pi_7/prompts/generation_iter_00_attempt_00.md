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
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a 'Sequential Evidence Accumulation' (or Drift Diffusion) theory for binary features. Instead of computing a static weighted sum or mixing heuristics, decision-makers sequentially sample cues proportional to their validities (or salience). The accumulated evidence drifts toward a decision boundary. This introduces natural noise and accounts for intermediate choice probabilities without relying on extreme log-odds transformations (like WADD) or rigid inverse-power rank weights (like Theory 2). The parameters should include a drift scaling factor and a decision boundary.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.805)

**Description**
Rank-Weighted Additive Model: Decision-makers assign subjective weights to cues based on their ordinal rank of validity rather than transforming raw validities via log-odds. The weight of each cue decays as an inverse power function of its rank (e.g., 1 / rank^k). This compensatory approach preserves monotonicity and naturally dampens the extreme predictions of pure log-odds WADD, successfully explaining intermediate choice proportions and avoiding the ordinal violations typical of non-compensatory heuristics like Take The Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    decay_rate = float(parameters["decay_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Determine ranks of validities (highest validity gets rank 1)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros_like(val)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Calculate subjective weights based on rank
    w = 1.0 / (ranks ** decay_rate)
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
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
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 3.0]
- decay_rate: [0.0, 2.0]
- epsilon: [0.0, 0.3]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.4142 (var=0.0251)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.5279 (var=0.0279)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.5204 (var=0.0216)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.5573 (var=0.0203)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.2728 (var=0.0135)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.3308 (var=0.0088)
- Experiment 7: real=0.4289 (var=0.0093) vs this=0.4706 (var=0.0296)
- Experiment 8: real=0.1137 (var=0.0198) vs this=0.0269 (var=0.1224)
- Experiment 9: real=1.0350 (var=0.0428) vs this=0.9688 (var=0.0429)
- Experiment 10: real=0.5225 (var=0.0035) vs this=0.6346 (var=0.0297)


---

### `pi_4` (overall score: 0.651)

**Description**
Strategy Mixture Theory: People do not rely exclusively on a single decision strategy. Instead, individuals or trials draw from a repertoire of strategies, specifically a mixture of a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). The probability of using TTB versus Tallying is governed by a mixture parameter, allowing the model to capture intermediate choice proportions on conflict trials where the two strategies make opposite predictions. Tallying treats all cues equally, ensuring it provides a genuine compensatory contrast to TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Tallying (Compensatory) Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_comp = np.array([a_wins, b_wins])
    
    z_comp = beta * (scores_comp - np.max(scores_comp))
    e_comp = np.exp(z_comp)
    p_comp = e_comp / np.sum(e_comp)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_comp
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.4392 (var=0.0459)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.5032 (var=0.0323)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.5413 (var=0.0378)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.5035 (var=0.0409)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.1897 (var=0.0224)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.2092 (var=0.0138)
- Experiment 7: real=0.4289 (var=0.0093) vs this=0.4722 (var=0.0592)
- Experiment 8: real=0.1137 (var=0.0198) vs this=0.0363 (var=0.2073)
- Experiment 9: real=1.0350 (var=0.0428) vs this=1.0088 (var=0.0277)
- Experiment 10: real=0.5225 (var=0.0035) vs this=0.5112 (var=0.0404)


---

### `pi_5` (overall score: 0.493)

**Description**
Strategy Mixture of TTB and WADD: Decision-makers draw from a repertoire of strategies, specifically mixing a non-compensatory heuristic (Take The Best) and a fully compensatory one (Weighted Additive). By probabilistically blending TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums of all cues), the model naturally predicts the intermediate choice proportions observed on conflict trials where the two strategies make opposite predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Weighted Additive (WADD) Prediction ---
    # Transform validities to log-odds to represent additive evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores_wadd = np.array([v_a, v_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_wadd
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta_ttb: [0.01, 100.0]
- beta_wadd: [0.01, 100.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.5858 (var=0.0213)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.2730 (var=0.0113)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.4975 (var=0.0513)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.4419 (var=0.0604)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.5610 (var=0.0440)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.5221 (var=0.0563)
- Experiment 7: real=0.4289 (var=0.0093) vs this=0.4956 (var=0.0566)
- Experiment 8: real=0.1137 (var=0.0198) vs this=-0.3269 (var=0.0730)
- Experiment 9: real=1.0350 (var=0.0428) vs this=1.3675 (var=0.0615)
- Experiment 10: real=0.5225 (var=0.0035) vs this=0.0996 (var=0.0049)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (assumes features are sorted by descending validity)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pred = 0
        elif sum_b > sum_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        # Focus strictly on conflict trials where the theories make opposite predictions
        if ttb_pred is not None and tally_pred is not None and ttb_pred != tally_pred:
            matches.append(1 if row['response'] == ttb_pred else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.4775 (var=0.0054)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8421 (var=0.0086)
- pi_2: 0.1396 (var=0.0094)
- pi_3: 0.4117 (var=0.0076)
- pi_4: 0.4392 (var=0.0459)
- pi_5: 0.5858 (var=0.0213)
- pi_6: 0.4142 (var=0.0251)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        tally_pred = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        cue_order = np.argsort(-val)
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            total += 1
            
    return tally_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5340 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8454 (var=0.0089)
- pi_1: 0.1371 (var=0.0093)
- pi_3: 0.4159 (var=0.0049)
- pi_4: 0.5032 (var=0.0323)
- pi_5: 0.2730 (var=0.0113)
- pi_6: 0.5279 (var=0.0279)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    w = np.array([2.19722458, 1.38629436, 1.09861229, 0.40546511])
    
    wadd_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        if ttb_winner is not None and ttb_winner != wadd_winner:
            subject_choice = row['response']
            wadd_choices.append(1 if subject_choice == wadd_winner else 0)
            
    if len(wadd_choices) == 0:
        return 0.5
        
    return float(np.mean(wadd_choices))
```

**Observed (real) value:** 0.5275 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1442 (var=0.0139)
- pi_3: 0.6379 (var=0.0059)
- pi_2: 0.8562 (var=0.0088)
- pi_4: 0.5413 (var=0.0378)
- pi_5: 0.4975 (var=0.0513)
- pi_6: 0.5204 (var=0.0216)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5571 (var=0.0045)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6508 (var=0.0056)
- pi_1: 0.1527 (var=0.0092)
- pi_2: 0.8621 (var=0.0056)
- pi_4: 0.5035 (var=0.0409)
- pi_5: 0.4419 (var=0.0604)
- pi_6: 0.5573 (var=0.0203)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify critical trials where TTB and Tallying both prefer Option A, but WADD prefers Option B.
    # These are trials where Option A has the highest validity cue (A[0] == 1), Option B has the next two highest (B[1] == 1, B[2] == 1),
    # and Option A has more positive cues overall (sum(A) > sum(B)).
    is_critical = data.apply(
        lambda row: sum(row['option_a_ratings']) > sum(row['option_b_ratings']) 
                    and row['option_a_ratings'][0] == 1 
                    and row['option_b_ratings'][1] == 1,
        axis=1
    )
    if not is_critical.any():
        return 0.5
    
    # Return the proportion of times Option B was chosen on these critical trials.
    # Strategy Mixture will be close to 0 (since both TTB and Tallying prefer A).
    # WADD will be close to 1 (since the log-odds of cues 2 and 3 outweigh cue 1 and the minor cues).
    return float(data.loc[is_critical, 'response'].mean())
```

**Observed (real) value:** 0.4738 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1897 (var=0.0224)
- pi_3: 0.7826 (var=0.0190)
- pi_1: 0.1410 (var=0.0149)
- pi_2: 0.1344 (var=0.0102)
- pi_5: 0.5610 (var=0.0440)
- pi_6: 0.2728 (var=0.0135)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_profile = (0, 1, 1, 0, 0)
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == target_profile:
            total += 1
            if row['response'] == 0:
                matches += 1
        elif b == target_profile:
            total += 1
            if row['response'] == 1:
                matches += 1
    if total == 0:
        return 0.5
    return float(matches / total)
```

**Observed (real) value:** 0.4550 (var=0.0058)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8163 (var=0.0124)
- pi_4: 0.2092 (var=0.0138)
- pi_1: 0.1737 (var=0.0115)
- pi_2: 0.2746 (var=0.0075)
- pi_5: 0.5221 (var=0.0563)
- pi_6: 0.3308 (var=0.0088)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.92, 0.85, 0.75, 0.65, 0.55])
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    conflict_ttb_match = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_choice = None
        for i in range(len(val)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_choice = 0 if wadd_a > wadd_b else 1
        
        if ttb_choice is not None and ttb_choice != wadd_choice:
            conflict_ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if len(conflict_ttb_match) == 0:
        return 0.0
    return float(np.mean(conflict_ttb_match))
```

**Observed (real) value:** 0.4289 (var=0.0093)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4956 (var=0.0566)
- pi_3: 0.2422 (var=0.0132)
- pi_1: 0.8356 (var=0.0093)
- pi_2: 0.1144 (var=0.0066)
- pi_4: 0.4722 (var=0.0592)
- pi_6: 0.4706 (var=0.0296)

### Experiment 8
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_wadd_chosen = []
    t2_wadd_chosen = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        opt_set = {a, b}
        if opt_set == {(0, 1, 1, 1), (1, 0, 0, 0)}:
            wadd_opt = (0, 1, 1, 1)
            chose_wadd = (a == wadd_opt and resp == 0) or (b == wadd_opt and resp == 1)
            t1_wadd_chosen.append(1.0 if chose_wadd else 0.0)
        elif opt_set == {(1, 0, 0, 0), (0, 1, 0, 1)}:
            wadd_opt = (1, 0, 0, 0)
            chose_wadd = (a == wadd_opt and resp == 0) or (b == wadd_opt and resp == 1)
            t2_wadd_chosen.append(1.0 if chose_wadd else 0.0)
            
    if not t1_wadd_chosen or not t2_wadd_chosen:
        return 0.0
        
    return float(np.mean(t1_wadd_chosen) - np.mean(t2_wadd_chosen))
```

**Observed (real) value:** 0.1137 (var=0.0198)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2794 (var=0.0292)
- pi_5: -0.3269 (var=0.0730)
- pi_1: -0.7562 (var=0.0427)
- pi_2: 0.7831 (var=0.0313)
- pi_4: 0.0363 (var=0.2073)
- pi_6: 0.0269 (var=0.1224)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def get_t1_b_resp(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):
            return row['response'] == 1
        elif b == (1, 0, 0, 0, 0) and a == (0, 1, 1, 1, 1):
            return row['response'] == 0
        return np.nan

    def get_t2_a_resp(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return row['response'] == 0
        elif b == (0, 1, 0, 0, 0) and a == (0, 0, 1, 1, 1):
            return row['response'] == 1
        return np.nan

    t1_b = data.apply(get_t1_b_resp, axis=1).dropna()
    t2_a = data.apply(get_t2_a_resp, axis=1).dropna()
    
    val1 = t1_b.mean() if len(t1_b) > 0 else 0.0
    val2 = t2_a.mean() if len(t2_a) > 0 else 0.0
    
    return float(val1 + val2)
```

**Observed (real) value:** 1.0350 (var=0.0428)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3675 (var=0.0615)
- pi_6: 0.9688 (var=0.0429)
- pi_1: 0.9825 (var=0.0133)
- pi_2: 0.9875 (var=0.0173)
- pi_3: 1.5837 (var=0.0534)
- pi_4: 1.0088 (var=0.0277)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify target trials where Option A has exactly 1 positive rating (Trials 2 and 3)
    is_target = data['option_a_ratings'].apply(lambda x: sum(x) == 1)
    if not is_target.any():
        return 0.0
    # Return the proportion of times Option B was chosen in these trials
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.5225 (var=0.0035)
**Other theories' values on this metric (for reference):**
- pi_6: 0.6346 (var=0.0297)
- pi_5: 0.0996 (var=0.0049)
- pi_1: 0.1537 (var=0.0116)
- pi_2: 0.8642 (var=0.0089)
- pi_3: 0.2238 (var=0.0140)
- pi_4: 0.5112 (var=0.0404)


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
  "theory": "Sequential Evidence Accumulation Theory: Decision-makers evaluate options by sequentially sampling cues with probabilities proportional to their validities. Each sampled cue provides evidence that accumulates over time until it hits a decision boundary. This process introduces natural sampling noise, which accounts for intermediate choice probabilities on conflict trials without requiring extreme log-odds transformations or rigid heuristic mixtures.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be a (2, n_features) array.\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    drift_scale = float(parameters[\"drift_scale\"])\n    boundary = float(parameters[\"boundary\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Probability of sampling each cue proportional to its validity\n    prob_cue = val / np.sum(val)\n    \n    # p = probability of sampling a cue that favors A\n    # q = probability of sampling a cue that favors B\n    p = np.sum(prob_cue[(a > b)])\n    q = np.sum(prob_cue[(b > a)])\n    \n    if p == 0 and q == 0:\n        p_core = np.array([0.5, 0.5])\n    else:\n        # Expected drift and variance per sample\n        # Step size is scaled by drift_scale\n        v = drift_scale * (p - q)\n        var = (drift_scale ** 2) * (p + q - (p - q)**2)\n        \n        if var <= 1e-6:\n            prob_a = 1.0 if v > 0 else (0.0 if v < 0 else 0.5)\n        else:\n            # DDM probability of hitting upper boundary before lower boundary\n            exponent = -2.0 * boundary * v / var\n            # Clip exponent for numerical stability\n            exponent = np.clip(exponent, -20.0, 20.0)\n            prob_a = 1.0 / (1.0 + np.exp(exponent))\n            \n        p_core = np.array([prob_a, 1.0 - prob_a])\n        \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / np.sum(probabilities)\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "drift_scale": "[0.1, 10.0]",
    "boundary": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's guidance, this theory implements a Sequential Evidence Accumulation (Drift Diffusion) model. Instead of relying on extreme log-odds transformations (WADD) or inverse-power rank weights, the model assumes decision-makers sequentially sample cues proportional to their validities. The evidence accumulates toward a decision boundary, and the inherent variance of this binomial sampling process provides 'natural noise'. This noise elegantly smooths the choice probabilities, naturally explaining intermediate choice proportions on conflict trials while remaining mathematically principled and avoiding the ordinal violations of non-compensatory heuristics."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13544,
  "candidates_token_count": 897,
  "total_token_count": 20114
}
```
