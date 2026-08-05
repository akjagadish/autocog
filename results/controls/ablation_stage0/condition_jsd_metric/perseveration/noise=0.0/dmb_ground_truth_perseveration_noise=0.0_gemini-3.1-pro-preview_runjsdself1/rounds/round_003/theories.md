# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Population-level Strategy Selection Mixture Model: The population consists of distinct subgroups of decision-makers. Some individuals consistently use a non-compensatory heuristic (Take The Best) while others consistently use a compensatory strategy (Weighted Additive). This discrete between-subject variation in strategy selection accounts for the high individual divergence from the population average, and low lapse rates ensure that the distinct strategy signatures are not washed out by noise.

**Rationale:** Following the critic's feedback, we build upon the accepted Iteration 2 base (the population-level strict mixture model). To increase the JSD and make subjects more distinctly divergent from the population average, we restrict the lapse rate (epsilon) to a much lower range [0.0, 0.1] and increase the maximum inverse temperature (beta) up to 30.0. High lapse rates wash out individual differences, causing choices to regress toward the population mean. By reducing epsilon and allowing sharper, more deterministic choices, subjects utilizing TTB and WADD will behave starkly differently, driving the between-subject variance (JSD) up toward the empirical targets.

**Parameters:**
  - `beta_wadd`: `[0.1, 30.0]`
  - `beta_ttb`: `[0.1, 30.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected state to be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- WADD Strategy ---
    score_a_wadd = np.sum(val * a)
    score_b_wadd = np.sum(val * b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- TTB Strategy ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Population-level Mixture & Lapse ---
    w_raw = float(parameters["w_wadd"])
    w_wadd = 1.0 if w_raw > 0.5 else 0.0
    p_mix = w_wadd * p_wadd + (1.0 - w_wadd) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Decision-makers use an 'Equal Weight' or Tallying heuristic to choose between options. Instead of weighting features by their validities (which is cognitively demanding) or relying on a single best cue, individuals simply count the total number of positive features for each option. The option with the higher total count of positive features is preferred. Choice probabilities are derived using a softmax function over these unweighted counts, reflecting bounded rationality by ignoring complex weights to reduce cognitive load.

**Rationale:** Based on the arbiter's feedback, this theory implements the Tallying (Equal Weight) heuristic. It differs from WADD by completely ignoring the expert validities, and it differs from Take-The-Best by integrating information across all cues equally. The decision variable is simply the unweighted sum of positive features for each option. A softmax function transforms these scores into choice probabilities, modulated by the inverse temperature parameter 'beta'. This provides a boundedly rational model that significantly reduces cognitive load while still considering all available features.

**Parameters:**
  - `beta`: `[0.0, 20.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Count the total number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    
    # Numerically stable softmax over the unweighted counts
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return p_core
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Sequential Evidence Accumulation with Deterministic Threshold and Cognitive Bounds. Decision-makers evaluate features sequentially in descending order of validity. They accumulate the validity-weighted difference between the two options. Evaluation stops early either when the absolute accumulated evidence exceeds a threshold (theta) or when a cognitive bound on the maximum number of features (max_features) is reached. If the threshold is hit, choice is strictly deterministic. If the cognitive bound is reached without hitting the threshold, choice is probabilistic via a softmax over the evidence accumulated up to that point. This truncation prevents late, low-validity features from diluting the evidence, generating sharper probabilities and higher behavioral variability while maintaining a baseline lapse rate for true errors.

**Rationale:** Following the critic's Iteration 8 advice, we revert to the successful Iteration 7 base (which uses a strict deterministic threshold rather than artificial parameter bounds) and introduce a cognitive bound on the number of features processed. We add a `max_features_prop` parameter that dictates the maximum proportion of features evaluated. The decision-maker stops evaluating either when the accumulated evidence hits `theta` OR when the `max_features` limit is reached. If the process stops due to `max_features` without hitting the threshold, the choice is made via the `beta` softmax based on the evidence accumulated up to that point. This truncation prevents low-validity features from diluting the accumulated evidence, naturally leading to starker probabilities and higher JSDs while allowing `epsilon` to remain wide enough to handle actual human lapses without massive log-loss penalties.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `theta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `max_features_prop`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Process features in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    max_features_prop = float(parameters["max_features_prop"])
    
    n_features = len(val)
    # Convert proportion to an integer number of features (at least 1)
    max_features = max(1, int(np.ceil(max_features_prop * n_features)))
    
    accumulated_evidence = 0.0
    hit_threshold = False
    
    for i, cue_idx in enumerate(cue_order):
        if i >= max_features:
            break
            
        # Accumulate evidence (difference in weighted feature values)
        accumulated_evidence += val[cue_idx] * (a[cue_idx] - b[cue_idx])
        
        # Threshold stopping rule
        if abs(accumulated_evidence) >= theta:
            hit_threshold = True
            break
            
    # Probability of choosing A based on accumulated evidence
    if hit_threshold:
        # Strictly deterministic choice if threshold is hit
        if accumulated_evidence > 0:
            p_a = 1.0
        elif accumulated_evidence < 0:
            p_a = 0.0
        else:
            p_a = 0.5
    else:
        # Softmax choice if max_features exhausted without hitting threshold
        z = beta * accumulated_evidence
        if z > 500:
            p_a = 1.0
        elif z < -500:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    # Incorporate baseline lapse rate
    p_a_final = (1.0 - epsilon) * p_a + epsilon * 0.5
    return np.array([p_a_final, 1.0 - p_a_final])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
