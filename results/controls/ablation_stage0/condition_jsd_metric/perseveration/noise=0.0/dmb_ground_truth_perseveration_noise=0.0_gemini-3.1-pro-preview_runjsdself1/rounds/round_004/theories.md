# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Decision-makers evaluate options by computing a weighted sum of their features based on subjective cue validities. However, instead of these validities remaining static, subjects exhibit a learning or adaptation mechanism over trials: they gradually adjust their decision weights to align with the features of the options they have previously chosen. This means that if a subject chooses an option with certain positive features, the subjective importance of those features increases (or decreases, depending on the learning rate) for future decisions. This psychologically plausible adaptation creates strong, idiosyncratic sequential dependencies—capturing the high between-subject variance in sequential choice behavior—while remaining deeply integrated with the utility computation.

**Rationale:** Following the critic's feedback, I replaced the pure A/B motor-response inertia with a simple learning/adaptation mechanism. The model now updates the subjective cue validities after each trial by adding the features of the chosen option scaled by a learning rate 'alpha'. This means subjects gradually specialize their decision weights to align with their past choices. This approach naturally generates strong, context-sensitive sequential dependencies that should inflate the sequence-aware JSD to match human levels, while maintaining the core utility evaluation framework to protect overall accuracy. I used a wide range for 'alpha' to allow for both strong reinforcement and alternation.

**Parameters:**
  - `beta`: `[0.0, 50.0]`
  - `alpha`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float).copy()
    alpha = float(parameters["alpha"])
    
    # Adapt subjective validities based on past choices
    if history and "response" in history and len(history["response"]) > 0:
        for i in range(len(history["response"])):
            resp = history["response"][i]
            opt_a = np.asarray(history["option_a_ratings"][i], dtype=float)
            opt_b = np.asarray(history["option_b_ratings"][i], dtype=float)
            chosen_features = opt_a if resp == 0 else opt_b
            val += alpha * chosen_features
            
    # Base utilities using Weighted Additive (WADD) rule with adapted validities
    u_a = np.dot(a, val)
    u_b = np.dot(b, val)
    
    scores = np.array([u_a, u_b])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion to probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
