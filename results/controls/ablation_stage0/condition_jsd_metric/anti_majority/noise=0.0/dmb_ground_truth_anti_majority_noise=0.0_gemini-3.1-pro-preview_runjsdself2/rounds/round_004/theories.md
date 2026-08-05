# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Decision makers evaluate options by sequentially processing features in descending order of their validity. They accumulate evidence in the form of log-odds weights for each feature. However, accumulation is not always exhaustive; it stops as soon as the absolute accumulated evidence exceeds a subjective threshold. A low threshold mimics a Take The Best heuristic (stopping at the first discriminating cue), while a high threshold mimics a Weighted Additive strategy (integrating all available cues).

**Rationale:** Applying the minimal-diff edit suggested by the critic: narrowing the threshold range to [0.0, 5.0] to provide higher resolution for the fitter, and widening the beta range to [0.1, 50.0] to allow for more deterministic choices. The mechanism remains exactly the same, as it was already correctly implementing Sequential Evidence Accumulation with Threshold.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Convert validities to log-odds weights (Bayesian evidence)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    threshold = float(parameters["threshold"])
    
    evidence = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            evidence += weights[j] * diff
            # Stop accumulating if evidence exceeds the subjective threshold
            if abs(evidence) >= threshold:
                break
                
    # The accumulated evidence represents the log-odds favoring option A over B
    scores = np.array([evidence, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Decision makers use a Weighted Additive (WADD) strategy to evaluate options, integrating all available features. Instead of raw validities or linear shifts, they weight each feature by its log-odds, which is the mathematically principled way to linearly accumulate independent evidence (equivalent to Naive Bayes). The total score for each option is the sum of these log-odds weights for the features it possesses. The option with the higher total score is chosen probabilistically via a softmax function over the scores, subject to a baseline lapse rate.

**Rationale:** Following the critic's advice, we replace the linear shift (val - 0.5) with the theoretically principled log-odds transformation. This corresponds to the correct Bayesian method for linearly integrating independent cues, scaling the evidence more aggressively for highly predictive cues. This minimal edit keeps the theory firmly within the WADD compensatory family while addressing the under-prediction of variance in several experiments.

**Parameters:**
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds of validities to represent the true Bayesian weight of evidence.
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax choice rule with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Strategy Selection (Blended Evidence of Take-The-Best and Tallying): Decision-makers do not invariably use exhaustive, weighted integration (like WADD). Instead, they rely on a repertoire of fast-and-frugal heuristics. On any given decision, rather than discretely switching between strategies, they integrate the evidence from a non-compensatory 'Take-The-Best' (TTB) strategy and a compensatory but unweighted 'Tallying' strategy. The unified evidence is then evaluated probabilistically. The weight placed on TTB versus Tallying is a subject-specific parameter that naturally adapts to the environment's cue validity dispersion and the subject's cognitive constraints.

**Rationale:** Following the critic's advice, this edit modifies the PREVIOUS CANDIDATE (Iteration 1 base) by mixing the pre-softmax scores (evidence) of the Take-The-Best and Tallying strategies, rather than their post-softmax probabilities. This 'blended' approach avoids the bimodal distributions and overly extreme probabilities caused by discretely flipping between two strategies trial-by-trial. By applying a single softmax over the unified evidence, the model can more smoothly capture intermediate levels of determinism, which should better predict the empirical JSD metric (behavioral variance) without introducing any new parameters.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `w_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take-The-Best (TTB) Strategy
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        s_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        s_ttb = np.array([0.0, 1.0])
    else:
        s_ttb = np.array([0.5, 0.5])
        
    # 2. Tallying Strategy (unweighted feature counting)
    # Normalized by n_features so the score difference is at most 1, 
    # keeping it on a similar scale to TTB for the softmax beta.
    s_tally = np.array([np.sum(a), np.sum(b)]) / n_features
    
    beta = float(parameters["beta"])
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Mix pre-softmax scores instead of post-softmax probabilities
    s_mix = w_ttb * s_ttb + (1.0 - w_ttb) * s_tally
    
    # Apply softmax to the blended evidence
    z = beta * s_mix
    z -= np.max(z)
    e = np.exp(z)
    p_mix = e / np.sum(e)
    
    # Apply baseline lapse rate
    n_opts = len(p_mix)
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
