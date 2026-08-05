# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Strategy Mixture Theory: Decision-makers do not uniformly rely on a single decision strategy. Instead, they exhibit variability in strategy selection, probabilistically mixing between a non-compensatory heuristic (Take The Best) and a compensatory strategy (Weighted Additive, WADD). The mixture parameter 'alpha' dictates the probability of using TTB on any given trial, while '1 - alpha' is the probability of using WADD. This allows the model to capture intermediate choice proportions that neither pure strategy can explain on its own, with a bias toward the empirically dominant non-compensatory strategy.

**Rationale:** Following the critic's advice, I applied three minimal edits to the Strategy Mixture model to prevent it from collapsing into uniform guessing (~0.5): 1) Swapped Tallying for Weighted Additive (WADD) as the compensatory strategy, which better accounts for cue validities. 2) Shifted the 'alpha' parameter range from [0.0, 1.0] to [0.5, 1.0] to reflect the empirical dominance of TTB-consistent choices. 3) Increased the lower bound of 'beta' to 1.0 to ensure deterministic choices don't wash out before mixing.

**Parameters:**
  - `alpha`: `[0.5, 1.0]`
  - `beta`: `[1.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Take The Best (TTB) prediction
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
        
    # WADD prediction
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Strategy Mixture
    alpha = float(parameters["alpha"])
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Rationale:** Added a normalization step for the exponentiated validities (val = val / np.sum(val)) as suggested by the critic. Since the raw validities are fractions, raising them to a large power shrinks their absolute values, causing the weighted sums to become very small and washing out the choice probabilities. Normalizing them ensures the scale of the scores remains stable, allowing the beta parameter to properly control the determinism of the choice.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[1.0, 30.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Sequential Evidence Accumulation Theory: Decision-makers sample cues sequentially in order of their validity, accumulating evidence for each option. The accumulation process stops as soon as the absolute difference in evidence between the two options reaches a subject-specific threshold. A low threshold mimics non-compensatory heuristics like Take-The-Best by stopping at the first discriminating cue, while a high threshold mimics compensatory strategies like Weighted Additive (WADD) by integrating all available cues. This natural stopping rule generates variable choice probabilities across different conflict geometries, naturally capturing moderate variance across stimuli without requiring non-linear scaling of validities.

**Rationale:** Following the critic's feedback, the core stopping threshold `theta` is gently widened to [0.0, 1.25]. This represents a middle ground between the overly strict [0.0, 1.0] from Iteration 2 and the overly loose [0.0, 1.5] from Iteration 3. This adjustment allows slightly more compensatory integration to pull Experiment 4's prediction closer to the observed values, without destroying the strong Take-The-Best behavior required for Experiments 1-3. The choice rule parameters (beta and epsilon) are reverted to their successful Iteration 2 values to avoid distorting fine-grained differences in variance metrics.

**Parameters:**
  - `theta`: `[0.0, 1.25]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    score_a = 0.0
    score_b = 0.0
    
    for j in cue_order:
        score_a += val[j] * a[j]
        score_b += val[j] * b[j]
        
        diff = abs(score_a - score_b)
        # Stop if threshold is reached AND there is a strict difference (to avoid stopping on ties if theta is 0)
        if diff >= theta and diff > 1e-9:
            break
            
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
