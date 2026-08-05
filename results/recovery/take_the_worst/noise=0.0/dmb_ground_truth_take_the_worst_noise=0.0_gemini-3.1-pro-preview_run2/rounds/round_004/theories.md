# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Inverse Validity Weighting: Subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. Instead of relying on them, they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity (1 - validity). This mechanism leads subjects to consistently prefer options that are supported by lower-validity features over those supported by high-validity features, explaining the strong anti-expert choices observed in the experiments.

**Rationale:** Following the critic's advice, we remove the non-linear scaling parameter `gamma` to strictly use `weights = 1.0 - val`, preventing distortion of the relative differences between options. To achieve the high choice magnitudes observed in Experiments 2, 5, and 8, we increase the upper bound of the softmax inverse temperature `beta` to 100.0, allowing for more deterministic choices based on the linear inverse-validity scores without distorting the underlying weight distribution.

**Parameters:**
  - `beta`: `[0.1, 100.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Inverse Validity Weighting expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Primary mechanism: Weighted additive using inverse validities
    # Lower validity features receive higher subjective weights
    weights = 1.0 - val
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Anti-Expertise Tallying: Decision-makers actively distrust or ignore high-validity 'expert' cues, perceiving them as overly salient, redundant, or manipulative. Instead of using them, subjects filter out features whose stated validity exceeds a certain personal threshold. After discarding these high-validity cues, subjects apply a standard Tallying heuristic (counting strict feature-wise wins) on the remaining lower-validity features to make their choice.

**Rationale:** Following the arbiter's suggestion, this theory instantiates a structurally distinct 'Anti-Expertise' mechanism that relies on filtering and attention rather than tie-breaking. Subjects discard features with validities above a personal threshold, actively ignoring 'expert' cues entirely, and then perform Tallying on the remaining low-validity features. This explains why subjects consistently go against Take-The-Best (Exp 1) and WADD (Exp 3 & 4), heavily favor options with multiple low-validity cues over a single high-validity cue (Exp 5 & 6), yet still perform well on general Tallying metrics when low-validity cues align with the tally (Exp 2).

**Parameters:**
  - `threshold`: `[0.5, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Anti-Expertise Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Filter out features with validity above the threshold
    mask = val <= threshold
    
    # Fallback to all features if the threshold is so low that all features are discarded
    if not np.any(mask):
        mask = np.ones_like(val, dtype=bool)
        
    a_masked = a[mask]
    b_masked = b[mask]
    
    # Tallying on the remaining features
    a_wins = float(np.sum(a_masked > b_masked))
    b_wins = float(np.sum(b_masked > a_masked))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Reverse Take-The-Worst (Reverse Lexicographic) Heuristic: Subjects actively distrust high-validity expert cues, viewing them as manipulative or overly salient. Instead of relying on them, subjects rank features from lowest validity to highest validity. They compare options starting with the absolute lowest validity feature and immediately choose the option that is superior on that feature, ignoring all higher-validity features. If there is a tie, they move to the next lowest validity feature. This explains extreme anti-expert choices where a single low-validity feature strictly dominates any number of high-validity features.

**Rationale:** Following the arbiter's instructions, this theory implements a 'Reverse Lexicographic' or 'Take-The-Worst' heuristic. Subjects rank the cues from lowest validity to highest validity and make their decision based solely on the first discriminating feature in this reversed order. This perfectly captures the extreme anti-expert behavior seen in Exps 1 and 2, where subjects favor options that are strictly worse on high-validity cues but better on the lowest-validity ones. It also naturally mimics the aggregate patterns of Inverse Validity Weighting on the other experiments, as the lowest-validity cues heavily dictate the choices.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Reverse TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features from lowest validity to highest validity
    order = np.argsort(val)
    
    score_a = 0.0
    score_b = 0.0
    
    # Take-The-Worst: find the first feature (starting from lowest validity) that discriminates
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
