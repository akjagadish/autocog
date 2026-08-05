# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Strategy Selection Mixture Theory with Normalized WADD Weights: Decision-makers probabilistically choose between a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Weighted Additive) based on individual preferences. To maintain stable compensatory integration across varying feature counts and validity scales, the log-odds weights in the WADD strategy are normalized by their sum, allowing the decision-maker to apply a consistent level of determinism (softmax temperature) regardless of the specific experimental context.

**Rationale:** Following the critic's advice, this edit reverts to the successful Iteration 1 base (subject-level probabilistic mixture of TTB and WADD) and introduces a crucial normalization step to the WADD component. By minimally clipping the validities to [1e-4, 1-1e-4] and normalizing the resulting log-odds weights by their sum, the WADD scores are placed on a consistent, bounded scale across all experiments, regardless of the number of features or overall validity magnitude. This prevents extreme weights from breaking the softmax scaling, allowing the full `beta` range ([0.1, 100.0]) to function reliably and improving experiment-invariance.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `beta`: `[0.1, 100.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb_prob = float(parameters["p_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # --- Take-The-Best (TTB) Strategy ---
    order = np.argsort(-validities, kind='mergesort')
    p_ttb_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb_core = np.array([0.0, 1.0])
            break
            
    # --- Weighted Additive (WADD) Strategy ---
    # Minimal clipping for numerical safety before log-odds
    v = np.clip(validities, 1e-4, 1.0 - 1e-4)
    w = np.log(v / (1.0 - v))
    
    # Normalize weights by their absolute sum to create a bounded scale for scores
    w_sum = np.sum(np.abs(w))
    if w_sum > 0:
        w = w / w_sum
        
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Translate scores to probabilities using a numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # --- Probabilistic Mixture ---
    p_mix = p_ttb_prob * p_ttb_core + (1.0 - p_ttb_prob) * p_wadd
    
    # --- Lapse Rate ---
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) Strategy with Bounded Weights and Expanded Temperature: Decision-makers evaluate options using a compensatory approach where all available features are integrated. Each feature is weighted by its log-odds validity, reflecting its normative diagnostic value. The subjective weight of a cue is bounded (equivalent to clipping validity at 0.05 and 0.95) to prevent extreme over-weighting. The decision-maker sums the bounded weighted feature values for each option and translates the resulting scores into choice probabilities via a softmax function, with a wide temperature range allowing for highly deterministic behavior when required.

**Rationale:** Following the critic's advice, we retain the [0.05, 0.95] clipping bounds for validities from Iteration 3, as Iteration 4 proved that narrowing them degrades the fits on Experiments 3 and 5. To give the model the flexibility to capture the sharper, more deterministic choices required by Experiment 6, we expand the upper bound of the `beta` parameter from 20.0 to 100.0. This minimal edit allows between-subject variance in temperature to better accommodate differences across experimental designs without altering the core compensatory mechanism.

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
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid extreme log-odds values and division by zero
    v = np.clip(validities, 0.05, 0.95)
    # Compute log-odds weights for each feature
    w = np.log(v / (1.0 - v))
    
    a, b = stim[0], stim[1]
    
    # Calculate the weighted sum of features for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Translate scores to probabilities using a numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Soft-Thresholded Additive Difference Model: Decision-makers evaluate options by directly comparing them feature-by-feature. Rather than using a hard cutoff to ignore less diagnostic features, they employ a soft thresholding mechanism: they apply a non-linear power function to the log-odds validities. This smoothly amplifies highly diagnostic cues and suppresses weaker ones, integrating them into a relative evidence score that translates to choice probabilities via a softmax function.

**Rationale:** Replaced the hard threshold `theta` with a soft thresholding parameter `gamma` applied as a power function to the log-odds validities (`w = np.sign(w) * (np.abs(w) ** gamma)`). This addresses the critic's observation that human decision-makers smoothly downweight less important features rather than abruptly ignoring them, avoiding discontinuities in the loss landscape.

**Parameters:**
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Clip validities for numerical stability before log-odds
    v = np.clip(validities, 1e-4, 1.0 - 1e-4)
    w = np.log(v / (1.0 - v))
    
    # Apply soft thresholding via non-linear power function
    w = np.sign(w) * (np.abs(w) ** gamma)
    
    a, b = stim[0], stim[1]
    
    # Calculate the additive difference between options A and B
    diff = np.sum(w * (a - b))
    
    # Translate the relative difference to choice probabilities
    # Equivalent to softmax over [diff, 0.0]
    scores = np.array([diff, 0.0])
    scores = beta * scores
    scores = scores - np.max(scores)
    e = np.exp(scores)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
