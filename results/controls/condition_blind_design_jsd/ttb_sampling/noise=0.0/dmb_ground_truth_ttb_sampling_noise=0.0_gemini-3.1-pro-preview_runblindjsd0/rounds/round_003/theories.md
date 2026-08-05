# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Take-The-Best (TTB) heuristic: Decision-makers evaluate options using a non-compensatory, rank-based approach. They first rank the available features by their validity (descending). They then sequentially compare the options on these features, stopping at the first feature that discriminates between them (i.e., one option has a higher value than the other). The option favored by this discriminating feature is chosen, and all remaining features are completely ignored. If no features discriminate between the options, the decision-maker resorts to a random guess.

**Rationale:** Following the arbiter's suggestion, we implement the Take-The-Best (TTB) heuristic. This model sorts features based on their provided validities and evaluates them one by one. The first feature that differentiates the two options determines the choice 100%, ignoring all subsequent features. This is a purely non-compensatory strategy, contrasting strongly with the exhaustive integration of Weighted Additive (WADD) and the equal-weight exhaustive summation of Tallying. Randomness is introduced solely via an epsilon lapse rate, reflecting execution noise or inattention, and uniform guessing is used if no features discriminate.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # Using mergesort for a stable sort in case of ties.
    order = np.argsort(-validities, kind='mergesort')
    
    a, b = stim[0], stim[1]
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequentially check features according to their validity ranking
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend the deterministic choice (or guess) with the lapse rate
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


### slot 2 — `pi_5` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
