# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

**Rationale:** Following the critic's feedback, the raw validities in the WADD model under-weighted the most valid cues compared to human behavior, leading to an over-prediction of Tallying-consistent choices and an under-prediction of TTB-consistent choices. To address this, a non-linear scaling parameter `gamma` was introduced to exponentiate the validities before computing the weighted sum (`weights = val ** gamma`). This allows the model to adjust the steepness of the weight distribution, bridging the gap between pure WADD and TTB to better match the empirical conflict resolution rates.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
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
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Rank-Based Weighting with Normalized Directional Cues: Decision-makers evaluate cues based on their informational value (distance from 0.5 validity) rather than raw validity. They sort cues by this importance and assign weights that decay exponentially based on normalized ordinal rank (w = direction * alpha ^ normalized_rank). Normalizing ranks to the [0, 1] interval ensures that the decay profile is invariant to the total number of features in the environment, allowing the same cognitive strategy to scale robustly across contexts with varying dimensionality.

**Rationale:** Following the critic's advice, we reverted to the accepted Iteration 2 base, which correctly handles cue directionality but struggles to generalize across experiments with different numbers of features. To fix this, we normalized the ordinal ranks by `len(val) - 1`, mapping them to the `[0, 1]` interval. This makes the exponential decay profile `alpha ** normalized_ranks` invariant to the dimensionality of the experiment, anchoring the most important cue at weight 1.0 and the least at weight `alpha`. This minimal edit allows a single `alpha` parameter to capture both highly compensatory and non-compensatory behaviors across diverse experimental designs.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute cue importance (distance from 0.5) and direction
    importance = np.abs(val - 0.5)
    direction = np.sign(val - 0.5)
    
    # Calculate ranks (0 is highest importance, 1 is second highest, etc.)
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(val), dtype=float)
    
    # Normalize ranks to [0, 1] so that decay is invariant to the number of features
    if len(val) > 1:
        normalized_ranks = ranks / (len(val) - 1)
    else:
        normalized_ranks = ranks
    
    alpha = float(parameters["alpha"])
    # Compute weights: direction * alpha^normalized_rank.
    weights = direction * (alpha ** normalized_ranks)
    
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
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Log-Odds Weighting (Naïve Bayes): Decision-makers intuitively convert cue validities into log-odds (i.e., weight = log(v / (1 - v))) rather than using raw validities or rank-based decay. This non-linear transformation naturally penalizes uninformative cues and strongly boosts extreme validities. It preserves the continuous, compensatory nature of Weighted Additive (WADD) models while producing the steep extreme-cue preference often attributed to non-compensatory or rank-based models. Restricting the exponentiation of these log-odds ensures the model remains appropriately compensatory and matches human behavior.

**Rationale:** Following the critic's feedback, the upper bound of the `gamma` parameter has been further reduced from 1.5 to 1.0. A gamma > 1.0 allows for convex exponentiation of log-odds, which can inadvertently push the model back toward lexicographic behavior and over-rely on top cues (as seen in Exp 1 and Exp 8). Keeping gamma <= 1.0 ensures the weighting remains concave or linear with respect to log-odds, preserving the compensatory nature required by the empirical data while maintaining the log-odds transformation.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid division by zero or log(0)
    val = np.clip(val, 1e-4, 1.0 - 1e-4)
    
    # Compute log-odds
    log_odds = np.log(val / (1.0 - val))
    
    # Apply gamma for additional flexibility in steepness of the weighting function
    gamma = float(parameters["gamma"])
    weights = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
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
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```
