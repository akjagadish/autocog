# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

**Rationale:** To address the model being too deterministic compared to human data, two minimal adjustments were made. First, the lower bound of `beta` was widened to 0.01 to allow for a flatter softmax (more noise). Second, the raw WADD scores were normalized by the sum of the validities before applying the softmax. This ensures that the scale of the score differences remains consistent across experiments with varying numbers of cues, allowing the single `beta` parameter to generalize better and capture the higher level of human noise observed.

**Parameters:**
  - `beta`: `[0.01, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Threshold-based Binarization (Satisficing WADD): Decision-makers simplify complex cardinal information by converting continuous or multi-level ratings into binary cues based on a satisficing threshold. A feature is considered satisfactory (1) if its rating meets or exceeds the threshold, and unsatisfactory (0) otherwise. The options are then evaluated by computing the validity-weighted sum of these binarized features (WADD on binary cues). This mechanism naturally explains why extreme cardinal advantages (e.g., 10 vs 5) might be ignored if both options exceed the satisficing threshold, allowing an option with distributed moderate advantages to win against an option with a single extreme advantage.

**Rationale:** Following the arbiter's feedback, this model instantiates a Threshold-based Binarization mechanism. Instead of integrating cardinal values linearly or transforming them continuously, it maps cardinal ratings to binary states (satisfactory vs. unsatisfactory) using a single threshold parameter. The options are then evaluated using a weighted additive approach on these binary cues. This perfectly captures the intuition that decision-makers might ignore large magnitude differences (e.g., 8 vs 10) if both values exceed their internal standard of 'good enough', which explains the moderation of large magnitude effects in Experiments 4, 5, and 8, and accounts for the negative metric in Experiment 6.

**Parameters:**
  - `threshold`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize validities
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val
        
    # Binarize features based on the threshold
    bin_a = (a >= threshold).astype(float)
    bin_b = (b >= threshold).astype(float)
    
    # Compute validity-weighted sum of binarized features
    score_a = np.sum(bin_a * val)
    score_b = np.sum(bin_b * val)
    
    scores = np.array([score_a, score_b])
    
    # Convert to probabilities via softmax with max-subtraction
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Diminishing Returns WADD posits that individuals evaluate options by applying a concave utility function to cardinal feature values before weighting them by their cue validities. By compressing large cardinal values via a power law transformation, extreme advantages on a single feature are discounted relative to multiple moderate advantages across several features. Parameterizing the shift applied before the power law allows the model to flexibly smooth the extreme marginal utility near zero, preventing over-sensitivity to small integer differences while maintaining the core concave utility mechanism.

**Rationale:** Following the critic's advice, we introduce a new free parameter `shift` with a range of `[0.1, 5.0]` to replace the fixed 1.0 shift. The utility function becomes `u(x) = (x + shift)^alpha - shift^alpha`. This minimal edit preserves the Diminishing Returns WADD mechanism but gives the model the flexibility to find the optimal amount of smoothing for small cardinal differences. This should help correct the over-prediction in Experiment 8 without compromising the gains achieved in other experiments.

**Parameters:**
  - `alpha`: `[0.1, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `shift`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    shift = float(parameters["shift"])
    
    # Normalize validities to stabilize softmax across different experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val
        
    # Apply concave utility transformation (diminishing returns)
    # Shifted by a parameterized value to flexibly smooth infinite marginal utility at zero
    u_a = np.power(a + shift, alpha) - np.power(shift, alpha)
    u_b = np.power(b + shift, alpha) - np.power(shift, alpha)
    
    # Compute validity-weighted sum of transformed features
    score_a = np.sum(u_a * val)
    score_b = np.sum(u_b * val)
    
    scores = np.array([score_a, score_b])
    
    # Convert to probabilities via softmax with max-subtraction
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
