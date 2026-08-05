# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Dynamic Strategy Selection (Threshold Model): Decision-makers dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy on a trial-by-trial basis. The choice of strategy depends on the maximum cardinal difference between the options across all features. If this difference exceeds a certain threshold, the decision-maker is more likely to use WADD to account for the large magnitude; otherwise, they default to the simpler Tallying heuristic. This is modeled as a probabilistic mixture where the weight of WADD is a soft step function (sigmoid) of the maximum feature difference.

**Rationale:** Following the critic's feedback, the fixed probabilistic mixture is replaced with a dynamic, stimulus-driven threshold mechanism. The model computes the maximum cardinal difference between options on any feature. A soft threshold (sigmoid) determines the mixture probability `w_wadd` for that trial based on this max difference. This allows the model to correctly favor Tallying when differences are small (like in Experiment 2) and switch to WADD when there are extreme cardinal differences (like in Experiment 3), addressing the specific failure modes of the previous fixed-mixture theory.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 10.0]`
  - `steepness`: `[0.1, 10.0]`
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
    
    # WADD score computation
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a_wadd = np.sum(a * val) / sum_val
        score_b_wadd = np.sum(b * val) / sum_val
    else:
        score_a_wadd = np.sum(a)
        score_b_wadd = np.sum(b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # Tallying score computation
    score_a_tally = float(np.sum(a > b))
    score_b_tally = float(np.sum(b > a))
    scores_tally = np.array([score_a_tally, score_b_tally])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Dynamic strategy selection based on max cardinal difference
    max_diff = np.max(np.abs(a - b))
    w_wadd = 1.0 / (1.0 + np.exp(-steepness * (max_diff - threshold)))
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Diminishing Returns WADD (Non-linear WADD): Subjects evaluate options by applying a concave utility transformation to the cardinal feature values before computing the validity-weighted sum. This naturally compresses large cardinal differences, making the evaluation process a smooth hybrid between Tallying (where magnitudes are ignored) and linear WADD (where magnitudes are fully integrated). To prevent extreme marginal utility jumps near zero, feature values are shifted by 1 before the power transformation is applied.

**Rationale:** Following the critic's advice, the minimal edit shifts the ratings by 1.0 before applying the power transformation `np.power(x + 1.0, alpha)`. This prevents the excessive marginal utility jump between 0 and 1 when alpha is small, which had distorted the WADD validity-weighted sums in Experiment 3 (where zero values are common). It preserves the intended diminishing returns compression for larger differences, maintaining the improved fits seen in the other experiments.

**Parameters:**
  - `alpha`: `[0.1, 1.0]`
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
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize validities to stabilize softmax across different experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val
        
    # Apply concave utility transformation (diminishing returns)
    # Ratings are shifted by 1.0 to prevent excessive marginal utility between 0 and 1
    u_a = np.power(a + 1.0, alpha)
    u_b = np.power(b + 1.0, alpha)
    
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
