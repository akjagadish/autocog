# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Rationale:** Following the arbiter's suggestion, this implements the Equal Weight (Tallying) model. By simply summing the unweighted feature values, the model heavily penalizes options that rely on a single high-validity cue when the alternative is supported by multiple lower-validity cues. This mechanism naturally explains the very low TTB-match in Experiment 2 and the specific compensatory patterns observed in Experiment 1, outperforming both the strict one-reason stopping rule of TTB and the exact validity weighting of WADD.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

**Rationale:** Expanded the lower bound of the `theta` parameter range to -1.0. This allows the parameter fitting routine to push the threshold well below the lowest cue validities, guaranteeing that the term (validities - theta) is large and positive. This fully saturates the soft sigmoid mask to 1.0 for all features, perfectly recovering pure Tallying behavior when subjects' data demands it.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[-1.0, 1.0]`
  - `k`: `[1.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Random Subset Tallying: Decision-makers use an equal-weight heuristic but are bounded by working memory, preventing them from processing all features simultaneously. Instead of calculating a complete tally and applying post-decision softmax noise, they stochastically sample a subset of the available features on each trial (each feature included independently with some probability) and perform pure tallying strictly on that subset. This provides a mechanistic, cognitive origin for choice variability while preserving the validity-agnostic, compensatory nature of the Tallying heuristic.

**Rationale:** Following the critic's advice, we further tighten the parameter ranges for Random Subset Tallying. By restricting `sample_prob` to [0.7, 1.0] and `epsilon` to [0.0, 0.1], we further reduce the excess noise on non-tie trials. This increases the probability of sampling enough features to reliably detect the tallying winner, driving the Experiment 3 prediction closer to 0.82 and the Experiment 4 prediction closer to 0.14, while preserving the structural 50% tie-breaking behavior seen in Experiments 5 and 6.

**Parameters:**
  - `sample_prob`: `[0.7, 1.0]`
  - `epsilon`: `[0.0, 0.1]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import itertools
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    n_features = stim.shape[1]
    p = float(parameters["sample_prob"])
    epsilon = float(parameters["epsilon"])
    
    prob_A = 0.0
    
    # Iterate over all possible subsets of features (2^n_features)
    for seq in itertools.product([0, 1], repeat=n_features):
        mask = np.array(seq)
        # Probability of sampling this specific subset
        subset_prob = np.prod(np.where(mask == 1, p, 1.0 - p))
        
        if subset_prob == 0:
            continue
            
        score_A = np.sum(stim[0] * mask)
        score_B = np.sum(stim[1] * mask)
        
        # Pure tallying on the sampled subset
        if score_A > score_B:
            prob_A += subset_prob
        elif score_A == score_B:
            prob_A += 0.5 * subset_prob
            
    prob_B = 1.0 - prob_A
    p_core = np.array([prob_A, prob_B])
    
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```
