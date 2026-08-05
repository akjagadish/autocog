# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People make choices by integrating all available feature information, weighting each cue by its subjective validity. Unlike Tallying (which weights all cues equally) or Take The Best (which stops at the first discriminating cue), the Weighted Additive (WADD) model computes an overall expected value for each option by summing the products of the feature values and their validities. Choice probabilities are then generated via a softmax function over these weighted sums, allowing for graded sensitivity to both the number of supporting features and their relative importance.

**Rationale:** Following the arbiter's suggestion, this Weighted Additive (WADD) model integrates all available features, weighting them proportionally to their validities. This provides a mechanism that accumulates evidence across all cues (unlike TTB) while remaining sensitive to the varying importance of different features (unlike Tallying). The weighted sums are translated into choice probabilities via a softmax function with a lapse rate, offering a robust parametric alternative that captures compensatory decision-making behavior.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
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
    
    # WADD: compute the weighted sum of features for each option
    score_a = np.sum(stim[0] * val)
    score_b = np.sum(stim[1] * val)
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
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** People employ a non-compensatory Take-The-Best (TTB) heuristic to make decisions under uncertainty. Instead of integrating all available features, decision-makers rank cues by their subjective validity and inspect them sequentially in descending order. The search stops at the very first cue that discriminates between the two options, and the choice is determined entirely by this single cue. If no cues discriminate after inspecting all of them, the decision-maker guesses randomly. Response stochasticity is modeled simply through a lapse rate (epsilon), rather than a softmax, reflecting a constant probability of execution error or random guessing regardless of the cue.

**Rationale:** Following the critic's advice, the previous attempt to scale scores by cue validity was rejected because TTB is strictly non-compensatory. Moreover, the original base model used both a softmax temperature (beta) and a lapse rate (epsilon) on a fixed 1 vs 0 score difference, which is mathematically redundant and causes identifiability issues. This minimal edit removes beta and the softmax entirely. If a cue discriminates, the model deterministically favors that option modulo the lapse rate epsilon. If no cues discriminate, it outputs a uniform distribution [0.5, 0.5].

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Rank cues by validity in descending order
    order = np.argsort(val)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    p_a = 0.5
    p_b = 0.5
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_a = 1.0 - epsilon
            p_b = epsilon
            break
        elif stim[1, idx] > stim[0, idx]:
            p_a = epsilon
            p_b = 1.0 - epsilon
            break
            
    return np.array([p_a, p_b])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Unweighted Threshold Evidence Accumulation Model: Decision-makers integrate feature information sequentially in order of cue validity, but they do not weight the features by their exact validities. Instead, they simply count the unweighted evidence (treating each feature as 1 or 0) and stop search early once the unweighted evidence difference between options exceeds a subjective threshold. This blends the bounded, heuristic nature of Take-The-Best with the equal-weight compensatory integration of Tallying.

**Rationale:** Following the critic's feedback, restricting the threshold and beta parameters degraded performance, indicating that subjects might be integrating more information or acting more deterministically. As suggested, I modified the Threshold model to integrate *unweighted* evidence difference rather than validity-weighted difference, while keeping the original broader parameter bounds. This shifts the model closer to the Tallying family (which has shown strong performance historically) while retaining the sequential, validity-ordered search and early stopping of the Threshold mechanism.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(val)[::-1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Integrate unweighted evidence until the difference threshold is met
    for i in order:
        score_a += stim[0, i]
        score_b += stim[1, i]
        if abs(score_a - score_b) >= theta:
            break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
