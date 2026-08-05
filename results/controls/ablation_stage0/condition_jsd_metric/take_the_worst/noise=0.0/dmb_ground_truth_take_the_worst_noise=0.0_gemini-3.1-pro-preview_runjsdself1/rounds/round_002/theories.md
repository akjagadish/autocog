# Round 2 — Theories

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Take The Best (TTB) heuristic: People make decisions by searching through features in descending order of their stated validities. The search stops at the first feature that discriminates between the options. The option favored by this single cue is chosen, ignoring all other features. If no cues discriminate, the choice is a random guess. A single error parameter epsilon governs the probability of making a choice contrary to the discriminating cue.

**Rationale:** Following the previous critic's advice, we revert to the pure Take The Best (TTB) mechanism where the first discriminating cue determines the choice. To resolve the parameter redundancy between `beta` and `epsilon` from the previous iteration, we eliminate the softmax temperature `beta` entirely. We introduce a single parameter `epsilon` that directly assigns the probability `1.0 - epsilon` to the favored option, simplifying the model while adhering strictly to the non-compensatory lexicographic decision rule.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validities
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    epsilon = float(parameters["epsilon"])
    
    if winner == 0:
        return np.array([1.0 - epsilon, epsilon])
    elif winner == 1:
        return np.array([epsilon, 1.0 - epsilon])
    else:
        return np.array([0.5, 0.5])
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Rank-Weighted Additive Model: People make choices by integrating all available feature information, but rather than weighting each cue by its exact subjective validity (as in WADD) or weighting all cues equally (as in Tallying), they weight cues by their ordinal rank. The most valid cue receives the highest integer weight, the next most valid receives the second highest, and so on. This captures a heuristic simplification where decision-makers respect the order of cue importance but fail to perfectly integrate precise cardinal probabilities.

**Rationale:** The arbiter requested a compensatory theory that differs from both WADD and Take The Best, suggesting either a Tallying or Rank-Weighted Additive Model. Since Tallying (unit weights) has already been proposed in a previous iteration (pi_2), I have instantiated the Rank-Weighted Additive Model. This model evaluates options by taking a weighted sum of their features, but substitutes the exact stated validities with their ordinal ranks. This tests whether subjects are integrating the precise validity values or just using a simplified, rank-based compensatory heuristic.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: ascending order, so highest validity gets largest integer weight.
    weights = rankdata(val)
    
    score_a = np.sum(stim[0] * weights)
    score_b = np.sum(stim[1] * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
