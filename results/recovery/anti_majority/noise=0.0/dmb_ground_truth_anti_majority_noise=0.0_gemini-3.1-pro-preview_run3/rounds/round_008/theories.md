# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** Conflict-Driven Strategy Selection: Decision-makers adaptively select their decision strategy based on the dissimilarity of the options' total cue counts. When options are highly dissimilar in the number of positive cues (indicating high conflict or complexity), individuals abandon compensatory processing and fall back entirely on a simple non-compensatory heuristic (Take The Best). Conversely, when the total cue counts are similar, individuals attempt to integrate all available information using a compensatory strategy (Tallying). This is modeled as a probabilistic mixture of TTB and Tallying, where the probability of using TTB scales directly with the absolute difference in total cue counts.

**Rationale:** Following the critic's advice, we revert to the exact Iteration 2 mechanism but adjust the parameter ranges to find a better global compromise. We constrain `beta_ttb` and `beta_tally` to `[0.1, 10.0]` to ensure both strategies maintain reasonable stochasticity and do not collapse into pure determinism. We also increase the upper bound of `epsilon` to `0.5` to allow for a higher baseline lapse rate. This gentle retuning preserves the stable core formulation of Iteration 2 while preventing the destructive extremes observed in Iteration 6.

**Parameters:**
  - `beta_ttb`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB Probabilities
    z_ttb = beta_ttb * scores_ttb
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta_tally * scores_tally
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally_dist = e_tally / e_tally.sum()
    
    # Conflict-Driven Weight
    gamma = float(parameters["gamma"])
    
    # Dissimilarity in total cue counts
    diff_cues = abs(np.sum(a) - np.sum(b))
    
    # Probability of using TTB increases linearly with diff_cues
    w_ttb = min(1.0, gamma * diff_cues / max(1.0, float(n_features)))
    
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb_dist + (1.0 - w_ttb) * p_tally_dist
    
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


### slot 2 — `pi_10` — KILLED ✗

**Description:** Rank-Dependent Discriminating Weighting (RDDW): Decision-makers first simplify the choice by performing feature-wise comparisons, completely canceling out cues where both options have the same value. They then rank the remaining discriminating cues by their subjective validity. Evidence is accumulated for each option using a rank-dependent weighting scheme, where the impact of each subsequent cue diminishes geometrically. This allows the strategy to smoothly interpolate between a purely non-compensatory Take-The-Best approach (steep decay) and a fully compensatory WADD or Tallying approach (no decay), naturally explaining why multiple weaker cues can sometimes override a single highly valid cue.

**Rationale:** The new theory, 'Rank-Dependent Discriminating Weighting (RDDW)', directly implements the arbiter's suggestion to use feature-wise comparison (cancellation) followed by integration with diminishing marginal impact. By filtering out non-discriminating cues and ranking the remaining ones by validity, the model applies a geometric decay (gamma^k) to the weight of each subsequent cue. This allows the model to flexibly capture pure Take-The-Best (gamma near 0), pure Tallying (gamma near 1, alpha near 0), and intermediate compensatory behaviors where multiple moderately valid cues can outweigh a single highly valid cue, while naturally handling high-conflict scenarios.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `alpha`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature-wise comparison: isolate discriminating cues
    diff = a - b
    discrim_indices = np.where(diff != 0)[0]
    
    if len(discrim_indices) == 0:
        return np.array([0.5, 0.5])
        
    # Rank discriminating cues by validity
    discrim_vals = val[discrim_indices]
    sorted_order = np.argsort(-discrim_vals, kind="stable")
    sorted_indices = discrim_indices[sorted_order]
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Accumulate evidence with rank-dependent diminishing returns
    for k, idx in enumerate(sorted_indices):
        v = val[idx]
        # Weight depends on scaled validity and rank k
        w = (max(0.0, v - 0.5)) ** alpha * (gamma ** k)
        
        if diff[idx] > 0:
            ev_a += w
        elif diff[idx] < 0:
            ev_b += w
            
    # Convert evidence to choice probabilities
    scores = np.array([ev_a, ev_b])
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


## Replacement

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Rightward/Recency Biased Compensatory Processing (Piecewise): Decision-makers use a two-regime evaluation strategy. In standard situations where one option has more positive cues than the other, they rely on a strictly decreasing, validity-based weight vector, prioritizing early, highly valid cues. However, when the options are tied in their total number of positive cues (zero-conflict), they discard strict validity prioritization and instead rely on a recency-biased weight vector, systematically preferring options supported by later cues.

**Rationale:** Following the critic's diagnosis, the model now explicitly separates the evaluation of standard trials and tied-cue trials. By implementing a piecewise function based on whether `sum_a == sum_b`, the model can apply a steep, exponential validity decay (`val_weight * (decay ** i)`) to correctly predict standard non-compensatory-like trials (Exps 1-4), while cleanly applying a recency/rightward bias (`pos_weight * (i / max(1, n - 1))`) to tied-cue trials (Exps 13-18) without the two regimes mathematically conflicting with one another.

**Parameters:**
  - `val_weight`: `[1.0, 10.0]`
  - `pos_weight`: `[0.0, 10.0]`
  - `decay`: `[0.1, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    n = len(val)
    
    val_weight = float(parameters["val_weight"])
    pos_weight = float(parameters["pos_weight"])
    decay = float(parameters["decay"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    w = np.zeros(n)
    if sum_a == sum_b:
        # Zero-conflict (tied cue count) scenario: use recency-biased weights
        for i in range(n):
            w[i] = pos_weight * (i / max(1, n - 1))
    else:
        # Standard scenario: use strictly decreasing validity-based weights
        for i in range(n):
            w[i] = val_weight * (decay ** i)
            
    ev_a = np.sum(a * w)
    ev_b = np.sum(b * w)
    
    scores = np.array([ev_a, ev_b])
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
