# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) strategy: Decision-makers evaluate options by integrating all available information, computing a weighted sum of the features for each option where the weights are the cue validities. This compensatory approach balances the number of features an option possesses against the importance (validity) of those features.

**Rationale:** Following the arbiter's instructions, we replace the non-compensatory Take The Best (TTB) model with the fully compensatory Weighted Additive (WADD) model. While TTB discards all cues past the first discriminating one, and Tallying ignores cue importance entirely, WADD strikes a balance by summing all features weighted by their respective validities. This allows high-validity cues to exert a stronger pull while still allowing a large number of lower-validity features to compensate for a missing high-validity cue.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

**Rationale:** Following the critic's diagnosis, the linear rank-weighting model still leaned too heavily toward WADD in conflict trials because the sum of the top ranks strictly dominated the bottom ranks. To address this, we introduce a non-linear scaling parameter `gamma` applied to the ranks (`weights = rankdata(val) ** gamma`). This minimal change allows the model to smoothly interpolate between Tallying (gamma=0) and steeper rank-weighting, providing the necessary flexibility to exactly match the ~0.50 split in conflict trials while remaining within the prescribed Rank-Weighted Additive family.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Strategy Mixture Theory (Tallying-Biased with Softened Determinism): Decision-makers do not universally rely on a single compensatory mechanism. Instead, the population consists of a mixture of strategies using fast-and-frugal heuristics: 'Take-The-Best' (lexicographic) and 'Tallying' (unweighted sum of strict feature-wise wins). The population shows a stronger preference for Tallying over Take-The-Best, but choices are also somewhat stochastic. By softening the determinism of the individual heuristics, extreme choice probabilities are tempered, allowing the model to fit intermediate conflict trial outcomes more robustly without drastically shifting the population mixture.

**Rationale:** Following the critic's advice, we revert to the successful iteration 2 base but make two minimal adjustments to balance the experiments. First, we relax `w_tally` to [0.4, 0.9] to maintain a strong preference for Tallying without completely saturating it. Second, we lower the upper bound of the softmax inverse-temperature `beta` to 5.0 (from 20.0). This naturally softens the deterministic predictions of both the TTB and Tallying branches, pulling extreme choice probabilities closer to 0.5 and preventing the massive overprediction of Tallying in Experiment 4 without ruining the fit for Experiments 1, 2, and 5.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `w_tally`: `[0.4, 0.9]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Order features by validity (highest first)
    order = np.argsort(val)[::-1]
    
    # Take-The-Best (TTB) evaluation
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying evaluation (count of strict feature-wise wins)
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    # TTB probabilities
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Response noise (epsilon-greedy)
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
