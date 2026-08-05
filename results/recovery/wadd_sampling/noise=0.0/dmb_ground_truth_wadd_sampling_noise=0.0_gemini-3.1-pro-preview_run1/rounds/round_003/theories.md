# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

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

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Raw Validity-Weighted Additive Strategy: Decision-makers do not rely on fast-and-frugal heuristics or non-linear rank transformations. Instead, they compute a simple compensatory weighted sum of the features, where the weights are directly proportional to the raw validities provided in the instructions. This naturally explains the ~0.50 split in conflict trials because the sum of a few high-validity cues roughly equals the sum of several lower-validity cues, bridging the gap between heuristic and rank-based models. A constrained temperature parameter ensures that small differences in these sums translate to intermediate choice probabilities rather than deterministic outcomes.

**Rationale:** Following the arbiter's feedback, this model implements a pure Raw Validity-Weighted Additive Strategy. By using the raw validities directly as weights rather than ordinal ranks or lexicographic rules, the model naturally equates the evidence from a few high-validity cues with the evidence from several lower-validity cues. We constrain the inverse temperature ('beta') to a smaller upper bound (5.0) compared to previous WADD instantiations to ensure that the small score differences on conflict trials correctly produce the observed ~0.50 choice splits, avoiding the over-determinism that caused prior weighted-additive models to fail.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute simple weighted sums using raw validities as weights
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([score_a, score_b])
    
    # Convert scores to probabilities using softmax
    z = beta * scores
    z = z - np.max(z)
    p_core = np.exp(z)
    p_core = p_core / np.sum(p_core)
    
    n_opts = len(p_core)
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
