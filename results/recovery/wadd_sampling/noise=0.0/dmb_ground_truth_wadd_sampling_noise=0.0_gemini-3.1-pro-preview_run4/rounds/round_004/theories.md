# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Validity Threshold Tallying with Sub-Threshold Influence: Decision-makers simplify the choice environment by setting a subjective threshold on cue validity. Cues with a validity above the threshold are considered 'important' and are tallied with unit weight. To account for weak compensatory behavior and tie-breaking, cues below the threshold are not completely ignored but receive a small, uniform sub-threshold weight.

**Rationale:** Following the critic's feedback, the strict binary thresholding was softened by introducing a sub-threshold weight parameter (`alpha`). Cues below the validity threshold now receive a small, non-zero weight instead of being completely discarded, allowing them to break ties and exert a weak compensatory influence. The threshold range was also widened to [0.0, 1.0] to better adapt to varying validity scales across experiments.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 1.0]`
  - `alpha`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity Threshold Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    alpha = float(parameters["alpha"])
    
    # Cues with validity >= threshold are given unit weight, others receive alpha weight
    w = np.where(val >= threshold, 1.0, alpha)
    
    # Compute tallied scores for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
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
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Decision-makers assign importance to features based on their ordinal rank in validity rather than their exact cardinal values. This Rank-Based Weighting heuristic avoids the extreme sensitivity to numerical validity differences seen in purely compensatory models, while still acknowledging that some cues are more diagnostic than others. Feature weights are computed as a power transformation of their inverse rank (e.g., 1 / rank^gamma). By restricting gamma to lower values, the model maintains a strong compensatory nature, ensuring that multiple lower-ranked cues can outweigh a single higher-ranked cue. Combined with a lower softmax temperature upper bound, it prevents overly deterministic choices and captures the noisier human behavior in conflicting trade-offs.

**Rationale:** Following the critic's feedback, the upper bound of the `beta` parameter has been restricted to 5.0. This minimal edit prevents the softmax from becoming overly deterministic when the weighted score differences are small, which naturally pulls the match rate in Experiment 3 down closer to the human 60% without requiring further changes to the core weighting logic.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.6]`
  - `gamma`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Based Weighting expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Determine the ordinal rank of each feature's validity (1 = highest validity)
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Weights are a power transformation of the inverse rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
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
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Probabilistic Take-The-Best (Cue-Skipping Lexicographic): Decision-makers inspect cues sequentially in descending order of validity. However, execution is probabilistic: each discriminating cue has a probability of being successfully 'trusted', which is proportional to a power transformation of its validity. If trusted, it determines the choice; if skipped, search continues to the next cue. If all cues are exhausted, the decision-maker guesses. This creates a validity-dependent noise mechanism that captures deviations from strict lexicographic behavior without using compensatory tallying.

**Rationale:** Following the critic's feedback, the `alpha` parameter is now restricted to [0.0, 1.0]. In the previous iteration, allowing `alpha` to reach 5.0 caused the trust probability (`p_trust`) to easily saturate at 1.0, effectively eliminating the cue-skipping mechanism and reverting the model to deterministic Take-The-Best. By constraining `alpha` to a maximum of 1.0, `p_trust` remains strictly fractional for typical validity values, ensuring that cue-skipping occurs frequently enough to match the empirical variance and deviations from strict lexicographic behavior observed in the human data.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `gamma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Probabilistic TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity in descending order
    order = np.argsort(-val, kind="stable")
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of trusting and acting on this discriminating cue
            p_trust = np.clip(alpha * (val[idx] ** gamma), 0.0, 1.0)
            
            if a[idx] > b[idx]:
                p_a += p_reach * p_trust
            else:
                p_b += p_reach * p_trust
                
            # Update the probability of reaching the next cue
            p_reach *= (1.0 - p_trust)
            
    # If all discriminating cues are skipped, guess
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Apply epsilon-greedy lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
