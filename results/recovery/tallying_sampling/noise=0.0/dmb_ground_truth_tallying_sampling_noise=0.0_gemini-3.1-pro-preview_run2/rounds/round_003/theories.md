# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Noisy Encoding Tallying: Decision-makers rely on the unweighted Tallying heuristic but suffer from noisy perception or encoding of the environment. Each binary feature has an independent probability of being misperceived (a 1 flipped to a 0, or a 0 flipped to a 1). Subjects then compute the tally of these perceived features and deterministically choose the option with the higher tally, breaking ties randomly. This naturally predicts that decision errors scale with the total number of features (capturing non-linear log-odds in certain experiments) because more features provide more opportunities for bit-flips to alter the tally difference.

**Rationale:** Following the critic's feedback, the core mechanism of Noisy Encoding Tallying remains entirely unchanged. However, the parameter ranges for `flip_prob` and `epsilon` have been restricted from [0.0, 0.5] down to [0.0, 0.2]. This prevents the model from injecting excessive noise that washes out the tallying signal, enabling it to better match the relatively low baseline error rates observed in Experiments 2, 4, and 7 while still capturing the feature-dependent scaling of errors.

**Parameters:**
  - `flip_prob`: `[0.0, 0.2]`
  - `epsilon`: `[0.0, 0.2]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    import math
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    p = float(parameters["flip_prob"])
    epsilon = float(parameters["epsilon"])
    
    def binom_pmf(k, n, prob):
        if n == 0:
            return 1.0 if k == 0 else 0.0
        if prob == 0.0:
            return 1.0 if k == 0 else 0.0
        if prob == 1.0:
            return 1.0 if k == n else 0.0
        return math.comb(n, k) * (prob ** k) * ((1 - prob) ** (n - k))
        
    def get_tally_dist(N1, N0, p):
        dist = np.zeros(N1 + N0 + 1)
        for x in range(N1 + 1):
            px = binom_pmf(x, N1, 1 - p)
            if px == 0.0:
                continue
            for y in range(N0 + 1):
                py = binom_pmf(y, N0, p)
                if py > 0.0:
                    dist[x + y] += px * py
        return dist
        
    N1_A = int(np.sum(stim[0]))
    N0_A = n_features - N1_A
    dist_A = get_tally_dist(N1_A, N0_A, p)
    
    N1_B = int(np.sum(stim[1]))
    N0_B = n_features - N1_B
    dist_B = get_tally_dist(N1_B, N0_B, p)
    
    prob_A_wins = 0.0
    prob_tie = 0.0
    for vA in range(len(dist_A)):
        if dist_A[vA] == 0.0:
            continue
        for vB in range(len(dist_B)):
            if dist_B[vB] == 0.0:
                continue
            if vA > vB:
                prob_A_wins += dist_A[vA] * dist_B[vB]
            elif vA == vB:
                prob_tie += dist_A[vA] * dist_B[vB]
                
    p_A_core = prob_A_wins + 0.5 * prob_tie
    p_B_core = 1.0 - p_A_core
    
    p_core = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
