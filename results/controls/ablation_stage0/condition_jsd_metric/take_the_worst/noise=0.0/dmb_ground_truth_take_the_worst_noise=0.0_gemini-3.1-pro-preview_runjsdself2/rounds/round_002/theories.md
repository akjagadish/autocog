# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) theory posits that decision-makers integrate all available cues by weighting them proportionally to their validities. The overall value for each option is the validity-weighted sum of its features. Choices are made probabilistically via a softmax over these aggregated values, incorporating response noise through an inverse temperature parameter and an independent lapse rate.

**Rationale:** Following the arbiter's recommendation, this theory implements the Weighted Additive (WADD) heuristic. Unlike Take The Best (which relies only on the single best discriminating cue) and Tallying (which weights all cues equally), WADD considers all features but weights them proportionally to their validities. This captures compensatory decision-making where a combination of weaker cues can override a single strong cue, which aligns with the observed human behavior in these experiments.

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
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Probabilistic-Stopping Take-The-Best (TTB)

**Rationale:** Following the critic's advice to implement a Probabilistic-Stopping TTB model. It maintains the arbiter-prescribed lexicographic search order (descending validity) but introduces an `alpha` parameter that dictates the probability of stopping at a discriminating cue. This allows downstream cues to analytically influence expected choice probabilities, bridging the gap between strict TTB and the partially compensatory patterns in human data without reverting to WADD.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Descending validity; argsort is stable so validity ties break toward earlier feature index
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    for j in cue_order:
        if a[j] > b[j]:
            p_A += p_continue * alpha
            p_continue *= (1.0 - alpha)
        elif b[j] > a[j]:
            p_B += p_continue * alpha
            p_continue *= (1.0 - alpha)
            
    # If search exhausts all cues or fails to stop, guess uniformly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Integrated Utility Theory (WADD + Tallying): Decision-makers integrate both compensatory validity weighting (WADD) and non-compensatory feature counting (Tallying) into a single, unified subjective value for each option. Rather than stochastically switching between strategies on a trial-by-trial basis, they form a composite score by taking a weighted average of the WADD and Tallying scores. This composite score is then translated into choice probabilities via a single softmax process subject to decision noise and lapses.

**Rationale:** Following the critic's advice, this minimal edit changes how the WADD and Tallying components are combined. Instead of applying a softmax to each strategy independently and mixing their resulting probabilities (which implies a trial-by-trial stochastic switching between strategies), the model now computes a weighted average of the raw scores (utilities) from WADD and Tallying before applying a single softmax. This integrated utility approach posits that individuals simultaneously weigh exact validities and simple feature counts to form a unified evaluation, potentially offering a better structural fit without adding new parameters.

**Parameters:**
  - `w`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = float(parameters["w"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # WADD scores
    wadd_scores = np.sum(stim * val, axis=1)
    
    # Tallying scores (strict wins)
    tally_scores = np.array([np.sum(a > b), np.sum(b > a)], dtype=float)
    
    # Mix scores instead of probabilities
    mixed_scores = w * wadd_scores + (1.0 - w) * tally_scores
    
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
