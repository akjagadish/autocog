# Round 10 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

**Rationale:** Following the arbiter's feedback, this theory implements a 'Tally-then-TTB' mechanism. It evaluates options primarily by Tallying, which correctly captures human behavior on unequal-tally trials where subjects tend to prefer the option with more winning cues regardless of their validity. When Tallying results in a tie, rather than guessing randomly (which fails to capture deterministic behavior on tied trials), the model breaks the tie by defaulting to Take-The-Best, selecting the option that wins on the highest-validity cue. This directly addresses the mechanistic failures of pure Tallying and pure TTB by combining their strengths in a sequential heuristic framework.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_12` — KILLED ✗

**Description:** Probabilistic Strategy Mixture (Tallying and Take-The-Best) with Strong Tallying Bias

**Rationale:** Following the critic's advice, we continue to push the mixture towards Tallying and reduce noise to close the remaining gaps in predicting extreme empirical magnitudes (e.g., >0.80 match rates in Experiments 2, 4, 14). By tightening the upper bound of p_ttb to 0.3, we reflect that TTB is only a minority strategy in this population. Similarly, tightening epsilon to [0.0, 0.15] allows the dominant deterministic choices to express themselves more clearly without being washed out by too much random noise.

**Parameters:**
  - `p_ttb`: `[0.0, 0.3]`
  - `beta_tally`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    p_ttb = float(parameters["p_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    ttb_a, ttb_b = 0.5, 0.5
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a, ttb_b = 1.0, 0.0
            break
        elif b[idx] > a[idx]:
            ttb_a, ttb_b = 0.0, 1.0
            break
    p_ttb_strat = np.array([ttb_a, ttb_b])
    
    # Fixed probabilistic mixture of the two pure strategies
    p_core = p_ttb * p_ttb_strat + (1.0 - p_ttb) * p_tally
    
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

### `pi_13` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Model with Power Transformation: Decision-makers evaluate options by taking the sum of the feature values weighted by a monotonically increasing power function of their cue validities. The power transformation allows the strategy to flexibly interpolate between equal-weighting (Tallying) and strict non-compensatory weighting (Take-The-Best), capturing variations in compensatory behavior across environments.

**Rationale:** Following the most recent critic feedback, we revert to the unnormalized power function weight transformation (`val ** gamma`) from the accepted Iteration 1 base, as all alternative transformations (exponential, normalized power, log-odds) failed to improve the aggregate loss. The power function remains the most empirically successful at interpolating between Tallying and TTB. To improve its ability to capture compensatory behavior without breaking the core mechanism, we adjust the parameter ranges: beta is widened to [0.01, 20.0] to allow for more uniform probabilistic choices, and epsilon is increased to [0.0, 0.5] to accommodate higher baseline noise.

**Parameters:**
  - `gamma`: `[0.0, 20.0]`
  - `beta`: `[0.01, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Monotonically increasing function of validities
    # Using a power function allows smooth interpolation from Tallying (gamma=0) 
    # to Take-The-Best-like lexicographic weighting (gamma -> large)
    weights = val ** gamma
    
    # Calculate Weighted Additive (WADD) scores
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    
    scores = np.array([score_a, score_b])
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
