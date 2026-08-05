# Round 11 — Theories

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


### slot 2 — `pi_13` — KILLED ✗

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


## Replacement

### `pi_14` → slot 2 (via `new_theory`)

**Description:** Thresholded WADD (Tally-then-WADD) with Normalized Exponentiated Validities: Decision-makers primarily rely on a non-compensatory Tallying heuristic, counting the number of winning features. If the difference in tally scores exceeds a certain threshold (tau), they make a choice based strictly on this tally difference. If the tally difference is small or zero, they switch to a compensatory Weighted Additive (WADD) strategy. To capture strong non-compensatory choices on tied trials without destabilizing the softmax temperature, the cue validities are exponentiated and normalized. This allows the WADD component to become steep enough to mimic Take-The-Best (TTB) behavior when needed, bridging the gap between compensatory evaluation and strict non-compensatory tie-breaking.

**Rationale:** Following the critic's feedback, the WADD component's exponentiated validities are now normalized to sum to 1. This prevents drastic scale changes that interfere with the beta_wadd softmax temperature. The gamma parameter range has been expanded to [1.0, 20.0] to allow for highly non-compensatory extremes, mimicking TTB on tied trials. Furthermore, the tau parameter is constrained to {1, 2} to ensure that clear tally differences of 2 or more reliably trigger the Tallying heuristic, restoring performance on experiments where subjects strongly follow large tally differences.

**Parameters:**
  - `tau`: `{1, 2}`
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[1.0, 20.0]`
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
    
    tau = float(parameters["tau"])
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Tallying scores
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_diff = abs(a_wins - b_wins)
    
    if tally_diff >= tau:
        # Use Tallying
        scores = np.array([a_wins, b_wins])
        beta = beta_tally
    else:
        # Use WADD with normalized exponentiated validities
        val_transformed = val ** gamma
        val_transformed = val_transformed / np.sum(val_transformed)
        wadd_a = np.sum(val_transformed * a)
        wadd_b = np.sum(val_transformed * b)
        scores = np.array([wadd_a, wadd_b])
        beta = beta_wadd
        
    # Softmax choice rule
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
