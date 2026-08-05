# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tallying with a Sub-optimal Tie-Breaker (Negative Cue-Weighting in Ties): Decision-makers primarily compare options using a frugal Tallying heuristic, choosing the option with the greater number of winning cues. However, when options are tied on the number of winning features, individuals do not simply guess or use a rational compensatory strategy. Instead, they exhibit a sub-optimal tie-breaking mechanism where they systematically favor the option with lower-ranked or lower-validity cues (effectively a Negative WADD score). This explains both the strong adherence to Tallying when cue counts differ, and the counter-intuitive preference for lower-validity options when cue counts are tied.

**Rationale:** The previous Mixture theory incorrectly assumed that ties in Tallying would be broken by a standard, positive WADD mechanism. However, the data from Experiments 5 and 6 reveal that when Tallying produces a tie, subjects actually favor the option with the lower WADD score (choosing the WADD winner only ~13-15% of the time). This new theory directly addresses this mechanistic failure by implementing a two-stage process: probabilistic Tallying as the primary driver, seamlessly falling back to a 'Negative WADD' tie-breaker when cue counts are equal. This allows the model to capture high Tallying consistency in Exp 2 and 4 while correctly predicting the counter-intuitive preference for lower-validity options in Exp 5 and 6.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `tie_beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    tie_beta = float(parameters["tie_beta"])
    epsilon = float(parameters["epsilon"])
    
    if a_wins != b_wins:
        # Primary strategy: Tallying
        scores = np.array([a_wins, b_wins])
        z = beta * scores
    else:
        # Tie-breaker: Negative WADD (favoring lower validity cues)
        val = np.asarray(parameters["validities"], dtype=float)
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        scores = np.array([-wadd_a, -wadd_b])
        z = tie_beta * scores
        
    z = z - np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Misinterpreted Expert Ratings (Mixture of Standard Tallying and Reverse WADD): Decision-makers consist of two sub-populations. One group understands the instructions correctly and uses a standard Tallying heuristic, choosing the option with more positive features. A second group misinterprets the binary expert ratings as 'defects' or 'costs' (1 = bad, 0 = good) and evaluates options using a Reverse WADD strategy, minimizing the validity-weighted sum of these perceived defects. In scenarios where Tallying and Reverse WADD agree, choice consistency is high. When they disagree, they pull the aggregate preference in opposite directions, weakening the apparent Tallying effect. Crucially, when options tie under Tallying, the standard group guesses uniformly, allowing the Reverse WADD group to entirely drive the aggregate behavior, resulting in a strong 'Negative WADD' effect.

**Rationale:** Following the critic's advice, the parameter ranges for beta_tally and beta_rev are further widened. Increasing the upper bound of beta_rev to 100.0 and beta_tally to 20.0 allows both subpopulations to make nearly deterministic choices when their respective strategies dictate a clear winner, compensating for mixture dilution and enabling the model to hit the extreme empirical probabilities.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_rev`: `[0.1, 100.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # Strategy 1: Standard Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([a_wins, b_wins])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Strategy 2: Reverse WADD (interpreting 1s as costs/defects)
    val = np.asarray(parameters["validities"], dtype=float)
    wadd_a = np.dot(a, val)
    wadd_b = np.dot(b, val)
    
    beta_rev = float(parameters["beta_rev"])
    # Negative WADD scores to penalize options with higher validity-weighted 1s
    z_rev = beta_rev * np.array([-wadd_a, -wadd_b])
    z_rev -= np.max(z_rev)
    p_rev = np.exp(z_rev) / np.sum(np.exp(z_rev))
    
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_rev
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Misinterpreted Expert Ratings (Mixture of Standard Tallying and Reverse Take-The-Best): Subjects fall into two distinct strategic groups. One group uses a standard Tallying heuristic, choosing the option with more positive features. A second group misinterprets the binary expert ratings as 'defects' or 'costs' (1 = bad, 0 = good) and processes the cues lexicographically in order of validity (Reverse TTB). They compare options on the most valid cue; if one option has a '1' (defect) and the other has a '0' (no defect), they immediately choose the option with the '0'. If they tie, they move to the next most valid cue. The population is heavily skewed toward the Reverse TTB strategy, explaining the strong preference for options with fewer high-validity 1s while maintaining a deterministic lexicographic mechanism.

**Rationale:** Following the critic's advice on the rejected Iteration 2, we revert to the deterministic Reverse TTB mechanism from the accepted Iteration 1 base, as it proved superior across most experiments. To address the underestimation of effect sizes in Experiments 9 and 10, we restrict the mixture weight `w_tally` to [0.0, 0.5] (forcing the model to be at least 50% Reverse TTB) and tighten the global lapse rate `epsilon` to [0.0, 0.1]. This minimal edit amplifies the deterministic defect-avoidance signal without introducing the disruptive softmax noise that degraded fits on the other experiments.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `w_tally`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # Strategy 1: Standard Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([a_wins, b_wins])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Strategy 2: Reverse Take-The-Best (Reverse TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
    order = np.argsort(-val, kind='stable')
    
    p_rev = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] < b[idx]:
            # A has 0 (no defect), B has 1 (defect) -> A is preferred
            p_rev = np.array([1.0, 0.0])
            break
        elif b[idx] < a[idx]:
            # B has 0 (no defect), A has 1 (defect) -> B is preferred
            p_rev = np.array([0.0, 1.0])
            break
            
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_rev
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
