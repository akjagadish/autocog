# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

**Rationale:** Following the arbiter's feedback, we replace Take The Best with the 'Tallying' (Equal Weight) theory. Tallying assumes that decision-makers do not weight cues by their validity, but rather just sum up the number of positive features for each option. This explains why subjects frequently choose the option with a higher total number of positive features (yielding a low value on the Experiment 1 metric) and why their choices strongly diverge from the Take The Best predictions (yielding a low value on the Experiment 2 metric). It serves as a strong, parameter-free (in terms of feature weights) compensatory alternative to WADD.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Weighted Additive (WADD) Model posits that decision-makers compute a fully compensatory weighted sum of features for each option. Unlike Tallying, which ignores cue validities, and Take-The-Best, which relies solely on the highest validity differing cue, WADD integrates all available information continuously by weighting each positive feature by its corresponding validity. The option with the highest overall weighted value is chosen.

**Rationale:** As per the arbiter's feedback, the previous model (Tallying with Lexicographic Tie-Breaking) systematically failed on tie trials because subjects simply guess rather than relying on the highest-validity differing feature. To provide a strong alternative baseline, we propose the classic Weighted Additive (WADD) model. WADD posits that subjects compute a fully compensatory weighted sum using the provided validities, integrating the cue validities continuously across all features instead of ignoring them or using them strictly for tie-breaking.

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
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive mechanism: sum of features weighted by their validities
    # This assumes subjects use the raw validities as compensatory weights.
    scores = np.sum(stim * validities, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Tallying Fallback posits that decision-makers attempt to use a non-compensatory strategy by relying on the single most valid cue (Take-The-Best), but due to cognitive constraints, time pressure, or low confidence, they frequently lapse into a simpler, compensatory equal-weight Tallying heuristic. The resulting behavior is a probabilistic mixture of Tallying and Take-The-Best, heavily skewed towards Tallying. This captures the overwhelmingly compensatory nature of the experimental data while preserving a residual sensitivity to cue validities that a pure Tallying model completely misses.

**Rationale:** Following the critic's advice, the parameter range for `w_tally` has been tightened from [0.5, 1.0] to [0.9, 1.0]. This minimal edit ensures that the mixture remains overwhelmingly dominated by Tallying, avoiding the large overestimations of Take-The-Best influence seen in the previous iteration, while still preserving the nuanced, residual sensitivity to cue validities.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_tally`: `[0.9, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Tallying scores (equal weight)
    tally_scores = np.sum(stim, axis=1)
    
    # 2. Take-The-Best scores (single highest validity cue that discriminates)
    diff = stim[0] - stim[1]
    valid_diffs = np.where(diff != 0)[0]
    ttb_scores = np.zeros(2)
    if len(valid_diffs) > 0:
        best_feature = valid_diffs[np.argmax(validities[valid_diffs])]
        if diff[best_feature] > 0:
            ttb_scores[0] = 1.0
        else:
            ttb_scores[1] = 1.0
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_tally = float(parameters["w_tally"])
    
    # Softmax for Tallying
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Softmax for TTB
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Mixture of strategies
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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
