# Round 2 — Theories

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Soft Threshold Tallying posits that decision-makers use a compensatory tallying heuristic, but down-weight cues that are less reliable. They evaluate each feature's subjective validity against a personal threshold using a smooth sigmoid function, creating a soft inclusion mask. Features that fall well below the threshold are effectively ignored, while those above are tallied with near-equal weight. This smooth transition provides a bridge between pure equal-weight tallying and selective attention to high-validity information, while allowing for continuous optimization.

**Rationale:** Expanded the `threshold` parameter range from [-1.0, 1.0] to [-5.0, 1.0] as suggested by the critic. This minimal edit ensures that the optimizer can push the threshold deeply negative so that the sigmoid soft-mask fully saturates to 1.0 for all cues, even at lower steepness values. This allows the model to cleanly and reliably recover the pure Tallying regime required to match human behavior in Experiments 3 and 4, eliminating residual variance.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[-5.0, 1.0]`
  - `steepness`: `[1.0, 50.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Threshold Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Soft threshold using a logistic sigmoid to create a smooth inclusion mask
    weights = 1.0 / (1.0 + np.exp(-steepness * (validities - threshold)))
    
    # Tally the features weighted by the soft threshold mask
    scores = np.sum(stim * weights, axis=1)
    
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Tallying with Lexicographic Tie-Breaking posits that decision-makers primarily rely on a compensatory equal-weight tallying heuristic, choosing the option with the highest number of positive features. However, when options are tied in their feature tallies, decision-makers do not guess randomly. Instead, they break the tie by comparing the options on the single most valid feature where the options differ (a Take-The-Best mechanism). This hybrid approach perfectly mimics Tallying on unequal feature counts but provides a deterministic, validity-based resolution for ties.

**Rationale:** Following the critic's feedback, we introduce a new parameter `w_tie` with a restricted upper bound of 0.05. Instead of adding 1.0 to the winner's score during a tie, we add `w_tie`. This drastically limits the maximum difference in softmax probabilities during tied trials, bringing the maximum expected score difference in Experiment 6 closer to the empirical ceiling of 0.5000, while preserving the deterministic fits on the primary tallying trials.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `w_tie`: `[0.0, 0.05]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Primary mechanism: Equal-weight tallying
    scores = np.sum(stim, axis=1)
    
    # Secondary mechanism: Lexicographic tie-breaking (Take-The-Best)
    if scores[0] == scores[1]:
        validities = np.asarray(parameters["validities"], dtype=float)
        w_tie = float(parameters["w_tie"])
        diff = stim[0] - stim[1]
        valid_diffs = np.where(diff != 0)[0]
        
        if len(valid_diffs) > 0:
            # Find the differing feature with the highest validity
            best_feature = valid_diffs[np.argmax(validities[valid_diffs])]
            if diff[best_feature] > 0:
                scores[0] += w_tie
            else:
                scores[1] += w_tie

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
