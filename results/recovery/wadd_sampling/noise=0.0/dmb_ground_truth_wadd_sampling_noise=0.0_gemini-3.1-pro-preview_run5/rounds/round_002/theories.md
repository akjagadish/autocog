# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Rationale:** Following the latest feedback, we restrict the parameter ranges to decouple the deterministic weighting strategy from choice noise. We limit `gamma` to [0.1, 2.0] to enforce Tally-like subjective weights, and simultaneously reduce the upper bound of the inverse temperature `beta` to 5.0 (down from 20.0). This higher choice noise softens the excessively deterministic predictions, pulling Experiment 2's Tally agreement down to empirical levels and pushing Experiment 1's TTB match up toward its empirical baseline, without changing the core WADD mechanism.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 2.0]`
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
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Validity-Thresholded Tallying: Decision-makers simplify choices by ignoring cues with low validities and applying a simple tallying heuristic to the remaining high-validity cues. A subjective threshold determines which cues are considered reliable enough to use. For the included cues, the decision-maker counts how many times each option has a higher feature value than the other. The option with more wins among the thresholded cues is chosen. This boundedly rational strategy combines the frugality and robustness of tallying with the validity-sensitivity of weighted additive models, effectively breaking ties in favor of options that excel on more valid cues without requiring complex mental arithmetic.

**Rationale:** Following the critic's feedback, the parameter range for the validity threshold `tau` has been expanded from [0.5, 1.0] to [0.0, 1.0]. The previous restricted range forced the model to exclude too many cues, making it behave too similarly to Take-The-Best (TTB) and failing to capture the strong Tallying-like behavior observed in Experiment 2. By allowing lower thresholds, the model can include more cues in its tally, achieving a better balance between Tallying and TTB.

**Parameters:**
  - `tau`: `[0.0, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Thresholded Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues meet the subjective validity threshold
    valid_mask = val >= tau
    
    if not np.any(valid_mask):
        # If no cues are reliable enough, the decision-maker has no preference
        scores = np.zeros(2)
    else:
        # Tally wins only on the included cues
        a_filtered = stim[0, valid_mask]
        b_filtered = stim[1, valid_mask]
        
        a_wins = float(np.sum(a_filtered > b_filtered))
        b_wins = float(np.sum(b_filtered > a_filtered))
        
        scores = np.array([a_wins, b_wins])
        
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Strategy Selection (WADD and Probabilistic Tallying): Decision-makers probabilistically alternate between a purely compensatory strategy (Weighted Additive) and a simpler Tallying heuristic on a trial-by-trial basis. The Tallying heuristic is probabilistic, using a softmax over win counts to generate choice probabilities rather than deterministic choices. This mixture allows individuals to exhibit graded sensitivity to cue evidence on some trials while defaulting to unweighted, softer cue-counting on others, effectively explaining both the high tallying agreement in certain environments and the near-zero extremeness differences in others.

**Rationale:** Following the critic's advice, the deterministic Tallying component was replaced with a probabilistic softmax over tally scores using a new `beta_tally` parameter. This softens the heuristic's predictions, addressing the over-prediction of extremeness contrast in Experiment 5 (which was caused by hard 1.0/0.0 probabilities on non-tie trials) while preserving the excellent fits on the other experiments achieved by the WADD + Tallying mixture.

**Parameters:**
  - `w_wadd`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD Strategy
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    wadd_weights = val ** gamma
    wadd_scores = np.dot(stim, wadd_weights)
    
    z = beta * (wadd_scores - np.max(wadd_scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Tallying Strategy (Probabilistic)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    tally_scores = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
        
    # Mixture
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
