# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Rank-Weighted Additive Theory posits that decision-makers ignore the exact numerical validities provided and instead assign exponentially decaying weights based solely on the ordinal rank of the cues. The lowest-ranked cue is assigned a weight equal to the second-lowest-ranked cue. This ensures that the sum of all lower-ranked cues exactly equals the highest-ranked cue. They then compute a weighted sum for each option and choose probabilistically (via softmax). Because the highest-ranked cue's weight perfectly equals the sum of all lower-ranked cues' weights, this theory naturally predicts the exact ~50% guessing rate observed on '1 vs all' adversarial trials without needing arbitrary thresholds or conflict-based guessing rules.

**Rationale:** Following the critic's diagnosis, the previous model correctly implemented the rank-weighted additive mechanism but used a beta parameter range up to 50.0. This caused the model to behave deterministically even for minor score differences, leading to severe over/under-predictions compared to the near-0.5 empirical values in Exps 5-8. This minimal edit tightens the beta parameter range from [0.1, 50.0] to [0.1, 5.0] to enforce softer, more probabilistic choices that better reflect human noise levels when evaluating rank-weighted sums.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Get validities and determine ordinal rank (0 is highest rank)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Assign exponentially decaying weights based on rank.
    # To ensure the sum of all lower-ranked cues exactly equals the highest-ranked cue,
    # the lowest-ranked cue gets the same weight as the second-lowest-ranked cue.
    w = np.zeros_like(val)
    n_cues = len(cue_order)
    for i, cue_idx in enumerate(cue_order):
        if i == n_cues - 1 and n_cues > 1:
            w[cue_idx] = 1.0 / (2.0 ** i)
        else:
            w[cue_idx] = 1.0 / (2.0 ** (i + 1))
        
    # Compute weighted sum for each option
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Threshold Tallying Theory posits that decision makers evaluate options by counting the number of features where one option strictly dominates the other (unweighted tallying). However, they only make a confident directional choice if the difference in winning features between the two options meets or exceeds a certain cognitive threshold. If the difference in evidence is below this threshold (e.g., a difference of 0 or 1), the decision maker finds the evidence too ambiguous or weak, leading to cognitive overload or uncertainty, and they resort to uniform guessing. This captures the persistent ~50% choice rates observed across many adversarial trials where the feature counts are closely matched.

**Rationale:** Following the arbiter's suggestion, this model instantiates the 'Threshold Tallying Theory'. It computes the unweighted tally of winning features for each option. If the absolute difference between the two tallies is strictly less than a threshold parameter (sampled between 1.5 and 3.5, effectively requiring a difference of at least 2 or 3 to trigger a directional choice), the model outputs uniform probabilities. This naturally drives the choice probabilities to exactly 0.50 on the majority of the adversarial trials where the tally difference is 0 or 1, perfectly explaining the empirical choice rates across all 6 experiments without needing extreme noise parameters.

**Parameters:**
  - `threshold`: `[1.5, 3.5]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins for each option
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    diff = abs(a_wins - b_wins)
    threshold = float(parameters["threshold"])
    
    # If the difference is below the threshold, the evidence is deemed too weak -> guessing
    if diff < threshold:
        p_core = np.array([0.5, 0.5])
    else:
        # Otherwise, make a choice based on the tally scores using softmax
        scores = np.array([float(a_wins), float(b_wins)])
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Theory posits that decision-makers probabilistically alternate between two simple heuristics: Take-The-Best and unweighted Tallying. To account for the different scales of evidence produced by these heuristics (e.g., tallying differences can be large, whereas TTB is always a binary 1 vs 0), decision-makers apply distinct sensitivities (inverse temperatures) to each strategy. This allows the model to capture the magnitude of tallying differences without them overwhelming the TTB predictions.

**Rationale:** Following the latest feedback, we reverted to using raw tally counts (preserving the magnitude of the tally difference) but introduced two separate inverse-temperature parameters: `beta_tally` and `beta_ttb`. This allows the model to independently calibrate the determinism of TTB and Tallying, balancing their scales without arbitrary normalization.

**Parameters:**
  - `p_tally`: `[0.0, 1.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    tally_scores = np.array([float(a_wins), float(b_wins)])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally_choice = e_tally / np.sum(e_tally)
    
    # Take-The-Best prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    ttb_scores = np.array([0.0, 0.0])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_scores = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_scores = np.array([0.0, 1.0])
            break
            
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb_choice = e_ttb / np.sum(e_ttb)
    
    # Mixture
    p_tally = float(parameters["p_tally"])
    p_core = p_tally * p_tally_choice + (1.0 - p_tally) * p_ttb_choice
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
