# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

**Rationale:** Modified the conflict condition so that if Tallying results in a tie (tally_winner is None), it no longer defaults to following TTB. Instead, a tie in the compensatory strategy fails to corroborate the non-compensatory strategy, which triggers the same guessing behavior as a direct contradiction. This resolves the failure in Experiment 4 where participants guessed on trials with a TTB winner but tied Tallying scores.

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
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
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


### slot 2 — `pi_5` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
