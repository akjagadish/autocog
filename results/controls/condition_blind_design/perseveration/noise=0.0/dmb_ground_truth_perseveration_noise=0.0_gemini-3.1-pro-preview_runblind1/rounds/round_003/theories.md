# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Subjects use a Tallying (Equal Weights) heuristic, where they ignore the explicit cue validities and simply count the total number of positive features for each option. They choose the option with the highest total number of positive features, and guess randomly if there is a tie. A highly restricted softmax temperature and lapse rate account for the overall noisy, near-chance behavior observed in the experiments.

**Rationale:** Following the critic's feedback, the Tallying mechanism is retained exactly as in the previous candidate, but the upper bound for the inverse temperature parameter (beta) has been tightened from 1.0 to 0.3. This minimal edit forces the softmax to be flatter, preventing the model from predicting higher accuracy on trials with large differences in total positive features. This adjustment aligns the model's predictions with the empirical data from Experiments 5 and 6, which show near-zero sensitivity to differences in WADD scores.

**Parameters:**
  - `beta`: `[0.0, 0.3]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Count total positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Noisy Take The Best: Decision-makers use a lexicographic heuristic (Take The Best), checking features in descending order of validity and stopping at the first feature that discriminates between options. However, human application of this rule is highly noisy. A heavily restricted inverse temperature parameter accounts for the empirical observation that subjects' choices on these trials hover near chance level (0.50-0.55), rather than the highly deterministic choices (0.85+) predicted by standard TTB.

**Rationale:** The arbiter recommended implementing Take The Best (TTB). However, as seen in the leaderboard (pi_1), a standard TTB model with a wide beta range severely overpredicts the metrics, scoring ~0.85-0.87 whereas human subjects score ~0.50-0.55. The data clearly shows that on these specific trials, subjects are operating very close to chance. To comply with the arbiter's suggestion while fixing the mechanistic failure of pi_1, I propose a Noisy Take The Best model. By restricting the softmax inverse temperature (beta) to a very low range [0.0, 0.5], the model captures the fact that subjects might be using lexicographic rules but with extreme inconsistency or high decision noise, bringing the predictions down to the ~0.50-0.55 range actually observed in humans.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity descending; stable sort preserves original order on ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Decision-makers evaluate options using a Weighted Additive (WADD) strategy. Instead of relying on a single best cue (like Take The Best) or ignoring cue importance (like Tallying), individuals integrate all available features by weighting each feature according to its validity. The overall value of an option is the sum of its validity-weighted features. Choices are then made probabilistically by comparing these weighted sums, with response consistency governed by a softmax temperature parameter and a base lapse rate. A highly restricted temperature parameter prevents over-sensitivity to small differences in the weighted sums, matching the near-chance behavior of human subjects.

**Rationale:** Following the arbiter's recommendation, this model implements a Weighted Additive (WADD) heuristic. It computes a weighted sum of positive features using the explicit cue validities as weights. To accurately capture the near-chance empirical performance (e.g., choice proportions hovering around 0.50-0.55), the model incorporates a highly restricted softmax inverse temperature (beta in [0.0, 0.5]) and a lapse rate (epsilon). This prevents the deterministic overconfidence seen in unconstrained WADD models and aligns the choice probabilities with the observed data.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate validity-weighted sums for both options
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
