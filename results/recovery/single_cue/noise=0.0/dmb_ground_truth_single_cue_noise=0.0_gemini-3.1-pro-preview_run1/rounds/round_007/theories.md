# Round 7 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_9` — KILLED ✗

**Description:** Tallying with Recency/Anti-Primacy Tie-Breaker: Decision-makers primarily use the Tallying heuristic, counting strict feature-wise wins to choose between options. However, when the win counts are tied, they do not guess uniformly; instead, they systematically break ties by favoring options whose positive features appear later in the sequence (a recency or anti-primacy bias). This is modeled by adding a secondary 'recency' score (a weighted sum with linearly increasing weights) to the tally score. The recency weight is constrained such that it can only determine the choice when the primary tallies are tied, preserving a strict Tallying baseline while robustly capturing tie-breaking behavior.

**Rationale:** The arbiter diagnosed that the previous model (Theory 1) used a messy 3-way mixture that compromised the strict Tallying baseline, suggesting a simpler approach where Tallying is primary and a recency/anti-primacy score acts as a tie-breaker. This theory implements exactly that: the primary score is the strict tally of feature wins, and a secondary recency score (using linearly increasing weights for later features) is added. By normalizing the recency weights and bounding the recency parameter `w_recency` to [0.0, 0.99], we mathematically guarantee that a tally difference of 1 or more will always overpower the recency score. Thus, the recency score only dictates the choice probabilities when the tallies are tied, perfectly capturing the anti-primacy tie-breaking effect while retaining the robust baseline of Theory 2.

**Parameters:**
  - `w_recency`: `[0.0, 0.99]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    w_recency = float(parameters["w_recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # 1. Tallying (Primary)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # 2. Recency / Anti-Primacy (Secondary Tie-Breaker)
    # Linearly increasing weights for later features
    weights = np.arange(1, n_features + 1, dtype=float)
    weights /= np.sum(weights) # Normalize so max difference is <= 1
    
    recency_a = np.sum(a * weights)
    recency_b = np.sum(b * weights)
    
    # Combine scores. Since w_recency < 1 and max recency diff <= 1,
    # a tally difference of >= 1 will always dominate the recency difference,
    # preserving the strict Tallying baseline.
    score_a = a_wins + w_recency * recency_a
    score_b = b_wins + w_recency * recency_b
    scores = np.array([score_a, score_b])
    
    # Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_2` — SURVIVED ✓

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_10` → slot 1 (via `new_theory`)

**Description:** Decision-makers use a 'Take-The-Best (TTB) with Thresholded Compensatory Tallying' strategy. They initially attempt to use a lexicographic TTB approach, favoring the option that wins on the first discriminating feature. However, they evaluate the remaining features: if the number of opposing wins on subsequent features reaches or exceeds a subject-specific threshold, they abandon TTB and switch to a compensatory Tallying strategy. When Tallying results in a tie, a secondary recency (anti-primacy) bias acts as a tie-breaker, which is explicitly constrained to only apply when the primary tally counts are exactly equal, preventing it from overriding strict tallying wins.

**Rationale:** Following the critic's feedback, the recency tie-breaker has been explicitly constrained to only alter the tally scores when `a_wins == b_wins`. This prevents the tie-breaker from overriding a strict 1-win difference in Tallying, which had ruined the model's performance on pure Tallying tasks (Exps 2 and 4) in Iteration 4. With this structural safeguard in place, the `w_recency` range has been safely expanded to [-5.0, 5.0] to ensure it can act as a sufficiently strong deterministic tie-breaker when needed. The `theta` range is kept at [0.0, 3.0] as in the successful Iteration 3 base, maintaining the model's sensitivity to switch to Tallying when opposing evidence exists.

**Parameters:**
  - `theta`: `[0.0, 3.0]`
  - `w_recency`: `[-5.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    theta = float(parameters["theta"])
    w_recency = float(parameters["w_recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # 1. Identify TTB winner and the first discriminating cue
    ttb_winner = None
    k = -1
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_winner = 0
            k = i
            break
        elif b[i] > a[i]:
            ttb_winner = 1
            k = i
            break
            
    if ttb_winner is None:
        return np.array([0.5, 0.5])
        
    # 2. Evaluate remaining features
    if ttb_winner == 0:
        opposing_wins = np.sum(b[k+1:] > a[k+1:])
    else:
        opposing_wins = np.sum(a[k+1:] > b[k+1:])
        
    diff_val = float(opposing_wins)
    
    # 3. Calculate Tallying scores with Recency tie-breaker
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    weights = np.arange(1, n_features + 1, dtype=float)
    weights /= np.sum(weights)
    recency_a = np.sum(a * weights)
    recency_b = np.sum(b * weights)
    
    # Explicitly restrict tie-breaker to only apply when tally counts are equal
    if a_wins == b_wins:
        tally_scores = np.array([
            a_wins + w_recency * recency_a,
            b_wins + w_recency * recency_b
        ])
    else:
        tally_scores = np.array([a_wins, b_wins])
    
    # 4. Apply Threshold Logic
    if diff_val >= theta:
        scores = tally_scores
    else:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
