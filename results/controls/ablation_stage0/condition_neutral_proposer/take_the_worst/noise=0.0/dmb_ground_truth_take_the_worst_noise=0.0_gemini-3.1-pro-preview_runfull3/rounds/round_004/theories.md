# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tallying with Right-to-Left Position-Based Tie-Breaking

**Rationale:** Applying a minimal edit to the tie-breaking logic based on critic feedback. The previous left-to-right search failed on Experiments 3 and 4's tie-trials. Reversing the search direction to right-to-left (perhaps reflecting a recency effect in evaluating sequentially presented features) explains the systematic deviations observed in those experiments while maintaining the core Tallying performance on conflict trials.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    # Position-based lexicographic tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in range(len(a) - 1, -1, -1):
            if a[i] > b[i]:
                scores[0] += 1.0
                break
            elif b[i] > a[i]:
                scores[1] += 1.0
                break
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Tallying with Recency-Driven Tie-Breaking: Decision makers primarily evaluate options using a Tallying heuristic, counting the number of features where one option strictly dominates the other. When the tallies are tied, they do not guess randomly. Instead, they rely on the cues most recently encoded into working memory to break the tie. Assuming a standard left-to-right reading order, the right-most features are the most recently processed. The model resolves ties by scanning features from right to left, favoring the option that wins on the first differentiating feature encountered.

**Rationale:** The arbiter pointed out that the standard Tallying model fails to capture systematic tie-breaking behavior, often reverting to random guessing when tallies are equal. Inspired by the success of right-to-left tie-breaking, this model introduces 'Tallying with Recency-Driven Tie-Breaking'. It posits that when primary tallying fails to distinguish between options, subjects fall back on the most recently encoded information in working memory. In a left-to-right reading paradigm, the right-most features are the most recent. This provides a psychologically grounded, memory-based explanation for the lexicographic tie-breaking effect without discarding the robust baseline of the Tallying heuristic.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary strategy: Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Recency-driven tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in reversed(range(len(a))):
            if a[i] > b[i]:
                score_a += 1.0
                break
            elif b[i] > a[i]:
                score_b += 1.0
                break
                
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with epsilon-greedy lapse
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Tallying with Validity-Based Tie-Breaking: Decision makers primarily evaluate options using a Tallying heuristic, counting the number of features where one option strictly dominates the other. When the tallies are tied, they do not guess randomly or rely on recency. Instead, they break ties by inspecting features in descending order of their objective validity (Take-The-Best), favoring the option that wins on the most valid differentiating feature.

**Rationale:** Following the arbiter's recommendation, this theory maintains Tallying as the dominant mechanism for decision making, as it best explains choice behavior in conflict trials. However, to better capture behavior when Tallying yields a tie, we replace the right-to-left recency tie-breaker with a Validity-Based Tie-Breaker (Take-The-Best). When tallies are equal, the model resolves the tie by examining features in decreasing order of their objective validities, choosing the option that dominates on the most valid differentiating feature.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary strategy: Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Validity-driven tie-breaking (Take-The-Best)
    if a_wins == b_wins:
        validities = np.array(parameters['validities'], dtype=float)
        sorted_indices = np.argsort(-validities)
        for i in sorted_indices:
            if a[i] > b[i]:
                score_a += 1.0
                break
            elif b[i] > a[i]:
                score_b += 1.0
                break
                
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with epsilon-greedy lapse
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
