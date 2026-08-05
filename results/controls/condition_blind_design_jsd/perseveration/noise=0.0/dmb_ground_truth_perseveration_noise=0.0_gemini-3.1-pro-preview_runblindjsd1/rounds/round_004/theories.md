# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Decision-makers evaluate options using a Weighted Additive (WADD) strategy, but their choices are subject to sequential dependencies, specifically choice inertia. The value of each option is computed as a weighted sum of its features, where weights correspond to cue validities. Additionally, a baseline bias (inertia) is added to the score of the option that was chosen in the immediately preceding trial. This reflects a psychological tendency to repeat previous actions (or avoid them, if inertia is negative). Choice probabilities are generated via a softmax function with an inverse temperature, along with a lapse rate for random guessing.

**Rationale:** Following the critic's advice to strengthen the lag-1 sequential effect without introducing multi-trial complexity (which was rejected), the parameter ranges for `inertia` and `beta` have been widened to [-20.0, 20.0] and [0.1, 50.0], respectively. This allows the model to rely more heavily on the previous choice and produce sharper transitions, helping to better capture the strong short-lived sequential dependence observed in the human data.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `inertia`: `[-20.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    # Incorporate choice inertia from the previous trial
    inertia = float(parameters["inertia"])
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        if prev_resp == 0:
            score_a += inertia
        elif prev_resp == 1:
            score_b += inertia
            
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Decision-makers use a boundedly rational strategy called Tallying with Validity-Ranked Tie-Breaking to minimize cognitive load while still making informed choices. They first evaluate options by simply counting the number of positive features for each (Tallying) and prefer the option with the highest tally. If the tallies are tied, they fall back to a non-compensatory tie-breaker, sequentially inspecting cues in descending order of their validity until one discriminates between the options. The resulting scores (tallies plus a potential tie-breaking bonus) are passed through a softmax function to generate choice probabilities, allowing for stochasticity, and a lapse rate is included to account for random guessing.

**Rationale:** Following the arbiter's guidance, this theory replaces strict one-reason decision making (Take-The-Best) and compensatory weighted sums (WADD) with a simpler boundedly rational approach: Tallying. Subjects count the number of positive features and choose the option with the higher tally. When a tie occurs, they rely on a non-compensatory tie-breaker, inspecting cues in descending order of validity. A separate 'tie_bonus' parameter controls how strongly the tie-breaker influences the softmax score, allowing the model to smoothly integrate the tie-breaker into the stochastic choice framework alongside a lapse rate.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `tie_bonus`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Tally positive features
    score_a = np.sum(a)
    score_b = np.sum(b)
    
    # Validity-Ranked Tie-Breaking
    if score_a == score_b:
        val = np.asarray(parameters["validities"], dtype=float)
        cue_order = np.argsort(-val, kind="stable").tolist()
        tie_bonus = float(parameters["tie_bonus"])
        
        for j in cue_order:
            if a[j] > b[j]:
                score_a += tie_bonus
                break
            elif b[j] > a[j]:
                score_b += tie_bonus
                break
                
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a purely non-compensatory Lexicographic strategy (Take-The-Best) to evaluate options. They search through the features strictly in descending order of cue validity. They stop and make a choice at the very first cue that discriminates between the two options. If no cues discriminate, they guess randomly. Response noise is modeled purely via an independent lapse rate (epsilon), as the heuristic yields a categorical preference.

**Rationale:** Following the critic's advice, we simplify the pure Take-The-Best (TTB) model by removing the redundant `beta` parameter. Since the TTB heuristic produces a categorical winner (or a tie), the softmax temperature `beta` merely acts as a constant scaling factor on the choice probabilities across all discriminating trials, which is completely degenerate with the lapse rate `epsilon`. By assigning a strict probability of 1.0 to the lexicographic winner (and 0.5 for ties) and mixing it with a uniform distribution via `epsilon`, we achieve the most parsimonious instantiation of the pure lexicographic family without losing any expressive power.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    p_core = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_core = np.array([0.0, 1.0])
            break

    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
