# Round 2 — Theories

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


### slot 2 — `pi_3` — KILLED ✗

**Description:** Decision-makers use a Weighted Additive (WADD) strategy, evaluating each option by computing a weighted sum of its features, where the weights are directly proportional to the provided cue validities. The option with the higher weighted sum is favored. Response noise is modeled via a softmax over these weighted sums with an inverse temperature, along with a lapse rate for random guessing.

**Rationale:** In accordance with the arbiter's feedback, this implements the Weighted Additive (WADD) theory. Instead of discarding cue validities (like Tallying) or ignoring less valid cues (like Take The Best), WADD integrates all available information by weighting each feature by its validity. This compensatory mechanism is captured by taking the dot product of the feature vectors and the validity vector, then mapping the resulting scores to choice probabilities using a softmax rule with a lapse rate.

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
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a non-compensatory heuristic, specifically Take-The-Best (TTB), but their choices are also influenced by sequential dependencies like choice inertia. They evaluate options by inspecting features sequentially in descending order of validity. The first feature that discriminates between the options determines the preferred option, which is assigned a base value. However, the final choice probability is modulated by an inertia bias added to the option chosen in the previous trial. This combines frugal, one-reason decision making with psychological momentum (inertia), passed through a softmax function with a lapse rate for random guessing.

**Rationale:** The arbiter recommended a non-compensatory heuristic theory such as Take-The-Best (TTB), moving away from exhaustive integration like WADD. While a pure TTB model (pi_1) captures the frugal nature of human decision-making, it fails to account for the sequential dependencies (choice inertia) that strongly characterize the data in these experiments, as seen in the success of pi_4. By combining the sequential cue search of TTB with a choice inertia term added to the previously chosen option's score, this model captures both the non-compensatory evaluation of features and the trial-to-trial psychological momentum.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `inertia`: `[-10.0, 10.0]`
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
    
    score_a, score_b = 0.0, 0.0
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner == 0:
        score_a = 1.0
    elif winner == 1:
        score_b = 1.0
        
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
