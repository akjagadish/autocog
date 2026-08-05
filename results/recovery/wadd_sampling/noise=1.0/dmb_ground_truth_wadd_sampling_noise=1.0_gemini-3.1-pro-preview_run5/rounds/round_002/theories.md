# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload. Instead of systematically integrating cue validities and feature vectors, they abandon structured decision strategies and resort to random guessing. Choice behavior is driven entirely by this stochasticity, with only a potential slight bias toward one spatial position (e.g., Option A or Option B) over the other.

**Rationale:** Across all experiments, the empirical metric values hover extremely close to 0.5, and the between-subject variance is very low. Previous deterministic or highly structured models (like Take The Best or Tallying) produce predictions that significantly deviate from 0.5 on specific trial types, leading to poor fits. This theory instantiates the arbiter's suggestion that subjects are cognitively overloaded and simply guess randomly, with a small parameter for side bias to capture any baseline spatial preference without over-committing to stimulus features.

**Parameters:**
  - `side_bias`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Under cognitive overload, subjects ignore the state (features) and just guess.
    # The choice probability is determined only by an intrinsic side bias.
    p_b = float(parameters.get('side_bias', 0.5))
    p_a = 1.0 - p_b
    return np.array([p_a, p_b])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** People compare options by computing a compensatory overall value for each option. This is done by summing the features of each option weighted by their respective validities (Weighted Additive rule). Because empirical behavior in these experiments is highly stochastic (near random guessing), choice is subject to significant softmax noise and lapse rates. Narrowing the inverse temperature bounds forces the model to capture this high level of noise.

**Rationale:** Following the critic's advice, the upper bound of beta is narrowed further from 2.0 to 1.0. This restricts the optimizer to a maximally noisy regime, allowing the WADD model to more easily capture the highly stochastic (~0.50) empirical targets observed in the experiments without being drawn to deterministic parameter points.

**Parameters:**
  - `beta`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
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
        raise ValueError("Length of validities must match number of features.")

    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    a_score = np.sum(a * val)
    b_score = np.sum(b * val)
    scores = np.array([a_score, b_score])

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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Tallying under Overload (Equal Weights): Under cognitive overload without trial-by-trial feedback, subjects abandon complex integration of cue validities. Instead, they fall back on a highly simplified Equal Weights heuristic, merely tallying the total number of positive features (1s) for each option. Even with this simplification, the high cognitive demand leads to near-random choice behavior, which is captured by extreme softmax noise and a very high lapse rate.

**Rationale:** Following the arbiter's feedback, this theory replaces the Weighted Additive (WADD) approach with an Equal Weights (Tallying) heuristic. By ignoring cue validities and merely counting the number of positive features per option, the model avoids overpredicting sensitivity to score differences. To capture the empirically observed near-random behavior, the tallying process is coupled with a very high lapse rate (epsilon in [0.8, 1.0]) and extreme softmax noise (beta in [0.0, 0.2]). This provides a structured but simplified alternative to pure random guessing.

**Parameters:**
  - `beta`: `[0.0, 0.2]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Equal Weights / Tallying: count the number of positive features (1s) for each option
    a, b = stim[0], stim[1]
    a_score = np.sum(a)
    b_score = np.sum(b)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Apply high lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
