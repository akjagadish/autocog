# Round 3 — Theories

**Verdict:** `new_model` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

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


### slot 2 — `pi_5` — SURVIVED ✓

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


## Replacement

### `pi_4_1` → slot 1 (via `new_model`)

**Description:** When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload. Instead of systematically integrating cue validities and feature vectors, they abandon structured decision strategies and resort to random guessing. Choice behavior is driven entirely by this stochasticity, with only a potential slight bias toward one spatial position (e.g., Option A or Option B) over the other.

**Rationale:** Following the arbiter's recommendation, the range for the `side_bias` parameter is heavily restricted to [0.48, 0.52]. The previous range of [0.3, 0.7] allowed for too much systematic side preference, leading to an overestimation of the mean absolute deviation from 0.5 in choice proportions. A tighter range closely mimics pure random guessing (with finite-trial binomial variance accounting for most of the observed ~0.04 deviation), better matching the empirical data in Experiments 7 and 8.

**Parameters:**
  - `side_bias`: `[0.48, 0.52]`

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
