# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4_1` — SURVIVED ✓

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


### slot 2 — `pi_5` — KILLED ✗

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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Sequential Choice Dependency under Overload: When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload and abandon structured decision strategies that integrate cue validities. Instead of a static side bias, their random-looking behavior is driven by sequential choice dependencies—specifically, a 'stay/switch' heuristic where their current choice is heavily influenced by their choice on the immediate previous trial.

**Rationale:** Following the arbiter's feedback, this theory maintains the core assumption of cognitive overload (ignoring the complex stimulus features) but replaces the static side bias with sequential choice dependencies. By introducing a 'p_stay' parameter, the model hypothesizes that subjects rely on a 'stay/switch' heuristic, meaning their current action is heavily influenced by their immediate previous action. This provides a distinct and testable mechanism for the stochastic behavior observed across experiments, without relying on cue integration.

**Parameters:**
  - `p_stay`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Under cognitive overload, subjects ignore the state (features).
    # Choice probability is determined by a sequential dependency on the previous choice.
    p_stay = float(parameters.get('p_stay', 0.5))
    
    if len(history['response']) == 0:
        # No previous choice, guess randomly
        return np.array([0.5, 0.5])
    
    last_response = history['response'][-1]
    
    if last_response == 0:
        p_a = p_stay
        p_b = 1.0 - p_stay
    else:
        p_a = 1.0 - p_stay
        p_b = p_stay
        
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
