# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People make decisions between options by computing a fully compensatory, weighted sum of each option's features, using the provided cue validities as weights (Weighted Additive Model, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), decision-makers integrate all available evidence proportionally to its reliability. Response selection is probabilistic, modeled as a softmax over the weighted sums with an inverse temperature parameter and an independent lapse rate. The inverse temperature parameter is constrained to very low values to reflect the high degree of stochasticity observed in human choices.

**Rationale:** Following the critic's advice, the upper bound of the inverse temperature parameter `beta` has been further lowered from 5.0 to 2.0. This minimal edit restricts the model from becoming too deterministic, enforcing the softer, more probabilistic choices necessary to fully close the gap with the empirical data where choice proportions sit closer to 0.5. The core WADD mechanism remains unchanged.

**Parameters:**
  - `beta`: `[0.01, 2.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {stim.shape[1]}."
        )

    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Noisy Take-The-Best: Decision-makers evaluate options lexicographically, starting with the most valid cue and stopping at the first cue that discriminates between the options. However, the execution of this heuristic is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic TTB predictions with heavy noise.

**Rationale:** Following the arbiter's suggestion, this model instantiates a 'Noisy Take-The-Best' (Lexicographic) theory. It uses the standard TTB stopping rule (checking cues in descending validity order until one discriminates) but incorporates massive response noise to capture the ~0.50 empirical metrics across experiments. By constraining the lapse rate `epsilon` to a very high range [0.7, 1.0], the model acknowledges the heavy stochasticity in human data, providing a non-compensatory heuristic alternative to WADD while correcting the deterministic failure of the previous TTB model.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.7, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.ones(2) / 2.0
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Noisy Equal Weights (Tallying): Decision-makers find integrating specific cue validities too cognitively demanding. Instead, they evaluate options by simply counting the number of positive features (1s) for each option, treating all cues equally, and preferring the option with the higher total. However, their choices are dominated by an extremely high lapse rate (epsilon ~0.90-1.0), meaning they almost always guess randomly. This accounts for the observed aggregate behavior being essentially at chance across multiple experimental metrics.

**Rationale:** Following the critic's advice, the Noisy Tallying mechanism is kept exactly the same, but the epsilon parameter range is shifted higher to [0.90, 1.0]. This minimal edit further washes out the deterministic tallying predictions, bringing the simulated metrics closer to the near-perfectly random behavior (0.50 choice proportion / 0.0 score difference) observed in the human data across experiments.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.9, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count the number of positive features (1s) for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

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
