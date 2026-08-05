# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Noisy Weighted Additive (WADD): Decision-makers compute a fully compensatory, weighted sum of the features for each option, using the provided cue validities as weights. They then choose based on the difference in these weighted sums. However, the execution of this strategy is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic WADD predictions with heavy noise.

**Rationale:** Following the arbiter's feedback, this theory replaces the Noisy Take-The-Best mechanism with a Noisy Weighted Additive (WADD) mechanism. It computes a fully compensatory weighted sum of features using the provided cue validities. To capture the near-random aggregate behavior observed across the experiments (where most metrics hover around 0.50), the model is constrained to have a very high lapse rate (epsilon in [0.8, 1.0]). This approach explicitly weights cues by their validity, capturing subtle deviations from 0.50 that Tallying might miss, while still respecting the heavily noise-dominated nature of the subjects' choices.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
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
    return np.random.choice(len(p), p=p)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Mixture of Take-The-Best and Tallying: Decision-makers rely on a mixture of two simple heuristics rather than a fully compensatory weighted additive strategy. Depending on the trial or individual, they either use Take-The-Best (evaluating features sequentially by validity and stopping at the first discriminating cue) or Tallying (simply counting the number of positive features). The overall behavior is highly stochastic due to a large lapse rate (frequent random guessing), which effectively washes out the deterministic heuristic predictions to match the near-chance aggregate observed across experiments.

**Rationale:** The previous WADD theory completely failed the adversarial test in Experiment 10, predicting a compensatory signal orthogonal to Tallying that was not present in the empirical data. To resolve this, this new theory replaces WADD with a mixture of two non-compensatory/equal-weight heuristics: Take-The-Best (TTB) and Tallying. By blending these two strategies and maintaining the high lapse rate (epsilon) needed to capture the nearly random empirical choices, the model can flexibly capture the subtle deviations from chance across all experiments without predicting the flawed compensatory WADD signal.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # TTB prediction
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner == 0:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Tallying prediction
    score_a = np.sum(a)
    score_b = np.sum(b)
    if score_a > score_b:
        p_tally = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return np.random.choice(len(p), p=p)
```
