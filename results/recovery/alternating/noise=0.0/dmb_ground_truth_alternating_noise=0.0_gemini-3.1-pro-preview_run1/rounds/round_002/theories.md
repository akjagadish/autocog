# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

**Rationale:** The arbiter requested a Random Choice or Zero-Intelligence theory because the empirical data consistently shows metric scores and choice proportions exactly at or very near 0.5 across all experiments. This indicates that subjects are not systematically relying on cue validities, tallying, or any complex integration of features. A simple 50/50 guessing model captures this task disengagement perfectly without requiring any parameters.

**Parameters:**
  (none)

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a weighted sum of their features, where the weights directly correspond to the cue validities. The probability of choosing an option is then determined by the difference in these weighted sums via a softmax function, subject to decision noise. To capture empirical data where choices appear highly noisy or evenly balanced across experimental manipulations, the decision noise parameters (softmax inverse temperature and random lapse rate) can span the full range from deterministic choice to complete uniform guessing. In this iteration, the inverse temperature is constrained to very low values, enforcing highly stochastic behavior.

**Rationale:** Following the critic's feedback, the upper bound of the inverse temperature parameter `beta` has been further tightened from 2.0 to 0.5. This almost entirely restricts the model to a highly stochastic regime, ensuring that the weighted differences translate into choices that are virtually indistinguishable from random guessing, effectively eliminating the residual deterministic bias to perfectly hit the 0.50 empirical baselines.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute the weighted sum of features for each option
    wadd_a = np.sum(stim[0] * val)
    wadd_b = np.sum(stim[1] * val)
    
    scores = np.array([wadd_a, wadd_b])
    
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
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Single-Cue Heuristic with Very High Lapse: Participants avoid the cognitive overload of integrating multiple cues by relying exclusively on the single most valid cue. They compare the two options on this single best feature, choosing the one that is superior. If the options tie on this cue, they guess randomly. However, due to task disengagement, lack of feedback, or cognitive fatigue, participants exhibit a near-total lapse rate, meaning they almost entirely guess randomly, with only a tiny residual trace of the single-cue strategy.

**Rationale:** Following the critic's advice, I restricted the range of the lapse parameter 'epsilon' from [0.0, 1.0] to [0.9, 1.0]. The empirical data consistently hovers around 0.5000, indicating near-random guessing. By constraining epsilon to be very high, the model correctly acts primarily as a zero-intelligence guesser, while still allowing for a minuscule residual effect of the single-cue heuristic. This prevents the large deviations from 0.5000 that were observed in the previous iteration.

**Parameters:**
  - `epsilon`: `[0.9, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Single-Cue expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single most valid cue
    best_cue_idx = np.argmax(val)
    
    a_val = stim[0, best_cue_idx]
    b_val = stim[1, best_cue_idx]
    
    epsilon = float(parameters["epsilon"])
    
    # Deterministic choice based on the single best cue
    if a_val > b_val:
        p_core = np.array([1.0, 0.0])
    elif b_val > a_val:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend with random guessing based on the lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
