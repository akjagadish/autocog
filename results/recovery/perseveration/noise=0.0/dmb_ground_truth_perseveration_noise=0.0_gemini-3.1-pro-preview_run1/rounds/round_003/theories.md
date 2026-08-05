# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) Model: Decision-makers evaluate options by computing a compensatory score for each, weighting every feature by its subjective importance (the log-odds of its validity). To balance extreme validities, decision-makers apply a smoothing or regularization process (Laplace smoothing) to the validities before computing their log-odds. This allows multiple weaker cues to perfectly balance a single strong cue, naturally yielding near-tied evaluations for adversarial choice pairs.

**Rationale:** Following the critic's advice, we introduced a Laplace smoothing parameter `kappa` to regularize the validities before applying the log-odds transform. This allows the model to compress the log-odds weights in a stable, principled way, giving the optimizer the flexibility to gently balance strong versus weak cues to achieve perfectly tied WADD scores on adversarial trials, without the instability of power transformations.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `kappa`: `[0.0, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    # Laplace smoothing to regularize extreme validities
    val_smoothed = (val + kappa) / (1.0 + 2.0 * kappa)
    
    # Transform smoothed validities to log-odds for normative compensatory weighting
    # Clip to avoid log(0) or division by zero
    val_clipped = np.clip(val_smoothed, 0.001, 0.999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute WADD scores as the dot product of features and weights
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
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
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Task Disengagement / Position Bias Theory: Subjects find the fictitious product choice task unengaging or too cognitively demanding to evaluate systematically. Instead of using the expert ratings, they adopt a completely deterministic, zero-effort heuristic of exclusively selecting the same option position (e.g., always choosing Option A or always Option B) on every single trial.

**Rationale:** Following the arbiter's insight, the data perfectly reflects a deterministic position bias rather than random guessing or compensatory evaluation. A random guessing model predicts a ~0 absolute deviation from 0.5 in Experiment 5 (as responses average out to 0.5 per trial type), whereas the real data shows exactly 0.5, meaning subjects are 100% deterministic. Because all other experiments balance the 'correct' or 'heuristic' option equally between A and B, a purely deterministic position bias naturally yields exactly 50% match rates across all metrics, perfectly aligning with the experimental results.

**Parameters:**
  - `preferred_option`: `{0, 1}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    pref = int(parameters["preferred_option"])
    n_opts = np.asarray(state).shape[0]
    
    probs = np.zeros(n_opts)
    if pref < n_opts:
        probs[pref] = 1.0
    else:
        probs[0] = 1.0
        
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** First Impression Anchoring: Subjects find the task repetitive and cognitively demanding. To minimize effort, they genuinely evaluate the options on the very first trial using a simple tallying heuristic (choosing the option with the most positive features, or guessing if tied). They then use this initial choice as an anchor, deterministically repeating the same position choice (e.g., always choosing A or always B) for the remainder of the experiment. This results in a completely zero within-subject variance while maintaining an even 50/50 split across the population.

**Rationale:** Theory 1 (pi_3) predicted that subjects actively evaluate every trial, which produced non-zero within-subject variance, falsifying it against the 0.0000 variance seen in Experiments 7 and 8. Following the arbiter's suggestion, this new theory implements 'First Impression Anchoring'. Subjects evaluate only the first trial using a simple sum-of-features heuristic and then deterministically repeat that exact choice index for the rest of the experiment. This perfectly captures the zero within-subject variance while providing a more plausible cognitive mechanism than an a priori position bias, as the bias is formed dynamically based on the first trial encountered.

**Parameters:**
  - `dummy`: `{1.0}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # Dummy parameter to satisfy the contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # On the first trial, evaluate based on simple feature tallying
    if len(history.get("response", [])) == 0:
        sums = np.sum(stim, axis=1)
        max_sum = np.max(sums)
        is_max = (sums == max_sum).astype(float)
        return is_max / np.sum(is_max)
    else:
        # On all subsequent trials, repeat the first trial's choice
        first_resp = int(history["response"][0])
        probs = np.zeros(n_opts)
        if first_resp < n_opts:
            probs[first_resp] = 1.0
        else:
            probs[0] = 1.0
        return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
