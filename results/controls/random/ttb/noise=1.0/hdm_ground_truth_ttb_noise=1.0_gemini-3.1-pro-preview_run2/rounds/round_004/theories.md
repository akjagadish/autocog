# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5_1` — KILLED ✗

**Description:** High-Temperature Weighted Additive (WADD) Theory: Subjects attempt to integrate all available features by computing a weighted sum of feature differences, using subjective validities as weights. However, their decision process is subject to extremely high cognitive noise (a very low inverse temperature, beta, in the softmax choice rule). This results in behavior that appears nearly random, capturing the high lapse rates observed across experiments, while still retaining a slight sensitivity to large aggregate differences in option quality.

**Rationale:** Following the arbiter's recommendation, the parameter range for the inverse temperature `beta` is strictly shrunk to [1e-5, 1e-4]. This drastically flattens the softmax probabilities to nearly uniform, closely matching the random guessing baseline while preserving a minuscule systematic sensitivity to large aggregate feature differences observed in the data.

**Parameters:**
  - `beta`: `[0.00001, 0.0001]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute scores as weighted sum of features
    scores = stim @ validities
    
    # Extremely high cognitive noise (low beta)
    beta = float(parameters["beta"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

**Rationale:** The arbiter noted that the previous WADD theory was degenerate because participants do not exhibit strong compensatory behavior, and aggregate choices are consistently near 0.5. This implies that subjects are likely overwhelmed by the cardinal feature magnitudes and simply guess. A pure Random Guessing model explicitly instantiates this hypothesis, capturing the ~0.5 metric values across all experiments without relying on a mismatched mechanism like TTB with inflated thresholds.

**Parameters:**
  - `bias`: `[0.48, 0.52]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])
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

**Description:** Magnitude-Insensitive Tallying Heuristic: Subjects ignore the cardinal magnitude of feature differences and instead simply tally the number of features that favor each option. They then probabilistically choose the option with the higher tally using a high-noise softmax choice rule. This explains why extreme rating scales do not produce deterministic choices, keeping behavior near random guessing for balanced or complex choices, while still allowing for slight deviations from 50% when one option has a clear majority of winning features.

**Rationale:** Following the arbiter's recommendation, this theory implements a Magnitude-Insensitive Tallying approach. By counting the number of winning features for each option rather than integrating their cardinal values, the model avoids the extreme deterministic predictions that WADD generates when exposed to massive rating scales (e.g., 10,000 or 100,000). A small beta parameter applies high cognitive noise over these tallies, ensuring that tied or closely matched tallies result in near 50% guessing, perfectly matching the high lapse rates observed across the experiments, while still capturing the slight behavioral pull toward an option when it clearly dominates on the feature count.

**Parameters:**
  - `beta`: `[0.0, 0.2]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Tally how many features favor each option
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    
    beta = float(parameters["beta"])
    
    # Softmax over tallies
    z = beta * np.array([tally_a, tally_b])
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
