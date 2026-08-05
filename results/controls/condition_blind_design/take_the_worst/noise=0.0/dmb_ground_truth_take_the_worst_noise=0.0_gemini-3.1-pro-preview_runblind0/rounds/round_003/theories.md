# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** People process feature arrays from right to left, exhibiting a recency or layout bias. Instead of consulting cues in order of their objective validities, subjects use a Right-to-Left Take-The-Best (R2L-TTB) heuristic. They evaluate features starting from the last index down to the first, choosing the option that wins on the first discriminating feature encountered in this reversed order. If no feature discriminates, they guess. Response noise is incorporated via a softmax over the binary winner score and a constant lapse rate.

**Rationale:** Following the arbiter's suggestion, this model instantiates a Right-to-Left Take-The-Best heuristic. Instead of using provided validities, it assumes subjects process features starting from the rightmost feature (index n-1) and move leftwards, stopping at the first discriminating feature. This recency-biased lexicographic approach captures the strong negative effect observed in Experiment 4 and aligns better with the lower rates of TTB-consistent choices in Experiments 1 and 2 compared to the objective-validity TTB model.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("R2L-TTB expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    # Right-to-left cue order
    cue_order = list(range(n_features - 1, -1, -1))
    
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
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for choice probability
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Standard Take-The-Best (Validity-Ordered Search)

**Rationale:** Following the arbiter's guidance, this proposes a standard Take-The-Best (TTB) model where subjects search cues in descending order of their objective validities. In the current experimental setups, this corresponds to a Left-to-Right search. This model serves as a strong normative baseline to contrast against the Right-to-Left TTB theory, allowing us to definitively arbitrate whether participants are following a rational validity-based search or a layout-driven recency bias.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Descending order of validities (Left-to-Right in these experiments)
    cue_order = np.argsort(val)[::-1]
    
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
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for choice probability
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** People integrate all available features but exhibit a strong spatial attention bias, weighting features more heavily the further to the right they appear. This Right-biased Compensatory Model evaluates options using a weighted additive sum where the subjective weight of each feature grows exponentially from left to right. This allows the model to capture behavior that appears strictly non-compensatory (like Right-to-Left Take-The-Best) when the growth factor is large, while still accommodating subjects who might integrate multiple cues in a compensatory manner with a heavy rightward bias. Decisions are made by applying a softmax choice rule over these weighted sums, alongside a baseline rate of random guessing.

**Rationale:** Following the arbiter's suggestion, this theory implements a Right-biased Compensatory Model. By weighting features exponentially higher from left to right, it bridges the gap between a purely non-compensatory heuristic (R2L-TTB, which it approximates when gamma > 2) and a fully compensatory tallying model (when gamma approaches 1). This allows us to test whether the layout bias strictly triggers a sequential non-compensatory process, or if subjects are still integrating multiple cues while merely shifting their attentional weights heavily to the rightmost features.

**Parameters:**
  - `gamma`: `[1.0, 10.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Right-biased WADD expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Weights increase exponentially from left to right
    weights = np.array([gamma ** j for j in range(n_features)])
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    # Softmax for choice probability
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
