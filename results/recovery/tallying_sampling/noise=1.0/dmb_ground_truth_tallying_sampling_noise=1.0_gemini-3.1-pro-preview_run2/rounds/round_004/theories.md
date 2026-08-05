# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) with Extreme Noise Theory: Subjects attempt to integrate all available cues by weighting them according to their provided validities. However, the cognitive demand of integrating multiple conflicting fictitious validities is overwhelming. This results in extreme decision conflict and distraction, leading to a near-total reliance on guessing (a very high lapse rate). Consequently, choice probabilities are pulled almost entirely toward chance (0.50), masking the underlying compensatory process in the aggregate behavioral data.

**Rationale:** Following the arbiter's suggestion, this theory implements a Weighted Additive (WADD) model combined with an extreme noise component (lapse rate between 0.95 and 1.0). The previous TTB with Extreme Noise model (pi_4) performed perfectly by recognizing that choices in these experiments are heavily dominated by guessing, clustering around 0.50. This new model tests the hypothesis that subjects might be attempting a fully compensatory integration of cues rather than a non-compensatory one, but the extreme difficulty of the task still drives them to guess almost all the time. This perfectly balances conflicting cues and reproduces the ~0.50 choice proportions across all metrics without relying on a one-reason heuristic.

**Parameters:**
  - `beta`: `[0.0, 2.0]`
  - `epsilon`: `[0.95, 1.0]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Weighted sum per option (dot product with per-feature weights)
    scores = stim @ (validities * w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with extreme uniform lapse (guessing)
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Pure Random Guessing (Complete Cognitive Overload): Subjects completely ignore the cues and validities due to the high cognitive demand and artificial nature of the fictitious task. This results in choices that are strictly driven by a 50/50 binomial coin flip, without any underlying compensatory or non-compensatory signal.

**Rationale:** Following the arbiter's feedback, this theory drops all complex heuristics in favor of a pure guessing model. It assumes that the cognitive overload of the task causes subjects to completely ignore cues and validities, resulting in a strict 50/50 coin flip. This explains why aggregate choice probabilities hover around 0.5 across most experiments, without relying on masked underlying signals.

**Parameters:**
  - `guessing_rate`: `{0.5}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    p = float(parameters['guessing_rate'])
    return np.array([p, 1.0 - p])
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Take-The-Best (TTB) with Extreme Cognitive Noise

**Rationale:** Following the arbiter's feedback, we implement Take-The-Best (TTB) with extreme cognitive noise. Subjects are heavily overwhelmed by the task and guess randomly on the vast majority of trials (represented by an epsilon parameter strictly between 0.95 and 1.0). On the rare occasions they do process the stimuli, they do not attempt to integrate all cues (which WADD assumes and which fails to capture the subtle non-compensatory signals). Instead, they rely strictly on the single most valid discriminating cue (TTB). This preserves the near-0.50 baseline across most experiments while injecting the correct underlying non-compensatory directional signal in the data.

**Parameters:**
  - `epsilon`: `[0.95, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    epsilon = float(parameters['epsilon'])
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind='stable').tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    p = np.array([0.5, 0.5])
    if winner is not None:
        p[winner] = 1.0
        p[1 - winner] = 0.0
        
    # Blend deterministic TTB choice with extreme uniform lapse (guessing)
    p_final = (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
