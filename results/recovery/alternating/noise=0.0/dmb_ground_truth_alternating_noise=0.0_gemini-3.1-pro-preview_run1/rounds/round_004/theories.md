# Round 4 — Theories

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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Position Bias / Option A Default Theory: Participants do not systematically integrate feature validities or use complex heuristics. Instead, due to reading order (e.g., left-to-right or top-to-bottom) or motor ease, they exhibit a slight systematic bias toward choosing the first option presented (Option A). This structural bias is mixed with random guessing, leading to a state-independent choice probability where Option A is chosen slightly more often than Option B, regardless of the features of the options.

**Rationale:** Following the arbiter's recommendation, this theory replaces the Single-Cue Heuristic with a 'Position Bias' or 'Option A Default' mechanism. It discards feature integration entirely and introduces a state-independent slight preference for Option A mixed with random guessing. This serves as a strong alternative null hypothesis to pure 50/50 random choice, allowing us to rigorously test whether the residual variance in the data is better explained by a structural/motor bias rather than residual cue usage.

**Parameters:**
  - `p_A`: `[0.5, 0.6]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the state and history, relying solely on a structural bias toward Option A.
    p_A = float(parameters["p_A"])
    return np.array([p_A, 1.0 - p_A])
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Sequential Pattern Theory: Participants facing difficult or unrewarded binary choices often fall back on sequential heuristics rather than evaluating the options independently on each trial. They may exhibit response persistence (repeating the same choice) or response alternation (switching back and forth between Option A and Option B). This strategy maintains an overall 50% choice proportion for each option, but introduces significant trial-by-trial autocorrelation, explaining sequential dependencies in the choice data that independent random guessing (Zero-Intelligence) fails to capture.

**Rationale:** Following the arbiter's suggestion, this theory replaces the Position Bias model with a Sequential Pattern model. Instead of an independent bias for Option A or B, participants are modeled as having a sequential dependency (autocorrelation) in their choices, represented by the parameter 'alpha'. An alpha > 0.5 indicates response persistence, while alpha < 0.5 indicates response alternation. This captures the exact 50% overall choice proportion while providing a mechanistic explanation for trial-by-trial choice sequences that deviate from pure independent random guessing.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    alpha = float(parameters["alpha"])
    
    if len(history["response"]) == 0:
        return np.array([0.5, 0.5])
        
    prev_response = history["response"][-1]
    
    p = np.zeros(2)
    p[prev_response] = alpha
    p[1 - prev_response] = 1.0 - alpha
    
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
