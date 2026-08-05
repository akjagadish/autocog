# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

**Rationale:** The arbiter feedback indicates that the previous mechanistic models (which assume subjects actively process features via TTB or Tallying) fail because the empirical alignment scores hover around chance (0.50). This suggests that subjects are not systematically using the features. By implementing a 'Position Bias / Task Disengagement' theory, the model predicts choices based purely on a spatial bias (or random guessing if bias_a ~ 0.5). This naturally captures the experiment-invariant finding that subjects' choices are largely uncorrelated with the feature-based heuristic predictions.

**Parameters:**
  - `bias_a`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Strict Random Guessing: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects completely disengage from the task. They ignore the stimulus features, spatial positions, and their own choice history, instead falling back on a pure uniform random guessing strategy (flipping a coin) on every single trial.

**Rationale:** Following the arbiter's feedback, the prior theories (such as spatial bias or sequential alternation) introduced inter-subject variances that were not present in the observed data, which had exactly 0.0000 variance on many metrics. A Strict Random Guessing theory with exactly p=0.5 and zero free parameters directly models complete task disengagement, aiming to eliminate the spurious variance introduced by subject-specific bias or history dependence parameters.

**Parameters:**
  (none)

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Predict exactly 0.5 for both options regardless of stimulus or history
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Extreme Spatial Strategy / Extreme Position Bias: In the absence of correctness feedback and when confronted with complex, multi-cue choices, subjects completely disengage from the task. They adopt a degenerate strategy of pressing exactly one button (either always Option A or always Option B) for the entirety of the experiment. This stimulus-independent behavior ignores all feature values and previous history.

**Rationale:** Following the arbiter's feedback, this theory implements an Extreme Spatial Strategy where subjects completely disengage from the task and press only one button (either A or B) consistently throughout the experiment. By sampling a 'preferred_side' that is strictly 0 or 1, the model captures the exact 0.5000 mean absolute deviation from 0.5 observed in Experiments 9 and 10 (since subjects are either 0% or 100% on one side). It also perfectly matches the 0.0 variance in transitions (Experiments 7 and 8) because subjects never switch responses, and yields a 0.5 accuracy on balanced designs (Experiments 1, 2, 3, 4) since the degenerate strategy will be correct exactly half the time.

**Parameters:**
  - `preferred_side`: `{0, 1}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The subject has a single preferred side for the entire experiment
    preferred_side = int(parameters['preferred_side'])
    
    # Predict exactly 1.0 for the preferred option and 0.0 for the other
    if preferred_side == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
