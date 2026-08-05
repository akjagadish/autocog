# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Sequential Dependency / Alternation: In the absence of correctness feedback and when facing complex multi-cue choices, subjects often disengage from evaluating the actual features of the options. Instead of relying on a static spatial position bias, subjects exhibit sequential dependencies in their choices, such as a tendency to repeat their previous choice (inertia) or to systematically alternate between Option A and Option B. This history-dependent strategy completely ignores the stimulus validities and values, making decisions solely based on the temporal sequence of past actions.

**Rationale:** The arbiter pointed out that Theory 2 failed because it assumed feature-based decision-making, which is not supported by the data across these experiments. Following the arbiter's recommendation, this new theory implements a 'Sequential Dependency / Alternation' mechanism. It posits that subjects ignore the features entirely (task disengagement) and instead base their current choice on their own previous response, governed by a single `p_repeat` parameter (where p > 0.5 implies repeating and p < 0.5 implies alternating). Because trial order is randomized per subject, this history-based strategy perfectly predicts 0.5 expected accuracy on feature-based metrics and 0.0 feature-driven variance, offering a strong, plausible alternative to the static position bias model.

**Parameters:**
  - `p_repeat`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    
    if not history["response"]:
        # First trial: no history, predict uniformly
        return np.array([0.5, 0.5])
    
    last_resp = history["response"][-1]
    
    if last_resp == 0:
        # Last response was Option A
        prob_a = p_repeat
        prob_b = 1.0 - p_repeat
    else:
        # Last response was Option B
        prob_a = 1.0 - p_repeat
        prob_b = p_repeat
        
    return np.array([prob_a, prob_b])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

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
