# Round 2 — Theories

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


### slot 2 — `pi_3` — KILLED ✗

**Description:** People make decisions by computing the expected value of each option, integrating both the magnitude of the features and their subjectively weighted validities. In a Weighted Additive (WADD) strategy, every feature contributes to an option's total score proportionally to a non-linear transformation of its cue validity, capturing subjective distortion of probabilities or weights. Choice probabilities are generated via a softmax function over these weighted sums, with an additional lapse rate to account for random errors.

**Rationale:** Following the critic's feedback, a non-linear weighting parameter 'gamma' is introduced to transform the provided validities before calculating the expected values. This allows the model to capture subjective distortion of the cue validities (such as overweighting weak cues or exaggerating strong cues) while remaining strictly within the WADD framework, helping bridge the remaining gap to the empirical baseline.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = np.power(val, gamma)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

### `pi_5` → slot 2 (via `new_theory`)

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
