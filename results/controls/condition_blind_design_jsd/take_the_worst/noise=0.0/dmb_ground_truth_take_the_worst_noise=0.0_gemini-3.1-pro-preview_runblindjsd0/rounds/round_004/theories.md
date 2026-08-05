# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Weighted Additive with Choice Inertia (WADD-CI): Decision-makers integrate all available feature information in a compensatory manner by weighting each feature by its cue validity. However, choices are not made in isolation; they are subject to sequential dependence. The model incorporates 'choice inertia' by adding a utility bonus to the option that was chosen on the immediately preceding trial, capturing the empirically observed conditional response distributions where people tend to repeat their past choices. The final preference is mapped to probabilities via a softmax function and an independent lapse rate.

**Rationale:** Following the arbiter's guidance, this theory implements a Weighted Additive (WADD) mechanism that integrates all features weighted by their validities, preserving the compensatory nature of the evaluation. To address the mechanistic failure of previous memoryless models that could not explain trial-to-trial dependencies, we introduce a 'choice inertia' parameter (phi). This parameter adds a utility bonus to the option chosen on the previous trial, allowing the model to naturally explain the distinct conditional response distributions seen in the experimental data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `phi`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD-CI expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compensatory integration: Weighted Additive (WADD)
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    
    # Sequential dependence: Choice Inertia
    phi = float(parameters["phi"])
    if len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        if prev_resp == 0:
            score_a += phi
        elif prev_resp == 1:
            score_b += phi
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with lapse rate
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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


### slot 2 — `pi_3` — KILLED ✗

**Description:** People integrate information across all features by weighting each feature by its subjective validity. The Weighted Additive (WADD) model computes an overall value for each option by summing the products of the feature values and their corresponding cue validities. This mechanism allows for compensatory decision-making, where multiple weak cues can collectively override a single strong cue, unlike non-compensatory heuristics (e.g., Take The Best) or unweighted integration (e.g., Tallying). The resulting option values are transformed into choice probabilities via a softmax function, with an independent lapse rate to capture random guessing or attentional errors.

**Rationale:** The Weighted Additive (WADD) model directly addresses the arbiter's feedback by integrating information across all features rather than relying on a single cue (like Take The Best) or ignoring cue validities (like Tallying). By weighting each feature's contribution by its validity, the model naturally supports compensatory decision-making, where a combination of lesser cues can outweigh a single highly valid cue. This approach provides a more nuanced and continuous evaluation of the options.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate WADD scores by weighting each feature by its validity
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Feature-Based Priming (FBP): Decision-makers integrate feature information using a compensatory strategy, but the subjective weighting of features is dynamically updated based on recent experience. Specifically, the features present in the previously chosen option become primed, temporarily increasing their salience and subjective weight on the subsequent trial. This mechanism accounts for sequential dependence not through simple motor or choice inertia (repeating the same action), but through a content-dependent attentional shift towards recently favored attributes.

**Rationale:** Following the arbiter's suggestion, this theory replaces simple choice inertia with Feature-Based Priming. Instead of adding a fixed utility bonus to the previously chosen side (which only captures raw action repetition), this model temporarily increases the subjective weight of the specific features that were present in the previously chosen option. This provides a more cognitively plausible, content-dependent sequential mechanism that can capture trial-to-trial dependencies even when the specific options change, as long as they share features with past choices.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("FBP expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Feature-based priming: boost weights of features present in the previously chosen option
    if len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        prev_a = np.asarray(history["option_a_ratings"][-1], dtype=float)
        prev_b = np.asarray(history["option_b_ratings"][-1], dtype=float)
        prev_chosen_features = prev_a if prev_resp == 0 else prev_b
    else:
        prev_chosen_features = np.zeros_like(validities)
        
    current_weights = validities + gamma * prev_chosen_features
    
    score_a = np.sum(a * current_weights)
    score_b = np.sum(b * current_weights)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with lapse rate
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
