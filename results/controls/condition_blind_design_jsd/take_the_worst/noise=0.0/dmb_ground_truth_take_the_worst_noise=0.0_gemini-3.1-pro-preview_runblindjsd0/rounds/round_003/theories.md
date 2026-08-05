# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Take-The-Best (Lexicographic) Heuristic: Decision-makers evaluate features sequentially in order of decreasing cue validity. The first feature that discriminates between the two options strictly determines the preference, and all remaining lower-validity features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory strategy is formalized by assigning a binary score to the winning option based on the first discriminating cue, which is then mapped to choice probabilities via a softmax function (to capture decision noise) and a uniform lapse rate (to capture random errors).

**Rationale:** Following the arbiter's recommendation, this theory strictly implements the Take-The-Best (Lexicographic) heuristic. It differs from Tallying and WADD by employing a non-compensatory mechanism: cues are checked sequentially from highest to lowest validity, and the search terminates at the first discriminating cue. This single cue dictates the choice, ignoring all other features. The deterministic rule is softened by applying a softmax over the binary TTB scores to capture decision noise, along with an independent lapse rate for random attentional errors.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-The-Best expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    # stable sort ensures consistent tie-breaking for equal validities
    cue_order = np.argsort(-validities, kind="stable")
    
    score_a = 0.0
    score_b = 0.0
    
    # Evaluate cues sequentially
    for idx in cue_order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break

    scores = np.array([score_a, score_b])
    
    # If no cues discriminate, guess randomly
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_3` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
