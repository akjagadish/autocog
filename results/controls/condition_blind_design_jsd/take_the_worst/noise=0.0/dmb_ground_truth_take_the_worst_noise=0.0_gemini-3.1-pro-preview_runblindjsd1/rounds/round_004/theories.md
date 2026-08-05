# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) with Choice Inertia: Decision-makers evaluate options by computing a weighted sum of their features, where the weights correspond to the given cue validities. However, choices are not independent across trials. Individuals exhibit a sequential dependency (choice inertia or alternation) where the probability of selecting an option is biased by whether it was chosen on the immediately preceding trial. This is modeled by adding an inertia parameter to the logit of the previously chosen option before applying the softmax response rule.

**Rationale:** Following the critic's advice, the upper bound of 'beta' is further restricted to 1.5 to modestly increase the softmax temperature, adding principled stochasticity. Additionally, the upper bound of 'inertia' is expanded to 4.0 to allow the model to capture stronger sequential dependencies, which directly drive the conditional choice probabilities. 'epsilon' is kept at [0.0, 0.2].

**Parameters:**
  - `beta`: `[0.1, 1.5]`
  - `epsilon`: `[0.0, 0.2]`
  - `inertia`: `[0.0, 4.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted additive evaluation
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    inertia = float(parameters["inertia"])
    
    logits = beta * np.array([score_a, score_b])
    
    # Apply choice inertia based on the previous trial's response
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        if 0 <= last_choice < 2:
            logits[last_choice] += inertia
            
    # Numerically stable softmax
    z = logits - np.max(logits)
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


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** WADD with Feature-based Inertia: Decision-makers evaluate options by computing a weighted sum of their features based on cue validities, but their evaluation is also influenced by attentional persistence from the previous trial. Instead of a simple spatial or index-based choice inertia, individuals exhibit feature-based inertia: they are biased toward options that share positive features with the option they chose on the immediately preceding trial. This reflects a tendency for recently attended attributes to carry extra weight in the current decision.

**Rationale:** Following the arbiter's suggestion, this theory replaces simple index-based choice inertia with feature-based inertia. In index-based inertia, the model simply assumes people repeat 'Left' or 'Right'. In feature-based inertia, we assume that the specific attributes of the previously chosen option remain salient in working memory, biasing the decision-maker toward new options that possess those same features. We model this by computing the dot product between the current option's features and the previously chosen option's features, scaled by an inertia parameter, and adding it to the base WADD score.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `inertia`: `[-2.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    inertia = float(parameters["inertia"])
    
    # Base Weighted Additive (WADD) scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Apply feature-based inertia based on the previous trial's chosen option
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        last_a = np.asarray(history["option_a_ratings"][-1], dtype=float)
        last_b = np.asarray(history["option_b_ratings"][-1], dtype=float)
        
        last_chosen_features = last_b if last_resp == 1 else last_a
        
        # Add inertia proportional to the number of shared positive features
        score_a += inertia * np.sum(a * last_chosen_features)
        score_b += inertia * np.sum(b * last_chosen_features)

    logits = beta * np.array([score_a, score_b])
    
    # Numerically stable softmax
    z = logits - np.max(logits)
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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Take-The-Best (TTB) / Lexicographic Heuristic: Decision-makers evaluate options using a non-compensatory, fast-and-frugal heuristic. They rank features by their given validities and compare the options sequentially, starting from the most valid feature. The first feature that discriminates between the options determines the choice. If all features tie, the decision-maker guesses randomly. A single noise parameter (epsilon) captures occasional lapses where the individual chooses randomly instead of applying the heuristic.

**Rationale:** Following the arbiter's feedback, this model implements a purely non-compensatory Lexicographic / Take-The-Best (TTB) strategy, completely removing Weighted Additive (WADD) and inertia components. It relies on ranking features by validity and deciding based on the first discriminating feature. A single noise parameter (epsilon) is included to account for lapses or exploration, ensuring the model remains simple and directly tests the non-compensatory hypothesis.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(val)[::-1]
    
    # TTB logic: find the first discriminating feature
    p_a = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a = 1.0
            break
        elif b[idx] > a[idx]:
            p_a = 0.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Apply lapse rate for random exploration/noise
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    
    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
