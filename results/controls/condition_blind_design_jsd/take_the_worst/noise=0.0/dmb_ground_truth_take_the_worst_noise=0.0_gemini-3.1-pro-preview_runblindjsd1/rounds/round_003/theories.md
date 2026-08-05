# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

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


### slot 2 — `pi_3` — KILLED ✗

**Description:** People make decisions using a Weighted Additive (WADD) strategy. Instead of relying on a single discriminating cue (Take The Best) or treating all cues equally (Tallying), decision makers compute a comprehensive value for each option by summing its feature values weighted by their respective cue validities. The option with the higher total weighted value is favored. This compensatory approach allows multiple weak cues to outweigh a single strong cue, reflecting a more exhaustive integration of available information. Choice stochasticity is modeled via a softmax over the weighted sums with an inverse temperature parameter (beta), along with an independent lapse rate (epsilon) for random guessing.

**Rationale:** Following the arbiter's recommendation, this theory implements the Weighted Additive (WADD) heuristic. Unlike Tallying (which ignores cue validities) and Take The Best (which ignores all but the first discriminating cue), WADD integrates all available information by weighting each feature by its validity. This allows for compensatory decision-making where multiple weak cues can overcome a single strong one. The model computes the weighted sum for each option and uses a softmax function with a lapse rate to produce choice probabilities, capturing both the compensatory nature of the strategy and the inherent noise in human decision-making.

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
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute the weighted sum of features for each option
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Convert scores to probabilities using a numerically stable softmax
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

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
