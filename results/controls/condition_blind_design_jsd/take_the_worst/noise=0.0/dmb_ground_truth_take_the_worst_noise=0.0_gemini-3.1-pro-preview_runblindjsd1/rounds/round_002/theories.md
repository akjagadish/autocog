# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Tallying (Equal Weights) Theory: Decision-makers evaluate options by simply counting the total number of positive features each option possesses, completely ignoring the continuous cue validities. This frugal, compensatory heuristic assumes all features are equally important. Choice probabilities are generated via a softmax function over these unweighted feature counts, with an added lapse rate for random guessing.

**Rationale:** Following the arbiter's recommendation, this theory implements the Tallying (Equal Weights) heuristic. Instead of relying on a single cue (like Take The Best) or weighting features by their validities (like WADD), the decision-maker simply counts the number of positive features for each option. This is formalized by taking the unweighted sum of the binary feature vectors for each option. A softmax rule with inverse temperature (beta) translates the difference in tallies into choice probabilities, while a lapse rate (epsilon) accounts for random guessing.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
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


### slot 2 — `pi_3` — SURVIVED ✓

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

### `pi_5` → slot 1 (via `new_theory`)

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
