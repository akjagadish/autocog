# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: People make decisions between multi-attribute options by ranking features according to their validities and comparing the options lexicographically. The decision-maker examines the feature with the highest validity first; if one option is strictly better on that feature, it is chosen immediately and no further features are considered. If there is a tie, they move to the second most valid feature, and so on. This non-compensatory, one-reason decision making process implies that a single highly valid cue can completely determine the choice, overriding any number of smaller advantages on less valid cues.

**Rationale:** Following the arbiter's recommendation, this proposes the Take-The-Best (TTB) heuristic. Unlike Tallying (which ignores validities) or WADD (which computes a compensatory sum), TTB is a non-compensatory lexicographic rule. It evaluates features sequentially by validity, making a deterministic choice based solely on the first discriminating feature. Response noise is handled via a standard lapse rate epsilon. This captures the strong directional effects seen in the data where decisions are driven entirely by the most informative cue.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    choice = None
    for idx in order:
        if a[idx] > b[idx]:
            choice = 0
            break
        elif b[idx] > a[idx]:
            choice = 1
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = 2
    
    if choice is None:
        p_core = np.ones(n_opts) / n_opts
    else:
        p_core = np.zeros(n_opts)
        p_core[choice] = 1.0
        
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Naïve Bayes / Log-Odds Weighted Additive (WADD) model: Decision makers integrate all available cues by weighting them according to their log-odds (the theoretically optimal Bayesian weights for independent cues). This transformation naturally amplifies cues with high validities, bridging compensatory integration and TTB-like non-compensatory behavior without requiring an extra free parameter for the non-linearity. A softmax choice rule then translates the difference in the options' overall subjective values into choice probabilities.

**Rationale:** Following the critic's advice, the arbitrary exponential transformation with parameter `gamma` has been replaced by the theoretically grounded log-odds (Naïve Bayes) transformation. This transformation naturally scales the validities such that high-validity cues are strongly amplified, mimicking TTB-like behavior, while still allowing compensatory integration of multiple lower-validity cues. By removing `gamma` and widening the `beta` range to [0.1, 25.0], the model retains the ability to make highly deterministic choices without the risk of over-compressing weights, leading to a more principled and robust soft-compensatory WADD formulation.

**Parameters:**
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities using log-odds (Naïve Bayes weights)
    v_clipped = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v_clipped / (1.0 - v_clipped))
    
    # Calculate overall subjective value for each option
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Mixture of Strategies (TTB and WADD): Decision makers are cognitively flexible and probabilistically alternate between a fast-and-frugal non-compensatory heuristic (Take-The-Best) and a compensatory, optimal evidence-integration strategy (Weighted Additive). On any given decision, an individual either relies solely on the most valid discriminating cue (TTB) or integrates all available cues weighted by their Bayesian log-odds (WADD). This hybrid approach explains both the strict adherence to high-validity cues on some trials and the compensatory influence of multiple weaker cues on others.

**Rationale:** Following the arbiter's suggestion, this theory replaces the pure WADD model with a probabilistic mixture of Take-The-Best (TTB) and Weighted Additive (WADD) strategies. Pure WADD fails to capture the extreme non-compensatory behavior seen on certain trials where large margins don't linearly increase choice probability, while pure TTB fails to account for trials where multiple weaker cues can overcome a single strong cue. By introducing a mixing weight (`w_ttb`), the model allows subjects to stochastically switch between a strict heuristic and compensatory integration, seamlessly bridging the gap between deterministic early-stopping and full evidence accumulation.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Strategy ---
    order = np.argsort(validities)[::-1]
    ttb_choice = None
    for idx in order:
        if a[idx] > b[idx]:
            ttb_choice = 0
            break
        elif b[idx] > a[idx]:
            ttb_choice = 1
            break
            
    if ttb_choice is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        p_ttb = np.zeros(2)
        p_ttb[ttb_choice] = 1.0
        
    # --- Weighted Additive (WADD) Strategy ---
    v_clipped = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v_clipped / (1.0 - v_clipped))
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    scores = np.array([score_a, score_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # --- Probabilistic Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
