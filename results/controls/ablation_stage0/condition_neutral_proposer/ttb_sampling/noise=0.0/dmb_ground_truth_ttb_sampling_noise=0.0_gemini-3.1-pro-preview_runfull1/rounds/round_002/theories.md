# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

**Rationale:** Following the arbiter's suggestion, this model implements the 'Take The Best' (TTB) heuristic. Instead of integrating all features compensatorily (like WADD) or equally (like Tallying), TTB searches through features in descending order of their validity and stops at the first feature that discriminates between the two options. The choice is made based entirely on this single cue. This provides a strong non-compensatory heuristic mechanism that tests whether subjects rely on the most valid discriminating cue rather than integrating multiple cues.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_2_1` — KILLED ✗

**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** Following the critic's advice, we apply a log-odds transformation to the objective validities to compute the subjective weights. This strictly monotonic transformation stretches out the weights for high-validity cues, increasing the difference in weighted scores between options. We also reduce the upper bound of the lapse rate (epsilon) to 0.2 to prevent artificial capping of the model's accuracy. These changes allow the model to confidently select the WADD-aligned option and match the high alignment (~0.86) observed in the data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Use the log-odds of objective validities as the subjective weights
    validities = np.asarray(parameters["validities"], dtype=float)
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)  # Prevent division by zero or log(0)
    weights = np.log(validities / (1.0 - validities))
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
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

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Probabilistic Take The Best (PTTB) posits that decision-makers search through features sequentially in order of their subjective validities. However, unlike deterministic Take The Best, the stopping rule is stochastic: upon finding a feature that discriminates between the two options, the decision-maker stops and bases their choice on that feature with probability gamma. If they do not stop, they continue to the next feature. If all features are evaluated without a decision, they guess randomly. This model captures the primarily non-compensatory nature of human decision-making while allowing for occasional deviations where lower-validity cues might influence the choice.

**Rationale:** Following the critic's feedback, the previous PTTB model exhibited too much stochasticity, pulling predictions toward 0.5. To address this, I tightened the parameter ranges: gamma is now constrained to [0.8, 1.0] to ensure a high stopping probability at the first discriminating cue, and epsilon is constrained to [0.0, 0.2] to reduce excessive uniform lapsing. This minimal edit retains the Probabilistic Take The Best mechanism while keeping the model closer to deterministic TTB, better matching the extreme non-compensatory choices observed in human data.

**Parameters:**
  - `gamma`: `[0.8, 1.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"PTTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    p_choice = np.array([0.0, 0.0])
    p_reach = 1.0
    
    # Search for discriminating cues with stochastic stopping
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_stop = gamma
            p_choice += p_reach * p_stop * np.array([1.0, 0.0])
            p_reach *= (1.0 - p_stop)
        elif stim[1, idx] > stim[0, idx]:
            p_stop = gamma
            p_choice += p_reach * p_stop * np.array([0.0, 1.0])
            p_reach *= (1.0 - p_stop)
            
    # If no decision is made after evaluating all cues, guess
    p_choice += p_reach * np.array([0.5, 0.5])
        
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_choice + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
