# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

**Rationale:** As advised by the critic, the Tallying mechanism underpredicts TTB consistency on these compensatory trials because it deterministically favors the non-TTB option. To pull the predictions closer to the observed ~0.40-0.42 range (closer to 0.5), we further restrict the parameters to enforce higher stochasticity. We lowered the upper bound of beta to 1.0 and shifted the epsilon range to [0.3, 0.8] to allow for even more uniform lapsing, bridging the remaining gap toward the observed data.

**Parameters:**
  - `beta`: `[0.01, 1.0]`
  - `epsilon`: `[0.3, 0.8]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) Decision Theory with Flexible Sigmoid Subjective Validity Transformation

**Rationale:** Following the critic's advice, a free parameter `delta` is introduced to represent the inflection point of the sigmoid curve used for subjective validity transformation. This allows the model to flexibly shift the psychophysical mapping of validities to weights, potentially restoring the fit on Experiments 1, 2, 5, and 6 while maintaining the improvements achieved on Experiments 3 and 4. The model remains strictly within the WADD framework.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 5.0]`
  - `delta`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    
    # Transform raw probabilities into subjective weights using a sigmoid function
    w = 1.0 / (1.0 + np.exp(-gamma * (validities - delta)))
    
    # Calculate the overall score for each option by multiplying cue values by subjective weights
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice based on the weighted scores
    z = beta * scores
    z = z - np.max(z)  # for numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate for uniform guessing
    n_opts = len(p_core)
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Decision-makers probabilistically sample between a compensatory Equal-Weight (Tallying) heuristic and a non-compensatory Lexicographic (Take-The-Best) heuristic on a trial-by-trial basis, but they heavily favor Tallying. Furthermore, when they do use Take-The-Best, the application of the rule is subject to decision noise (modeled via a softmax temperature), reflecting uncertainty or stochasticity in identifying the most valid cue rather than a perfectly deterministic lexicographic choice.

**Rationale:** Following the critic's diagnosis, the previous model overpredicted TTB-consistent choices because it allowed too much reliance on a strictly deterministic TTB rule. To fix this, I made a minimal edit: I constrained `w_tally` to [0.5, 1.0] so the model defaults to Tallying the majority of the time. Additionally, I softened the TTB component by assigning it a new softmax temperature (`beta_ttb`), making the application of the lexicographic rule itself probabilistic rather than a rigid [1.0, 0.0] choice. This preserves the hybrid mechanism but calibrates it closer to the successful pure-Tallying baseline.

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `beta_ttb`: `[0.01, 5.0]`
  - `w_tally`: `[0.5, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying: sum the unweighted feature values for each option
    scores_tally = stim.sum(axis=1)
    beta = float(parameters["beta"])
    z = beta * (scores_tally - scores_tally.max())
    e = np.exp(z)
    p_tally = e / e.sum()
    
    # Take-The-Best (TTB): find the first discriminating cue ordered by validity
    order = np.argsort(validities)[::-1]
    diff = stim[0, order] - stim[1, order]
    non_zero = np.where(diff != 0)[0]
    
    p_ttb = np.array([0.5, 0.5])
    if len(non_zero) > 0:
        first_diff = diff[non_zero[0]]
        score_ttb = np.array([1.0, 0.0]) if first_diff > 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * score_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
            
    # Mixture of Tallying and TTB
    w_tally = float(parameters["w_tally"])
    p_mix = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Global lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mix)
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
