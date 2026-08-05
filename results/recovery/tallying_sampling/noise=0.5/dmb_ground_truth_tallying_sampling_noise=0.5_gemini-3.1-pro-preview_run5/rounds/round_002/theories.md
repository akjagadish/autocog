# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying (Equal-Weight) Theory: People compare two options by simply counting the number of positive cues (features) for each option and choosing the one with the higher count. This theory posits that decision-makers ignore the varying validities of the cues, treating all features as equally important. It is a compensatory strategy because multiple cues can outweigh a single cue, but it is more frugal than a Weighted Additive (WADD) rule because it avoids multiplying by or storing cardinal validities. Response noise is modeled via a softmax function over the tally scores, along with an independent lapse rate.

**Rationale:** Following the critic's feedback, the Tallying mechanism is retained exactly as before, but the parameter ranges have been constrained to induce more stochasticity. Specifically, the upper bound of beta has been reduced to 5.0, and the lower bound of epsilon has been raised to 0.1. This prevents the model from making overly deterministic predictions and brings the expected choice shares closer to the empirical observations (e.g., lowering the Exp 1 prediction from ~0.87 toward ~0.69).

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.1, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: count the number of positive cues for each option
    # Since cues are binary (0 or 1), we can just sum them.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Noisy-Validity Take-The-Best (NV-TTB) Model: Decision makers use a strict lexicographic search (Take-The-Best), consulting cues in descending order of their validity and stopping at the first cue that discriminates between options. However, subjects have noisy internal representations of cue validities. On each choice, Gaussian noise is added to the objective validities, and cues are sorted based on these noisy values. This probabilistic cue ordering allows the model to capture aggregate deviations from pure TTB (and approach Tallying-like behavior when noise is high) while preserving the non-compensatory, one-reason decision mechanism at the single-trial level.

**Rationale:** The previous accepted candidate (Power-Weighted Additive) was a compensatory model, which explicitly violated the arbiter's instruction to use the Take-The-Best (TTB) mechanism family. To comply with the critic's instruction to implement TTB with probabilistic cue ordering, a larger rewrite of `predict` was necessary to replace the additive weighted sum with a sequential lexicographic search. In this Noisy-Validity TTB model, Gaussian noise (parameterized by `sigma`) is added to the validities on each simulated trial to determine the search order, and the search stops strictly at the first discriminating cue. By marginalizing over these noisy orderings, the model introduces probabilistic variation into TTB, allowing it to capture aggregate deviations from pure TTB (bridging the gap to Tallying when noise is high) without using the previously rejected probabilistic stopping rule.

**Parameters:**
  - `sigma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    epsilon = float(parameters["epsilon"])
    
    n_features = stim.shape[1]
    
    # Marginalize over noisy validities via sampling to produce choice probabilities
    n_samples = 200
    noise = np.random.normal(0, sigma + 1e-9, size=(n_samples, n_features))
    noisy_validities = validities + noise
    
    # Sort cues for each sample (descending order of noisy validity)
    cue_orders = np.argsort(-noisy_validities, axis=1)
    
    a, b = stim[0], stim[1]
    wins = np.zeros(2)
    
    for i in range(n_samples):
        winner = None
        for j in cue_orders[i]:
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            wins += 0.5
        else:
            wins[winner] += 1.0
            
    p_core = wins / n_samples
    
    # Incorporate lapse rate
    n_opts = 2
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Validity-Biased Tallying: Decision-makers evaluate options using a compensatory strategy where they compute a weighted sum of the features. However, instead of using the objective cue validities as weights (as in pure WADD) or completely ignoring them (as in pure Tallying), they use subjective weights that are heavily compressed toward equality. This means the weight of each cue is a mixture of a uniform value and its actual validity, capturing the dominant tallying behavior while allowing for a slight, noisy pull from the objective cue validities.

**Rationale:** The arbiter suggested a 'Validity-Biased Tallying' theory where decision-makers use a compensatory approach but with weights compressed heavily toward equality. This model implements this by introducing an `alpha` parameter that interpolates between pure Tallying (equal weights of 1.0) and Weighted Additive (WADD, weights equal to validities). By allowing `alpha` to vary, the model can capture the dominant tallying behavior while accounting for the slight pull of cue validities seen in the observed data, overcoming the rigidity of strict lexicographic or pure equal-weight models.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Interpolate between uniform weights (Tallying) and objective validities (WADD)
    w = (1.0 - alpha) * 1.0 + alpha * validities
    
    # Calculate weighted sum of features for each option
    scores = stim @ w
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate uniform lapse rate
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
