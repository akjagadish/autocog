# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture: Tallying and Take-The-Best (TTB) with Elevated Noise. Decision-makers do not use a single compensatory mechanism that blends validities and uniform weights. Instead, they probabilistically sample between two distinct, non-compensatory heuristics on a trial-by-trial basis: pure Tallying and pure Take-The-Best. When Tallying, they simply count the number of positive cues and choose the option with the higher count (guessing if tied). When using TTB, they consult cues in descending order of objective validity and choose based on the first discriminating cue. Tallying is heavily favored over TTB in the mixture. Furthermore, subjects exhibit a substantial baseline level of random guessing (lapse rate), which dampens the extremity of both Tallying and TTB predictions.

**Rationale:** Following the critic's feedback, the parameter range for `epsilon` has been shifted from [0.0, 0.5] to [0.2, 0.6]. The previous model captured the relative proportion of Tallying vs. TTB correctly but was slightly too confident across all experiments compared to human data. By elevating the uniform lapse rate, the model's predictions are pulled closer to 0.50, aligning the peak Tallying rates and tied-trial validity biases perfectly with the empirical baselines.

**Parameters:**
  - `p_tally`: `[0.7, 1.0]`
  - `epsilon`: `[0.2, 0.6]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    p_tally = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Pure Tallying: count cues, ignoring validities
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        tally_p = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        tally_p = np.array([0.0, 1.0])
    else:
        tally_p = np.array([0.5, 0.5])
        
    # Pure Take-The-Best (TTB): lexicographic search based on validity
    order = np.argsort(-validities)
    ttb_p = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_p = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_p = np.array([0.0, 1.0])
            break
            
    # Mixture of the two heuristics
    p_core = p_tally * tally_p + (1.0 - p_tally) * ttb_p
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
