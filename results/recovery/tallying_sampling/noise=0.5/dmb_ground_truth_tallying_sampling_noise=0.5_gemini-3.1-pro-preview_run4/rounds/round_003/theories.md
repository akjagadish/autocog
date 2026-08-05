# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Strategy Mixture Theory: Decision-makers are heterogeneous in their strategies. A proportion of choices are made using a compensatory Weighted Additive (WADD) strategy that integrates cue validities and subjective weights, while the remainder rely on a simpler Tallying (Equal Weight) heuristic that merely counts the number of positive features. Behavior on any given trial is a probabilistic mixture of these two strategies, with the mixture proportion varying across individuals. This naturally explains why aggregate behavior falls between the pure predictions of WADD and Tallying.

**Rationale:** Based on the arbiter's feedback, neither pure WADD nor pure Tallying perfectly accounts for the aggregate experimental metrics, which consistently fall between the predictions of the two isolated theories. The Strategy Mixture theory directly instantiates the arbiter's suggestion by combining WADD and Tallying. It uses a continuous parameter `w_mix` to represent the probability of applying WADD vs Tallying, which models inter-subject heterogeneity in strategy selection and naturally bridges the gap between the two endpoints.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_mix`: `[0.0, 1.0]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_mix * p_wadd + (1.0 - w_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Non-linear Validity Scaling Theory: Decision-makers use a single compensatory strategy to integrate cues, but they subjectively transform the provided cue validities by raising them to a fractional exponent (gamma). This non-linear scaling flattens the differences between cue validities. As gamma approaches 0, the validities become equal, naturally mimicking the Tallying (equal-weight) heuristic. As gamma approaches 1, the model recovers pure Weighted Additive (WADD) behavior. This provides a mathematically elegant, single-process account of the empirical pull towards equal weighting without assuming a discrete mixture of distinct decision strategies.

**Rationale:** Following the critic's advice, the unconstrained 'weights' array parameter has been removed. The subjective weights are now computed strictly as `validities ** gamma`. This reduces overparameterization and forces the model to rely entirely on the non-linear scaling of the instructed validities, which should lower variance and improve cross-experiment generalization while remaining faithful to the arbiter's recommended Non-linear Validity Scaling mechanism.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform cue validities via fractional exponent
    subjective_validities = validities ** gamma
    
    # Calculate compensatory scores
    scores = stim @ subjective_validities
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Heuristic Toolbox (TTB + Tallying Mixture): Decision-makers are equipped with a repertoire of fast-and-frugal heuristics rather than complex compensatory algorithms. Specifically, individuals probabilistically switch between a non-compensatory Take-The-Best (TTB) heuristic and an equal-weight Tallying heuristic. TTB sequentially searches cues in order of validity and decides based on the first discriminating cue, capturing lexicographic decision-making. Tallying simply counts the number of positive features, capturing instances where cues are integrated equally. This 2-way mixture accounts for choices where subjects either rely on a single dominant cue or fall back to simple feature counting, without the excessive noise introduced by subjective compensatory weighting.

**Rationale:** Following the critic's recommendation, we simplify the Heuristic Toolbox model to a 2-way probabilistic mixture between Take-The-Best (TTB) and Tallying. Previous attempts to fit a 3-way mixture (including WADD) suffered from miscalibration and excessive noise due to subjective weighting and scale mismatches. By restricting the mixture to TTB and Tallying, we drastically simplify the parameter space and eliminate the problematic WADD component. This aligns with the arbiter's original suggestion that a TTB+Tally mixture can better capture empirical deviations from compensatory predictions, specifically modeling instances where subjects either rely on a single dominant cue or fall back to simple feature counting.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    
    # 1. Take-The-Best (TTB) strategy component
    p_ttb = np.array([0.5, 0.5])
    # Sort indices by validity descending
    order = np.argsort(validities)[::-1]
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            if stim[0, idx] > stim[1, idx]:
                p_ttb = np.array([1.0, 0.0])
            else:
                p_ttb = np.array([0.0, 1.0])
            break
            
    # 2. Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of TTB and Tallying
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```
