# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Rationale:** Following the critic's feedback, the Tallying mechanism remains exactly the same, but the range of the `beta` parameter has been further restricted from [0.1, 5.0] to [0.1, 1.5]. This will continue to soften the softmax choices, pulling the Experiment 2 prediction down closer to 0.67 and the Experiment 1 prediction up closer to 0.35, resulting in an even better fit across both datasets.

**Parameters:**
  - `beta`: `[0.1, 1.5]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Probabilistic Strategy Mixture: Decision-makers do not blend strategies into a single value score, nor do they use deterministic tie-breaking. Instead, they probabilistically sample a decision strategy on each trial. The predominant strategy is Pure Tallying (Equal Weight), which counts positive features and ignores validities. However, on a small proportion of trials, they employ a fully compensatory Weighted Additive (WADD) strategy that integrates explicit cue validities. This mixture of pure heuristics preserves the strong tallying dominance and low validity bias in certain environments, while the occasional use of WADD produces the graded validity correlations observed in others.

**Rationale:** Following the arbiter's recommendation, this model implements a Probabilistic Strategy Mixture. Rather than blending tally scores and validity scores into a single composite utility (which alters the fundamental choice dynamics on every trial) or using a deterministic gate, the model computes the choice probabilities for Pure Tallying and WADD independently. It then takes a weighted average of these probabilities, representing the likelihood of using each strategy on a given trial. This preserves Tallying's exact indifference when counts are tied (explaining near-zero validity bias in Exp 1 and 7) while the WADD mixture component captures the graded correlation with validity differences seen in Exp 2 and 8.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `p_val`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Strategy 1: Pure Tallying
    tally_scores = stim.sum(axis=1)
    
    # Strategy 2: Weighted Additive (WADD)
    validities = np.asarray(parameters["validities"], dtype=float)
    val_scores = stim @ validities
    
    beta = float(parameters["beta"])
    p_val = float(parameters["p_val"])
    epsilon = float(parameters["epsilon"])
    
    # Probabilities under Tallying
    z_t = beta * (tally_scores - np.max(tally_scores))
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Probabilities under WADD
    z_w = beta * (val_scores - np.max(val_scores))
    e_w = np.exp(z_w)
    p_wadd = e_w / e_w.sum()
    
    # Probabilistic strategy mixture
    p_mix = (1.0 - p_val) * p_tally + p_val * p_wadd
    
    # Add uniform lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Compressive Weighted Additive: Decision-makers evaluate options using a weighted additive strategy, but their subjective feature weights are systematically compressed toward uniform weighting. This means they treat all features as more equal than the objective validities imply. By interpolating between objective validities and equal weights, the model naturally produces choices that heavily resemble Tallying (due to strong compression) while still exhibiting a subtle, reliable sensitivity to cue validities across all trials. The decision noise (beta) is constrained to prevent tiny residual score differences from artificially amplifying into overly deterministic choices, and the compression (gamma) is forced to be strong.

**Rationale:** Reverting to the linear interpolation mechanism from the accepted Iteration 2 base, as the non-linear compression caused regressions. To address the remaining overpredictions of validity bias in Experiments 7 and 8 without relying solely on increased decision noise, we restrict the gamma parameter range to [0.5, 1.0]. This forces the model to heavily weight the Tallying component, guaranteeing strong weight compression and better capturing the subtle nature of the validity bias.

**Parameters:**
  - `beta`: `[0.1, 2.5]`
  - `gamma`: `[0.5, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compress weights towards equal weighting (1.0)
    # gamma = 1.0 represents pure equal weighting (Tallying)
    # gamma = 0.0 represents exact objective validities (WADD)
    subjective_weights = (1.0 - gamma) * validities + gamma * 1.0
    
    # Calculate options scores using the compressed weights
    scores = stim @ subjective_weights
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
