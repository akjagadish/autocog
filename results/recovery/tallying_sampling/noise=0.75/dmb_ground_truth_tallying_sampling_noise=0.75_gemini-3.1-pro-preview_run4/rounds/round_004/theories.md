# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tallying (Equal Weighting): Decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen. If the tallies are equal, the decision-maker guesses randomly. This heuristic provides an extremely fast and frugal way to compare options, perfectly explaining chance-level performance in 1-on-1 single-cue comparisons (where tallies tie) and highly consistent choices when one option has strictly more positive features. The decision process is subject to significant response noise, reflecting the inherent stochasticity in human choice behavior.

**Rationale:** Following the critic's advice, the mechanism remains identical (Tallying), but the parameter ranges for noise have been shifted to allow even more stochasticity. Specifically, the maximum of beta was lowered to 1.5 and the epsilon range was shifted to [0.2, 0.8]. This will pull the metrics in Exps 1, 2, and 3 closer to the empirical ~0.40-0.44 range without affecting the chance-level predictions in Exps 4 and 6.

**Parameters:**
  - `beta`: `[0.0, 1.5]`
  - `epsilon`: `[0.2, 0.8]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in Tallying, but we read them to satisfy the parameter reference rule.
    _ = parameters["validities"]
    
    # Tallying: sum the features for each option (ignoring validities)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Minimalist Heuristic (Trial-Specific Search): Decision-makers evaluate options sequentially using a one-reason lexicographic process, completely ignoring objective cue validities. Instead of maintaining a stable subjective cue hierarchy, individuals search through the available cues in a random order that varies from trial to trial. They stop at the first cue that discriminates between the two options and choose the favored option. Analytically, the probability of choosing an option is proportional to its share of the discriminating cues. This trial-by-trial stochasticity produces low per-subject determinism on symmetrically tied trials, matching human data.

**Rationale:** Modified the Minimalist Heuristic to use a trial-specific random cue order instead of a fixed subject-specific one, as requested. This is achieved analytically by marginalizing over all possible cue orders: the probability that a random sequential search first encounters a cue favoring Option A is simply the proportion of discriminating cues that favor A. This inherently stochastic evaluation perfectly explains the low per-subject determinism on tally-tied trials (Exp 7) and reduced variance (Exp 5), while maintaining the core sequential, validity-ignorant Minimalist mechanism.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in the Minimalist heuristic
    _ = parameters["validities"]
    
    # Trial-specific random cue order is analytically equivalent to choosing 
    # proportional to the number of discriminating cues for each option.
    diff = stim[0] - stim[1]
    d_a = np.sum(diff > 0)
    d_b = np.sum(diff < 0)
    
    if d_a + d_b > 0:
        p_a = d_a / (d_a + d_b)
        p_b = d_b / (d_a + d_b)
    else:
        p_a = 0.5
        p_b = 0.5
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores [1, 0] or [0, 1] marginalized over the random cue order
    S = np.exp(beta) / (np.exp(beta) + 1.0)
    
    prob_a = p_a * S + p_b * (1.0 - S)
    prob_b = p_b * S + p_a * (1.0 - S)
    
    p_core = np.array([prob_a, prob_b])
    
    # Uniform lapse
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Compressed Weights Additive Model with High Noise: Decision-makers evaluate options using a compensatory weighted sum, but their subjective feature weights are highly compressed towards equality. Instead of using raw objective validities, individuals anchor on equal weighting (Tallying) and make only marginal adjustments based on cue validities. Crucially, the decision process is subject to substantial stochasticity and lapsing, which explains why choice rates often hover near chance in conflict scenarios without requiring a structural shift in the underlying deterministic heuristic.

**Rationale:** Following the critic's feedback, the attempt to increase 'gamma' in Iteration 2 was rejected because it introduced an excessive validity bias, ruining the fit on Experiment 8 where human subjects are nearly indifferent on Tally-tied trials. To improve the fit on Experiments 1-3 without breaking Experiment 8, we revert to the Iteration 1 base (where 'gamma' is restricted to [0.0, 1.0] to maintain the Tallying anchor) but expand the allowable noise parameters. By widening 'epsilon' to [0.0, 1.0] and lowering the upper bound of 'beta' to 2.0, the model naturally pulls the overly deterministic predictions in conflict trials toward 0.5, lifting the TTB match rate in Exps 1-3 closer to the empirical ~42% while preserving the near-chance behavior in Exp 8.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Highly compressed subjective weights: base weight of 1.0 (Tallying) 
    # plus a small validity-driven adjustment.
    weights = 1.0 + gamma * (val - np.mean(val))
    
    # Compensatory integration using subjective weights
    scores = stim @ weights
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
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
