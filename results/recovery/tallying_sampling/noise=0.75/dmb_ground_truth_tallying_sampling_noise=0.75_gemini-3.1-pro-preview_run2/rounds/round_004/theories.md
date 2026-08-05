# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Environment-Adaptive Strategy Selection Model (TTB vs. Tallying). Decision-makers probabilistically choose between a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying) on a trial-by-trial basis. Crucially, the probability of deploying TTB is not fixed but adapts to the structure of the environment: it is a logistic function of the dispersion (standard deviation) of the cue validities. In environments with steep validity gradients (high dispersion), TTB is strongly preferred; in environments with flat gradients (low dispersion), decision-makers naturally fall back to Tallying.

**Rationale:** To address the severe misprediction in Experiment 7, the purely latent `w_ttb` parameter has been replaced with an environment-adaptive mechanism. The probability of selecting TTB (`w_ttb`) is now dynamically computed from the dispersion (standard deviation) of the cue validities in the current experiment using a logistic function parameterized by a `disp_slope` and `disp_threshold`. This allows the model to naturally suppress TTB in environments with flat validity gradients (like Experiment 7), while retaining the successful mixture behavior in environments that warrant non-compensatory heuristics.

**Parameters:**
  - `disp_slope`: `[0.0, 100.0]`
  - `disp_threshold`: `[0.0, 0.5]`
  - `beta_tally`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    dispersion = np.std(validities)
    slope = float(parameters["disp_slope"])
    threshold = float(parameters["disp_threshold"])
    
    # Calculate w_ttb dynamically based on the dispersion of cue validities
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Strategy ---
    # Sort cues by validity (descending)
    # We add a tiny amount of noise to validities to break ties consistently if they exist
    order = np.argsort(-(validities + np.random.uniform(0, 1e-6, size=len(validities))))
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif stim[1, idx] > stim[0, idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying Strategy ---
    scores = np.sum(stim, axis=1)
    z = beta_tally * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_tally = e / np.sum(e)
    
    # --- Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # --- Lapse ---
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Rationale:** Follows the critic's advice to implement a non-compensatory Take-The-Best model with a probabilistic cue search order. The model stops at the first discriminating cue, maintaining a strict non-compensatory mechanism at the single-trial level. However, the order in which cues are searched is sampled without replacement using a softmax over cue validities (via the Gumbel-max trick). The temperature parameter `tau` controls the determinism of the search order. When `tau` is large, the search order becomes uniform, which on average mimics the compensatory tallying-like behavior observed in Experiment 4, bridging the gap between non-compensatory heuristics and compensatory empirical patterns.

**Parameters:**
  - `tau`: `[0.01, 100.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Stochastic Weighted Additive Model (SWADD) with bounded exponential cue weighting. Decision-makers integrate all available cues in a compensatory manner, weighting them according to their subjective importance. The subjective weight of each cue is modeled as an exponential function of its validity: w_i = exp(gamma * v_i). The parameter gamma controls the degree of compensatoriness. When gamma = 0, all weights are equal (pure Tallying). As gamma increases, the weights diverge exponentially, allowing higher-validity cues to dominate. Constraining gamma ensures that the exponential divergence does not become so extreme that it washes out moderate compensatory strategies, better matching human behavior on conflict trials.

**Rationale:** Following the critic's feedback, the upper bound of the `gamma` parameter is reduced from 5.0 to 2.5. The exponential weight mapping (w = exp(gamma * validities)) scales up differences between validities very aggressively. A maximum gamma of 5.0 created too much disparity (e.g., exp(5) > 148), heavily oversampling extreme non-compensatory regimes and washing out the compensatory behavior necessary to explain choices in Exp 4 and Exp 10. By restricting gamma to [0.0, 2.5], the uniform prior samples moderate compensatory regimes more frequently, allowing the model to better capture human behavior on conflict trials while still being able to approximate TTB when gamma approaches its upper bound. The SWADD architecture, softmax choice rule, and lapse rate remain unchanged.

**Parameters:**
  - `gamma`: `[0.0, 2.5]`
  - `beta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate weights using an exponential function of validities
    w = np.exp(gamma * validities)
    
    # Normalize weights so the maximum weight is 1.0
    # This keeps the weighted sums on a consistent scale across different gamma values
    w = w / np.max(w)
    
    # Weighted sum for each option
    scores = np.sum(stim * w, axis=1)
    
    # Softmax choice rule
    z = beta * scores
    z = z - np.max(z) # numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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
