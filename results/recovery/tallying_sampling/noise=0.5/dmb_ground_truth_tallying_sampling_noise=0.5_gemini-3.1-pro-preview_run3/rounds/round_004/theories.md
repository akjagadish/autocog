# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Rationale:** Following the critic's feedback, the predict and policy logic of the Mixture of Subjective WADD and Tallying theory are kept identical. The only change is an adjustment to the parameter bounds for noise: lowering the `beta` range to `[0.01, 5.0]` and widening the `epsilon` range to `[0.0, 1.0]`. This allows the model to produce softer maximization and rely more on uniform lapsing, directly addressing the systematic under-prediction of minority choice rates observed in Experiments 1, 2, 4, and 6.

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers probabilistically mix between two cognitively simple heuristics. Take-The-Best (TTB) is a strict lexicographic rule that inspects cues in order of their validity and chooses based entirely on the first discriminating cue. Tallying is an equal-weight rule that simply counts the number of positive features for each option. The mixture is biased towards Tallying to account for the strong equal-weighting pull observed in human data, while still allowing TTB to explain some non-compensatory choices. A small uniform lapse rate accounts for general response errors.

**Rationale:** Following the critic's advice on Iteration 4, we retain the exact 'Mixture of TTB and Tallying' mechanism from the accepted Iteration 1 base, but adjust the parameter ranges to prevent the heuristics from canceling each other out and washing out the signal. Specifically, `w_tally` is shifted to [0.5, 1.0] to bias the mixture heavily toward Tallying, which aligns with human preferences in key experiments (like Exps 3 and 5) where TTB predicts the opposite direction. Additionally, `epsilon` is restored but restricted to [0.0, 0.3] to provide realistic lapse rates without causing the uniform-guessing washout seen in Iteration 1.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 0.3]`
  - `w_tally`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    # Order cues by validity (descending)
    cue_order = np.argsort(val)[::-1]
    diff = stim[0] - stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for cue in cue_order:
        if diff[cue] > 0:
            p_ttb = np.array([1.0, 0.0])
            break
        elif diff[cue] < 0:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Mixture of Tallying and Noisy Objective WADD: Decision-makers evaluate options by probabilistically mixing between a simple equal-weight heuristic (Tallying) and a Weighted Additive (WADD) rule. Unlike strict Take-The-Best, the compensatory WADD component uses the objective validities, but applies a power transformation (validity compression or expansion) to them. To maintain stable scaling relative to Tallying, the power-transformed validities are normalized by their maximum value. This mechanism allows the model to reflect objective validity differences without making overly deterministic predictions on tie-breaking trials, as the noise and compression soften the WADD predictions closer to 50% when feature counts are tied. A uniform lapse rate further accounts for response noise.

**Rationale:** Following the latest feedback, we revert to the iter 1 base and apply a minimal fix to the scaling issue. Instead of over-normalizing both Tallying and WADD to [0, 1] (which failed in iter 3), we only normalize the power-transformed validities by their maximum value (`v_adj = (val ** gamma) / np.max(val ** gamma)`). This prevents the WADD scores from vanishing toward zero when gamma is large, keeping them on a numerical scale comparable to the unnormalized Tallying sums (which naturally range from 0 to N). This allows the single shared `beta` to properly and stably calibrate the softmax determinism for both strategies without distortion.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_tally`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Objective WADD with Validity Compression ---
    # A power parameter 'gamma' scales the objective validities.
    # gamma < 1 compresses differences (closer to equal weights),
    # gamma > 1 expands differences (closer to lexicographic/TTB).
    v_adj = val ** gamma
    # Normalize by max to keep WADD scores on a stable scale comparable to Tallying
    v_adj = v_adj / np.max(v_adj)
    scores_wadd = stim @ v_adj
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
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
