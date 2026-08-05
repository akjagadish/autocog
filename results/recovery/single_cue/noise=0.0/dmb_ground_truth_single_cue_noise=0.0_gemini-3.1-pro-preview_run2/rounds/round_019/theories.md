# Round 19 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_19` — SURVIVED ✓

**Description:** Primacy-Dominant Anchoring: Decision-makers use the first cue as a powerful anchor that overwhelmingly dominates the evaluation process. While the final cue may receive a secondary recency boost due to short-term memory, the primacy weight is structurally much larger than both the recency weight and the middle cue validities. This explicitly enforces a hierarchy where primacy is the primary driver of choice.

**Rationale:** Based on the arbiter's feedback, this theory models a 'Primacy-Dominant Anchoring' mechanism. Instead of treating primacy and recency as symmetric boundary effects or purely overweighting recency, this model explicitly enforces a hierarchy where the first cue acts as a massive anchor (weight range [10, 30]), while the final cue can have a secondary recency effect (weight range [0, 9]), and middle cues carry their base validities. This captures the strong empirical phenomenon where subjects base choices almost entirely on the initial anchor cue, overriding both compensatory tallying and recency when they conflict.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `primacy_weight`: `[10.0, 30.0]`
  - `recency_weight`: `[0.0, 9.0]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Enforce Primacy-Dominant Anchoring hierarchy
    w[0] = primacy_weight
    if len(w) > 1:
        w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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


### slot 2 — `pi_21` — KILLED ✗

**Description:** Validity-Weighted Evidence Accumulation with Normalized Attention Decay (Simplified)

**Rationale:** Initial logic and parameters are validated. Removed the redundant `gamma` parameter to streamline the dimensionality of the optimization space. Weights are now computed directly as `val * (decay ** positions)` prior to normalization, allowing the single decay parameter to govern the interpolation between primacy and tallying behaviors without getting stuck in pathological local minima.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `decay`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    decay = float(parameters["decay"])
    
    # Apply exponential attention decay based on cue position (0-indexed)
    positions = np.arange(len(val))
    attention_weights = decay ** positions
    
    # Scale explicitly stated validities and apply attention decay directly
    w = val * attention_weights
    
    # Normalize weights to prevent exponential blowup from dominating the softmax temperature
    sum_w = np.sum(w)
    if sum_w > 0:
        w = w / sum_w
    else:
        w = np.ones_like(w) / len(w)
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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

### `pi_22` → slot 2 (via `new_theory`)

**Description:** Adaptive Strategy Selection (Extended Meta-Features): Decision-makers dynamically select between Tallying, WADD, and Take-The-Best (TTB) depending on the environment's structure. The selection mechanism uses a softmax function over learnable baseline logits for each strategy. These logits are adjusted by four meta-features: validity dispersion, maximum available validity, the number of available cues, and the dominance of the top cue (difference between highest and second-highest validity). This gives the strategy-selection mechanism the capacity to correctly identify environments that demand lexicographic behavior.

**Rationale:** Following the critic's advice, two additional meta-features have been introduced to the strategy selection logits: the number of cues (`len(val)`) and the top-cue dominance (`top_diff = sorted_val[0] - sorted_val[1]`). This extends the capacity of the Adaptive Strategy Selection model to differentiate between experimental environments, allowing it to more accurately trigger Take-The-Best behavior when the environment structure demands it.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `bias_tally`: `[-10.0, 10.0]`
  - `bias_wadd`: `[-10.0, 10.0]`
  - `bias_ttb`: `[-10.0, 10.0]`
  - `w_disp_tally`: `[-20.0, 20.0]`
  - `w_disp_wadd`: `[-20.0, 20.0]`
  - `w_disp_ttb`: `[-20.0, 20.0]`
  - `w_max_tally`: `[-20.0, 20.0]`
  - `w_max_wadd`: `[-20.0, 20.0]`
  - `w_max_ttb`: `[-20.0, 20.0]`
  - `w_len_tally`: `[-20.0, 20.0]`
  - `w_len_wadd`: `[-20.0, 20.0]`
  - `w_len_ttb`: `[-20.0, 20.0]`
  - `w_diff_tally`: `[-20.0, 20.0]`
  - `w_diff_wadd`: `[-20.0, 20.0]`
  - `w_diff_ttb`: `[-20.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    bias_tally = float(parameters["bias_tally"])
    bias_wadd = float(parameters["bias_wadd"])
    bias_ttb = float(parameters["bias_ttb"])
    
    w_disp_tally = float(parameters["w_disp_tally"])
    w_disp_wadd = float(parameters["w_disp_wadd"])
    w_disp_ttb = float(parameters["w_disp_ttb"])
    
    w_max_tally = float(parameters["w_max_tally"])
    w_max_wadd = float(parameters["w_max_wadd"])
    w_max_ttb = float(parameters["w_max_ttb"])
    
    w_len_tally = float(parameters["w_len_tally"])
    w_len_wadd = float(parameters["w_len_wadd"])
    w_len_ttb = float(parameters["w_len_ttb"])
    
    w_diff_tally = float(parameters["w_diff_tally"])
    w_diff_wadd = float(parameters["w_diff_wadd"])
    w_diff_ttb = float(parameters["w_diff_ttb"])
    
    dispersion = np.std(val)
    max_val = np.max(val)
    n_cues = len(val)
    sorted_val = np.sort(val)[::-1]
    top_diff = sorted_val[0] - sorted_val[1] if n_cues > 1 else 0.0
    
    logit_tally = bias_tally + w_disp_tally * dispersion + w_max_tally * max_val + w_len_tally * n_cues + w_diff_tally * top_diff
    logit_wadd = bias_wadd + w_disp_wadd * dispersion + w_max_wadd * max_val + w_len_wadd * n_cues + w_diff_wadd * top_diff
    logit_ttb = bias_ttb + w_disp_ttb * dispersion + w_max_ttb * max_val + w_len_ttb * n_cues + w_diff_ttb * top_diff
    
    logits_strat = np.array([logit_tally, logit_wadd, logit_ttb])
    logits_strat -= np.max(logits_strat)
    w_strat = np.exp(logits_strat)
    w_strat /= np.sum(w_strat)
    
    w_tally, w_wadd, w_ttb = w_strat
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins) / len(a)
    tally_b = np.sum(b_wins) / len(b)
    
    sum_val = np.sum(val)
    if sum_val == 0: sum_val = 1.0
    wadd_a = np.sum(val * a_wins) / sum_val
    wadd_b = np.sum(val * b_wins) / sum_val
    
    ttb_a = 0.5
    ttb_b = 0.5
    stable_val = val - np.arange(len(val)) * 1e-5
    order = np.argsort(stable_val)[::-1]
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            ttb_b = 0.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            ttb_a = 0.0
            break
            
    score_a = w_tally * tally_a + w_wadd * wadd_a + w_ttb * ttb_a
    score_b = w_tally * tally_b + w_wadd * wadd_b + w_ttb * ttb_b
    
    scores = np.array([score_a, score_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
