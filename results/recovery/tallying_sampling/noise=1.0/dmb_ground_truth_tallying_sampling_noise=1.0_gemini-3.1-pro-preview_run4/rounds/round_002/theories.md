# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People are heterogeneous in their decision-making strategies, with some choices driven by a non-compensatory heuristic (Take The Best) and others by a compensatory strategy (Weighted Additive). The population consists of individuals who employ a mixture of these strategies, governed by a subjective mixture weight. By blending a frugal, single-reason strategy with a fully compensatory evaluation, the model captures both the variance and the balanced aggregate behavior observed across decision-making experiments.

**Rationale:** Following the critic's advice, the parameter ranges for `wadd_prob` and `epsilon` have been adjusted. Because WADD sometimes agrees with TTB, a uniform 0-1 mixture over-predicts TTB matches. Shifting `wadd_prob` to `[0.4, 1.0]` slightly favors WADD, and expanding `epsilon` to `[0.0, 1.0]` allows for higher levels of pure guessing, which helps pull the aggregate metrics closer to the true 0.50 levels observed in the data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`
  - `wadd_prob`: `[0.4, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    n_features = stim.shape[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # --- TTB (Take The Best) ---
    cue_order = np.argsort(-validities, kind="stable").tolist()
    a, b = stim[0], stim[1]
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    beta = float(parameters["beta"])
    
    if winner_ttb is None:
        p_ttb = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * scores_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
        
    # --- WADD (Weighted Additive) ---
    scores_wadd = stim @ (validities * w)
    z_wadd = beta * scores_wadd
    e_wadd = np.exp(z_wadd - np.max(z_wadd))
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- MIXTURE ---
    wadd_prob = float(parameters["wadd_prob"])
    p_core = wadd_prob * p_wadd + (1.0 - wadd_prob) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Equal Weights with High Lapse: Subjects often find the integration of complex cue validities too cognitively demanding or disengaging in these conflict paradigms. As a result, they ignore the provided validities entirely and fall back on a simple 'Tallying' (Equal Weights) heuristic, where they just count the number of positive features for each option. Furthermore, due to the high cognitive friction or confusion, subjects exhibit a very high baseline guessing rate (lapse), meaning that on the vast majority of trials they simply guess randomly. This explains why behavior across various conflict and agreement metrics hovers so closely to 0.5 or 0 difference.

**Rationale:** Across all four experiments, the observed metrics heavily center around 0.5 for choice proportions and 0 for differences between conditions. This indicates that subjects are not consistently applying WADD or TTB, but rather exhibiting near-random behavior. Following the arbiter's suggestion, this model implements an Equal Weights (Tallying) strategy with an exceptionally high lapse rate (epsilon bounded between 0.8 and 1.0). By abandoning complex validity integration and injecting massive stochasticity, the model naturally predicts the ~0.50 matching rates in Exp 1, 2, and 3, and the ~0 difference in Exp 4, overcoming the deterministic biases of previous WADD and TTB models.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: equal weights for all features (counting positive cues)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallied scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # High uniform lapse blended in
    n_opts = len(p_core)
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

**Rationale:** Following the arbiter's guidance, this theory posits that subjects rely on a fragile, single-cue heuristic (looking only at the highest-validity cue) rather than tallying or weighted additive strategies. To capture the fact that empirical metrics stay tightly bounded near 0.5 (or 0 difference), the model assumes an overwhelmingly high lapse rate (epsilon between 0.9 and 1.0). This generates predictions that are mostly random guessing, with slight deviations in the direction of the single best cue, explaining the subtle effects observed in the data without over-predicting the influence of secondary cues.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.9, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
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
