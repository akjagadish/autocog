# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** Sequential Evidence Accumulation with Lateral Inhibition, Leak, and a Collapsing Decision Threshold: Decision-makers sample features sequentially with probability proportional to their validities. Options accumulate evidence subject to leak and lateral inhibition. To prevent indecision when evidence is mixed or tied, the decision threshold collapses over time (urgency signal). A highly valid cue can quickly cross the initially high threshold (mimicking Take-The-Best), but if early cues conflict, the threshold gradually lowers, forcing a decision based on the integrated evidence of later sampled cues (mimicking Tallying or WADD).

**Rationale:** Following the critic's feedback on Iteration 3's rejection, we revert to the stable LCA base from Iteration 2 (avoiding excessive inhibition and noise) and introduce a 'collapsing decision threshold' via the `collapse_rate` parameter. This urgency signal forces a decision when evidence is mixed or tied (mimicking WADD/Tallying over longer deliberations), while still allowing highly valid early cues to cross the initially high threshold quickly (mimicking TTB). This minimal edit cleanly implements the requested mechanism to better capture the compensatory dynamics in Experiments 7 and 9 without destabilizing the rest of the fits.

**Parameters:**
  - `gamma`: `[0.1, 20.0]`
  - `theta`: `[1.0, 15.0]`
  - `inhibition`: `[0.0, 1.0]`
  - `leak`: `[0.0, 1.0]`
  - `collapse_rate`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    inhibition = float(parameters["inhibition"])
    leak = float(parameters["leak"])
    collapse_rate = float(parameters["collapse_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Sampling probability based on validity
    w = val ** gamma
    if np.sum(w) == 0:
        p_sample = np.ones_like(w) / len(w)
    else:
        p_sample = w / np.sum(w)
        
    n_sims = 2000
    max_steps = 100
    
    # Pre-sample all cues for all sims and steps for speed
    samples = np.random.choice(len(val), size=(n_sims, max_steps), p=p_sample)
    
    inc_a = a[samples]
    inc_b = b[samples]
    
    ea = np.zeros(n_sims)
    eb = np.zeros(n_sims)
    
    decided = np.zeros(n_sims, dtype=bool)
    winner = np.full(n_sims, -1)
    
    for step in range(max_steps):
        mask = ~decided
        if not np.any(mask):
            break
            
        curr_inc_a = inc_a[mask, step]
        curr_inc_b = inc_b[mask, step]
        
        # LCA update rule with lateral inhibition and leak
        new_ea = np.maximum(0.0, ea[mask] * (1.0 - leak) + curr_inc_a - inhibition * eb[mask])
        new_eb = np.maximum(0.0, eb[mask] * (1.0 - leak) + curr_inc_b - inhibition * ea[mask])
        
        ea[mask] = new_ea
        eb[mask] = new_eb
        
        # Collapsing threshold
        curr_theta = max(0.01, theta - step * collapse_rate)
        cross_a = new_ea >= curr_theta
        cross_b = new_eb >= curr_theta
        
        just_decided = cross_a | cross_b
        
        if np.any(just_decided):
            jd_indices = np.where(just_decided)[0]
            
            for idx in jd_indices:
                if cross_a[idx] and not cross_b[idx]:
                    winner_val = 0
                elif cross_b[idx] and not cross_a[idx]:
                    winner_val = 1
                else:
                    if new_ea[idx] > new_eb[idx]:
                        winner_val = 0
                    elif new_eb[idx] > new_ea[idx]:
                        winner_val = 1
                    else:
                        winner_val = np.random.choice([0, 1])
                
                orig_idx = np.where(mask)[0][idx]
                winner[orig_idx] = winner_val
                decided[orig_idx] = True

    undecided = ~decided
    if np.any(undecided):
        ea_un = ea[undecided]
        eb_un = eb[undecided]
        
        # For those that haven't crossed threshold, highest evidence wins
        ties = ea_un == eb_un
        win_un = np.where(ea_un > eb_un, 0, 1)
        if np.any(ties):
            win_un[ties] = np.random.choice([0, 1], size=ties.sum())
            
        winner[undecided] = win_un
        
    p_A = np.sum(winner == 0) / n_sims
    p_B = np.sum(winner == 1) / n_sims
    
    p = np.array([p_A, p_B])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Rationale:** Added a normalization step for the exponentiated validities (val = val / np.sum(val)) as suggested by the critic. Since the raw validities are fractions, raising them to a large power shrinks their absolute values, causing the weighted sums to become very small and washing out the choice probabilities. Normalizing them ensures the scale of the scores remains stable, allowing the beta parameter to properly control the determinism of the choice.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[1.0, 30.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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

### `pi_8` → slot 1 (via `new_theory`)

**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Rationale:** Following the critic's advice, I reverted to the accepted Iteration 2 base (Evidence Dilution) and widened the upper bounds of the parameters, specifically increasing the upper bound of `gamma` to 10.0 and `lambda_val` to 20.0. This gives the optimizer the flexibility to apply a much stronger dilution penalty to options with many weakly-valid cues, which is necessary to capture the severe negative preference observed in Experiment 9, without fundamentally altering the mathematical formulation that successfully preserved core TTB/Tallying performance in other experiments.

**Parameters:**
  - `lambda_val`: `[1.0, 20.0]`
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
