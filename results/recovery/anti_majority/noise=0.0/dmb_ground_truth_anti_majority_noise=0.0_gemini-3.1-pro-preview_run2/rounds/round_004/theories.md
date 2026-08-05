# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Dual-Process Strategy Selection Theory: Decision-makers probabilistically select between a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive, WADD) on a trial-by-trial basis. The probability of employing the non-compensatory heuristic is a logistic function of the absolute validity of the highest-ranking discriminating cue. When the top discriminating cue is highly valid, subjects are highly likely to rely solely on it (TTB). However, when the top discriminating cue is weak, confidence in the heuristic drops, and subjects fall back to integrating all available information (WADD).

**Rationale:** Following the critic's feedback, we revert to the absolute validity mechanism for the dual-process strategy selection (Iteration 1 base) because it successfully captured core dynamics but underpredicted Experiment 7. To address this underprediction, we widen the parameter ranges for the logistic transition function (`theta` to [0.0, 1.0] and `tau` to [1.0, 100.0]). This allows the model to learn a sharper, step-like transition threshold that clearly distinguishes trials where the top discriminating cue is highly valid (relying heavily on TTB) from trials where it is weak (falling back to WADD), without distorting the underlying compensatory strategy.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[0.0, 1.0]`
  - `tau`: `[1.0, 100.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    tau = float(parameters["tau"])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    top_cue = None
    for j in cue_order:
        if a[j] != b[j]:
            top_cue = j
            break
            
    if top_cue is None:
        p_mix = np.array([0.5, 0.5])
    else:
        v_top = val[top_cue]
        
        # Probability of using TTB is a logistic function of the top cue's validity
        p_ttb_use = 1.0 / (1.0 + np.exp(-tau * (v_top - theta)))
        
        # Take-The-Best (TTB) prediction
        winner_ttb = 0 if a[top_cue] > b[top_cue] else 1
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
        # Weighted Additive (WADD) prediction
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
        scores_wadd = np.array([score_a, score_b])
        z_wadd = beta * (scores_wadd - np.max(scores_wadd))
        e_wadd = np.exp(z_wadd)
        p_wadd = e_wadd / np.sum(e_wadd)
        
        # Mix the two strategies based on the top cue validity
        p_mix = p_ttb_use * p_ttb + (1.0 - p_ttb_use) * p_wadd
        
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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

### `pi_7` → slot 1 (via `new_theory`)

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
