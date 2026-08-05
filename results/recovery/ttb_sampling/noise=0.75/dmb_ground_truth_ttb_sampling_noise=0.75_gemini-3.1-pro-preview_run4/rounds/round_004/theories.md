# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

**Rationale:** Following the critic's advice, I widened the epsilon parameter range from [0.0, 0.5] to [0.0, 1.0]. The human responses in the given experiments are closer to 0.5 than strict TTB allows with a small lapse rate. Allowing epsilon to range up to 1.0 enables the model to fit higher levels of behavioral noise (random guessing) while preserving the core TTB lexicographic mechanism.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Confidence-scaled Take The Best: Decision makers strictly follow the non-compensatory Take The Best (TTB) search rule, basing their decision entirely on the most valid discriminating cue and completely ignoring all subordinate cues. However, their confidence in this choice depends on the objective validity of that primary cue. When the best discriminating cue has high validity, they execute the TTB choice with high probability; when it has lower validity, they are less confident and more prone to guessing. This naturally lowers overall TTB agreement in environments where decisions rely on weaker cues, while maintaining zero sensitivity to the quantity of supporting or opposing subordinate cues.

**Rationale:** Following the arbiter's suggestion, this theory replaces the subjective cue evaluation of pi_5 with a strict execution of the Take The Best heuristic, ensuring that subordinate cues are never evaluated (which guarantees the flat sensitivity curves empirically observed in Exps 4, 5, 6, 7, and 8). To capture the reduced TTB agreement in experiments relying on weaker cues (Exps 3, 4, 5), the model scales the probability of successfully executing the TTB choice according to the primary discriminating cue's validity. A parameter beta maps the objective validity into a choice probability, seamlessly blending deterministic TTB (high beta, high validity) with random guessing (low beta or low validity).

**Parameters:**
  - `beta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    v_best = 0.5
    
    # Find the first discriminating feature
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            v_best = validities[idx]
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            v_best = validities[idx]
            break
            
    if chosen == -1:
        p_core = np.array([0.5, 0.5])
    else:
        # Confidence is a function of the primary cue's validity
        # using a softmax-like probability matching function
        num = v_best ** beta
        den = num + (1.0 - v_best) ** beta
        p_ttb = num / den if den > 0 else 0.5
        
        if chosen == 0:
            p_core = np.array([p_ttb, 1.0 - p_ttb])
        else:
            p_core = np.array([1.0 - p_ttb, p_ttb])
            
    # Apply general response noise (lapse rate)
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Non-linear Weighted Additive (WADD) Model: Decision makers integrate all available cues in a compensatory manner, but their subjective weighting of cues follows an exponential scaling of log-odds validities. By transforming the log-odds weights (equivalent to raising the odds ratio to a power), the model approximates a steeper, near-lexicographic weighting hierarchy, reducing over-sensitivity to accumulations of subordinate cues while retaining a fully compensatory architecture and avoiding erratic distortions.

**Rationale:** Following the critic's advice, we widened the range of the `alpha` parameter from `[0.0, 5.0]` to `[0.0, 10.0]` while keeping the predictive model completely intact. This minimal edit gives the model the flexibility to learn an even steeper weighting hierarchy, which will further suppress the residual compensatory effects observed in Experiments 5 and 7 by pushing the weights of primary cues higher relative to subordinate cues.

**Parameters:**
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.0, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    # Clip validities to avoid log(0) or division by zero
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    
    # Calculate log-odds weights and apply exponential scaling
    base_weights = np.log(v_clipped / (1.0 - v_clipped))
    weights = np.exp(alpha * base_weights)
    
    # Calculate values for options A and B
    val_a = np.sum(weights * stim[0])
    val_b = np.sum(weights * stim[1])
    
    # Softmax choice rule
    logits = gamma * np.array([val_a, val_b])
    logits -= np.max(logits) # for numerical stability
    probs = np.exp(logits)
    probs /= np.sum(probs)
    
    # Apply lapse rate
    p_final = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
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
