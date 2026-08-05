# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Take-The-Best (TTB) heuristic: People make binary choices by evaluating features sequentially in descending order of their validity. The first feature that discriminates between the two options (i.e., one option has a higher value than the other) entirely determines the choice, and all remaining features are ignored. This strictly non-compensatory strategy allows decision makers to heavily weight highly predictive cues without needing to compute complex compensatory trade-offs.

**Rationale:** Following the arbiter's guidance, this model replaces the compensatory Tallying heuristic with the Take-The-Best (TTB) heuristic. Instead of summing up all winning features, TTB evaluates cues sequentially based on their validity. The first cue that discriminates between the two options completely determines the choice, and all subsequent cues are ignored. This captures the strong preference for high-validity cues observed in the data in a purely non-compensatory (lexicographic) manner. Response noise is handled via a softmax over the binary decision outcome and an independent lapse rate.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    scores = np.array([0.0, 0.0])
    
    # Evaluate features one by one in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse
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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

**Rationale:** Following the latest feedback, I reverted to the Iteration 1 base which uses a single beta parameter for both TTB and WADD. To fix the scaling mismatch that forces the optimization into a compromise, I normalized the WADD scores by dividing them by the sum of subjective weights. This strictly bounds the WADD scores to [0, 1], perfectly aligning with the 0/1 scale of the TTB scores. This minimal edit resolves the issue without introducing the degrees of freedom that caused the previous decoupling attempt to fail.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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

**Description:** Probabilistic Cue Search (Sequential Evidence Accumulation): Decision-makers evaluate features sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue (as in pure Take-The-Best) or exhaustively accumulating all cues (as in Weighted Additive), they exhibit a probabilistic stopping rule. After evaluating each discriminating cue and updating their internal evidence, they stop searching and make a choice with probability `theta`. If they do not stop, they continue to the next discriminating cue, accumulating its evidence. This naturally produces a graded interpolation between non-compensatory and compensatory decision-making without relying on a discrete mixture of distinct strategies.

**Rationale:** Following the critic's feedback, the parameter ranges have been restored to their original wide bounds (e.g., gamma up to 5.0, beta up to 20.0) to ensure the model retains its ability to fit the non-compensatory majority in Exps 5-8. To address the underprediction of compensatory choices in Exps 1-3, the weight normalization has been changed from max-normalization (`w / np.max(w)`) to sum-normalization (`w / np.sum(w)`). This ensures the total accumulated score is always bounded within [0, 1], allowing the softmax `beta` parameter to act consistently across different experimental designs and validity structures, improving the model's balance between TTB-like and WADD-like behavior.

**Parameters:**
  - `theta`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Subjective weights normalized so the sum of weights is 1.0
    w = validities ** gamma
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Sort all features by validity descending
    order = np.argsort(validities)[::-1]
    
    # Find discriminating cues in order of validity
    discrim_indices = [i for i in order if a[i] != b[i]]
    
    if len(discrim_indices) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_A_total = 0.0
        p_B_total = 0.0
        p_continue = 1.0
        
        score_A = 0.0
        score_B = 0.0
        
        for j, i in enumerate(discrim_indices):
            # Accumulate evidence from the current discriminating cue
            score_A += w[i] * a[i]
            score_B += w[i] * b[i]
            
            # Determine stopping probability
            if j < len(discrim_indices) - 1:
                p_stop = theta
            else:
                p_stop = 1.0  # Must stop at the last discriminating cue
                
            # Softmax over accumulated scores so far
            z_A = beta * score_A
            z_B = beta * score_B
            max_z = max(z_A, z_B)
            e_A = np.exp(z_A - max_z)
            e_B = np.exp(z_B - max_z)
            p_A_given_stop = e_A / (e_A + e_B)
            p_B_given_stop = e_B / (e_A + e_B)
            
            # Marginalize over the stopping probability
            p_A_total += p_continue * p_stop * p_A_given_stop
            p_B_total += p_continue * p_stop * p_B_given_stop
            
            # Update the probability of continuing to the next cue
            p_continue *= (1.0 - p_stop)
            
        p_core = np.array([p_A_total, p_B_total])
        
    # Apply uniform lapse
    n_opts = 2
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
