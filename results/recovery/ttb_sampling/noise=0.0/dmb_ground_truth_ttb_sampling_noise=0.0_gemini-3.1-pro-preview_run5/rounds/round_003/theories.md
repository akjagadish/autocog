# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

**Rationale:** Following the arbiter's feedback, this model replaces Tallying with the non-compensatory 'Take The Best' (TTB) heuristic. Instead of aggregating features (which fails to capture how strongly subjects rely on the single most valid feature in Experiment 2), TTB sorts the features by validity and bases the decision entirely on the first cue that discriminates between the two options. This perfectly replicates the extreme >90% choice rate for the option favoured by the highest-validity cue, a magnitude that WADD and Tallying fail to capture due to their compensatory nature.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Strategy Selection / Mixture Model: Decision-makers maintain a repertoire of distinct heuristics. On any given trial, a subject probabilistically samples a strategy from this repertoire—mixing a non-compensatory lexicographic rule (Take The Best) and a compensatory rule (Weighted Additive). The WADD rule incorporates a softmax decision process to gracefully handle near-ties and deviations. Global response noise is heavily restricted because the probabilistic mixture and the WADD softmax already provide sufficient stochasticity without artificially flattening predictions.

**Rationale:** Following the critic's feedback, we revert to the successful iteration 3 base (deterministic TTB + softmax WADD) but heavily restrict the global noise parameter `epsilon` to `[0.0, 0.05]`. The previous attempt to soften TTB over-parameterized the stochasticity and flattened predictions. The mixture probability `p_ttb` combined with the WADD softmax `tau` inherently guarantees that no choice probability is exactly zero (unless p_ttb=1.0 and TTB is decisive). Restricting epsilon prevents the likelihood optimization from inflating global noise, thereby sharpening the model's predictions across all experiments.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.05]`
  - `tau`: `[0.01, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(-validities, kind='stable')
    score_a_ttb = 0.5
    score_b_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            score_a_ttb = 1.0
            score_b_ttb = 0.0
            break
        elif b[idx] > a[idx]:
            score_a_ttb = 0.0
            score_b_ttb = 1.0
            break
    p_ttb = np.array([score_a_ttb, score_b_ttb])
    
    # Strategy 2: Weighted Additive (WADD) with softmax
    score_a_wadd = np.sum(a * validities)
    score_b_wadd = np.sum(b * validities)
    tau = float(parameters["tau"])
    z = np.array([score_a_wadd, score_b_wadd]) / tau
    z -= np.max(z)
    e = np.exp(z)
    p_wadd = e / np.sum(e)
        
    # Mixture
    p_mix = float(parameters["p_ttb"])
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_wadd
    
    # Response noise
    epsilon = float(parameters["epsilon"])
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Probabilistic Cue Retrieval TTB: Decision-makers follow the non-compensatory, sequential search architecture of Take The Best (TTB), but execution is noisy at the cue level. When a discriminating cue is evaluated, there is a probability of skipping it or failing to retrieve its value. This skip probability is a function of a base skip rate and is inversely proportional to the cue's validity (operationalized as proportional to 1 - validity). If a cue is skipped, the decision-maker continues to the next most valid cue. This produces slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration.

**Rationale:** Following the critic's diagnosis, the previous model overcorrected by enforcing bounds that were too tight, preventing the model from matching the ~15-17% deviation rate and ~85% strict adherence rate observed in human data. The functional form of the skip probability is maintained, but the upper bounds for `p_skip` and `epsilon` have been widened from 0.2 to 0.4. This minimal edit gives the optimizer the necessary flexibility to find the correct balance between strict TTB adherence and human-like cue evaluation noise.

**Parameters:**
  - `p_skip`: `[0.0, 0.4]`
  - `epsilon`: `[0.0, 0.4]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='stable')
    
    p_skip_base = float(parameters["p_skip"])
    
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    # Sequential search with probabilistic cue retrieval
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of skipping this cue is proportional to its invalidity
            p_skip_cue = p_skip_base * (1.0 - validities[idx])
            p_eval = 1.0 - p_skip_cue
            
            if a[idx] > b[idx]:
                p_A += p_continue * p_eval
            else:
                p_B += p_continue * p_eval
                
            p_continue *= p_skip_cue
            
    # If all discriminating cues are skipped, or no cues discriminate, guess randomly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    # Global response noise
    epsilon = float(parameters["epsilon"])
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
