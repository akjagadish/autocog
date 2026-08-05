# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Weighted Additive Model (WADD) with Subjective Validities: Decision-makers integrate all available features by computing a weighted sum of cue values for each option. The weights are subjective validities, modeled as a power transformation of the objective validities. Choices are made probabilistically using a softmax function over the options' weighted sums, along with an independent lapse rate. Limiting the softmax inverse temperature ensures higher decision noise, which tempers overconfidence when cue validities conflict.

**Rationale:** Following the critic's advice, we keep the identical WADD mechanism with the power-law validity transformation and lapse rate, but further restrict the 'beta' parameter range to [0.0, 1.0]. This minimal edit enforces an even higher baseline of softmax noise, which should help to further suppress the remaining overprediction of accuracy in Experiments 4 and 6, bringing the predictions closer to the near-random human behavior observed on these specific trial types.

**Parameters:**
  - `gamma`: `[0.0, 5.0]`
  - `beta`: `[0.0, 1.0]`
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
    
    # Transform objective validities into subjective weights
    subj_weights = validities ** gamma
    
    # Compute WADD scores
    scores = stim @ subj_weights
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
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

**Description:** Take-The-Best (TTB) Heuristic with High Decision Noise: Decision-makers evaluate cues sequentially in descending order of their validities. They stop at the first cue that discriminates between the options and choose the option favored by that cue. To account for the extremely high degree of randomness observed in the experimental data (where aggregate choices often hover near 0.50), the model forces a significantly high lapse rate (between 0.4 and 1.0), reflecting instances where decision-makers guess randomly instead of strictly applying the lexicographic rule.

**Rationale:** Reverting to the simple lapse rate TTB model (as the probabilistic stopping rule was rejected) but strictly enforcing a high degree of noise by restricting the epsilon parameter range to [0.4, 1.0]. This forces the optimizer to apply a high lapse rate, thereby bringing the aggregated predictions closer to the ~0.5 mark consistently observed across the experiments while remaining within the prescribed TTB family.

**Parameters:**
  - `epsilon`: `[0.4, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-validities)
    
    # Iterate through cues to find the first that discriminates
    decision = np.array([0.5, 0.5])
    for cue_idx in cue_order:
        val_a = stim[0, cue_idx]
        val_b = stim[1, cue_idx]
        if val_a > val_b:
            decision = np.array([1.0, 0.0])
            break
        elif val_b > val_a:
            decision = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Return the TTB decision mixed with random guessing (lapse rate)
    return (1.0 - epsilon) * decision + epsilon * np.array([0.5, 0.5])
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

**Description:** Tallying (Equal Weights) Heuristic with Decision Noise

**Rationale:** Following the critic's feedback, the softmax decision rule evaluated in Iteration 2 exacerbated the extreme predictions by amplifying small tally differences, leading to a worse aggregate loss. Therefore, we revert to the exact epsilon-lapse formulation of the Tallying heuristic from Iteration 1. This model counts the positive features for each option, chooses the one with the higher tally, and guesses on ties. By strictly maintaining the epsilon parameter with a full range of [0.0, 1.0], the model is free to fit the high degree of decision noise required to capture the near-0.50 choice proportions in heavily mixed human data, without the distorting effects of a temperature parameter.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    
    # Count the number of positive features (tally) for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    # Choose the option with the higher tally, or guess if tied
    if tally_a > tally_b:
        p_core = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    
    # Mix the deterministic tallying decision with random guessing (lapse rate)
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
