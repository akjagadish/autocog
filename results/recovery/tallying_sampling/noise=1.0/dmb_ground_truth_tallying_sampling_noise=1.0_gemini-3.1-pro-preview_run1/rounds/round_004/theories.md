# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Biased Logistic Tallying: Subjects make decisions by counting the total number of positive features for each option (Tallying), ignoring cue validities. However, their choices are highly noisy and subject to a baseline spatial/option bias. The decision process is modeled as a logistic function over the difference in feature tallies between the two options, parameterized by a sensitivity weight and a baseline bias. The sensitivity to tally differences is weak, reflecting empirical data where feature counts only slightly sway noisy guessing.

**Rationale:** Following the critic's advice, I am restricting the `beta` parameter range from [-1.0, 1.0] to [-0.2, 0.2] to prevent over-sensitivity to the feature tally difference. This ensures the model behaves closer to the empirical data, where tally differences only weakly influence choices, reducing the overshoot observed in Experiments 2 and 4.

**Parameters:**
  - `beta`: `[-0.2, 0.2]`
  - `bias`: `[-2.0, 2.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    
    # Logistic choice rule based on tally difference and baseline bias
    diff = scores[0] - scores[1]
    logit = beta * diff + bias
    
    # Numerically stable sigmoid
    if logit >= 0:
        p_a = 1.0 / (1.0 + np.exp(-logit))
    else:
        exp_logit = np.exp(logit)
        p_a = exp_logit / (1.0 + exp_logit)
        
    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** In complex, multi-attribute binary choice tasks without trial-by-trial feedback, subjects largely fail to integrate cue validities or even simple feature counts. Instead, their decisions are dominated by high response noise, effectively reducing their behavior to random guessing with a slight baseline bias for one option over the other. This explains the ~0.5 choice rates, ~0.5 TTB match rates, and ~0.0 differences across experimental conditions.

**Rationale:** The arbiter diagnosed that deterministic TTB and structured WADD models fail because human behavior in these complex, no-feedback tasks is much noisier and less systematic than these models assume. The empirical results show choice rates very close to 0.5, TTB match rates near 0.5, and cross-condition differences near 0.0. A pure random-guessing model with a baseline spatial/option bias predicts these flat empirical patterns much better than structured integration models.

**Parameters:**
  - `bias_a`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
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

**Description:** High-Lapse Take-The-Best: Subjects largely rely on noisy guessing with a baseline spatial/option bias, but on a small fraction of trials, they employ the Take-The-Best heuristic, deciding based on the single most valid cue that discriminates between the options. This captures the slightly positive effects of cue validity and feature differences observed in the data without over-predicting the reliance on full-profile tallying.

**Rationale:** Following the arbiter's suggestion, this theory models subjects as mostly guessing with a high lapse rate, but occasionally using the Take-The-Best heuristic (relying on the first discriminating cue). This accounts for the near-zero but slightly positive cue reliance and TTB match rates seen across the experiments, outperforming models that strictly tally all features or guess completely randomly without any cue integration.

**Parameters:**
  - `lapse_rate`: `[0.8, 1.0]`
  - `bias_a`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a = stim[0]
    b = stim[1]
    
    lapse_rate = float(parameters['lapse_rate'])
    bias_a = float(parameters['bias_a'])
    
    # Take-The-Best: find the first cue that discriminates
    ttb_pred = -1
    for i in range(len(a)):
        if a[i] > b[i]:
            ttb_pred = 0
            break
        elif a[i] < b[i]:
            ttb_pred = 1
            break
            
    p_guess = np.array([bias_a, 1.0 - bias_a])
    
    if ttb_pred == 0:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_pred == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = p_guess
        
    return (1.0 - lapse_rate) * p_ttb + lapse_rate * p_guess
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
