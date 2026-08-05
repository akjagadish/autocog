# Round 10 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

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


### slot 2 — `pi_12` — KILLED ✗

**Description:** Threshold-Gated Dilution Theory: Decision-makers integrate cues differently depending on the number of active cues. When an option has only a few active cues (below a threshold), they are integrated additively (acting like a non-compensatory strategy if weights are steep). However, when a coalition of many cues is present, it triggers a shift to an averaging process, where the total evidence is diluted by the number of active cues. This explains why adding many weak cues to an option can paradoxically reduce its subjective value.

**Rationale:** Following the arbiter's suggestion, this theory implements a Threshold-Gated Dilution mechanism. Instead of applying a global dilution penalty to all options, the penalty is gated by a threshold parameter ('tau'). If an option has only a few active cues (e.g., 1 or 2), the decision-maker merely sums their non-linearly scaled validities, mimicking a Take-The-Best heuristic without dilution. But if an option is supported by a coalition of many cues, the evaluation shifts to a compensatory, averaging mode, dividing the accumulated evidence by the number of active cues raised to a power ('gamma'). This captures the empirical finding that options with numerous weak cues are heavily penalized (explaining the dramatic effects in Experiments 9, 13, and 20), while isolated weak cues are ignored.

**Parameters:**
  - `lambda_val`: `[1.0, 20.0]`
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `tau`: `{1, 2, 3, 4}`
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
    tau = int(parameters["tau"])
    
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    if n_a <= tau:
        v_a = np.sum(w * a)
    else:
        v_a = np.sum(w * a) / (n_a ** gamma)
        
    if n_b <= tau:
        v_b = np.sum(w * b)
    else:
        v_b = np.sum(w * b) / (n_b ** gamma)
        
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


## Replacement

### `pi_13` → slot 2 (via `new_theory`)

**Description:** Sequential Search with Relative Evidence Thresholding: Decision-makers evaluate cues in descending order of validity. Each active cue adds to an evidence accumulator, but the subjective weight of each subsequent cue decays exponentially. Instead of a fixed absolute threshold for early stopping, subjects use a relative stopping rule: they stop searching when the accumulated evidence difference between options exceeds a threshold proportional to the maximum possible remaining evidence. This relative threshold seamlessly bridges Take-The-Best (stopping when remaining evidence cannot change the outcome) and Tallying (integrating all cues), explaining dynamic strategy shifts without relying on artificial cue penalties.

**Rationale:** Following the critic's feedback, `cue_cost` has been completely removed to align with the arbiter's original vision that weak cues simply decay rather than actively penalizing strong ones. To balance Take-The-Best and Tallying behaviors without breaking the Sequential Search family, the early stopping condition was modified to be relative to the remaining available evidence. The search now stops if the absolute difference in accumulated evidence is greater than or equal to `threshold` times the sum of the remaining decayed validities. When `threshold=1.0`, this perfectly mimics TTB logic (stopping when the remaining cues cannot possibly flip the decision). When `threshold > 1.0`, it forces deeper search (Tallying), and when `< 1.0`, it allows more impulsive decisions. This relative thresholding provides the flexibility needed to capture behavior across different experiments without artificial penalties.

**Parameters:**
  - `decay`: `[0.0, 1.0]`
  - `threshold`: `[0.0, 5.0]`
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
    
    decay = float(parameters["decay"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    acc_a = 0.0
    acc_b = 0.0
    
    # Evaluate cues sequentially
    for k, j in enumerate(cue_order):
        # Exponential decay of subjective weight based on cue rank
        weight = val[j] * (decay ** k)
        
        # Add weight for each active cue
        if a[j] == 1:
            acc_a += weight
        if b[j] == 1:
            acc_b += weight
        
        # Calculate the maximum possible remaining evidence
        remaining_evidence = sum(val[cue_order[m]] * (decay ** m) for m in range(k + 1, len(cue_order)))
        
        # Early stopping if the evidence difference reaches the relative threshold
        if abs(acc_a - acc_b) >= threshold * remaining_evidence:
            break
            
    scores = np.array([acc_a, acc_b])
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
