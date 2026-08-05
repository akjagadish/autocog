# Round 16 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_16` — SURVIVED ✓

**Description:** Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

**Rationale:** Following the critic's feedback, the predict and policy functions are kept exactly the same as the previous running-best base. The only change is widening the parameter bounds for `alpha`, `gamma`, and `theta` to allow the model to fully express the extreme non-compensatory choice behavior (Take-The-Best-like reversals) observed in Experiments 20, 27, and 28. By expanding `alpha` to 10.0, `gamma` to [-10.0, 10.0], and `theta` to 50.0, the model is granted the necessary flexibility to stretch the initial evidence differences and translate them into highly deterministic choice probabilities.

**Parameters:**
  - `gamma`: `[-10.0, 10.0]`
  - `leak`: `[0.1, 2.0]`
  - `inhibition`: `[0.0, 5.0]`
  - `theta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
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


### slot 2 — `pi_18` — KILLED ✗

**Description:** Dual-Process Dynamic Attention Model: Decision-makers start by evaluating options using a fast, unweighted tallying process. If the relative tally difference (normalized by the number of cues) is large, this simple cue count drives the choice. However, when the initial relative tally difference is small or cues are conflicting, attention dynamically shifts toward the validities of the cues. In this later phase, cues are integrated proportionally to their reliability (validity). The decision-maker integrates the evidence (logits) from both processes before making a final choice, allowing for smooth compensatory behavior in high-conflict trials that scales consistently across environments with varying numbers of cues.

**Rationale:** Following the critic's advice, we normalized the tally difference by the total number of cues (`n_cues`). This ensures that the dynamic attention shift mechanism (`p_shift`) operates on a consistent, relative scale across experiments with varying numbers of features (from 3 to 8). We also expanded the upper bound of `gamma` to 20.0 to compensate for the smaller normalized tally difference values, giving the optimizer enough range to fit the shift.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_val`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_val = float(parameters["beta_val"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying process (unweighted)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    z_tally = beta_tally * np.array([tally_a, tally_b])
    
    # Validity-weighted process
    val_a = np.sum(a * val)
    val_b = np.sum(b * val)
    
    z_val = beta_val * np.array([val_a, val_b])
    
    # Dynamic attention shift based on relative tally difference
    n_cues = len(a)
    tally_diff = abs(tally_a - tally_b) / max(1, n_cues)
    p_shift = np.exp(-gamma * tally_diff)
    
    # Mixture of evidence (logits) rather than probabilities
    z_mix = (1.0 - p_shift) * z_tally + p_shift * z_val
    z_mix = z_mix - np.max(z_mix)
    p_mix = np.exp(z_mix) / np.sum(np.exp(z_mix))
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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

### `pi_19` → slot 2 (via `new_theory`)

**Description:** Rank-Weighted Sequential Sampling with Forward-Looking Stopping: Decision-makers evaluate cues sequentially in descending order of validity, accumulating evidence at each step. To decide whether to stop or continue, they compare the currently accumulated evidence against the maximum possible evidence that could be provided by the remaining cues. The probability of halting increases as the current evidence outweighs the potential remaining evidence. This forward-looking dynamic prevents premature stopping when remaining cues could overturn the decision, seamlessly bridging Take-The-Best and Tallying-like compensatory behavior.

**Rationale:** To address the severe under-prediction of tallying behavior in Exps 9, 15, 18, and 34, I modified the stopping probability mechanism. The previous version stopped too eagerly based solely on the absolute magnitude of accumulated evidence. I scaled the accumulated evidence by the maximum possible remaining evidence (`max_remaining`) and reduced the upper bound of `lambda_stop` from 10.0 to 3.0. This 'forward-looking' modification ensures that the model is much less likely to halt after just one or two cues if the remaining cues hold enough weight to potentially overturn the current lead, allowing it to better capture the compensatory integration seen in those specific experiments.

**Parameters:**
  - `lambda_stop`: `[0.0, 3.0]`
  - `beta`: `[0.1, 25.0]`
  - `gamma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    lambda_stop = float(parameters["lambda_stop"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending (stable sort to preserve original order on ties)
    order = np.argsort(-val, kind='stable')
    
    E = 0.0
    p_continue = 1.0
    p_choose_A = 0.0
    
    for i, idx in enumerate(order):
        # Accumulate evidence weighted by validity
        w = val[idx] ** gamma
        E += w * (a[idx] - b[idx])
        
        # Stopping probability depends on absolute accumulated evidence relative to remaining possible evidence
        if i < len(order) - 1:
            max_remaining = sum((val[j] ** gamma) for j in order[i+1:])
            S = 1.0 - np.exp(-lambda_stop * (abs(E) / (max_remaining + 1e-3)))
        else:
            S = 1.0  # Must stop at the last cue
            
        p_stop_here = p_continue * S
        p_continue *= (1.0 - S)
        
        # Choice probability if stopped at this step
        z = beta * E
        if z > 20.0:
            p_A_given_stop = 1.0
        elif z < -20.0:
            p_A_given_stop = 0.0
        else:
            p_A_given_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_choose_A += p_stop_here * p_A_given_stop
        
    # Apply lapse rate
    p_A_final = (1.0 - epsilon) * p_choose_A + epsilon * 0.5
    
    return np.array([p_A_final, 1.0 - p_A_final])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
