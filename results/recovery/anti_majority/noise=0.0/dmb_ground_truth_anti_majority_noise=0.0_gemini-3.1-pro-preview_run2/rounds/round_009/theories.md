# Round 9 — Theories

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


### slot 2 — `pi_11` — KILLED ✗

**Description:** Heuristic Switching Theory with Rank-based Tallying: Decision-makers probabilistically alternate between a non-compensatory strategy (Take-The-Best) and a compensatory Rank-based Tallying heuristic. Take-The-Best relies entirely on the most valid discriminating cue. Rank-based Tallying evaluates options by summing rank-based weights (e.g., 1/rank) of all active features, completely avoiding global dilution. This mixture naturally accounts for why adding weak cues has a minimal or slightly positive impact (as weak cues add small positive values) rather than causing massive subjective devaluation.

**Rationale:** Following the arbiter's explicit instructions and the most recent critic feedback, this minimal edit abandons the 'Averaging' mechanism (which divided by the number of active cues, causing global dilution) and replaces it with 'Rank-based Tallying'. In this new compensatory component, cues are weighted by their inverse rank (1/rank), and an option's score is the simple sum of the weights of its active cues. By summing instead of averaging, adding weak cues has a strictly non-negative (slightly positive) impact, fulfilling the theoretical requirement.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    alpha = float(parameters["alpha"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Rank-based Tallying prediction (sum of rank weights, NO division by count)
    ranks = np.empty_like(cue_order, dtype=float)
    ranks[cue_order] = np.arange(1, len(val) + 1)
    weights = 1.0 / ranks
    
    score_a = np.sum(weights[a > 0.5])
    score_b = np.sum(weights[b > 0.5])
    scores_tally = np.array([score_a, score_b])
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy Mixture
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Apply lapse rate
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

### `pi_12` → slot 2 (via `new_theory`)

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
