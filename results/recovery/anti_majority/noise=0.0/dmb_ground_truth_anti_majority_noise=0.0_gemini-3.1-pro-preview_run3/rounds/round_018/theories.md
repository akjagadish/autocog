# Round 18 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_17` — SURVIVED ✓

**Description:** Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

**Rationale:** Following the critic's advice, this edit retains the exact same lateral inhibition mechanism but further widens the parameter ranges for self-dilution (`alpha`) up to 5000.0 and decision temperature (`theta`) up to 500.0. This allows the optimizer to find a regime where multi-cue options are entirely squashed by massive self-dilution, reducing their evidence to near exactly zero, while single-cue options remain unsuppressed and deterministically win due to the high theta. This extreme contrast is needed to capture the deep negative reversals seen in Experiments 12, 29, and 30.

**Parameters:**
  - `alpha`: `[0.0, 5000.0]`
  - `beta`: `[0.0, 100.0]`
  - `gamma`: `[0.1, 5.0]`
  - `theta`: `[0.1, 500.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_20` — KILLED ✗

**Description:** Relative Evidence Accumulation with Conflict Discounting: Decision-makers evaluate options by accumulating evidence from their features, but this evidence is dynamically discounted by the presence of conflicting features in the competing option (cross-option inhibition). This avoids perverse self-dilution (adding cues to an option never hurts it) while capturing conflict-driven non-linearities and context effects through mutual suppression.

**Rationale:** Following the arbiter's recommendation, this model implements 'Relative Evidence Accumulation with Conflict Discounting'. Instead of normalizing by the total number of features within an option (which causes perverse self-dilution in Exp 1 and 2), the evidence for an option is discounted by the number of conflicting features in the competing option (cross-option inhibition). This ensures that adding supporting cues strictly benefits an option, while the mutual suppression mechanism naturally captures the conflict-driven non-linearities and context effects required for Experiments 13, 14, 31, 32, and 33.

**Parameters:**
  - `alpha`: `[0.0, 5.0]`
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of cue validities (centered at chance 0.5)
    w = np.maximum(0.0, val - 0.5) ** alpha
    
    # Cross-option conflict: number of features present in the competitor but absent in the focal option
    conflict_a = np.sum(b * (1.0 - a))
    conflict_b = np.sum(a * (1.0 - b))
    
    # Evidence accumulation with exponential conflict discounting
    ev_a = np.sum(a * w) * np.exp(-gamma * conflict_a)
    ev_b = np.sum(b * w) * np.exp(-gamma * conflict_b)
    
    # Softmax choice rule
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_21` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture with Validity-Scaled TTB, Mean-Normalized Tallying, and Logistic Dispersion Modulation

**Rationale:** Following the critic's advice, we build strictly on the accepted Iteration 7 base. To resolve the scale mismatch between the TTB and Tallying evidence vectors, we normalize the Tallying evidence by taking the mean of the cues rather than the sum, ensuring both operate roughly on a [0, 1] scale. This prevents Tallying from disproportionately dominating the mixture simply due to larger raw values. Additionally, we widen the bounds for the logistic parameters (w_base to [-20.0, 20.0] and gamma to [-50.0, 50.0]) to allow the model to push the mixture weight even closer to 1.0 in TTB-dominant environments, enabling it to fully capture the extreme primacy effects observed in the conflict experiments.

**Parameters:**
  - `w_base`: `[-20.0, 20.0]`
  - `gamma`: `[-50.0, 50.0]`
  - `beta_ttb`: `[0.1, 50.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy - scaled by the validity of the discriminating cue
    ev_ttb = np.array([0.0, 0.0])
    for i in cue_order:
        if a[i] > b[i]:
            ev_ttb = np.array([val[i], 0.0])
            break
        elif b[i] > a[i]:
            ev_ttb = np.array([0.0, val[i]])
            break
            
    # Tallying Strategy (Unit-Weight Additive) - normalized by total cues to match scale
    ev_tally = np.array([np.mean(a), np.mean(b)])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB probabilities
    z_ttb = beta_ttb * ev_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb = p_ttb / np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta_tally * ev_tally
    z_tally = z_tally - np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally = p_tally / np.sum(p_tally)
    
    # Strategy Mixture Weight (Logistic modulation based on dispersion)
    w_base = float(parameters["w_base"])
    gamma = float(parameters["gamma"])
    dispersion = np.std(val) if len(val) > 1 else 0.0
    
    logit_w = w_base + gamma * dispersion
    w = 1.0 / (1.0 + np.exp(-logit_w))
    
    p_mix = w * p_ttb + (1.0 - w) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
