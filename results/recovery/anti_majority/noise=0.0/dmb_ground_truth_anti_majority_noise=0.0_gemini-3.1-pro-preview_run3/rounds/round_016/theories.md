# Round 16 — Theories

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


### slot 2 — `pi_18` — KILLED ✗

**Description:** Feature-Averaging with Cue-wise Cancellation and Soft Normalization

**Rationale:** Following the arbiter's feedback, we roll back the changes from the rejected iteration 2 and return to the successful iteration 1 base. We restore the 0.5 baseline thresholding for validities, which prevents low-validity cues from being over-weighted. To address the discrepancies in experiments like 8 and 30, we replace the strict `n ** alpha` normalization with a softer penalty: `1 + alpha * (n - 1)`. This smoothly interpolates between summation (alpha=0) and pure averaging (alpha=1) without artificially inflating or deflating the evidence when exactly one unique cue remains, providing a better fit for trials contrasting single-cue vs multi-cue unique features.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `alpha`: `[0.0, 2.0]`
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
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities to reflect subjective evidence weight, keeping the 0.5 baseline
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Cue-wise cancellation: only consider unique features
    diff = a - b
    unique_a = (diff > 0)
    unique_b = (diff < 0)
    
    # Feature-averaging: soft normalization interpolating between sum (alpha=0) and mean (alpha=1)
    n_a = np.sum(unique_a)
    sum_w_a = np.sum(w[unique_a])
    ev_a = sum_w_a / (1.0 + alpha * (n_a - 1.0)) if n_a > 0 else 0.0
    
    n_b = np.sum(unique_b)
    sum_w_b = np.sum(w[unique_b])
    ev_b = sum_w_b / (1.0 + alpha * (n_b - 1.0)) if n_b > 0 else 0.0
    
    # Convert evidence to probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
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

### `pi_19` → slot 2 (via `new_theory`)

**Description:** Salience-Weighted Evidence Accumulator: Decision-makers process options by accumulating evidence from their features, but features shared between options receive reduced attentional weight (salience down-weighting) rather than being perfectly cancelled. This partially discounted shared evidence is added to the unique evidence, and the total accumulated evidence for each option undergoes a mild non-linear squashing (e.g., a parameterized power function). This compressive non-linearity ensures that adding shared features subtly reduces the discriminability between options, naturally predicting the small but non-zero choice shifts observed when common cues are introduced.

**Rationale:** Following the arbiter's feedback, this model instantiates the 'Salience-Weighted Evidence Accumulator'. Instead of completely cancelling shared cues (which fails to predict shifts when shared cues are added) or using extreme divisive normalization (which suppresses distinctiveness too much), this model applies a salience down-weighting parameter (`phi`) to shared features. The accumulated evidence then passes through a power-function squashing (`gamma`). When gamma < 1, the presence of shared cues increases the base evidence for both options, pushing them into the flatter region of the power function and thereby reducing their relative discriminability. This mechanism captures context-dependent choice shifts robustly without losing sensitivity to overall cue differences.

**Parameters:**
  - `theta`: `[0.1, 5.0]`
  - `phi`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 2.0]`
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
    theta = float(parameters["theta"])
    phi = float(parameters["phi"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities to subjective weights
    w = np.maximum(0.0, val - 0.5) ** theta
    
    # Identify unique and shared positive features
    shared = (a == 1) & (b == 1)
    unique_a = (a == 1) & (b == 0)
    unique_b = (a == 0) & (b == 1)
    
    # Accumulate evidence with salience down-weighting for shared features
    ev_a = np.sum(w[unique_a]) + phi * np.sum(w[shared])
    ev_b = np.sum(w[unique_b]) + phi * np.sum(w[shared])
    
    # Mild non-linear squashing (power function)
    sq_a = ev_a ** gamma if ev_a > 0 else 0.0
    sq_b = ev_b ** gamma if ev_b > 0 else 0.0
    
    # Convert to probabilities via softmax
    z = beta * np.array([sq_a, sq_b])
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
