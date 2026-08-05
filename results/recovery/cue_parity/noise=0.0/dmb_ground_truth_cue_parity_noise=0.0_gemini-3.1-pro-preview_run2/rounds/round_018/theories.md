# Round 18 — Theories

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


### slot 2 — `pi_20` — KILLED ✗

**Description:** Tally-Difference Modulated Feature Differencing with Exponential Validity Scaling: Decision-makers evaluate options in parallel by computing feature differences, but the weights applied to these differences are dynamically modulated by the aggregate conflict between the options. When the tally difference is small (high conflict), decision-makers shift toward simple equal-weighting (tallying). When the tally difference is large (low conflict), they rely on validity weights. To capture non-compensatory (Take-The-Best) behavior during validity-weighted decisions, the validities are scaled exponentially, allowing the single best cue to mathematically dominate the sum of lesser cues without squashing the evidence scores.

**Rationale:** Following the latest feedback, I reverted to the Iteration 4 base which successfully modulates strategies based on the tally difference, but replaced the previous validity transformation (`val ** kappa`) with an exponential transformation (`np.exp(kappa * val)`). The power transformation caused all fractional validities to shrink toward zero for large kappa, compressing the evidence differences and preventing the softmax from making sharp, deterministic choices. The exponential transformation naturally creates large, non-vanishing weights where the highest validity cue exponentially dominates the lesser cues, perfectly capturing Take-The-Best behavior without squashing the scores or requiring complex normalization.

**Parameters:**
  - `gamma`: `[0.0, 10.0]`
  - `threshold`: `[0.0, 10.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `kappa`: `[0.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    kappa = float(parameters["kappa"])
    
    # Calculate conflict based on the absolute difference in simple tallies
    tally_diff = abs(np.sum(a) - np.sum(b))
    
    # Dynamic weight interpolation:
    # High tally difference -> validity weighting (p_val close to 1)
    # Low tally difference -> equal weighting / tallying (p_val close to 0)
    p_val = 1.0 / (1.0 + np.exp(-gamma * (tally_diff - threshold)))
    
    # Exponential scaling of validities to allow non-compensatory (lexicographic) dominance
    transformed_val = np.exp(kappa * val)
    
    # Tallying uses equal weights (1.0 for each feature)
    w = p_val * transformed_val + (1.0 - p_val) * 1.0
    
    # Parallel evaluation of options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    
    # Softmax choice
    z = beta * np.array([score_a, score_b])
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


## Replacement

### `pi_21` → slot 2 (via `new_theory`)

**Description:** Sequential Cue Sampling with Conflict-Modulated Threshold: Decision-makers sample cues sequentially in order of their validities. Instead of integrating all cues simultaneously, they accumulate evidence until it reaches a decision threshold. Crucially, the level of conflict between the options (inversely related to the absolute tally difference) dynamically modulates this threshold. Under low conflict, the threshold is low, leading to fast, non-compensatory decisions (like Take-The-Best). Under high conflict, the threshold increases, enforcing extensive sampling and yielding compensatory behavior. This provides a natural speed-accuracy tradeoff without mathematical artifacts.

**Rationale:** Following the arbiter's feedback, this new model discards the problematic tally-difference-modulated exponential scaling that caused mathematical artifacts and non-monotonic reversals in Theory 2. Instead, it implements a Sequential Cue Sampling framework where cues are evaluated in validity order. The core innovation is that the decision threshold (rather than cue weights) is modulated by the level of conflict between options. When conflict is low (large tally difference), the threshold drops, allowing the model to quickly terminate sampling and produce non-compensatory choices. When conflict is high (small tally difference), the threshold rises, forcing the model to sample extensively and act in a compensatory manner. This naturally captures the speed-accuracy tradeoff and seamlessly bridges TTB-like and Tallying-like behaviors.

**Parameters:**
  - `theta_0`: `[0.0, 3.0]`
  - `theta_1`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta_0 = float(parameters["theta_0"])
    theta_1 = float(parameters["theta_1"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate conflict based on the absolute tally difference of discriminating cues
    N_a = np.sum((a > b).astype(int))
    N_b = np.sum((b > a).astype(int))
    tally_diff = abs(N_a - N_b)
    
    # High tally difference means low conflict; zero tally difference means maximal conflict
    conflict = 1.0 / (1.0 + tally_diff)
    
    # Decision threshold is modulated by the level of conflict
    theta = theta_0 + theta_1 * conflict
    
    # Sequential sampling of cues prioritized by validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j]) * val[j]
        if abs(E) >= theta:
            break
            
    # Softmax choice probability based on accumulated evidence
    z = beta * np.array([E, -E])
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Incorporate lapse rate
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
