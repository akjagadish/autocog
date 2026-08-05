# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Rationale:** Applying the minimal diff requested by the critic: narrowing the range of the `gamma` parameter from `[0.0, 10.0]` to `[0.0, 2.0]`. This restricts the model from selecting overly high inverse temperatures, forcing it to distribute its sampling more evenly across cues and bringing the choice proportions closer to the empirical ~0.50 mark on incongruent trials.

**Parameters:**
  - `gamma`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Mixture of Simple Heuristics (Take-The-Best and Tallying) with Softened Determinism

**Rationale:** Following the critic's feedback, the attempt to decouple the temperatures of TTB and Tallying was rejected because it failed to improve the fit and increased the aggregate loss. We revert to the highly successful single-temperature mixture model from Iteration 2 (which dropped the loss from 0.58 to 0.22). To address the over-determinism in Experiment 8 where both heuristics strongly align, we adjust the parameter bounds: restricting `beta` to a lower range [0.0, 2.0] enforces softer probability bounds, and widening `epsilon` to [0.0, 1.0] allows the model to rely more heavily on random guessing if needed. This minimal edit softens the predictions while preserving the core mechanism that successfully captured near-chance behavior in the other experiments.

**Parameters:**
  - `beta`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # TTB Heuristic: find the first discriminating cue in descending order of validity
    order = np.argsort(validities)[::-1]
    ttb_diff = 0.0
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            ttb_diff = stim[0, idx] - stim[1, idx]
            break

    # Tallying Heuristic: unweighted count of winning features
    a_wins = np.sum(stim[0] > stim[1])
    b_wins = np.sum(stim[1] > stim[0])
    tally_diff = a_wins - b_wins

    # Translate differences into probabilities using softmax (decision noise)
    # Bound the differences to prevent overflow
    z_ttb = np.clip(beta * ttb_diff, -100, 100)
    p_ttb_A = 1.0 / (1.0 + np.exp(-z_ttb))
    
    z_tally = np.clip(beta * tally_diff, -100, 100)
    p_tally_A = 1.0 / (1.0 + np.exp(-z_tally))

    # Mixture of the two heuristics
    p_core_A = w_ttb * p_ttb_A + (1.0 - w_ttb) * p_tally_A
    p_core_B = 1.0 - p_core_A

    # Apply uniform lapse rate
    p_final = (1.0 - epsilon) * np.array([p_core_A, p_core_B]) + epsilon * np.array([0.5, 0.5])

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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Stochastic Leaky Competing Accumulator (LCA): Decision-makers evaluate all features in parallel, accumulating evidence for each option proportional to the objective cue validities. The accumulation process is subject to leakage, lateral inhibition, and within-trial Gaussian noise. Due to these stochastic dynamics and a non-negativity constraint, the system frequently dampens small net differences in inputs, producing near-chance behavior when net evidence is weak or closely matched, while still robustly discriminating when one option is strongly superior.

**Rationale:** Following the critic's feedback, we modified the deterministic simplified LCA by introducing within-trial stochasticity (Gaussian noise) to the evidence accumulation updates. By simulating multiple noisy trajectories per trial and averaging the resulting choice probabilities, the model naturally captures the observed near-chance behavior when net evidence is closely matched, without relying exclusively on the uniform lapse rate.

**Parameters:**
  - `leak`: `[0.0, 5.0]`
  - `inhibition`: `[0.0, 5.0]`
  - `theta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `noise_std`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    noise_std = float(parameters["noise_std"])

    # Parallel evaluation: compute total input evidence for each option
    I_A = np.sum(validities * stim[0])
    I_B = np.sum(validities * stim[1])

    # Stochastic Leaky Competing Accumulator (LCA) dynamics
    # We simulate multiple paths to compute a stable expected choice probability
    n_sims = 100
    x_A = np.zeros(n_sims)
    x_B = np.zeros(n_sims)
    dt = 0.1
    steps = 50
    sqrt_dt = np.sqrt(dt)

    for _ in range(steps):
        dx_A = I_A - leak * x_A - inhibition * x_B
        dx_B = I_B - leak * x_B - inhibition * x_A
        
        # Add Gaussian noise at each time step (Euler-Maruyama method)
        noise_A = np.random.normal(0, noise_std, n_sims) * sqrt_dt
        noise_B = np.random.normal(0, noise_std, n_sims) * sqrt_dt
        
        # Update with non-negativity constraint
        x_A = np.maximum(0.0, x_A + dx_A * dt + noise_A)
        x_B = np.maximum(0.0, x_B + dx_B * dt + noise_B)

    # Translate final activation difference into choice probabilities
    diff = x_A - x_B
    # Numerically stable logistic function
    z = np.clip(theta * diff, -100, 100)
    p_A_sim = 1.0 / (1.0 + np.exp(-z))
    
    # Expected probability over stochastic paths
    p_A = np.mean(p_A_sim)
    p_B = 1.0 - p_A

    # Apply uniform lapse rate to account for execution errors or guessing
    return (1.0 - epsilon) * np.array([p_A, p_B]) + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
