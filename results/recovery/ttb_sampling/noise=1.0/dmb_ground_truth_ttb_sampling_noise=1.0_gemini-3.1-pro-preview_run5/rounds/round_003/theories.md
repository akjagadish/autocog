# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Noisy Weighted Additive (WADD) with Regularized Non-linear Weighting: Decision-makers compute a global utility for each option by summing the subjective weights of all cues that favor it. Subjective weights are formed by applying a non-linear transformation to the objective validities (via an exponent gamma), allowing decision-makers to either amplify or compress the relative importance of high-validity cues. To prevent extreme lexicographic behavior, the degree of non-linear amplification and the choice determinism are bounded. These utilities are then translated into choice probabilities via a softmax function parameterized by an inverse temperature and a lapse rate.

**Rationale:** Following the critic's advice, we reduce the upper bound of the `gamma` parameter to 3.0 and `beta` to 5.0. In the previous iteration, allowing gamma to grow up to 10.0 enabled the model to become overly non-compensatory (pseudo-lexicographic), which severely hurt performance on Experiment 1 by making the highest-validity cue completely dominate. Tightening these bounds regularizes the model, forcing it to find a better balance between compensatory and non-compensatory behavior across all experiments while retaining the successful `validities ** gamma` mechanism.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 3.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])

    # Compute a global utility for each option by summing the non-linearly weighted validities
    weights = validities ** gamma
    utilities = stim @ weights

    # Translate utilities into choice probabilities using a softmax function
    z = beta * utilities
    z = z - np.max(z)  # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Apply lapse rate
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

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

### `pi_6` → slot 2 (via `new_theory`)

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
