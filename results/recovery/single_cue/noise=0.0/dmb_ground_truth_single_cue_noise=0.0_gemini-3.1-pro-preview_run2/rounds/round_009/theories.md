# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

**Description:** Weighted Additive (WADD) Theory with Dampened Log-Odds: Decision-makers evaluate options by computing a weighted sum of their features. The subjective weight of each feature interpolates between a flat weight (Tallying) and the normative log-odds of its cue validity. To prevent extreme validities from completely overpowering the compensatory tallying behavior, the log-odds are dampened using a squashing function (tanh). This allows the strategy to capture both tally-like behavior and validity-driven normative evidence integration without creating extreme, non-compensatory weights.

**Rationale:** Following the critic's advice, we reverted to the successful log-odds weighting scheme from Iteration 2 but introduced a `tanh` dampening function to the log-odds before mixing. This squashing function bounds the log-odds values, preventing high-validity features from creating extreme weights that entirely overpower the flat (Tallying) component on critical conflict trials. This minimal edit preserves the normative evidence-accumulation properties of the log-odds transformation while ensuring the model can still capture the strong compensatory tallying preferences observed in the human data.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into log-odds, clipping to avoid infinities
    val_clipped = np.clip(val, 0.001, 0.999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    # Dampen the log-odds using tanh to prevent extreme values from overpowering tallying
    dampened_log_odds = np.tanh(log_odds)
    
    # Interpolate between flat weighting (Tallying, gamma=0) and dampened log-odds (WADD, gamma=1)
    w = (1.0 - gamma) * 1.0 + gamma * dampened_log_odds
    
    # Compute the weighted additive value for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_2` — SURVIVED ✓

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_12` → slot 1 (via `new_theory`)

**Description:** Decision-makers employ a mixture of two distinct fast-and-frugal heuristics: Tallying and Take-The-Best (TTB). Tallying counts the number of features where an option strictly dominates the other, ignoring cue validities. TTB searches through features sequentially in descending order of validity, stopping at the first feature that discriminates between the options. Rather than integrating these into a single compensatory utility score, individuals apply these non-compensatory rules strictly. The population behavior is modeled by blending the deterministic probability vectors of these two heuristics via a mixing parameter, along with a lapse rate for random guessing.

**Rationale:** Previous models struggled to capture empirical behavior because they applied continuous, compensatory weighting (such as Softmax over tallies or non-linear WADD scores). This approach fails to capture the strict, non-compensatory nature of fast-and-frugal heuristics. Following the arbiter's guidance, this theory treats both Tallying and TTB as strict deterministic rules that output degenerate probability vectors (with uniform guessing on ties). By blending these predictions at the population level with a mixing parameter 'w_tally', the model naturally captures the 50/50 split in conflict trials (Exp 10) and TTB-like tie-breaking behavior (Exp 20) without over-smoothing the choices.

**Parameters:**
  - `w_tally`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying component (strict heuristic, no softmax)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_pred = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_pred = np.array([0.0, 1.0])
    else:
        tally_pred = np.array([0.5, 0.5])
        
    # Take-The-Best (TTB) component
    order = np.argsort(-val, kind="stable")
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Blend the deterministic predictions of the two heuristics
    p_core = w_tally * tally_pred + (1.0 - w_tally) * ttb_pred
    
    # Apply lapse rate for random guessing
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
