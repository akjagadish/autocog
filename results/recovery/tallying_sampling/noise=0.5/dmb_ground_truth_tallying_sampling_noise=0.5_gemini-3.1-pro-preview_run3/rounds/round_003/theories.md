# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Rationale:** Following the critic's feedback, the predict and policy logic of the Mixture of Subjective WADD and Tallying theory are kept identical. The only change is an adjustment to the parameter bounds for noise: lowering the `beta` range to `[0.01, 5.0]` and widening the `epsilon` range to `[0.0, 1.0]`. This allows the model to produce softer maximization and rely more on uniform lapsing, directly addressing the systematic under-prediction of minority choice rates observed in Experiments 1, 2, 4, and 6.

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_2` — KILLED ✗

**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers probabilistically mix between two cognitively simple heuristics. Take-The-Best (TTB) is a strict lexicographic rule that inspects cues in order of their validity and chooses based entirely on the first discriminating cue. Tallying is an equal-weight rule that simply counts the number of positive features for each option. The mixture is biased towards Tallying to account for the strong equal-weighting pull observed in human data, while still allowing TTB to explain some non-compensatory choices. A small uniform lapse rate accounts for general response errors.

**Rationale:** Following the critic's advice on Iteration 4, we retain the exact 'Mixture of TTB and Tallying' mechanism from the accepted Iteration 1 base, but adjust the parameter ranges to prevent the heuristics from canceling each other out and washing out the signal. Specifically, `w_tally` is shifted to [0.5, 1.0] to bias the mixture heavily toward Tallying, which aligns with human preferences in key experiments (like Exps 3 and 5) where TTB predicts the opposite direction. Additionally, `epsilon` is restored but restricted to [0.0, 0.3] to provide realistic lapse rates without causing the uniform-guessing washout seen in Iteration 1.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 0.3]`
  - `w_tally`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    # Order cues by validity (descending)
    cue_order = np.argsort(val)[::-1]
    diff = stim[0] - stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for cue in cue_order:
        if diff[cue] > 0:
            p_ttb = np.array([1.0, 0.0])
            break
        elif diff[cue] < 0:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
