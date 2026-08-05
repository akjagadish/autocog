# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Strategy Mixture Model: Decision-making in multi-attribute choice is heterogeneous. Individuals (or single decisions) rely on a probabilistic mixture of non-compensatory heuristics (like Take The Best) and compensatory integration (like Weighted Additive). Because TTB and WADD produce scores on fundamentally different scales, their choice determinism is parameterized independently. This allows the mixture to flexibly capture the intermediate match rates observed in empirical data without being constrained by an artificial coupling of noise levels across strategies.

**Rationale:** Following the critic's advice, the single inverse temperature parameter `beta` has been replaced by two independent parameters, `beta_ttb` and `beta_wadd`. Since TTB scores are exactly 0 or 1, while WADD scores scale with the sum of validities, sharing a single beta forced a compromise in the fit. Decoupling them allows the fitting procedure to independently calibrate the determinism of each strategy, freeing the `w_ttb` parameter to accurately capture the true mixture proportion and bring the TTB match rates closer to the empirical ~0.41-0.44 range.

**Parameters:**
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) --- 
    cue_order = np.argsort(-val, kind="stable").tolist()
    a, b = stim[0], stim[1]
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        
    # --- Weighted Additive (WADD) ---
    scores_wadd = stim @ val
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Softmax for TTB
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for WADD
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Mixture ---
    w_ttb = float(parameters["w_ttb"])
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # --- Lapse ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_2` — SURVIVED ✓

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

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Non-linear Weighting Model: Instead of probabilistically mixing distinct heuristics like Take The Best and Weighted Additive, decision-makers rely on a single compensatory integration process where cue validities are transformed non-linearly. By exponentiating the objective validities by a subjective parameter gamma, the model dynamically scales between equal-weighting (gamma near 0), objective WADD (gamma = 1), and effectively lexicographic/TTB-like behavior (gamma > 1) where the highest validity cue dominates the sum of all lower cues. This allows the model to capture the intermediate TTB-match rates on conflict trials while still producing compensatory ties (and thus guessing) when the non-linearly weighted sums are close.

**Rationale:** Following the critic's feedback, the previous adjustment to gamma's upper bound (down to 3.0) over-corrected and suppressed the TTB match rate too far below the empirical ~0.40-0.45 range. By setting the gamma range to [0.1, 5.0], we strike a balance: it prevents the model from sampling extreme convexities that perfectly mimic TTB (which drove Exp 4's guess rate down and Exps 1-3's match rates up), while still providing enough non-linearity to hit the observed intermediate TTB choice proportions on conflict trials.

**Parameters:**
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validities
    weights = val ** gamma
    
    # Compensatory integration using subjective weights
    scores = stim @ weights
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
