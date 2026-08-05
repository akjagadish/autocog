# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Take The Best with Soft Compensatory Check: Decision-makers default to a non-compensatory heuristic (Take The Best) by relying on the most valid discriminating cue. However, instead of a deterministic shift, if the combined evidence (sum of log-odds weights) from cues opposing this initial choice grows, the probability of switching to a fully compensatory integration strategy (Weighted Additive) increases smoothly. This is modeled via a logistic function centered on a subject-specific evidence threshold, allowing for graded, conflict-driven transitions between strategies.

**Rationale:** Following the critic's advice, the hard deterministic threshold was replaced with a soft, probabilistic transition via a logistic function (`scipy.special.expit`). The probability of switching to WADD (`p_wadd`) smoothly increases as the opposing evidence exceeds the threshold, controlled by a new steepness parameter `k`. This smooth transition preserves the core logic of the previously accepted base model while providing the flexibility needed to capture the graded deviations and variance seen in Experiment 4.

**Parameters:**
  - `threshold`: `[0.0, 10.0]`
  - `k`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import scipy.special
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds weights for WADD and opposing evidence
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # --- Take The Best (TTB) Phase ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is not None:
        # Calculate opposing evidence: sum of weights where the TTB loser beats the TTB winner
        if winner_ttb == 0:
            opposing_evidence = np.sum(weights * (b > a))
        else:
            opposing_evidence = np.sum(weights * (a > b))
            
        k = float(parameters["k"])
        threshold = float(parameters["threshold"])
        # Soft transition: probability of using WADD is a logistic function of opposing evidence
        p_wadd = float(scipy.special.expit(k * (opposing_evidence - threshold)))
    else:
        # If no cue discriminates, default to WADD (which will tie)
        p_wadd = 1.0
        
    # --- Decision Formulation ---
    # WADD probabilities
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_t = np.exp(z_t)
    p_core_ttb = e_t / np.sum(e_t)
    
    # Mix strategies based on soft threshold
    p_core = p_wadd * p_core_wadd + (1.0 - p_wadd) * p_core_ttb
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
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


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a bounded evidence accumulation process that integrates sequential dependencies. They default to a non-compensatory heuristic (Take The Best) but smoothly transition to a compensatory strategy (Weighted Additive) when opposing evidence exceeds a soft threshold. Furthermore, choices are subject to sequential dependencies: the baseline preference for an option is dynamically adjusted based on the previous trial's choice, capturing choice inertia or auto-correlation. This moving baseline influences the final decision probabilities alongside the strategy-specific evidence.

**Rationale:** The new theory directly implements the arbiter's suggestion to incorporate sequential dependencies by adding a choice inertia mechanism alongside the bounded evidence accumulation process (soft compensatory check). The evaluation metric computes JSD conditioned on the previous trial's response, implying that the history of choices influences the current choice probabilities. By adding a dynamic baseline shift (`inertia`) to the logits of the previously chosen option, the model naturally produces auto-correlation in choices, improving its ability to match the conditional JSD metrics across experiments.

**Parameters:**
  - `threshold`: `[0.0, 10.0]`
  - `k`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `inertia`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import scipy.special
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds weights for WADD and opposing evidence
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # --- Take The Best (TTB) Phase ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is not None:
        # Calculate opposing evidence: sum of weights where the TTB loser beats the TTB winner
        if winner_ttb == 0:
            opposing_evidence = np.sum(weights * (b > a))
        else:
            opposing_evidence = np.sum(weights * (a > b))
            
        k = float(parameters["k"])
        threshold = float(parameters["threshold"])
        # Soft transition: probability of using WADD is a logistic function of opposing evidence
        p_wadd = float(scipy.special.expit(k * (opposing_evidence - threshold)))
    else:
        # If no cue discriminates, default to WADD (which will tie)
        p_wadd = 1.0
        
    # --- Sequential Dependency (Inertia) ---
    inertia = float(parameters["inertia"])
    z_inertia = np.array([0.0, 0.0])
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        z_inertia[last_resp] = inertia
        
    # --- Decision Formulation ---
    # WADD probabilities with inertia
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * scores_wadd + z_inertia
    z_w = z_w - np.max(z_w)
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities with inertia
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * scores_ttb + z_inertia
    z_t = z_t - np.max(z_t)
    e_t = np.exp(z_t)
    p_core_ttb = e_t / np.sum(e_t)
    
    # Mix strategies based on soft threshold
    p_core = p_wadd * p_core_wadd + (1.0 - p_wadd) * p_core_ttb
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
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
