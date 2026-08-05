# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Rank-Weighted Additive Theory: Individuals integrate all cues but weight them according to an exponential decay based solely on their rank-order of validity. This creates a 'soft' lexicographic rule that acts primarily like Take-The-Best, but allows multiple secondary cues to exert a small, non-zero compensatory pull on the decision. Response variability is captured via a softmax choice rule and a lapse rate.

**Rationale:** Following the latest critic feedback, the previous attempt to restrict noise parameters failed because it amplified deterministic preferences, exacerbating the overshoot in Exps 3 and 4 and dropping Exp 6 too low. To flatten the choice probabilities and compress the exaggerated compensatory differences in Exps 3/4 while pulling Exp 6 closer to 0.5, we expand the lower bound of `beta` to [0.01, 5.0] to allow softer maximization, while maintaining the Rank-Weighted Additive mechanism and the `epsilon` range.

**Parameters:**
  - `decay`: `[0.01, 1.0]`
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: 0 is highest validity
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    # Exponential decay based on rank
    decay = float(parameters["decay"])
    weights = decay ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

**Rationale:** The previous WADD model used raw objective validities, which resulted in options with many lower-validity cues easily dominating those with fewer, higher-validity cues, failing to capture the ~0.50 choice proportions observed in adversarial experiments. Following the critic's feedback, we introduce a subjective validity scaling parameter, `gamma`. By exponentiating the validities by `gamma`, the model can disproportionately weight the highest-validity cues (when gamma > 1), allowing a single highly valid cue to mathematically balance out multiple lower-validity cues. This brings the weighted sums of the options closer together, yielding the observed ~50/50 choice proportions via the softmax decision rule.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Dynamic Dual Process Strategy Mixture: Individuals probabilistically switch between a non-compensatory 'Take-The-Best' (TTB) heuristic and a compensatory 'Tallying' strategy. Crucially, the probability of deploying TTB is not fixed but depends on the structural clarity of the choice—specifically, the validity of the best discriminating cue. When the best discriminating cue is highly valid, individuals are more likely to rely on TTB; when it is weaker, they shift towards Tallying (which integrates all positive cues with equal weight). To account for trials where choice behavior strongly diverges from both heuristics, the model allows for a wide range of decision noise (lapse rate) and potentially inverted or very soft Tallying temperatures.

**Rationale:** Following the critic's feedback, both the static mixture and logistic mixture iterations were rejected because they failed to capture behavior in experiments like Experiment 6 where individuals frequently choose the option that contradicts BOTH Take-The-Best and Tallying. To accommodate this, we revert to the successful clipped linear dynamic mixture formulation from Iteration 1, but we widen the parameter bounds for noise. Specifically, `epsilon` is expanded to [0.0, 1.0] and `beta_tally` is expanded to [-1.0, 10.0]. This allows the model to leverage higher lapse rates or softer/inverted tallying to accurately reflect the substantial decision noise or idiosyncratic weighting observed in these challenging trials, without altering the foundational TTB + Tallying architecture.

**Parameters:**
  - `w_base`: `[0.0, 1.0]`
  - `alpha`: `[-2.0, 2.0]`
  - `beta_tally`: `[-1.0, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the best discriminating cue for Take-The-Best (TTB)
    cue_order = np.argsort(-val, kind="stable")
    winner_ttb = None
    v_disc = 0.5  # default if no cues discriminate
    
    for j in cue_order:
        if a[j] != b[j]:
            winner_ttb = 0 if a[j] > b[j] else 1
            v_disc = val[j]
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Strategy 2: Tallying (unit-weight additive)
    scores_tally = np.array([np.sum(a), np.sum(b)])
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Dynamic Mixture Weight
    # w_ttb depends on a base rate and scales with the validity of the discriminating cue
    w_base = float(parameters["w_base"])
    alpha = float(parameters["alpha"])
    
    w_ttb = w_base + alpha * (v_disc - 0.5)
    w_ttb = np.clip(w_ttb, 0.0, 1.0)
    
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
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
