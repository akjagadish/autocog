# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Decision makers in multi-attribute choice tasks exhibit strategic heterogeneity. Rather than uniformly applying a single heuristic, the population consists of a mixture of strategies, predominantly Take The Best (TTB) and Tallying. On any given trial, a decision maker integrates the choice probabilities of these two heuristics according to their individual propensity (w_ttb). This mixture explains why aggregate choices in adversarial designs—where TTB and Tallying make opposing predictions—often hover around 50% alignment with either single heuristic.

**Rationale:** The arbiter noted that pure TTB and pure Tallying models fail completely on adversarial experimental designs where the two heuristics make opposite predictions, resulting in ~50% population alignment with either one. By positing a Strategy Mixture model where individuals weigh TTB and Tallying according to a parameter w_ttb (drawn from [0, 1]), the population average naturally centers around 0.50 in these adversarial cases. This effectively captures the observed variance and mean alignment across both experiments better than any single-heuristic formulation.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    
    # --- TTB Prediction ---
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tal = np.array([a_wins, b_wins])
    z_tal = beta * (scores_tal - np.max(scores_tal))
    e_tal = np.exp(z_tal)
    p_tal = e_tal / np.sum(e_tal)
    
    # --- Mixture & Noise ---
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tal
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. While they may occasionally attempt to tally features or look at the most valid cue, the lack of grounding leads to a near-uniform choice probability across all stimulus pairs.

**Rationale:** The real experimental data shows choice alignments hovering extremely close to 0.5 across all metrics (e.g., 0.5025, 0.5121, 0.5117, 0.4842). Previous models like pure TTB or Tallying vastly overpredicted decisive choices (reaching ~0.85). Even the mixture model predicted ~0.67 in some experiments. By explicitly modeling a massive lapse rate (epsilon between 0.8 and 1.0) and very low inverse temperature (beta near 0), this theory posits that subjects are essentially guessing due to the abstract task and lack of feedback, directly capturing the near-0.5 alignment observed in the empirical data.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Calculate a weak tallying signal
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Massive lapse rate dominates the choice
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
