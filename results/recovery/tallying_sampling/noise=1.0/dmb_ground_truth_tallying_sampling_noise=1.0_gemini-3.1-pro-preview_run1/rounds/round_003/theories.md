# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

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


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** In complex, multi-attribute binary choice tasks without trial-by-trial feedback, subjects largely fail to integrate cue validities or even simple feature counts. Instead, their decisions are dominated by high response noise, effectively reducing their behavior to random guessing with a slight baseline bias for one option over the other. This explains the ~0.5 choice rates, ~0.5 TTB match rates, and ~0.0 differences across experimental conditions.

**Rationale:** The arbiter diagnosed that deterministic TTB and structured WADD models fail because human behavior in these complex, no-feedback tasks is much noisier and less systematic than these models assume. The empirical results show choice rates very close to 0.5, TTB match rates near 0.5, and cross-condition differences near 0.0. A pure random-guessing model with a baseline spatial/option bias predicts these flat empirical patterns much better than structured integration models.

**Parameters:**
  - `bias_a`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Biased Logistic Tallying: Subjects make decisions by counting the total number of positive features for each option (Tallying), ignoring cue validities. However, their choices are highly noisy and subject to a baseline spatial/option bias. The decision process is modeled as a logistic function over the difference in feature tallies between the two options, parameterized by a sensitivity weight and a baseline bias. The sensitivity to tally differences is weak, reflecting empirical data where feature counts only slightly sway noisy guessing.

**Rationale:** Following the critic's advice, I am restricting the `beta` parameter range from [-1.0, 1.0] to [-0.2, 0.2] to prevent over-sensitivity to the feature tally difference. This ensures the model behaves closer to the empirical data, where tally differences only weakly influence choices, reducing the overshoot observed in Experiments 2 and 4.

**Parameters:**
  - `beta`: `[-0.2, 0.2]`
  - `bias`: `[-2.0, 2.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    
    # Logistic choice rule based on tally difference and baseline bias
    diff = scores[0] - scores[1]
    logit = beta * diff + bias
    
    # Numerically stable sigmoid
    if logit >= 0:
        p_a = 1.0 / (1.0 + np.exp(-logit))
    else:
        exp_logit = np.exp(logit)
        p_a = exp_logit / (1.0 + exp_logit)
        
    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
