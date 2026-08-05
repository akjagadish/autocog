# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

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


### slot 2 — `pi_2` — KILLED ✗

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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Faced with abstract options and no trial-by-trial feedback, subjects abandon feature-based compensatory or non-compensatory strategies. Instead, they fall back on a strong, idiosyncratic side bias (e.g., a spatial preference for the left or right option, or a sequential preference for the first or second option). Each subject has a fixed probability of choosing Option A, which varies uniformly across the population from 0 to 1. This mechanism produces behavior that is completely independent of the feature validities or values, resulting in an aggregate alignment of ~50% with any feature-based heuristic (like Take The Best or Tallying), but driven by individual-level spatial/sequential biases rather than trial-level random guessing.

**Rationale:** Following the arbiter's suggestion, this theory replaces the deterministic Tallying mechanism of Theory 2 with an idiosyncratic side bias. Rather than capturing the ~50% baseline via a massive lapse rate (pure trial-by-trial random guessing as in Theory 1), this model assumes subjects consistently favor one option (e.g., Option A or Option B) regardless of the feature values. Across the population, these individual biases average out, naturally yielding ~0.5 on multi-cue metrics. Furthermore, because the experimental designs are mostly counterbalanced with respect to which option is favored by heuristics, a consistent side bias produces very low between-subject variance on these metrics, closely matching the real experimental variance.

**Parameters:**
  - `p_a`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The subject ignores the stimulus features and relies entirely on their idiosyncratic side bias.
    p_a = float(parameters["p_a"])
    
    # Return the fixed choice probabilities for Option A and Option B
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
