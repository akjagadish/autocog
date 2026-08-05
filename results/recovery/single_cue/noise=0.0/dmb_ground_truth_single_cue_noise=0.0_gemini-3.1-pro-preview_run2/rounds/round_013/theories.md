# Round 13 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_15` — KILLED ✗

**Description:** Tallying with Probabilistic Least-Valid Cue Dropping: To save cognitive effort while maintaining robust integration of information, decision-makers predominantly evaluate all available features (Pure Tallying). However, on a small fraction of evaluations, they boundedly drop the single least valid feature and tally exclusively on the remaining subset. This probabilistic dropping naturally produces subtle Take-The-Best or WADD-like biases in edge cases by occasionally ignoring the lowest-validity cue that might otherwise balance the tally, without destroying the overall pure-tallying majority.

**Rationale:** Following the critic's advice, the discrete parameter M was replaced with a continuous parameter `p_drop` representing the probability of dropping the single least valid cue on any given evaluation. To avoid the over-aggressive cue dropping that caused the previous iteration to be rejected, `p_drop` is strictly bounded to `[0.0, 0.25]`. This ensures that subjects predominantly rely on pure tallying, preserving the strong tallying majorities seen in key experiments, while still allowing the optimization to find the small, precise amount of cue-dropping needed to explain subtle edge-case biases. The prediction is implemented as a deterministic mixture of the two states (drop 0 vs drop 1) to ensure smooth optimization.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `p_drop`: `[0.0, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    p_drop = float(parameters["p_drop"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(-val, kind="stable")
    n_features = len(val)
    
    # State 0: Pure Tallying (drop 0 cues)
    a_wins_0 = 0.0
    b_wins_0 = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            a_wins_0 += 1.0
        elif b[idx] > a[idx]:
            b_wins_0 += 1.0
            
    scores_0 = np.array([a_wins_0, b_wins_0])
    z_0 = beta * (scores_0 - np.max(scores_0))
    e_0 = np.exp(z_0)
    p_0 = e_0 / np.sum(e_0)
    
    # State 1: Drop 1 least-valid cue
    a_wins_1 = 0.0
    b_wins_1 = 0.0
    K = max(1, n_features - 1)
    top_cues = order[:K]
    for idx in top_cues:
        if a[idx] > b[idx]:
            a_wins_1 += 1.0
        elif b[idx] > a[idx]:
            b_wins_1 += 1.0
            
    scores_1 = np.array([a_wins_1, b_wins_1])
    z_1 = beta * (scores_1 - np.max(scores_1))
    e_1 = np.exp(z_1)
    p_1 = e_1 / np.sum(e_1)
    
    # Expected choice probabilities (mixture)
    p_core = (1.0 - p_drop) * p_0 + p_drop * p_1
    
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

### `pi_16` → slot 1 (via `new_theory`)

**Description:** Recency-Biased Cue Overweighting: Decision-makers evaluate options by attempting to integrate all available features, but due to visual recency and short-term memory effects, the final feature in the sequence is disproportionately salient. While the first N-1 features are weighted according to their stated validities (subject to non-linear scaling), the final feature is assigned an independent, often much larger weight. This mechanism explains boundary cases where subjects' choices are driven by the nominally least valid cue, effectively overriding both compensatory tallying and the expected Take-The-Best hierarchy.

**Rationale:** The arbiter provided feedback indicating a need for a model where the final feature in the sequence is highly salient due to visual recency, causing it to disproportionately drive choice. This model instantiates 'Recency-Biased Cue Overweighting' by computing a weighted additive score where the first N-1 features are weighted by their validities (scaled by a parameter gamma), but the final feature is assigned an independent, freely varying 'recency_weight'. This enables the model to capture empirical boundary cases where the nominally least valid feature dominates the decision, explaining variance that pure TTB or Tallying fail to account for.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `recency_weight`: `[0.0, 10.0]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    # Overweight the final feature due to recency
    w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
