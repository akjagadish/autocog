# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Hybrid Tallying: Decision-makers primarily rely on a frugal Tallying heuristic, counting the number of features where one option dominates another. However, they are not completely blind to cue validities; they give a slight premium to the single most valid cue. This premium acts strictly as a tie-breaker or a soft modulator and is never large enough to overcome a strict difference in tallies, blending the robustness of Tallying with a minimal sensitivity to cue hierarchy.

**Rationale:** Following the critic's advice, the upper bound of the `premium` parameter is restricted from 3.0 to 0.99. This minimal edit ensures that the premium applied to the most valid cue acts strictly as a tie-breaker or a slight nudge. Because tally differences are integers ≥ 1, a premium < 1.0 can never override a strict tally majority. This restores the robust tallying behavior necessary to match the empirical direction in Experiments 3 and 4, while still allowing the top cue to softly modulate choices when tallies are tied.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `premium`: `[0.0, 0.99]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid Tallying expects a (2, n_features) stimulus.")

    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    premium = float(parameters["premium"])
    
    a, b = stim[0], stim[1]
    
    # Pure tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Identify the most valid cue (stable sort to break ties by original index)
    top_cue = np.argsort(-val, kind="stable")[0]
    
    # Premium for winning the most valid cue
    a_top = float(a[top_cue] > b[top_cue])
    b_top = float(b[top_cue] > a[top_cue])
    
    score_a = a_wins + premium * a_top
    score_b = b_wins + premium * b_top
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Take-The-Best (TTB): Decision-makers use a strict lexicographic heuristic. They evaluate cues one by one in descending order of their subjective validities. The first cue that discriminates between the two options strictly determines the choice. All remaining lower-validity cues are completely ignored. If no cue discriminates, they guess randomly. Response noise is modeled as a simple uniform lapse rate.

**Rationale:** Based on the arbiter's feedback, the Hybrid Tallying theory is unsupported because subjects do not exhibit top-cue tie-breaking behavior. To provide a sharper, canonical contrast against Pure Tallying, this proposal introduces a pure Take-The-Best (TTB) baseline. TTB operates strictly lexicographically, relying entirely on the single most valid discriminating cue without any compensatory integration or tallying.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    val = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Sort cues by validity descending
    order = np.argsort(-val, kind="stable")
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    n_opts = 2
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
