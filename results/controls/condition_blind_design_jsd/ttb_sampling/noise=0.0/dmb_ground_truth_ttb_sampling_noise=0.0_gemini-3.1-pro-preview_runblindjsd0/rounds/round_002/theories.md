# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: Decision-makers evaluate options using a non-compensatory, rank-based approach. They first rank the available features by their validity (descending). They then sequentially compare the options on these features, stopping at the first feature that discriminates between them (i.e., one option has a higher value than the other). The option favored by this discriminating feature is chosen, and all remaining features are completely ignored. If no features discriminate between the options, the decision-maker resorts to a random guess.

**Rationale:** Following the arbiter's suggestion, we implement the Take-The-Best (TTB) heuristic. This model sorts features based on their provided validities and evaluates them one by one. The first feature that differentiates the two options determines the choice 100%, ignoring all subsequent features. This is a purely non-compensatory strategy, contrasting strongly with the exhaustive integration of Weighted Additive (WADD) and the equal-weight exhaustive summation of Tallying. Randomness is introduced solely via an epsilon lapse rate, reflecting execution noise or inattention, and uniform guessing is used if no features discriminate.

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
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # Using mergesort for a stable sort in case of ties.
    order = np.argsort(-validities, kind='mergesort')
    
    a, b = stim[0], stim[1]
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequentially check features according to their validity ranking
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend the deterministic choice (or guess) with the lapse rate
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Probabilistic Lexicographic (ProbLex) Heuristic: Decision-makers evaluate options by sequentially checking features in descending order of their validity, similar to Take-The-Best. However, at each discriminating feature, there is a probability (gamma) that the decision-maker skips the cue or makes an execution error, failing to terminate the search. If a cue is skipped, they move to the next most valid feature. If all discriminating features are skipped or no features discriminate, they resort to guessing. This captures both the rank-based nature of human decision-making and the stochasticity of cue evaluation.

**Rationale:** Following the arbiter's suggestion, we replace WADD with a Probabilistic Lexicographic (ProbLex) model. ProbLex is a non-compensatory heuristic similar to Take-The-Best (TTB), where cues are checked in descending order of validity. However, it introduces a probability `gamma` of skipping a discriminating cue. This provides a more competitive heuristic benchmark against TTB, as it captures the sequential nature of decision-making while allowing for stochasticity in cue evaluation. Unlike a simple lapse rate which falls back to pure random guessing, skipping a cue in ProbLex allows the model to fall back on lower-validity cues, potentially capturing human errors or incomplete cue processing more accurately.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("ProbLex expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='mergesort')
    a, b = stim[0], stim[1]
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    # Sequentially check features according to their validity ranking
    for idx in order:
        if a[idx] > b[idx]:
            p_a += p_reach * (1.0 - gamma)
            p_reach *= gamma
        elif b[idx] > a[idx]:
            p_b += p_reach * (1.0 - gamma)
            p_reach *= gamma
            
    # If all cues are skipped or none discriminate, guess.
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Blend the deterministic choice (or guess) with the lapse rate
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

**Description:** Weighted Additive (WADD) Strategy with Bounded Weights and Expanded Temperature: Decision-makers evaluate options using a compensatory approach where all available features are integrated. Each feature is weighted by its log-odds validity, reflecting its normative diagnostic value. The subjective weight of a cue is bounded (equivalent to clipping validity at 0.05 and 0.95) to prevent extreme over-weighting. The decision-maker sums the bounded weighted feature values for each option and translates the resulting scores into choice probabilities via a softmax function, with a wide temperature range allowing for highly deterministic behavior when required.

**Rationale:** Following the critic's advice, we retain the [0.05, 0.95] clipping bounds for validities from Iteration 3, as Iteration 4 proved that narrowing them degrades the fits on Experiments 3 and 5. To give the model the flexibility to capture the sharper, more deterministic choices required by Experiment 6, we expand the upper bound of the `beta` parameter from 20.0 to 100.0. This minimal edit allows between-subject variance in temperature to better accommodate differences across experimental designs without altering the core compensatory mechanism.

**Parameters:**
  - `beta`: `[0.1, 100.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid extreme log-odds values and division by zero
    v = np.clip(validities, 0.05, 0.95)
    # Compute log-odds weights for each feature
    w = np.log(v / (1.0 - v))
    
    a, b = stim[0], stim[1]
    
    # Calculate the weighted sum of features for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Translate scores to probabilities using a numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (random guessing)
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
