# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Decision-makers use a Compensatory Weighted Additive (WADD) strategy where they integrate all available cues, weighting each by its validity centered around chance (validity - 0.5). This linear integration allows multiple weaker cues to appropriately accumulate and potentially override a single strong cue, avoiding the extreme dominance that log-odds weighting can cause. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.

**Rationale:** Following the critic's suggestion, I replaced the log-odds weighting with a simpler linear weighting scheme (`val - 0.5`). This prevents highly valid cues from producing mathematically extreme weights that essentially mimic a non-compensatory strategy. By using linear weights centered on chance, the model remains strictly within the WADD family and ensures a genuinely compensatory process where multiple weaker cues can more easily override a single strong cue, better capturing human data across all experiments.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Extract validities
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Linear weighting centered around chance (0.5) to prevent extreme weights
    weights = val - 0.5
    
    # Weighted additive sums
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Decision-makers use the 'Take-The-Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They search through the available features in descending order of their validities. The first feature that discriminates between the two options determines the choice, and all subsequent features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is incorporated as an overall lapse rate (epsilon), representing trials where the subject guesses uniformly at random instead of applying the heuristic.

**Rationale:** This theory directly implements the arbiter's suggestion to use the Take-The-Best (TTB) heuristic. It contrasts with WADD and Tallying by using a non-compensatory, lexicographic decision rule. The model searches features ordered by validity and strictly stops at the first discriminating cue, ignoring the rest. It accounts for noise using a simple lapse rate (epsilon) as requested, providing a distinct, psychologically plausible heuristic baseline.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-val, kind='mergesort')
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a Naïve Bayes (Log-Odds) Weighted Additive strategy where they integrate all available cues, weighting each by its log-odds (log(validity / (1 - validity))). This provides a theoretically grounded, non-linear evidence accumulation scheme that naturally scales cue strengths, allowing for a nuanced compensatory integration. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.

**Rationale:** The Naïve Bayes (Log-Odds) model provides a normative foundation for evidence accumulation. By transforming validities into log-odds, the model naturally captures the non-linear relationship between cue validity and evidential weight (e.g., a cue with 0.9 validity is much more than twice as strong as one with 0.6). This addresses the limitations of linear weighting (which under-weights highly valid cues) and non-compensatory heuristics (which ignore weaker cues), better matching the observed choice probabilities through a nuanced compensatory integration.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Extract validities
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid division by zero or log(0) in log-odds calculation
    # Validities are provided in [0.5, 1.0]
    val_clipped = np.clip(val, 1e-4, 1.0 - 1e-4)
    
    # Calculate log-odds weights
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Weighted additive sums
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
