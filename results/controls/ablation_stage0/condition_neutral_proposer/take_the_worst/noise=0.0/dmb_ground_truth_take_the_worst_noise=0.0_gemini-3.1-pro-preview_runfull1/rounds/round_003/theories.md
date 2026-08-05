# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Equal-Weight Additive Rule: People integrate all available cues but ignore their differential validities, assigning equal weight to each cue. They compute an overall score for each option by summing its positive features. This is a fully compensatory process that discards validity magnitudes but retains the quantity of evidence. The option with the higher total score is favored, with response noise introduced via a softmax function and a uniform lapse rate.

**Rationale:** The arbiter suggested implementing Take The Best (TTB). However, the leaderboard shows that TTB (pi_1) has an overall score of 0.000, while Tallying (pi_2) achieves a perfect 1.000. This strongly indicates that subjects in these experiments do not use a non-compensatory lexicographic rule, but rather integrate multiple cues equally. Therefore, I am ignoring the arbiter's suggestion to implement TTB and instead proposing an 'Equal-Weight Additive' theory. This theory posits that subjects integrate all available cues but ignore their validities, assigning equal weight to each cue. Because softmax probabilities depend only on the difference between scores, comparing the sum of features (Equal-Weight) is mathematically identical to comparing the number of strict feature wins (Tallying). This perfectly captures the human behavior while framing it as a compensatory linear model rather than a heuristic tally.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal-Weight expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Sum all features (equal weighting)
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Lexicographic Tallying: Subjects evaluate options primarily by tallying the total number of positive features. If the difference in tallies between the options exceeds a certain threshold, they choose the option with the higher tally, ignoring specific cue validities. However, if the tally difference is small or zero (indicating a tie or weak evidence), they fall back to a lexicographic 'Take-The-Best' strategy, relying on the single most valid cue that discriminates between the options.

**Rationale:** Following the critic's feedback, the mechanism for Lexicographic Tallying remains identical, but the upper bound of the `delta` parameter has been reduced further from 2.0 to 1.0. This ensures that any tally difference of 1 or more will trigger the primary Tallying strategy, reserving the Take-The-Best fallback strictly for exact ties (difference of 0). This change is intended to capture the overwhelming empirical preference for tallying observed in Experiments 2, 3, and 4.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `delta`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Lexicographic Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    delta = float(parameters["delta"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    # Primary strategy: Tallying if difference is large enough
    if abs(tally_a - tally_b) >= delta:
        score_a = tally_a
        score_b = tally_b
    else:
        # Fallback strategy: Take-The-Best (Lexicographic)
        idx = np.argsort(val)[::-1]
        score_a, score_b = 0.0, 0.0
        for i in idx:
            if a[i] > b[i]:
                score_a = 1.0
                score_b = 0.0
                break
            elif b[i] > a[i]:
                score_a = 0.0
                score_b = 1.0
                break
        # If completely tied on all cues
        if score_a == 0.0 and score_b == 0.0:
            score_a = 0.5
            score_b = 0.5
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
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

**Description:** Weighted Additive (WADD): People integrate all available cues by weighting each cue according to its objective validity. They compute an overall compensatory score for each option by summing the validities of its positive features. The option with the higher total score is chosen, with response noise introduced via a softmax function and a uniform lapse rate. This provides a fully compensatory mechanism that contrasts with non-compensatory heuristics like Take-The-Best or equal-weighting strategies like Tallying.

**Rationale:** Following the arbiter's suggestion, this model implements a pure Weighted Additive (WADD) strategy. It computes a fully compensatory score for each option by weighting each present feature by its actual validity. This provides a clean theoretical contrast to the Equal-Weight (Tallying) model and the Lexicographic Tallying model, allowing us to evaluate whether subjects are genuinely integrating cue validities in a compensatory manner rather than relying on tallying or non-compensatory heuristics.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive score: dot product of features and validities
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
