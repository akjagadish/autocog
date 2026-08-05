# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

**Rationale:** The arbiter suggested replacing Take The Best (TTB) with Tallying (Equal Weight heuristic). In Tallying, subjects ignore explicit cue validities and simply count the number of positive features for each option. The option with the higher tally is chosen. This strongly compensatory heuristic easily overrides the single most valid cue if it is outnumbered by other positive cues. This perfectly explains why subjects in Experiment 1 heavily anti-align with TTB (they choose the option with more positive cues, even if it lacks the highest-validity cue) and why subjects in Experiment 2 align closely with WADD when it disagrees with TTB (since WADD and Tallying both favor the option with a greater number of less-valid cues over a single highly-valid cue).

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
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

**Description:** Take-The-Best (TTB) Heuristic with Flexible Adherence: Decision-makers evaluate options using a lexicographic, non-compensatory strategy by searching through features in descending order of their explicit validities. The search stops at the first discriminating feature. However, because empirical behavior often systematically deviates from strict TTB (e.g., due to compensatory processes or anti-alignment on specific conflict trials), the model allows for full-range guessing (epsilon up to 1.0) and potential inversion of the lexicographic preference (negative beta) to capture aggregate deviations while maintaining the core TTB mechanism.

**Rationale:** Following the critic's advice, the parameter bounds have been widened to allow the model to better fit the data, which is heavily anti-aligned with strict TTB predictions. Specifically, epsilon was expanded to [0.0, 1.0] to permit high rates of guessing or baseline non-TTB behavior, and beta was changed to [-5.0, 5.0] to allow the softmax to invert if subjects systematically choose opposite to the TTB prediction. The core lexicographic search mechanism itself was kept exactly intact.

**Parameters:**
  - `beta`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) state.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-validities)
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic search
    for cue in cue_order:
        if stim[0, cue] > stim[1, cue]:
            score_a = 1.0
            break
        elif stim[1, cue] > stim[0, cue]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary win/loss/tie outcomes
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Model: People integrate all available cues to evaluate options, weighting each cue directly by its explicit validity. Unlike Tallying, which assumes equal weights, WADD assumes a fully compensatory process where higher validity cues have proportionally greater impact on the final decision. Unlike previous WADD instantiations that fit free parameters per cue, this model strictly uses the provided objective validities as the subjective weights.

**Rationale:** Following the arbiter's recommendation, this proposes a strict Weighted Additive (WADD) model. While the previous WADD instantiation (pi_2) included free 'weights' parameters for every single feature, leading to overparameterization and poor generalization, this version strictly weights features by their explicit validities. This provides the correct fully compensatory benchmark against Tallying, allowing us to test whether subjects use the explicit cue validities provided in the instructions.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: score is the dot product of features and their explicit validities
    scores = stim @ validities
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
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
