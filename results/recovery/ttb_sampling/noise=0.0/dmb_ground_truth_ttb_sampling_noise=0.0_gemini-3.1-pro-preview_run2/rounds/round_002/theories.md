# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

**Rationale:** Following the arbiter's feedback, this theory replaces the compensatory Tallying mechanism with the 'Take The Best' (TTB) heuristic. Instead of counting total feature wins, TTB assumes a non-compensatory lexicographic search: subjects evaluate cues in descending order of validity and make their choice based entirely on the first cue that discriminates between the options. This directly captures the subjects' strong preference for the option that wins on the most valid feature, resolving Tallying's inverse predictions on the key diagnostic trials. A lapse rate (epsilon) is included to account for execution errors or guessing.

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
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Strategy Selection: Decision makers probabilistically switch between a lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). On any given trial, an individual predominantly uses TTB (with high probability p_ttb) but occasionally lapses into Tallying, reflecting a mixture of non-compensatory and compensatory decision strategies driven by individual differences or cognitive load.

**Rationale:** Following the critic's feedback, the parameter ranges for the Strategy Selection model have been tightened. By restricting `p_ttb` to `[0.7, 1.0]` and `epsilon` to `[0.0, 0.1]`, the model is forced to predominantly rely on the Take The Best heuristic, which aligns better with the robust lexicographic behavior observed in the empirical data, while still allowing for occasional compensatory Tallying behavior.

**Parameters:**
  - `p_ttb`: `[0.7, 1.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(-validities, kind='stable')
    p_ttb_choice = np.array([0.5, 0.5])
    for i in order:
        if a[i] > b[i]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of strategies
    p_core = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Take The Best with Validity-Dependent Confidence: Decision makers use a non-compensatory, lexicographic strategy (Take The Best) to choose between options. However, their execution of this strategy is noisy, and the degree of noise (lapse rate) depends on the validity of the discriminating cue. When options are discriminated by a highly valid cue, confidence is high and the lapse rate is low. When the discriminating cue has low validity, confidence is lower, leading to a higher probability of guessing or lapsing. This maintains the non-compensatory nature of TTB while naturally explaining variations in performance across different trial structures.

**Rationale:** Following the arbiter's suggestion, this model implements 'Take The Best with Validity-Dependent Confidence'. It retains the purely non-compensatory lexicographic search of TTB but replaces the uniform lapse rate with a validity-dependent one. The lapse rate is computed as `epsilon + gamma * (1.0 - v)`, meaning that discriminating cues with lower validities (closer to 0.5) induce higher noise and more frequent guessing, whereas highly valid cues yield more deterministic choices. This successfully accounts for performance variance across trial types without introducing compensatory tallying.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(-validities, kind='stable')
    
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    p_core = np.array([0.5, 0.5])
    v_discriminating = None
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            v_discriminating = validities[i]
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            v_discriminating = validities[i]
            break
            
    if v_discriminating is not None:
        # Lapse rate increases as validity of the discriminating cue decreases
        lapse = epsilon + gamma * (1.0 - v_discriminating)
        lapse = max(0.0, min(1.0, lapse))
        return (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    else:
        # Guess if no cues discriminate
        return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
