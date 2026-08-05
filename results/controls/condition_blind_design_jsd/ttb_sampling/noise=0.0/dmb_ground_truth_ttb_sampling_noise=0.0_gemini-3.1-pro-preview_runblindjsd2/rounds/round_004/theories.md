# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use the 'Take The Best' (TTB) heuristic to make binary choices. They search through features in descending order of their subjective or stated validities. The first feature that discriminates between the two options (i.e., one option has a positive feature and the other does not) strictly determines the choice, ignoring all other features. If no features discriminate, they guess uniformly. Response noise is modeled via a simple lapse rate where the individual occasionally guesses randomly instead of following the deterministic rule.

**Rationale:** Following the arbiter's recommendation, this model instantiates the Take The Best (TTB) heuristic. It orders features by their validities and makes a strict choice based on the first discriminating feature, acting as a non-compensatory lexicographic rule. This prevents lower-validity cues from overriding the most valid available cue. A lapse rate 'epsilon' is added to account for response noise and unmodeled errors.

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
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Take The Best with Validity-Dependent Confidence

**Rationale:** I am ignoring the arbiter's suggestion to propose a Tallying (Equal Weight) heuristic. As demonstrated by the prior theory `pi_1`, Tallying completely ignores cue validities and results in a very poor fit to the data (overall score 0.078). The high score of `pi_3` (Take The Best, 0.942) strongly indicates that subjects use cue validities to prioritize features. Instead of Tallying, I propose a refinement of Take The Best called 'Take The Best with Validity-Dependent Confidence'. In this theory, decision-makers still search through features in descending order of validity and stop at the first discriminating cue. However, rather than making a strictly deterministic choice, their probability of choosing the favored option scales with the log-odds validity of that discriminating cue. This captures the intuition that people are more confident and less noisy when the deciding cue is highly valid, and more equivocal when it is weak, subsuming strict TTB as a special case when beta is large.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] != b[idx]:
            # Convert validity to log-odds weight
            v = np.clip(validities[idx], 0.501, 0.999)
            w = np.log(v / (1.0 - v))
            
            # Assign weight to the option that has the feature
            scores = np.zeros(2)
            if a[idx] > b[idx]:
                scores[0] = w
            else:
                scores[1] = w
                
            # Softmax to convert to probability
            z = beta * scores
            z -= np.max(z)
            e = np.exp(z)
            p_core = e / np.sum(e)
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Theory: Decision-makers integrate all available information by weighting each feature according to its stated validity. The validities are converted to log-odds to represent evidence, and the total value of each option is the sum of the weights of its positive features. A softmax function converts these values into choice probabilities, combined with a uniform lapse rate to account for random errors.

**Rationale:** This theory replaces the non-compensatory Take The Best heuristic with a fully compensatory Weighted Additive (WADD) model, directly following the arbiter's feedback. By converting validities to log-odds weights and summing them across all features, it allows multiple weak cues to outweigh a single strong cue. The softmax decision rule and lapse rate accommodate response variability, satisfying the arbiter's request for a strong compensatory contrast.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities using log-odds for evidence weighting
    v_clipped = np.clip(validities, 0.501, 0.999)
    weights = np.log(v_clipped / (1.0 - v_clipped))
    
    # Calculate the weighted additive score for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax over the total scores
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
