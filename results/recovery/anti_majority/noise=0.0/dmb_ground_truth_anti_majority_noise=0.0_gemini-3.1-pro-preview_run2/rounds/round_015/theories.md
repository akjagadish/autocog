# Round 15 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_17` — KILLED ✗

**Description:** Evidence Averaging Heuristic Theory: Decision-makers evaluate an option by taking the strict average of the validities of its active cues, rather than summing them. This naturally explains the 'less-is-more' effect, as adding weak cues to an option with strong cues drags down its overall average validity, severely penalizing it without the need for an explicit dilution or capacity parameter.

**Rationale:** Following the arbiter's suggestion, this theory implements the Evidence Averaging Heuristic. Instead of summing cue validities (which requires complex parameterized dilution to explain the 'less is more' effect), this theory posits that decision-makers evaluate options by the strict average of their active cues' validities. A non-linear parameter (lambda_val) allows for transforming validities before averaging. Adding weaker cues to an option with strong cues drags down its average validity, inherently penalizing options that rely on many weak arguments. This elegant mechanism captures why options with a single highly valid cue are often preferred over those with multiple cues of mixed validities.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Strict average of active cues
    v_a = np.sum(w * a) / n_a if n_a > 0 else 0.0
    v_b = np.sum(w * b) / n_b if n_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_15` — SURVIVED ✓

**Description:** Diminishing Returns and Cue Count Dilution Theory: Decision-makers evaluate options by integrating the validities of active cues, but they perceive validity through a concave transformation that reflects diminishing marginal returns for extremely high-validity cues. Furthermore, the total integrated evidence for an option is diluted by the raw count of its active features. This captures the heuristic that presenting 'more arguments' can reduce the perceived average strength of the evidence, penalizing options that rely on many weak cues without imposing a strict capacity limit or a hard zero-bound.

**Rationale:** Following the arbiter's diagnosis, this theory replaces the strict top-K capacity bounds and convex weighting functions with a concave validity transformation (lambda < 1) and a continuous dilution mechanism. The concave transformation naturally models diminishing marginal returns, allowing a coalition of moderate cues to sometimes outweigh a single extreme cue, which directly explains the negative observed metric in Experiment 2. The dilution mechanism divides the total evidence by the number of active features raised to a power (gamma), capturing the 'more arguments' penalty without causing unrealistic zero-bounds or extreme negative evaluations.

**Parameters:**
  - `lambda_val`: `[0.1, 1.0]`
  - `gamma`: `[0.0, 3.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Concave transformation of validities for diminishing returns
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Integrate evidence and dilute by the count of active features
    v_a = np.sum(w * a) / (n_a ** gamma) if n_a > 0 else 0.0
    v_b = np.sum(w * b) / (n_b ** gamma) if n_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_18` → slot 1 (via `new_theory`)

**Description:** Lexicographic with Tolerance and Dilution: Decision-makers evaluate options sequentially by inspecting cues in descending order of validity. A running difference of accumulated evidence is maintained, and evaluation stops early if the absolute difference exceeds a tolerance threshold. The final accumulated difference is then diluted by the number of cues evaluated before being converted to choice probabilities. This ensures that early stopping (based on a few strong cues) yields highly confident decisions, whereas integrating many conflicting cues dilutes the average evidence, leading to less deterministic choices.

**Rationale:** Following the critic's advice, we revert to the accepted base mechanism (Lexicographic with Tolerance) but introduce a dilution factor based on the number of cues evaluated. The accumulated difference is divided by `cues_evaluated ** gamma` before being passed into the softmax function. This naturally solves the problem of early stopping yielding weak probabilities: when the model stops early, `cues_evaluated` is small, so the evidence remains strong, resulting in a confident prediction. When many cues are integrated without crossing the tolerance, the evidence dilutes, producing a less deterministic probability, mirroring human behavioral noise under conflict.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `tolerance`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 3.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    tolerance = float(parameters["tolerance"])
    epsilon = float(parameters["epsilon"])
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Order cues by validity (descending)
    order = np.argsort(val)[::-1]
    
    diff = 0.0
    cues_evaluated = 0
    
    # Sequential evidence accumulation
    for idx in order:
        cues_evaluated += 1
        if a[idx] != b[idx]:
            diff += w[idx] * (a[idx] - b[idx])
            # Stop evaluating if the evidence difference exceeds the tolerance threshold
            if abs(diff) >= tolerance:
                break
                
    # Dilute evidence by the number of evaluated cues
    diff = diff / (cues_evaluated ** gamma)
    
    # Convert the final accumulated difference into choice probabilities
    scores = np.array([diff, 0.0])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
