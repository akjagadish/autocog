# Round 14 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_16` — KILLED ✗

**Description:** Risk-Averse Evidence Integration (Logistic Saturation): Decision-makers evaluate options by integrating the validities of active cues, but they are 'risk-averse' regarding extreme validities. Instead of an explicit variance penalty, subjects apply a logistic saturation function to cue validities before summing them. This heavily bounds the maximum contribution of any single extreme cue ('all eggs in one basket'), naturally favoring options supported by a balanced coalition of moderately strong cues without ever violating monotonicity when new cues are added.

**Rationale:** Following the critic's diagnosis, the explicit variance penalty (`np.var`) has been removed because it inherently violates monotonicity (adding a weak supporting cue to a single strong cue spikes the variance and drops the overall subjective value). Instead, the model now applies a logistic saturation function to the cue validities before summing them. This bounds the impact of any single extreme cue, penalizing high-variance 'all eggs in one basket' options and favoring coalitions of moderately strong cues, perfectly capturing the risk-averse heuristic while remaining strictly monotonic.

**Parameters:**
  - `k`: `[1.0, 20.0]`
  - `x0`: `[0.5, 1.0]`
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
    
    k = float(parameters["k"])
    x0 = float(parameters["x0"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Logistic saturation function on validities to bound extreme cues
    w = 1.0 / (1.0 + np.exp(-k * (val - x0)))
    
    # Calculate sum of saturated evidence
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
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

### `pi_17` → slot 1 (via `new_theory`)

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
