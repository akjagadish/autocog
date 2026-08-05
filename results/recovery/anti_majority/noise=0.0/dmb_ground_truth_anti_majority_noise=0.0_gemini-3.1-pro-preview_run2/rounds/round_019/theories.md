# Round 19 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_19` — KILLED ✗

**Description:** Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

**Rationale:** Following the arbiter's recommendation, this theory implements the 'Relative Average Validity' model. Instead of relying on a lexicographic stopping rule or a complex dilution mechanism with a hard threshold, decision-makers simply compute the average validity of the active cues for each option. A base prior (represented by `prior_count` and `prior_sum`) is included to prevent zero-division and reflect a baseline expectation of quality. This formulation naturally accounts for the 'less-is-more' effect since adding weak cues directly dilutes the average evidence of an option. It also allows shared cues to exert a mild, context-dependent effect by modifying both the numerator and denominator simultaneously, balancing the observed phenomena in the experimental data.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `prior_count`: `[0.01, 10.0]`
  - `prior_sum`: `[0.0, 10.0]`
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
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
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


### slot 2 — `pi_21` — SURVIVED ✓

**Description:** Shared-Feature Cancellation and Evidence Averaging: Decision-makers evaluate options by first identifying and canceling out features that are shared (active) in both options. After removing these shared elements to simplify the task, they evaluate each option by averaging the validities of its remaining unique active cues. A base prior is included in the average to establish a baseline expectation, but its lower bounds are extended to exactly zero to allow subjects to completely turn off the prior if they rely purely on the raw average of unique cues.

**Rationale:** Following the critic's advice, we revert to the exact mathematical formulation of the accepted Iteration 1 base, maintaining both `prior_sum` and `prior_count`. To achieve a strict in-family improvement without disrupting the model's structural balance, we lower the bound for `prior_count` (and `prior_sum`) to exactly 0.0. This grants the optimization algorithm the flexibility to completely turn off the prior for subjects whose choices better reflect a pure raw average of unique cues, while preventing zero-division with a simple `max(..., 1e-6)` safeguard.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `prior_count`: `[0.0, 10.0]`
  - `prior_sum`: `[0.0, 10.0]`
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
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Shared-Feature Cancellation
    shared = a * b
    a_unique = a - shared
    b_unique = b - shared
    
    n_a = np.sum(a_unique)
    n_b = np.sum(b_unique)
    
    # Calculate average validity of unique features with a base prior (safeguarded against div-by-zero)
    v_a = (np.sum(w * a_unique) + prior_sum) / max(n_a + prior_count, 1e-6)
    v_b = (np.sum(w * b_unique) + prior_sum) / max(n_b + prior_count, 1e-6)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
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


## Replacement

### `pi_22` → slot 1 (via `new_theory`)

**Description:** Shared-Feature Cancellation with Contrastive Dilution: Decision-makers evaluate options by first canceling out features shared by both options. They then assess the remaining unique features by averaging their validities. Crucially, the expectation or 'baseline' they regress to during this averaging is dynamically set by the maximum validity among all active cues in the trial. This contrastive dilution naturally penalizes options that rely on weak unique cues, as these weak cues dilute the high expectation established by the strong (often canceled) cues, elegantly capturing both strict invariances and significant less-is-more effects.

**Rationale:** Following the arbiter's feedback, this theory implements 'Shared-Feature Cancellation with Contrastive Dilution'. It explicitly cancels shared features to simplify the decision space, which is strictly necessary for explaining invariances in experiments like Exp 2 and Exp 40. Instead of using a static prior, the baseline expectation for averaging unique features is dynamically anchored to the maximum validity of the active cues in the trial. When an option adds a weak unique cue to a set of strong shared cues, this dynamic baseline ensures the weak cue heavily dilutes the option's overall score, perfectly capturing the massive 'less-is-more' effect without relying on hard capacity limits.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `prior_weight`: `[0.01, 20.0]`
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
    prior_weight = float(parameters["prior_weight"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Identify maximum validity of active cues in the trial to serve as the dynamic baseline
    active = (a + b) > 0
    w_max = np.max(w[active]) if np.any(active) else 0.0
    
    # Shared-Feature Cancellation
    shared = a * b
    a_unique = a - shared
    b_unique = b - shared
    
    n_a = np.sum(a_unique)
    n_b = np.sum(b_unique)
    
    # Contrastive Dilution: average unique features with w_max as the dynamic prior
    v_a = (np.sum(w * a_unique) + prior_weight * w_max) / max(n_a + prior_weight, 1e-6)
    v_b = (np.sum(w * b_unique) + prior_weight * w_max) / max(n_b + prior_weight, 1e-6)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
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
