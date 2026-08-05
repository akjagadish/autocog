# Round 18 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_19` — SURVIVED ✓

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


### slot 2 — `pi_20` — KILLED ✗

**Description:** Reference-Dependent Quality Inference: Decision-makers maintain a default reference point (aspiration level) for evaluating product cues. Cues with validities above this reference point are perceived as strengths and add to the option's overall value, while cues below the reference point are perceived as defects and subtract from it. An option with zero features remains at the neutral baseline. This naturally explains why an option with zero features can be strictly preferred over an option with multiple weak cues, as the latter accumulates negative value. A loss aversion mechanism further allows negative signals to be weighted more heavily than positive ones.

**Rationale:** Following the critic's advice, we reverted the `reference_point` range to [0.0, 1.0] to let the model naturally find the balance threshold without overly penalizing strong cues. We restricted `lambda_val` to [0.1, 3.0] to prevent extreme compression of cue validities, which caused issues when the reference point was high. We also constrained `loss_aversion` to [1.0, 5.0] to enforce that penalties for weak cues are applied but do not completely overwhelm the choice probabilities. The Reference-Dependent Quality Inference mechanism remains identical.

**Parameters:**
  - `lambda_val`: `[0.1, 3.0]`
  - `reference_point`: `[0.0, 1.0]`
  - `loss_aversion`: `[1.0, 5.0]`
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
    reference_point = float(parameters["reference_point"])
    loss_aversion = float(parameters["loss_aversion"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    def get_score(opt):
        # Evaluate each cue against the reference point
        diff = w - reference_point
        # Apply loss aversion to cues that fall below the reference point (defects)
        weighted_diff = np.where(diff > 0, diff, loss_aversion * diff)
        # Sum the evidence only for active features
        return np.sum(opt * weighted_diff)
        
    v_a = get_score(a)
    v_b = get_score(b)
    
    # Convert scores to choice probabilities using softmax
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)  # numerical stability
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

### `pi_21` → slot 2 (via `new_theory`)

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
