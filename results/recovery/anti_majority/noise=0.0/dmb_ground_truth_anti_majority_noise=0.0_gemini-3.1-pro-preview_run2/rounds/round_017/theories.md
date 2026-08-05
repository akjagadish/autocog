# Round 17 — Theories

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


### slot 2 — `pi_15` — KILLED ✗

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

### `pi_20` → slot 2 (via `new_theory`)

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
