# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Autocorrelated Weighted Additive (WADD) Theory: Decision-makers evaluate options by integrating all available cues proportionally to their validities, but their choices are also subject to sequential dependencies (inertia or alternation). A 'stickiness' parameter biases the subjective value of the option that corresponds to the action chosen in the immediately preceding trial, capturing autocorrelation in choice behavior.

**Rationale:** Following the arbiter's suggestion, this theory builds upon the WADD model by explicitly incorporating sequential dependencies. A new 'stickiness' parameter is introduced, which biases the current choice by modifying the subjective value of the option corresponding to the previously chosen action. By actively using the `history` argument, this modification directly models the conditional choice probabilities observed in the data, accounting for trial-to-trial autocorrelation (such as choice inertia or alternation) that stationary models miss.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `stickiness`: `[-3.0, 3.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    stickiness = float(parameters["stickiness"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply choice stickiness from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        prev_choice = int(history["response"][-1])
        if 0 <= prev_choice < len(scores):
            scores[prev_choice] += stickiness
    
    # Softmax with max-subtraction for numerical stability
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
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Integrated Utility Theory (WADD + Tallying): Decision-makers integrate both compensatory validity weighting (WADD) and non-compensatory feature counting (Tallying) into a single, unified subjective value for each option. Rather than stochastically switching between strategies on a trial-by-trial basis, they form a composite score by taking a weighted average of the WADD and Tallying scores. This composite score is then translated into choice probabilities via a single softmax process subject to decision noise and lapses.

**Rationale:** Following the critic's advice, this minimal edit changes how the WADD and Tallying components are combined. Instead of applying a softmax to each strategy independently and mixing their resulting probabilities (which implies a trial-by-trial stochastic switching between strategies), the model now computes a weighted average of the raw scores (utilities) from WADD and Tallying before applying a single softmax. This integrated utility approach posits that individuals simultaneously weigh exact validities and simple feature counts to form a unified evaluation, potentially offering a better structural fit without adding new parameters.

**Parameters:**
  - `w`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = float(parameters["w"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # WADD scores
    wadd_scores = np.sum(stim * val, axis=1)
    
    # Tallying scores (strict wins)
    tally_scores = np.array([np.sum(a > b), np.sum(b > a)], dtype=float)
    
    # Mix scores instead of probabilities
    mixed_scores = w * wadd_scores + (1.0 - w) * tally_scores
    
    z = beta * (mixed_scores - np.max(mixed_scores))
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
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Integrated Heuristic-Inertia Theory: Decision-makers evaluate options by integrating both compensatory cue weighting (WADD) and non-compensatory feature counting (Tallying) into a single subjective value. In addition, their choices exhibit temporal dependence, where the log-odds of choosing an option are biased by whether it was chosen on the immediately preceding trial. This forms a composite utility that drives choice through a noisy maximization process, with inertia applied independent of the softmax temperature.

**Rationale:** Following the critic's advice, we retain the successful structural change from the previous iteration (applying stickiness directly to the logits after scaling by beta) but expand the stickiness parameter bounds from [-3.0, 3.0] to [-5.0, 5.0]. This provides the model with the capacity to capture stronger sequential inertia effects, which should help fit the high JSD values observed in the empirical data.

**Parameters:**
  - `w`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `stickiness`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = float(parameters["w"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    stickiness = float(parameters["stickiness"])
    
    a, b = stim[0], stim[1]
    
    # Compensatory WADD scores
    wadd_scores = np.sum(stim * val, axis=1)
    
    # Non-compensatory Tallying scores (strict wins)
    tally_scores = np.array([np.sum(a > b), np.sum(b > a)], dtype=float)
    
    # Form composite subjective value
    mixed_scores = w * wadd_scores + (1.0 - w) * tally_scores
    
    # Scale by temperature
    z = beta * mixed_scores
    
    # Apply sequential choice stickiness from the previous trial directly to logits
    if history and "response" in history and len(history["response"]) > 0:
        prev_choice = int(history["response"][-1])
        if 0 <= prev_choice < len(z):
            z[prev_choice] += stickiness
            
    # Softmax with max-subtraction for numerical stability
    z = z - np.max(z)
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
    return np.random.choice(len(probabilities), p=probabilities)
```
