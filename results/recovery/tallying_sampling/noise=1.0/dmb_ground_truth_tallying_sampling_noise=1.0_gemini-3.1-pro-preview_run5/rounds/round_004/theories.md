# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

**Rationale:** Following the arbiter's feedback, the observed metrics across all four experiments consistently hover around 0.5. This indicates a lack of systematic preference for the choices predicted by either Take-The-Best or Tallying on their respective critical trials. A pure Random Guessing baseline (parameterized by a slight side bias to account for minor individual differences) naturally yields a ~0.5 expected value on all these metrics, providing a highly parsimonious explanation for the empirical data.

**Parameters:**
  - `bias`: `[0.4, 0.6]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Tallying with High Lapse: Subjects evaluate multi-attribute choices by simply counting the total number of positive features for each option (Tallying), ignoring the validity weights. However, because the task lacks feedback and demands cognitive effort, subjects rely on random guessing (with a potential idiosyncratic bias) for the vast majority of trials (>95%). This creates a behavior that is overwhelmingly noisy but retains a faint, residual sensitivity to the overall quantity of positive attributes.

**Rationale:** Following the arbiter's feedback, this theory replaces the first-feature heuristic with a Tallying heuristic, combined with an extremely high lapse rate (>= 95%). Pure random guessing explains the data well, but this model tests if there is any residual, faint sensitivity to the overall number of positive attributes (a weak compensatory strategy) that pure guessing might miss, without overpredicting the influence of the leftmost feature or any single specific cue.

**Parameters:**
  - `epsilon`: `[0.95, 1.0]`
  - `bias`: `[0.4, 0.6]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected stimulus of shape (2, n_features), got {stim.shape}")
    
    # Tallying: count the number of positive features for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    if tally_a > tally_b:
        p_core = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    
    # The baseline guessing distribution with idiosyncratic bias
    p_guess = np.array([bias, 1.0 - bias])
    
    # Blend deterministic tallying rule with an extremely high random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * p_guess
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Temporal Dependency Baseline (Sticky Choice / Alternation)

**Rationale:** Following the arbiter's recommendation, this theory replaces the unsupported assumption of a faint Tallying signal with a purely temporal baseline model. Subjects are assumed to completely ignore the multi-attribute stimuli due to the lack of feedback and high cognitive demand. Instead, their decisions are driven by a 'Sticky Choice' or 'Alternation' heuristic, where the choice on the current trial depends solely on the choice made on the immediately preceding trial (parameterized by a probability of repeating the last choice). On the first trial, subjects rely on an idiosyncratic side bias. This provides a stronger, purely stimulus-independent null model that accounts for temporal dependencies often observed in low-effort decision-making.

**Parameters:**
  - `p_repeat`: `[0.0, 1.0]`
  - `bias`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    bias = float(parameters["bias"])
    
    responses = history.get("response", [])
    if len(responses) == 0:
        return np.array([bias, 1.0 - bias])
    else:
        last_resp = responses[-1]
        if last_resp == 0:
            return np.array([p_repeat, 1.0 - p_repeat])
        else:
            return np.array([1.0 - p_repeat, p_repeat])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```
