# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** First-Feature Heuristic (Left-to-Right Reading Bias) with High Lapse: Subjects minimize cognitive effort by inspecting only the first (leftmost) feature in the array. However, because this task is demanding and feedback is absent, subjects rely heavily on random guessing on the vast majority of trials, only occasionally applying the first-feature rule. This results in behavior that is near-random but retains a faint trace of the leftmost feature's influence.

**Rationale:** Following the critic's diagnosis, the First-Feature Heuristic mechanism was producing metric values too far from the ~0.5 observed in humans due to structural correlations in the experimental designs. To fix this while retaining the arbiter's prescribed mechanism, I constrained the lapse rate (epsilon) to [0.8, 1.0]. This ensures the model guesses randomly on most trials, diluting the extreme predictions of the first-feature rule and bringing the aggregate metrics down to the empirical ~0.5 baseline.

**Parameters:**
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a_first = stim[0, 0]
    b_first = stim[1, 0]
    
    epsilon = float(parameters["epsilon"])
    
    if a_first > b_first:
        p_core = np.array([1.0, 0.0])
    elif b_first > a_first:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend deterministic first-feature rule with a random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_6` → slot 2 (via `new_theory`)

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
