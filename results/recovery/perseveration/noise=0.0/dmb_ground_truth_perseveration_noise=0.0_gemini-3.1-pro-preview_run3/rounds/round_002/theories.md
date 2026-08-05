# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Subjects exhibit complete disengagement from the multi-attribute decision task. Rather than evaluating the options based on their features and the validities of those features, subjects ignore all stimulus information and simply guess at random on every trial, choosing Option A or Option B with equal probability.

**Rationale:** The arbiter feedback highlights that the observed experimental data perfectly aligns with random guessing across all experiments. The metrics for TTB agreement, Tallying agreement, and WADD/TTB divergence all center exactly on 0.5, while the difference in choice probabilities between symmetrical trials in Experiment 4 is exactly 0.0. This indicates that subjects are not using any systematic feature-based decision strategy, but are instead completely disengaged and flipping a coin for each choice. This Random Guessing model implements this by always returning a uniform probability distribution over the options, which perfectly matches the observed data without requiring any free parameters.

**Parameters:**
  (none)

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model completely ignores the state and history, 
    # reflecting total task disengagement.
    return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

**Rationale:** Following the arbiter's feedback, this theory implements a Weighted Additive (WADD) model that includes a strong spatial/positional bias ('bias_A') and a baseline lapse rate ('epsilon'). The experimental results show exactly 0.5 consistency with both TTB (Experiment 1) and Tallying (Experiment 2) with zero variance. This suggests subjects are either guessing randomly or exhibiting a strict side bias (clicking through), making their choices completely independent of the feature values. The inclusion of a wide-ranging side bias and lapse rate allows the model to capture this feature-ignoring behavior perfectly.

**Parameters:**
  - `beta`: `[0.0, 10.0]`
  - `bias_A`: `[-20.0, 20.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.

**Rationale:** Following the arbiter's suggestion, this model instantiates the 'Extreme Spatial Bias / Key Perseveration' theory. A pure random guessing model (pi_4) fails to capture the variance structure of the data, particularly in Experiment 6 where it yields a squared deviation of ~0.0 instead of 0.25. By assuming subjects are individually deterministic but perfectly split at the population level (50% Always-A, 50% Always-B), we perfectly recover the 0.5 accuracy metrics (since the experimental designs are balanced, always picking one option yields exactly 50% agreement with any balanced strategy like TTB or Tallying), the 0.0 differences in conditional probabilities, and the exact 0.25 squared deviation in Experiment 6 (since P(A) is 1.0 or 0.0 for each subject, yielding (1-0.5)^2 = 0.25 or (0-0.5)^2 = 0.25).

**Parameters:**
  - `preferred_option`: `{0, 1}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The subject adopts a deterministic response strategy, always choosing the same option.
    pref = int(parameters["preferred_option"])
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
