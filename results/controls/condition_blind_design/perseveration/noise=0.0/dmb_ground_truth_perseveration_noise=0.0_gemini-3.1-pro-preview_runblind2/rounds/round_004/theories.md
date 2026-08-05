# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Subject-Specific Single-Cue Heuristic and Strong Spatial Bias: Subjects largely ignore the objective validities provided in the instructions. Instead of engaging in complex compensatory or non-compensatory integration of multiple cues, individuals adopt highly simplified, deterministic strategies. The population is split: some subjects rely on a strong spatial bias (e.g., always choosing the left or right option), while others adopt a single-cue heuristic, randomly selecting one feature to follow deterministically and ignoring all others. This extreme simplification explains the ~50% aggregate choice proportions on critical divergence trials, the failure to align with objective-consensus trials, and the extremely high between-subject variance, as choices are driven by idiosyncratic, deterministic biases rather than shared objective validities.

**Rationale:** Following the arbiter's feedback, this theory abandons the assumption that subjects internalize the objective cue validities. Instead, it models the population as a mix of individuals who either adopt a strict spatial bias (always picking Option A or Option B) or a single-cue heuristic (picking one random cue to follow deterministically). This naturally accounts for the chance-level (~50%) aggregate means across most experiments, including the consensus trials in Experiment 8 where normative theories predict high agreement. It also inherently produces the extremely high single-trial variance (~0.25) observed in the data, as half the population deterministically picks A and the other half deterministically picks B based on their idiosyncratic bias or chosen cue.

**Parameters:**
  - `strategy_class`: `{0, 1}`
  - `spatial_dir`: `{0, 1}`
  - `cue_weights`: `[(0.0, 1.0)] * n_features`
  - `epsilon`: `[0.0, 0.1]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    strategy_class = int(parameters["strategy_class"])
    spatial_dir = int(parameters["spatial_dir"])
    cue_weights = np.asarray(parameters["cue_weights"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    if strategy_class == 0:
        # Spatial Bias: deterministically choose Left (0) or Right (1)
        p_core = np.array([1.0, 0.0]) if spatial_dir == 0 else np.array([0.0, 1.0])
    else:
        # Single-Cue Heuristic: follow a single randomly preferred cue
        k = int(np.argmax(cue_weights))
        if a[k] > b[k]:
            p_core = np.array([1.0, 0.0])
        elif b[k] > a[k]:
            p_core = np.array([0.0, 1.0])
        else:
            # If the chosen cue is tied, guess randomly
            p_core = np.array([0.5, 0.5])
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Heterogeneous Cue Hierarchies: Individuals all employ a lexicographic, non-compensatory decision strategy (Take-The-Best), but they differ in how they construct their cue hierarchies. Rather than perfectly internalizing the objective cue validities provided in instructions, subjects form subjective cue validities by combining the objective validities with idiosyncratic, subjective weights. The degree to which a subject relies on their idiosyncratic weights versus objective validities varies across the population. This results in some subjects strictly following the objective hierarchy, while others use idiosyncratic hierarchies (e.g., heavily overweighting a specific cue). This heterogeneity explains the high between-subject variance and why objectively dominant options in aggregate only receive partial choice shares.

**Rationale:** Following the critic's advice, the parameter ranges for `noise_level` and `epsilon` have been expanded. `noise_level` is increased to [0.0, 10.0] to allow for even stronger idiosyncratic overrides of the objective validities, which should further disrupt aggregate preference for objectively dominant cues. `epsilon` is increased to [0.0, 0.5] to allow a higher base rate of guessing, which will help pull extreme choice proportions (like the 0.75 in Experiment 6 and 0.35 in Experiment 4) closer to the observed ~0.50 mark.

**Parameters:**
  - `noise_level`: `[0.0, 10.0]`
  - `subjective_weights`: `[(0.0, 1.0)] * n_features`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    subjective_weights = np.asarray(parameters["subjective_weights"], dtype=float)
    noise_level = float(parameters["noise_level"])
    epsilon = float(parameters["epsilon"])
    
    # Construct subjective validities by adding individual-specific noise/preferences to objective validities
    # subjective_weights are in [0, 1], so we center them around 0
    subj_val = validities + noise_level * (subjective_weights - 0.5)
    
    # Lexicographic strategy based on subjective hierarchy
    cue_order = np.argsort(-subj_val, kind="stable").tolist()
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Pure Spatial Bias: Subjects completely ignore the objective validities and the feature values of the options. Instead, 100% of the subjects adopt a deterministic spatial bias, choosing either exclusively Option A (Left) or exclusively Option B (Right) across all trials. This lack of engagement with the task features accounts for the ~0.50 baseline match rates for any feature-based rules across experiments, and perfectly explains the 0.500 mean absolute deviation in subject choice proportions.

**Rationale:** The arbiter analysis correctly points out that in Experiment 9, the mean absolute deviation of subject choice proportions from 0.5 is exactly 0.5000 with 0 variance. This means every single subject had a mean choice proportion of 0.0 or 1.0, indicating they deterministically chose either Option A or Option B on every trial. Implementing a pure spatial bias model with minimal noise perfectly captures this phenomenon and explains why feature-based rule predictions hover around chance (0.50) across all other experiments.

**Parameters:**
  - `spatial_dir`: `{0, 1}`
  - `epsilon`: `[0.0, 0.05]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    spatial_dir = int(parameters["spatial_dir"])
    epsilon = float(parameters["epsilon"])
    
    p_core = np.zeros(2)
    p_core[spatial_dir] = 1.0
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
