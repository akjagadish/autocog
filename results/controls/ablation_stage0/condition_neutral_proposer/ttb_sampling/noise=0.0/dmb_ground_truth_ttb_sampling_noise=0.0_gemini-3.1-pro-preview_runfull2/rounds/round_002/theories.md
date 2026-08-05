# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Rationale:** Following the arbiter's instructions, this model implements the Take The Best (TTB) heuristic. Instead of computing a compensatory score (like WADD) or an unweighted sum (like Tallying), TTB sequentially evaluates features in decreasing order of validity. It stops at the first cue that discriminates and bases the choice entirely on it. This explains the high variance in Experiment 1 because trials with identical tally differences can have their decisions driven by different cues depending on the specific feature patterns. It also explains the strong preference in Experiment 2 for the 'fewer but better' option, as the single highest-validity cue dominates the decision regardless of deficits on multiple lower-validity cues.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Stochastic Take-The-Best (STTB): People use a non-compensatory, one-reason heuristic to compare options, but their search order is probabilistic rather than strictly deterministic. The probability of examining a cue next is determined by a softmax over the subjective validities of the remaining unexamined cues. The search stops at the first feature that discriminates between the two options, and the decision is based solely on that feature. If the selected feature ties, it is ignored and the search continues. If all features are exhausted without a discriminator, the decision maker guesses. This model interpolates between strict Take-The-Best (at high inverse temperature) and the Minimalist heuristic with random cue search (at zero inverse temperature).

**Rationale:** Following the critic's feedback, the upper bound of the softmax inverse temperature (beta) for cue selection is increased from 20.0 to 200.0. In the previous iteration, a beta of 20.0 was insufficient to produce the highly deterministic search orders required to capture the strong Take-The-Best compliance observed in Experiments 3 and 4, since validities often differ by only small amounts (e.g., 0.1). Expanding the upper bound allows the model to closely approximate strict deterministic TTB when fit to subjects who exhibit that behavior, while preserving the ability to model more stochastic search paths via lower beta values.

**Parameters:**
  - `beta`: `[0.0, 200.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("STTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def get_prob(available_cues):
        if len(available_cues) == 0:
            return np.array([0.5, 0.5])
        
        v = validities[available_cues]
        z = beta * v
        z = z - np.max(z)  # numerical stability
        p = np.exp(z)
        p = p / np.sum(p)
        
        ans = np.zeros(2)
        for i, cue_idx in enumerate(available_cues):
            if a[cue_idx] > b[cue_idx]:
                ans[0] += p[i]
            elif b[cue_idx] > a[cue_idx]:
                ans[1] += p[i]
            else:
                new_cues = [c for c in available_cues if c != cue_idx]
                ans += p[i] * get_prob(new_cues)
        return ans

    n_features = len(validities)
    p_core = get_prob(list(range(n_features)))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture: Decision-makers predominantly use a deterministic non-compensatory heuristic (Take-The-Best) but probabilistically substitute it with a simple compensatory heuristic (Tallying) on a trial-by-trial basis. This mixture captures the overwhelming adherence to TTB while accounting for systematic deviations toward options with a higher quantity of positive cues in extreme conflict scenarios.

**Rationale:** Tightened the parameter ranges for p_ttb to [0.6, 1.0] and epsilon to [0.0, 0.2] as suggested by the critic. This ensures the model predominantly relies on Take-The-Best, yielding the ~83-85% TTB adherence observed in the data, while still allowing the Tallying strategy to explain the systematic deviations seen in extreme conflict trials.

**Parameters:**
  - `p_ttb`: `[0.6, 1.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) Strategy
    order = np.argsort(validities, kind='stable')[::-1]
    p_ttb_choice = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying Strategy (Equal Weights)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_choice = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    # Mixture of the two strategies
    p_core = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
