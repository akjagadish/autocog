# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Random Guessing: Due to task complexity, lack of trial-by-trial correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial.

**Rationale:** Following the arbiter's feedback, the theory posits that subjects are guessing randomly. This is strongly supported by the experimental results: the real metric values across all four experiments are consistently around 0.5 (0.4844, 0.4800, 0.4954, 0.5046). Previous models (like TTB, WADD, and Tallying) predicted much more extreme values and failed to capture this central tendency. A pure random guessing model predicts exactly 0.5 for all metrics, perfectly aligning with the observed data.

**Parameters:**
  (none)

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Random Feature Heuristic (Minimalist): Subjects do not integrate multiple features or weight them by validity. Instead, on each trial, they randomly sample exactly one feature uniformly at random. They choose the option with the higher value on that sampled feature. If the options tie on that feature, they guess randomly. This acts as a highly noisy but boundedly rational alternative to pure guessing, predicting choices that slightly favor the option with more positive features overall. A lapse rate (epsilon) accounts for additional purely random choices.

**Rationale:** Following the arbiter's suggestion, this implements the Random Feature Heuristic. It assumes subjects are minimally engaged, selecting just one feature at random per trial and deciding based on it, ignoring validities entirely. This provides a mechanism that is very close to random guessing (since features often tie or conflict) but systematically deviates slightly in favor of options with more positive features, serving as a boundedly rational alternative to pure noise.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # The marginal probability of choosing A when picking one feature at random:
    # P(A) = 1/n * sum(a_i > b_i) + 0.5/n * sum(a_i == b_i)
    # For binary features, this simplifies exactly to 0.5 + 0.5 * (sum(a) - sum(b)) / n
    diff = np.sum(a) - np.sum(b)
    p_a_core = 0.5 + 0.5 * diff / n_features
    p_b_core = 1.0 - p_a_core
    
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b_core + epsilon * 0.5
    
    return np.array([p_a, p_b])
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Spatial Bias / Option Preference: Subjects completely ignore the complex feature information provided on each trial. Instead, their choices are driven by an idiosyncratic, subject-level baseline preference for Option A versus Option B (e.g., a left/right spatial bias or an order effect). Each subject has a fixed probability of choosing Option A on any given trial, which varies between subjects but remains constant across trials for a given subject. This explains why choices are invariant to feature differences while accounting for individual differences in baseline choice rates.

**Rationale:** Following the arbiter's feedback, this theory implements a 'Spatial Bias / Option Preference' model. It discards all feature-based evaluation, predicting that subjects completely ignore the ratings and validities. Instead, each subject has an idiosyncratic, baseline probability of choosing Option A (`bias_A`). The parameter range [0.3, 0.7] is chosen because it produces between-subject variance in choice rates that closely matches the empirical variance (around 0.016), which is higher than pure random guessing but lower than a uniform [0, 1] distribution. This theory captures the experiment-invariant finding that choices are insensitive to feature differences while accounting for the observed individual differences in baseline option preference.

**Parameters:**
  - `bias_A`: `[0.3, 0.7]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    p_a = float(parameters["bias_A"])
    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
