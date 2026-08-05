# Round 4 — Theories

**Verdict:** `new_model` (slot 2 replaced)

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


### slot 2 — `pi_6` — KILLED ✗

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


## Replacement

### `pi_6_1` → slot 2 (via `new_model`)

**Description:** Spatial Bias / Option Preference: Subjects completely ignore the complex feature information provided on each trial. Instead, their choices are driven by an idiosyncratic, subject-level baseline preference for Option A versus Option B (e.g., a left/right spatial bias or an order effect). Each subject has a fixed probability of choosing Option A on any given trial, which varies between subjects but remains constant across trials for a given subject. This explains why choices are invariant to feature differences while accounting for individual differences in baseline choice rates.

**Rationale:** Following the arbiter's recommendation, we restrict the range of `bias_A` from a wider distribution to a tightly bounded Uniform(0.4, 0.6). The variance of a Uniform(0.4, 0.6) distribution is approximately 0.0033, which closely matches the slight overdispersion in subject choice proportions observed in Experiments 9 and 10 (0.0018 to 0.0047). This allows the model to capture individual differences in baseline choice rates without overpredicting the between-subject variance.

**Parameters:**
  - `bias_A`: `[0.4, 0.6]`

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
