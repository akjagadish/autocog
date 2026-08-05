# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

**Rationale:** Following the arbiter's guidance, this theory posits that subjects rely on a fragile, single-cue heuristic (looking only at the highest-validity cue) rather than tallying or weighted additive strategies. To capture the fact that empirical metrics stay tightly bounded near 0.5 (or 0 difference), the model assumes an overwhelmingly high lapse rate (epsilon between 0.9 and 1.0). This generates predictions that are mostly random guessing, with slight deviations in the direction of the single best cue, explaining the subtle effects observed in the data without over-predicting the influence of secondary cues.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.9, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Equal Weights with High Lapse: Subjects often find the integration of complex cue validities too cognitively demanding or disengaging in these conflict paradigms. As a result, they ignore the provided validities entirely and fall back on a simple 'Tallying' (Equal Weights) heuristic, where they just count the number of positive features for each option. Furthermore, due to the high cognitive friction or confusion, subjects exhibit a very high baseline guessing rate (lapse), meaning that on the vast majority of trials they simply guess randomly. This explains why behavior across various conflict and agreement metrics hovers so closely to 0.5 or 0 difference.

**Rationale:** Across all four experiments, the observed metrics heavily center around 0.5 for choice proportions and 0 for differences between conditions. This indicates that subjects are not consistently applying WADD or TTB, but rather exhibiting near-random behavior. Following the arbiter's suggestion, this model implements an Equal Weights (Tallying) strategy with an exceptionally high lapse rate (epsilon bounded between 0.8 and 1.0). By abandoning complex validity integration and injecting massive stochasticity, the model naturally predicts the ~0.50 matching rates in Exp 1, 2, and 3, and the ~0 difference in Exp 4, overcoming the deterministic biases of previous WADD and TTB models.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: equal weights for all features (counting positive cues)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallied scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # High uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Weak Weighted Additive with Extreme Noise: In complex multi-cue decision environments, subjects attempt to integrate all available information by weighting each cue according to its provided validity (a compensatory Weighted Additive strategy). However, the cognitive demands of multiplying and summing multiple cues lead to severe cognitive overload and disengagement. Consequently, subjects exhibit an overwhelmingly high lapse rate, effectively guessing on the vast majority of trials. The underlying Weighted Additive evaluation only weakly shines through the noise, explaining why choice behavior hovers very close to random guessing (0.5) across conflict metrics while maintaining a slight systematic preference for higher-validity cue combinations.

**Rationale:** Per the arbiter's suggestion, this theory replaces the equal-weights (Tallying) assumption with a Weighted Additive (WADD) mechanism where subjects use the provided validities. However, to capture the observed empirical data where conflict metrics center tightly around 0.5, we maintain an extremely high lapse (guessing) rate. This reflects cognitive overload: subjects try to use the validities but mostly end up guessing, producing a very faint WADD signal. This specifically helps capture the very slight systematic deviations in experiments like Experiment 8, where subjects favor higher validity combinations slightly more than simple counting would predict.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.9, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted Additive (WADD): sum of cues weighted by their validities
    scores = stim @ validities
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
