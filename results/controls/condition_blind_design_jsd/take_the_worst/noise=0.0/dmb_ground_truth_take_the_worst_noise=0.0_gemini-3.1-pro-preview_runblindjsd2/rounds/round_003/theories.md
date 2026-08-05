# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a Weighted Additive (WADD) strategy to compare options. They compute a subjective value for each option by summing its feature values weighted by their explicit validities. Decisions are made probabilistically using a softmax function over these values, with occasional random lapses.

**Rationale:** Implements the Weighted Additive (WADD) model as prescribed by the arbiter. By weighting each feature by its given validity, WADD integrates all available information in a compensatory manner. This contrasts with Take The Best, which only uses the single most valid discriminating cue, and Tallying, which ignores validity magnitudes.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** People use a Thresholded Tallying heuristic to compare options. To simplify decision making, individuals ignore cues whose validity falls below a certain subjective threshold. For the remaining cues, they disregard the exact validities and simply tally the number of positive features for each option. Decisions are then made probabilistically based on the difference in tallies using a softmax function, with occasional random lapses. The choice rule operates in a softer, probabilistic regime to account for human behavioral variance.

**Rationale:** Following the most recent feedback, we revert to the Thresholded Tallying model that serves as the running-best base (Iteration 1). To increase the simulated variance (dispersion) and better match the human JSD values across experiments, we restrict the range of the softmax inverse-temperature `beta` from [0.1, 20.0] to [0.0, 5.0]. This prevents the choice rule from becoming overly deterministic, forcing a more probabilistic response regime while maintaining the core thresholded tallying mechanism.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Select features with validity at or above the subjective threshold
    mask = val >= threshold
    
    # Tally positive features for each option among selected cues
    score_a = np.sum(a[mask])
    score_b = np.sum(b[mask])
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Decisions are driven by a Weighted Additive (WADD) evaluation of the options, combined with a sequential 'choice inertia' mechanism. Individuals compute the subjective value of each option by weighting its features by their respective validities. However, when translating these values into choices, people exhibit a tendency to repeat their physical response from the immediately preceding trial. This inertia acts as a baseline shift in the evidence required to choose the previously selected option, capturing sequential dependencies in decision-making.

**Rationale:** The arbiter pointed out the need to capture sequential dependencies in choices, specifically choice inertia. The previous models evaluated trials independently. By incorporating a 'rho' parameter that adds a logit bonus to the option chosen on the previous trial, this model directly captures the tendency to stick with the same response side. This is combined with a Weighted Additive core strategy to evaluate the options' merits, balancing stimulus-driven value with sequential response biases.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `rho`: `[-2.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive value computation
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    beta = float(parameters["beta"])
    rho = float(parameters["rho"])
    epsilon = float(parameters["epsilon"])
    
    # Choice inertia: boost the logit of the previously chosen option
    bias_a = 0.0
    bias_b = 0.0
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = history["response"][-1]
        if last_resp == 0:
            bias_a = rho
        elif last_resp == 1:
            bias_b = rho
            
    logits = np.array([beta * score_a + bias_a, beta * score_b + bias_b])
    
    # Numerically stable softmax
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    p_core = exp_logits / np.sum(exp_logits)
    
    # Trembling hand lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
