# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Validity-Scaled Take-The-Best: Individuals use a lexicographic, non-compensatory heuristic to compare options sequentially. They rank features by their explicit validities and stop at the first feature that discriminates between the two options. Their confidence in the winning option scales with how much the validity of that discriminating feature exceeds chance (0.5), modeled via a softmax function, along with a baseline lapse rate.

**Rationale:** I applied the critic's suggested minimal edit: adjusted the confidence scaling logit from `z = beta * val[j]` to `z = beta * (val[j] - 0.5)`. Since validities are in the range [0.5, 1.0], this centers the confidence at 0 (meaning a 50/50 choice) when the discriminating feature's validity is exactly at chance (0.5), which provides a more theoretically sound mapping from validity to choice probability.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by descending validity; stable sort preserves original order for ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    for j in cue_order:
        if a[j] != b[j]:
            # First discriminating feature found
            # Confidence scales with the validity of this specific feature above chance
            z = beta * (val[j] - 0.5)
            p_win = 1.0 / (1.0 + np.exp(-z))  # Numerically stable for z >= 0
            
            p = np.zeros(2)
            if a[j] > b[j]:
                p[0] = p_win
                p[1] = 1.0 - p_win
            else:
                p[1] = p_win
                p[0] = 1.0 - p_win
                
            return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
            
    # If no features discriminate, guess uniformly
    return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
