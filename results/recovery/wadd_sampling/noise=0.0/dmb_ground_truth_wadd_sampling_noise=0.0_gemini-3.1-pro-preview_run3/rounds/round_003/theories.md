# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory strategy. They multiply each feature's value by its corresponding cue validity and sum these products to form an overall subjective value for each option. The option with the higher weighted sum is chosen. This allows multiple lower-validity cues to collectively outweigh a single high-validity cue, capturing behavior that falls between pure Take The Best and pure Tallying. To account for empirical response noise, the decision process incorporates a moderate degree of stochasticity.

**Rationale:** Following the critic's feedback, the WADD mechanism is preserved entirely, but the parameter ranges for decision noise have been adjusted. The empirical metrics lie closer to 0.5 than the previous model's predictions. By restricting `beta` to [0.1, 5.0] and raising the lower bound of `epsilon` to 0.1, the model naturally injects more stochasticity, pulling its predictions closer to the observed human data without altering the core compensatory decision rule.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.1, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

**Description:** Strategy Mixture Theory: Decision-makers are heterogeneous, employing different strategies on different trials or across different individuals. The population's behavior is best modeled as a probabilistic mixture of three distinct decision rules: Weighted Additive (WADD) for compensatory evaluation, Tallying for unweighted cue counting, and Take The Best (TTB) for fast lexicographic choice. Each strategy computes its own preference, and the final choice probabilities are a weighted average of these underlying strategies' predictions, plus a uniform lapse rate. This mixture naturally accounts for the intermediate levels of consistency observed with any single heuristic.

**Rationale:** Following the critic's feedback, the model is shifted from a pure Log-Odds Bayesian formulation (which was too WADD-heavy) to a Strategy Mixture Theory. By computing independent choice probabilities for WADD, Tallying, and TTB, and then taking a weighted average based on mixture parameters (w_wadd, w_tally, w_ttb), the model can flexibly capture the intermediate use of compensatory and non-compensatory heuristics across the population. This directly resolves the overprediction of WADD-consistency and underprediction of Tallying-consistency observed in the previous iteration.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_wadd`: `[0.0, 1.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # 1. WADD Strategy
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # 2. Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    # 3. TTB Strategy
    cue_order = np.argsort(-val, kind='stable')
    scores_ttb = np.array([0.0, 0.0])
    for cue in cue_order:
        if a[cue] > b[cue]:
            scores_ttb[0] = 1.0
            break
        elif b[cue] > a[cue]:
            scores_ttb[1] = 1.0
            break
    if np.sum(scores_ttb) == 0:
        scores_ttb = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    
    def get_probs(scores):
        z = beta * scores
        z_shifted = z - np.max(z)
        e = np.exp(z_shifted)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd)
    p_tally = get_probs(scores_tally)
    p_ttb = get_probs(scores_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    
    total_w = w1 + w2 + w3
    if total_w == 0:
        w1, w2, w3 = 1.0/3.0, 1.0/3.0, 1.0/3.0
        total_w = 1.0
        
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / total_w
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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

**Description:** Contingent Heuristic Theory (Difference-based Compensatory): Decision-makers conditionally apply decision strategies based on the relative strength of the best available evidence compared to the next best piece of evidence. They identify the most valid cue that discriminates between the two options and the second most valid discriminating cue. If the difference in validity between these two cues exceeds a subjective threshold (indicating the top cue is highly salient relative to the rest), they rely entirely on the top cue (Take-The-Best). If the difference is not sufficiently large, they fall back to a fully compensatory Weighted Additive (WADD) strategy.

**Rationale:** Following the critic's advice, the gating condition for relying on the Take-The-Best (TTB) heuristic was changed from an absolute validity threshold to a relative validity difference (`delta`). The previous model over-predicted TTB choices because absolute validities are often high across the board. By evaluating `v_max - v_next >= delta`, TTB is only triggered when the best discriminating cue is saliently better than the next best available evidence. This minimal edit directly addresses the critic's diagnosis while keeping the model within the prescribed Contingent Heuristic family.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `delta`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    cue_order = np.argsort(-val, kind='stable')
    
    ttb_winner = None
    v_max = 0.0
    v_next = 0.0
    found_first = False
    
    for cue in cue_order:
        if a[cue] != b[cue]:
            if not found_first:
                v_max = val[cue]
                ttb_winner = 0 if a[cue] > b[cue] else 1
                found_first = True
            else:
                v_next = val[cue]
                break
                
    delta = float(parameters["delta"])
    
    if ttb_winner is not None and (v_max - v_next) >= delta:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
    else:
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        scores = np.array([score_a, score_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
