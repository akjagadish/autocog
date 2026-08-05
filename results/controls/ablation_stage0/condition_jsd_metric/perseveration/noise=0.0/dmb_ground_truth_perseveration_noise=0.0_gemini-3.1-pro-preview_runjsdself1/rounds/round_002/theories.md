# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Population-level Strategy Selection Mixture Model: The population consists of distinct subgroups of decision-makers. Some individuals consistently use a non-compensatory heuristic (Take The Best) while others consistently use a compensatory strategy (Weighted Additive). This discrete between-subject variation in strategy selection accounts for the high individual divergence from the population average, and low lapse rates ensure that the distinct strategy signatures are not washed out by noise.

**Rationale:** Following the critic's feedback, we build upon the accepted Iteration 2 base (the population-level strict mixture model). To increase the JSD and make subjects more distinctly divergent from the population average, we restrict the lapse rate (epsilon) to a much lower range [0.0, 0.1] and increase the maximum inverse temperature (beta) up to 30.0. High lapse rates wash out individual differences, causing choices to regress toward the population mean. By reducing epsilon and allowing sharper, more deterministic choices, subjects utilizing TTB and WADD will behave starkly differently, driving the between-subject variance (JSD) up toward the empirical targets.

**Parameters:**
  - `beta_wadd`: `[0.1, 30.0]`
  - `beta_ttb`: `[0.1, 30.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected state to be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- WADD Strategy ---
    score_a_wadd = np.sum(val * a)
    score_b_wadd = np.sum(val * b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- TTB Strategy ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Population-level Mixture & Lapse ---
    w_raw = float(parameters["w_wadd"])
    w_wadd = 1.0 if w_raw > 0.5 else 0.0
    p_mix = w_wadd * p_wadd + (1.0 - w_wadd) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** People evaluate options using a Weighted Additive (WADD) strategy. Decision-makers compute an overall utility for each option by summing its feature values weighted by their respective cue validities. This compensatory process allows multiple weaker cues to jointly override a single stronger cue if their combined weight is greater. Choice probabilities are then generated via a softmax function over these weighted sums, without an additional lapse rate, allowing for strong deterministic commitments when beta is high.

**Rationale:** Following the critic's advice on the rejected iteration, we remove the epsilon lapse rate entirely to simplify the model and prevent the WADD signal from washing out. We adjust the beta range to [0.0, 20.0] to allow the model to make more extreme, deterministic predictions. This should help capture the higher individual-level variance (JSD metric) observed in the real data without pulling probabilities too close to 0.5.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
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
    
    # Weighted additive sums for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return p_core
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Decision-makers use an 'Equal Weight' or Tallying heuristic to choose between options. Instead of weighting features by their validities (which is cognitively demanding) or relying on a single best cue, individuals simply count the total number of positive features for each option. The option with the higher total count of positive features is preferred. Choice probabilities are derived using a softmax function over these unweighted counts, reflecting bounded rationality by ignoring complex weights to reduce cognitive load.

**Rationale:** Based on the arbiter's feedback, this theory implements the Tallying (Equal Weight) heuristic. It differs from WADD by completely ignoring the expert validities, and it differs from Take-The-Best by integrating information across all cues equally. The decision variable is simply the unweighted sum of positive features for each option. A softmax function transforms these scores into choice probabilities, modulated by the inverse temperature parameter 'beta'. This provides a boundedly rational model that significantly reduces cognitive load while still considering all available features.

**Parameters:**
  - `beta`: `[0.0, 20.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Count the total number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    
    # Numerically stable softmax over the unweighted counts
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return p_core
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
