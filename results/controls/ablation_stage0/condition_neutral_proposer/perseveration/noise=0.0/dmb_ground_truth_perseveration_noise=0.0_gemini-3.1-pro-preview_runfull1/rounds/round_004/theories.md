# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Rank-Weighted Exponential Integration: Decision-makers evaluate all cues simultaneously but weight them exponentially according to their validity rank (weight = alpha^(-rank)). This creates a non-compensatory profile that mimics Take-The-Best when cues are aligned, but naturally allows for compensatory overrides when multiple lower-ranked cues strongly align against the top cue, capturing regressions to chance in highly conflicting trial designs.

**Rationale:** Following the latest feedback, we reverted to the exact Rank-Weighted Exponential Integration mechanism from the Iteration 2 base, as it successfully balanced TTB-like behavior and compensatory overrides. To avoid the over-determinism that hurt previous iterations, we capped the softmax inverse temperature `beta` at 5.0 and the lapse rate `epsilon` at 0.5. By restricting `beta`, we prevent the model from exaggerating tiny score differences into 1.0/0.0 choices, thereby preserving the chance-level (~0.5) regressions required in perfectly symmetric conflict trials (Exps 2, 4, 6, 8).

**Parameters:**
  - `alpha`: `[1.0, 2.5]`
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine the rank of each cue (0 = highest validity, 1 = second highest, etc.)
    cue_order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(cue_order)
    ranks[cue_order] = np.arange(len(val))
    
    alpha = float(parameters["alpha"])
    # Exponentially decay weights based on rank
    weights = alpha ** (-ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

**Rationale:** Following the critic's advice, the log-odds weights are now normalized by their sum. This minimal edit bounds the maximum possible accumulated evidence to 1.0 (or -1.0), ensuring consistent scaling across experiments with different validities. Consequently, the threshold parameter is restricted to [0.0, 1.0]. This makes the threshold parameter experiment-invariant, allowing the model to robustly transition between early-stopping (Take-The-Best) and full-integration (Weighted Additive) behavior regardless of the specific validities presented in a given experiment.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `threshold`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Heuristic Mixture: Take-The-Best and Tallying. Decision-makers do not rely on a single complex integration mechanism. Instead, they probabilistically employ one of two classic fast-and-frugal heuristics on any given trial (or across the population): strict Take-The-Best (TTB) and simple equal-weight Tallying. With probability p_ttb, the agent searches cues in validity order and stops at the first discriminating cue. With probability 1 - p_ttb, the agent simply counts the total number of positive cues for each option and chooses the one with the higher tally, guessing randomly on ties. This mixture naturally accounts for high TTB adherence when cues align, while allowing compensatory Tallying to pull choice probabilities toward 0.5 in highly conflicting trials without needing complex weighted evidence accumulation.

**Rationale:** Following the arbiter's feedback, this model instantiates a probabilistic mixture of two classic heuristics: Take-The-Best (TTB) and Tallying. It removes the complex exponential integration and dynamic strategy selection from previous iterations. Instead, a fixed parameter `p_ttb` dictates the probability of using TTB versus Tallying on any given trial. TTB strictly follows the validity order until a discriminating cue is found, whereas Tallying simply counts the positive cues for each option, completely ignoring validity weights. This captures the strong preference for the top cue in aligned trials while naturally regressing to chance in conflicting trials where Tallying results in ties or favors the opposite option.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Take-The-Best (TTB) heuristic
    ttb_prob = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_prob = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_prob = np.array([0.0, 1.0])
            break
            
    # Tallying heuristic (equal weights)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    if tally_a > tally_b:
        tally_prob = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        tally_prob = np.array([0.0, 1.0])
    else:
        tally_prob = np.array([0.5, 0.5])
        
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of heuristics
    p_core = p_ttb * ttb_prob + (1.0 - p_ttb) * tally_prob
    
    # Add lapse rate
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
