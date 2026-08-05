# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Mixture of Tallying and Weighted Tallying: Decision makers do not universally ignore explicit validities (as in pure Tallying) nor do they perfectly weight every feature by its numerical validity (as in pure Weighted Tallying). Instead, the population consists of a mixture of strategies, or individuals use a blended strategy. Some individuals rely on a simple count of winning features (Tallying), while others incorporate the provided validities to weight those wins. By modeling choice as a convex combination of the Tallying score and the Weighted Tallying score, the model can capture intermediate levels of consensus and account for why human behavior often falls between the predictions of these two pure heuristics.

**Rationale:** Following the arbiter's suggestion, this theory models a mixture between pure Tallying and Weighted Tallying. Tallying completely ignores validities, resulting in overly extreme predictions on experiments where subjects do use validity information (e.g., Exp 6). Conversely, Weighted Tallying relies exclusively on numerical validities, which can underestimate the human tendency to simply count winning features. By introducing a subject-specific blending parameter `w_tally`, the model effectively forms a population-level mixture that smoothly interpolates between these two heuristics. This hybrid score mechanism should yield intermediate consensus levels, bringing predictions much closer to the observed human behavior across the experimental suite.

**Parameters:**
  - `w_tally`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying scores: simple count of strictly winning features
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    
    # Weighted Tallying scores: sum of validities for winning features
    wt_a = np.sum(validities[a > b])
    wt_b = np.sum(validities[b > a])
    
    # Blend the two strategies
    w = float(parameters["w_tally"])
    score_a = w * tally_a + (1.0 - w) * wt_a
    score_b = w * tally_b + (1.0 - w) * wt_b
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the blended scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Weighted Additive (WADD) Model: Decision-makers evaluate options by multiplying the full cardinal rating of each feature by its explicitly provided expert validity, and then summing these products to form an overall expected value for each option. Unlike tallying heuristics that binarize feature differences into strict wins and losses, WADD integrates both the magnitude of the feature ratings and the explicit cue weights. This compensatory strategy allows a large advantage on a lower-validity feature to outweigh a small deficit on a higher-validity feature. Choices are made probabilistically via a softmax over the integrated values, combined with a uniform lapse rate.

**Rationale:** Following the arbiter's recommendation, this theory instantiates the Weighted Additive (WADD) model. A previous attempt at WADD (`pi_2`) failed because it treated validities as free parameters to be fitted per-subject (`[(0.0, 1.0)] * n_features`), completely ignoring the explicit expert validities provided in the experiment. By correctly anchoring the weights to the explicit validities (`validities: validities`), this model accurately captures how decision-makers integrate cardinal feature magnitudes with explicit cue weights, providing a robust compensatory benchmark that accounts for magnitude-driven trade-offs.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    # Use the explicitly provided expert validities from the experiment
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD score: sum of (cardinal rating * explicit validity)
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the integrated scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Stochastic Take-The-Best (Plackett-Luce Search): Decision makers employ a non-compensatory lexicographic strategy, but their search order through cues is probabilistic rather than strictly deterministic. Cues are sampled sequentially without replacement, with selection probabilities proportional to a softmax over their explicit validities. The search terminates at the first cue that discriminates between the options. By the properties of the Plackett-Luce model, the probability that a specific discriminating cue is the one that drives the choice is exactly its relative softmax weight among the subset of all discriminating cues, seamlessly interpolating between strict Take-The-Best (at low temperatures) and uniform Tallying (at high temperatures).

**Rationale:** Following the critic's advice, we implement a Stochastic Take-The-Best heuristic where the search order is sampled probabilistically. Instead of a deterministic sort by validity, features are selected sequentially proportional to a softmax over their validities (temperature `tau`). The decision is made by the *first* discriminating feature encountered. Mathematically, this is a Plackett-Luce model over the search order. A known property of Luce's Choice Axiom is that the probability of drawing item i before item j is independent of the presence of other non-discriminating items. Thus, the probability that a specific discriminating cue is the deciding one is exactly its softmax weight normalized *only* over the subset of discriminating cues. This elegantly and efficiently implements the requested mechanism without requiring noisy Monte Carlo simulations, interpolating smoothly between pure TTB (tau -> 0) and Tallying (tau -> inf).

**Parameters:**
  - `tau`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    discrim_a = a > b
    discrim_b = b > a
    discrim_any = discrim_a | discrim_b
    
    if not np.any(discrim_any):
        p_core = np.array([0.5, 0.5])
    else:
        # Under Plackett-Luce sampling without replacement, the probability that 
        # a specific discriminating feature is drawn *before* any other 
        # discriminating feature is exactly its relative weight among the 
        # discriminating features. Non-discriminating features skipped along the 
        # way do not affect this relative probability.
        v_discrim = validities[discrim_any]
        z = v_discrim / tau
        z = z - np.max(z)  # for numerical stability
        w = np.exp(z)
        w = w / np.sum(w)
        
        w_all = np.zeros_like(validities)
        w_all[discrim_any] = w
        
        score_a = np.sum(w_all[discrim_a])
        score_b = np.sum(w_all[discrim_b])
        
        p_core = np.array([score_a, score_b])
        
    n_opts = p_core.shape[0]
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
