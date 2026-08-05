# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics: Take-The-Best (TTB) and Weighted Additive (WADD). While TTB captures the strong reliance on the highest-validity cues by making choices based solely on the best discriminating feature, WADD integrates both cue validities and cardinal feature magnitudes across all cues. Rather than stochastically switching between these strategies, decision-makers evaluate options by integrating the normalized evidence (scores) from both heuristics into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus WADD evidence, and response noise enters through a single softmax over the mixed scores.

**Rationale:** Based on the arbiter's feedback, this theory replaces the Tallying + WADD mixture (Theory 1) with a mixture of Take-The-Best (TTB) and Weighted Additive (WADD) strategies. By integrating the normalized evidence from TTB (which explains heavy reliance on the highest-validity cues, critical for Exps 2 and 8) and WADD (which retains sensitivity to cardinal feature magnitudes, critical for Exp 4) before applying a softmax decision rule, the model captures a broader range of human behaviors across experiments. This structure directly mirrors the successful formulation of Theory 2 but uses WADD instead of Weighted Tallying to better account for magnitude sensitivity.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `validities`: `[(0.0, 1.0)] * n_features`
  - `rating_max`: `rating_max`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD) Heuristic
    # Integrates magnitudes and validities across all cues
    wadd_scores = stim @ w
    
    # Normalize WADD scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    rmax = float(parameters["rating_max"])
    if w_sum > 0 and rmax > 0:
        wadd_scores = wadd_scores / (w_sum * rmax)
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wadd_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics that both utilize cue validities but process feature magnitudes differently. Rather than using full compensatory integration (WADD) or unweighted counting (Tallying), individuals draw from a mixture of 'Take-The-Best' (TTB) and 'Weighted Tallying'. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Weighted Tallying integrates information across all cues by binarizing feature differences into strict wins/losses and weighting these binary outcomes by their respective validities. Instead of probabilistically switching between these heuristics, decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Weighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.

**Rationale:** Following the critic's advice, I changed the mixture mechanism from mixing probability distributions to mixing the underlying scores (evidence) of the TTB and Weighted Tallying heuristics. By combining the normalized [0, 1] scores first via `alpha` and then applying a single softmax with `beta`, the model forms a single consistent preference per trial. This avoids the inherent stochasticity of mixing probabilities when heuristics disagree, allowing the model to achieve the near-deterministic behavior needed for Experiment 4 while preserving the probabilistic blending required for the other experiments.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `validities`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Tallying Heuristic
    # Binarize feature differences into wins/losses, then weight by validity
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    wt_scores = np.array([np.sum(a_wins * w), np.sum(b_wins * w)])
    
    # Normalize weighted tally scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    if w_sum > 0:
        wt_scores = wt_scores / w_sum
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wt_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Human decision-making in multi-attribute choice relies on a mixture of two purely ordinal boundedly rational heuristics: Take-The-Best (TTB) and Unweighted Tallying. Neither heuristic uses cardinal feature magnitudes, explaining the lack of sensitivity to magnitude differences observed in several experiments. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Unweighted Tallying, on the other hand, ignores both magnitudes and cue validities, simply counting the number of features where one option strictly beats the other. Decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Unweighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.

**Rationale:** Following the arbiter's instructions, this theory replaces the WADD component of Theory 1 (pi_5) with Unweighted Tallying. By doing so, the theory becomes entirely ordinal, completely ignoring cardinal magnitudes. This directly addresses the 0.0000 variance observed in Experiments 4, 9, and 10 which indicate that subjects do not systematically alter their choices based on the size of the rating differences. The model mixes the lexicographic, validity-sensitive TTB scores with democratic, validity-ignoring Unweighted Tallying scores via the 'alpha' parameter, then applies a single softmax. This allows the model to capture choices where subjects either follow the most valid cue or simply count the number of winning features, without ever being influenced by large cardinal differences.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `validities`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Unweighted Tallying Heuristic
    # Count strict feature-wise wins, ignoring validities and magnitudes
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    n_features = len(a)
    
    # Normalize tally scores so they occupy a similar [0, 1] scale as TTB
    tally_scores = np.array([a_wins, b_wins], dtype=float) / n_features
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * tally_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
