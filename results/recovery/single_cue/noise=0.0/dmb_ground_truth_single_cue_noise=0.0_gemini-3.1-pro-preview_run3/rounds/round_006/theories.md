# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

**Rationale:** Following the arbiter's feedback, this theory implements a 'Tally-then-TTB' mechanism. It evaluates options primarily by Tallying, which correctly captures human behavior on unequal-tally trials where subjects tend to prefer the option with more winning cues regardless of their validity. When Tallying results in a tie, rather than guessing randomly (which fails to capture deterministic behavior on tied trials), the model breaks the tie by defaulting to Take-The-Best, selecting the option that wins on the highest-validity cue. This directly addresses the mechanistic failures of pure Tallying and pure TTB by combining their strengths in a sequential heuristic framework.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_8` — KILLED ✗

**Description:** Generalized Weighted Additive Model (WADD). Decision-makers integrate all available cues simultaneously, but weight each cue according to an exponentiated function of its validity. This single continuous mechanism smoothly interpolates between three classic heuristics depending on the exponent theta: when theta = 0, it reduces to pure Tallying (all cues weighted equally); when theta = 1, it represents standard WADD; and when theta is large, it approximates Take-The-Best (lexicographic choice where the most valid cue dominates the sum of all others). This completely discards the 'Tallying-first' assumption and allows the model to flexibly capture both compensatory and non-compensatory decision-making across different subjects and experimental contexts without needing a two-stage or tie-breaking process.

**Rationale:** The previous theory (`pi_7`) relied on a two-stage 'Tallying-first' heuristic with a validity-based tie-breaker. The arbiter correctly pointed out that this mechanism fundamentally fails to capture scenarios where a single high-validity cue systematically overrides a larger tally of lower-validity cues on non-tied trials. To address this, we propose a Generalized Weighted Additive (WADD) model that completely discards the Tallying-first assumption. Instead, decision-makers integrate all cues simultaneously, weighting each cue by an exponentiated function of its validity (`validities ** theta`). This continuous parameterization elegantly subsumes three classic strategies: when `theta = 0`, it reduces to pure Tallying (equal weights); when `theta = 1`, it represents standard WADD; and for large `theta`, the highest-validity cue dominates, mimicking Take-The-Best (lexicographic choice). This allows the model to flexibly adapt to varying degrees of compensatory and non-compensatory behavior across subjects and experiments without relying on ad-hoc tie-breakers.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[0.0, 30.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate weights by exponentiating validities
    weights = val ** theta
    
    # Option scores based on weighted sum of winning cues
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(weights * a_wins)
    score_b = np.sum(weights * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_9` → slot 2 (via `new_theory`)

**Description:** Rank-Based Threshold Tallying: Decision-makers do not integrate all available cues indiscriminately, nor do they rely on a single cue. Instead, they form a consideration set of the top-K most valid cues and apply a compensatory tallying heuristic strictly within this subset. If this primary tally yields a clear preference, they choose that option. If the high-validity cues result in a tie, decision-makers fall back to a secondary tally of the remaining low-validity cues to break the tie probabilistically. This captures the robust tallying behavior in most tasks while restricting integration to 'important' cues, explaining deviations in environments where unweighted tallying over all cues breaks down.

**Rationale:** Following the critic's feedback, we revert to the iteration 1 `top_k_prop` base, which successfully passed the accept gate. The only change is tightening the `fallback_weight` range from `[0.0, 0.99]` to `[0.0, 0.1]`. This ensures the secondary tally strictly acts as a tie-breaker without compressing primary score differences, which previously diluted the softmax probabilities and degraded fit on experiments where primary tallying is dominant.

**Parameters:**
  - `top_k_prop`: `[0.1, 1.0]`
  - `fallback_weight`: `[0.0, 0.1]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    top_k_prop = float(parameters["top_k_prop"])
    k = max(1, int(np.round(top_k_prop * n_features)))
    
    # Rank validities: lowest = 0, highest = n_features - 1
    ranks = np.argsort(np.argsort(val))
    primary_mask = ranks >= (n_features - k)
    secondary_mask = ~primary_mask
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    a_wins_primary = np.sum(a_wins[primary_mask])
    b_wins_primary = np.sum(b_wins[primary_mask])
    
    a_wins_secondary = np.sum(a_wins[secondary_mask])
    b_wins_secondary = np.sum(b_wins[secondary_mask])
    
    fallback = float(parameters["fallback_weight"])
    
    # Normalize secondary wins by number of secondary features 
    # so its maximum contribution is strictly < 1.0, ensuring it acts only as a tie-breaker.
    num_secondary = np.sum(secondary_mask)
    if num_secondary > 0:
        sec_a = a_wins_secondary / num_secondary
        sec_b = b_wins_secondary / num_secondary
    else:
        sec_a = 0.0
        sec_b = 0.0
        
    score_a = a_wins_primary + fallback * sec_a
    score_b = b_wins_primary + fallback * sec_b
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
