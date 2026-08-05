# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** Conflict-Driven Strategy Selection: Decision-makers adaptively select their decision strategy based on the dissimilarity of the options' total cue counts. When options are highly dissimilar in the number of positive cues (indicating high conflict or complexity), individuals abandon compensatory processing and fall back entirely on a simple non-compensatory heuristic (Take The Best). Conversely, when the total cue counts are similar, individuals attempt to integrate all available information using a compensatory strategy (Tallying). This is modeled as a probabilistic mixture of TTB and Tallying, where the probability of using TTB scales directly with the absolute difference in total cue counts.

**Rationale:** Following the critic's advice, we revert to the exact Iteration 2 mechanism but adjust the parameter ranges to find a better global compromise. We constrain `beta_ttb` and `beta_tally` to `[0.1, 10.0]` to ensure both strategies maintain reasonable stochasticity and do not collapse into pure determinism. We also increase the upper bound of `epsilon` to `0.5` to allow for a higher baseline lapse rate. This gentle retuning preserves the stable core formulation of Iteration 2 while preventing the destructive extremes observed in Iteration 6.

**Parameters:**
  - `beta_ttb`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB Probabilities
    z_ttb = beta_ttb * scores_ttb
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta_tally * scores_tally
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally_dist = e_tally / e_tally.sum()
    
    # Conflict-Driven Weight
    gamma = float(parameters["gamma"])
    
    # Dissimilarity in total cue counts
    diff_cues = abs(np.sum(a) - np.sum(b))
    
    # Probability of using TTB increases linearly with diff_cues
    w_ttb = min(1.0, gamma * diff_cues / max(1.0, float(n_features)))
    
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb_dist + (1.0 - w_ttb) * p_tally_dist
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_7` — KILLED ✗

**Description:** Sequential Evidence Accumulation with Flexible Threshold: Decision-makers search through cues sequentially in order of their validity. As they evaluate each cue, they accumulate evidence for the favored option. The search stops as soon as the absolute difference in accumulated evidence between the two options reaches a subjective threshold. If the threshold is low (or zero), this mechanism perfectly mimics Take-The-Best by stopping at the first discriminating cue. If the threshold is high, it evaluates all available cues, naturally transitioning into compensatory strategies like Weighted Additive (WADD) or Tallying.

**Rationale:** Following the critic's advice on Iteration 5, the model was still slightly too deterministic, yielding sharp step-function probabilities that overpredicted Take-The-Best agreement in Experiment 3 and overestimated the metric in Experiment 8. To soften this deterministic edge without altering the core Sequential Evidence Accumulation stopping behavior, we reduce the upper bound of the softmax inverse temperature `beta` from 20.0 to 5.0. This prevents extreme scaling of evidence differences, naturally pulling extreme predictions closer to observed human baselines.

**Parameters:**
  - `gamma`: `[0.0, 3.0]`
  - `theta`: `[0.0, 1.0]`
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
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Search through cues in order of validity (highest first)
    cue_order = np.argsort(-val, kind="stable")
    
    # Scale validities non-linearly to represent subjective evidence weights
    w = val ** gamma
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for i in cue_order:
        if a[i] > b[i]:
            ev_a += w[i]
        elif b[i] > a[i]:
            ev_b += w[i]
            
        # Stop search if the evidence difference reaches the threshold
        # (and ensure we don't stop prematurely if no discriminating evidence has been found yet)
        if abs(ev_a - ev_b) >= theta and abs(ev_a - ev_b) > 0:
            break
            
    # Convert accumulated evidence into choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

**Description:** Recency-Biased Evidence Accumulation: Decision-makers process cue arrays sequentially (e.g., from left to right) and exhibit a recency or primacy effect, largely overriding instructed objective validities. The subjective weight of a cue grows or shrinks exponentially with its position in the sequence. By framing this exponential growth/decay factor on a log scale, the population symmetrically exhibits both strong primacy and extreme recency, allowing the model to capture diverse behavioral patterns across different experimental contexts.

**Rationale:** Following the previous rejection, we return to the Iteration 3 base but shift the `log_lambda` distribution to slightly favor primacy `[-3.5, 2.5]`. This dedicates more probability mass to simulating the Take-The-Best-like behavior that dominates Experiments 1-12, while still retaining enough positive mass to capture the recency effects in Experiments 13 and 14. We retain `epsilon` at `[0.0, 0.5]` to maintain necessary baseline noise and set `beta` to `[0.1, 20.0]`.

**Parameters:**
  - `log_lambda`: `[-3.5, 2.5]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # Use log_lambda to symmetrically balance primacy and recency
    log_lam = float(parameters["log_lambda"])
    lam = np.exp(log_lam)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate position-biased weights based on sequential position
    # Position 0 is the first (leftmost) cue, n_features-1 is the last (rightmost)
    weights = np.array([lam ** j for j in range(n_features)])
    
    # Normalize weights to prevent overflow and keep scale consistent for beta
    weights = weights / np.sum(weights)
    
    # Accumulate evidence for each option
    ev_a = np.sum(a * weights)
    ev_b = np.sum(b * weights)
    
    scores = np.array([ev_a, ev_b])
    
    # Convert evidence to choice probabilities via softmax
    z = beta * scores
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    # Incorporate decision noise (epsilon)
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
