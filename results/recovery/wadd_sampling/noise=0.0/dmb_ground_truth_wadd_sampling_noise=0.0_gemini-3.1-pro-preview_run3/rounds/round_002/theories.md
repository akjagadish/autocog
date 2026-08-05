# Round 2 — Theories

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Take The Best (TTB) Theory: Decision-makers evaluate options using a fast-and-frugal, non-compensatory lexicographic heuristic. Instead of integrating all available information (like WADD) or treating all cues equally (like Tallying), they search through features in descending order of cue validity. The search stops at the first feature that discriminates between the two options, and the option favored by that cue is chosen. Any lower-validity cues are completely ignored, and the magnitude of the difference on the discriminating cue does not matter. If no features discriminate, the decision-maker guesses uniformly. Choice stochasticity is modeled via a softmax over the binary TTB preference and an independent lapse rate.

**Rationale:** Following the arbiter's instructions, this proposes the Take The Best (TTB) theory. TTB represents a lexicographic, non-compensatory decision process that directly contrasts with both WADD (which integrates all features) and Tallying (which ignores validities). By evaluating features strictly in order of their validity and stopping at the first discriminating cue, TTB captures the fast-and-frugal nature of human heuristic decision-making, where individuals often rely on a single good reason rather than exhaustive computation.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind='stable')
    
    winner = -1
    for cue in cue_order:
        if a[cue] > b[cue]:
            winner = 0
            break
        elif b[cue] > a[cue]:
            winner = 1
            break
            
    if winner == -1:
        # No cues discriminate; pure guess
        return np.array([0.5, 0.5])
        
    scores = np.zeros(2)
    scores[winner] = 1.0
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * scores
    z_shifted = z - np.max(z)
    probs = np.exp(z_shifted) / np.sum(np.exp(z_shifted))
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
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

### `pi_5` → slot 2 (via `new_theory`)

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
