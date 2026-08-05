# Round 12 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_13` — SURVIVED ✓

**Description:** Inverse Validity Tie-Breaking with Non-linear Scaling: Decision-makers primarily rely on a Tallying heuristic. When this primary mechanism results in a tie, subjects do not guess randomly, but exhibit a recency bias or systematically misinterpret the cue validities, breaking the tie by heavily weighting the lower-validity (or right-most) features. A non-linear scaling parameter exaggerates this inverse-validity preference to better capture the magnitude of the recency effect.

**Rationale:** Following the critic's advice, we introduced a non-linear scaling parameter (`gamma`) to the inverse validity weights. By raising `(1.0 - validities)` to the power of `gamma`, the model can dynamically sharpen the tie-breaking weights, exaggerating the recency bias and strengthening the negative correlation observed in Experiment 22 without disrupting the strict Tallying baseline.

**Parameters:**
  - `validities`: `validities`
  - `w_tie`: `[0.0, 0.95]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 10.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tie = float(parameters["w_tie"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Primary mechanism: Tallying (count of strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Inverse Validity Tie-Breaker
    # Weight lower-validity features more heavily, with a non-linear scaling (gamma)
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # Combine scores. Since w_tie < 1.0 and tie_score difference is <= 1.0,
    # the tie-breaker will never override a strict Tallying win (difference >= 1.0).
    score_a = a_wins + w_tie * tie_score_a
    score_b = b_wins + w_tie * tie_score_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_14` — KILLED ✗

**Description:** Decision-making in binary choice tasks is driven by a probabilistic mixture of two primary heuristics: Tallying and Take-The-Best (TTB). On any given trial, a subject may evaluate the options by counting the number of winning features (Tallying) or by relying solely on the single highest-validity discriminating cue (TTB). When Tallying results in a tie, its choice probabilities become uniform, allowing the TTB preference to naturally act as a tie-breaker without requiring a separate, contrived tie-breaking mechanism. To maintain the strong dominance of Tallying observed in non-tie scenarios, the prior probability of using TTB is constrained to be relatively low.

**Rationale:** Following the critic's feedback, the most recent attempt to mix evidence (logits) was rejected, so we revert to the probabilistic mixture of strategies (mixing probabilities) from iteration 1. To address the issue of TTB diluting Tallying in non-tie scenarios (like Experiments 2, 3, and 4), the parameter range for `p_ttb` is constrained strictly to [0.0, 0.3]. This ensures that Tallying remains the dominant strategy overall, allowing TTB to shine primarily as a reliable tie-breaker when Tallying produces a 50/50 tie, without overriding the strict Tallying preferences in other experiments.

**Parameters:**
  - `validities`: `validities`
  - `p_ttb`: `[0.0, 0.3]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB): rely on the highest validity discriminating cue
    # Use stable sort to preserve left-to-right order for equal validities
    order = np.argsort(-validities, kind='stable')
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying: count of strict feature-wise wins
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    # Compute probabilities for each heuristic
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    probs_ttb = np.exp(z_ttb - np.max(z_ttb))
    probs_ttb /= np.sum(probs_ttb)
    
    z_tally = beta_tally * np.array([tally_a, tally_b])
    probs_tally = np.exp(z_tally - np.max(z_tally))
    probs_tally /= np.sum(probs_tally)
    
    # Mix the strategies
    mixed_probs = p_ttb * probs_ttb + (1.0 - p_ttb) * probs_tally
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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

### `pi_15` → slot 2 (via `new_theory`)

**Description:** Tallying with Attention Decay: Decision-makers primarily rely on a Tallying heuristic, counting the number of strictly winning features for each option. However, due to cognitive load and working memory constraints, features processed earlier in the sequence decay from memory. When Tallying results in a tie or a very weak difference, subjects rely disproportionately on the most recently processed (lower-validity, right-most) features. The secondary attention-decay score can occasionally override a weak Tallying difference, but typically acts as a tie-breaker.

**Rationale:** Following the feedback, we revert to the accepted Iteration 1 base, maintaining the strict recency effect by restricting `retention_rate` to [0.01, 0.99]. To address the failure on Experiment 6, where early salient cues sometimes override the tally, we expand `w_tie` slightly to [0.0, 1.5]. This allows the attention decay mechanism to occasionally override a weak 1-point Tallying difference without completely destroying the primary Tallying mechanism on larger differences, preserving the strong fit on the tie-breaker experiments.

**Parameters:**
  - `retention_rate`: `[0.01, 0.99]`
  - `w_tie`: `[0.0, 1.5]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    retention_rate = float(parameters["retention_rate"])
    w_tie = float(parameters["w_tie"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Primary mechanism: Tallying (count of strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Attention Decay / Recency
    # Features are processed left-to-right. The last feature (index n_features - 1)
    # is the most recent and decays the least. Feature i decays for (n_features - 1 - i) steps.
    weights = np.array([retention_rate ** (n_features - 1 - i) for i in range(n_features)])
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        weights = np.ones(n_features) / n_features
        
    recency_a = np.sum(a * weights)
    recency_b = np.sum(b * weights)
    
    # Combine scores. w_tie is allowed to slightly exceed 1.0, enabling the recency
    # mechanism to occasionally override a weak 1-point Tallying difference.
    score_a = a_wins + w_tie * recency_a
    score_b = b_wins + w_tie * recency_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
