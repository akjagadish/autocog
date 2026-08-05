# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

**Rationale:** Following the critic's feedback, the upper bound of the `beta` parameter range is further reduced from 5.0 to 2.5. This tighter range forces the model to produce softer choice probabilities on the discriminating features, bringing the overpredicted determinism in Experiment 1 and the overpredicted preference for the single most valid feature in Experiment 2 closer to the empirical levels.

**Parameters:**
  - `beta`: `[0.0, 2.5]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Strategy Mixture Model (TTB and Tallying): Decision-makers exhibit heterogeneous strategy use, with the population consisting of a mix of Take-The-Best (TTB) users and Tallying users. Rather than a pure one-reason heuristic or a fully compensatory process, aggregate behavior reflects a probabilistic mixture. On any given trial, a subject's choice is a weighted blend of a lexicographic TTB process (which chooses based on the single most valid discriminating cue) and a Tallying process (which counts the number of feature-wise wins for each option). Allowing the mixture weight to vary freely between 0 and 1 across individuals captures the empirical finding that aggregate behavior is predominantly non-compensatory but softened by a subset of subjects who rely more heavily on compensatory tallying.

**Rationale:** Following the critic's feedback, the only change made is expanding the parameter range of `w_tally` from [0.0, 0.5] to [0.0, 1.0]. Because parameters are fitted per subject, this wider bound allows the model to properly represent a population with a mixture of strategies across individuals, including subjects who predominantly or exclusively use the Tallying strategy. This addresses the mechanistic limitation that artificially prevented the model from capturing pure Tallying behavior, which should improve fits on Experiment 1 and 4 while preserving the overall success of the mixture approach.

**Parameters:**
  - `beta_ttb`: `[0.0, 10.0]`
  - `beta_tally`: `[0.0, 5.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Prediction ---
    order = np.argsort(validities)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for f in order:
        if a[f] > b[f]:
            ttb_scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            ttb_scores = np.array([0.0, validities[f]])
            break
            
    if ttb_scores[0] == ttb_scores[1]:
        p_ttb = np.array([0.5, 0.5])
    else:
        z = beta_ttb * (ttb_scores - ttb_scores.max())
        e = np.exp(z)
        p_ttb = e / e.sum()
        
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    if tally_scores[0] == tally_scores[1]:
        p_tally = np.array([0.5, 0.5])
    else:
        z = beta_tally * (tally_scores - tally_scores.max())
        e = np.exp(z)
        p_tally = e / e.sum()
        
    # --- Mixture ---
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Probabilistic Cue Search: Decision-makers rely on a strict one-reason heuristic but search through cues probabilistically rather than in a fixed deterministic order. The probability of inspecting a cue is proportional to its subjective validity (via a softmax function). The decision-maker inspects cues one by one without replacement until a discriminating cue is found, which then deterministically dictates the choice. Mathematically (via the Independence of Irrelevant Alternatives in the Plackett-Luce model), the probability that a specific discriminating cue is the first one encountered is exactly its softmax weight divided by the sum of the softmax weights of all discriminating cues. This preserves the non-compensatory nature of the process while introducing cognitively plausible noise through stochastic attention and memory retrieval.

**Rationale:** Applying the minimal edit suggested by the critic: restricting the range of `beta` to [1.0, 10.0]. The previous iteration's attempt to use [5.0, 30.0] made the model too deterministic, ruining the fits for Experiments 1-4. A moderate range of [1.0, 10.0] balances the need for low tallying effects in Exps 5-6 (by ensuring high-validity cues still dominate the softmax weights) with the probabilistic noise required to capture the variance in Exps 1-4.

**Parameters:**
  - `beta`: `[1.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues discriminate between the two options
    disc = (a != b)
    
    if not np.any(disc):
        # If no cues discriminate, guess randomly
        p_core = np.array([0.5, 0.5])
    else:
        # Compute softmax weights for discriminating cues
        w = np.zeros_like(validities)
        max_v = np.max(validities[disc])
        w[disc] = np.exp(beta * (validities[disc] - max_v))
        
        sum_w = np.sum(w)
        
        # The probability of choosing an option is the sum of the weights 
        # of the discriminating cues that favor it, normalized by the sum
        # of weights of all discriminating cues.
        p_a = np.sum(w[a > b]) / sum_w
        p_b = np.sum(w[b > a]) / sum_w
        
        p_core = np.array([p_a, p_b])
        
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
