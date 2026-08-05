# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Rationale:** Following the critic's advice, the minimal edit adjusts the `p_ttb` range to `[0.35, 0.95]` to perfectly split the difference between the previous two iterations. This should balance the slight over- and under-predictions of TTB consistency across the four experiments while keeping epsilon noise low.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `p_ttb`: `[0.35, 0.95]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
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
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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


### slot 2 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) theory with Non-Linear Cue Scaling posits that decision makers compute a total score for each option by summing the validities of the features it possesses, but they may non-linearly amplify or dampen the differences between cue validities. By exponentiating cue validities with a free parameter, the model can naturally interpolate between Tallying (where all cues are weighted equally) and Take The Best (where the highest-validity cue dominates).

**Rationale:** Following the critic's diagnosis, the previous non-linear WADD model still behaved too much like Tallying because the upper bound of the gamma parameter (5.0) was not high enough to allow the highest-validity cue to fully dominate multiple lower-validity cues. By expanding the gamma range to [1.0, 20.0], the model is granted the capacity to heavily amplify the primary cue, shifting predictions closer to the empirical Take The Best-like behavior observed in humans.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[1.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly to allow amplification of the best cues
    val_scaled = val ** gamma
    
    # Compute WADD scores: sum of scaled validities for features possessed by the option
    score_a = np.sum(val_scaled * a)
    score_b = np.sum(val_scaled * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Probabilistic Cue Selection (Random Dictator): Decision-makers do not deterministically follow a strict cue hierarchy (as in Take The Best) nor do they exhaustively sum all available cues (as in Tallying or WADD). Instead, on any given trial, they probabilistically sample a single cue from the set of discriminating cues to determine their choice. The probability of sampling a specific cue is proportional to its validity raised to a non-linear scaling parameter (gamma). If gamma is very high, the most valid cue is almost always sampled, perfectly mimicking TTB. If gamma is near zero, cues are sampled uniformly, producing a soft Tallying-like behavior where choice probability reflects the proportion of cues favoring an option. This provides a mathematically elegant, single-process probabilistic mechanism that gracefully spans the spectrum of non-compensatory to compensatory decision-making without requiring ad-hoc response noise or explicit strategy mixtures.

**Rationale:** Following the arbiter's guidance, this model replaces the previous WADD/mixture theories with a Probabilistic Cue Selection mechanism. By isolating discriminating cues and sampling one proportional to a power of its validity (gamma), the model directly generates probabilistic choice predictions without needing an external softmax function. This elegantly captures the variance in human decision-making: high gamma subjects behave like strict TTB users, while low gamma subjects exhibit compensatory-like behavior proportional to the cue tally. This avoids the mechanistic failures of summing scores and produces a more natural, psychologically plausible account of bounded rational choice.

**Parameters:**
  - `gamma`: `[0.0, 20.0]`
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
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify cues that discriminate between option A and option B
    diff = a - b
    disc_mask = diff != 0
    
    if not np.any(disc_mask):
        # No cues discriminate; guess uniformly
        p_core = np.array([0.5, 0.5])
    else:
        # Scale validities of discriminating cues non-linearly
        w = val[disc_mask] ** gamma
        w_sum = np.sum(w)
        
        if w_sum == 0:
            p_core = np.array([0.5, 0.5])
        else:
            # Probability of sampling each discriminating cue
            p = w / w_sum
            
            # The choice is determined entirely by the sampled cue.
            # Thus, the probability of choosing A is the sum of sampling probabilities
            # for cues where A > B.
            p_a = np.sum(p[diff[disc_mask] > 0])
            p_b = np.sum(p[diff[disc_mask] < 0])
            p_core = np.array([p_a, p_b])
            
    # Incorporate uniform lapse rate (guessing)
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
