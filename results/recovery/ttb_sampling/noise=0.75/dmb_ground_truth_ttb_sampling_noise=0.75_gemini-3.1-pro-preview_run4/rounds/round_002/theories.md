# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

**Rationale:** Following the critic's advice, I widened the epsilon parameter range from [0.0, 0.5] to [0.0, 1.0]. The human responses in the given experiments are closer to 0.5 than strict TTB allows with a small lapse rate. Allowing epsilon to range up to 1.0 enables the model to fit higher levels of behavioral noise (random guessing) while preserving the core TTB lexicographic mechanism.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Strategy Selection (Mixed Heuristic) with Linear Tallying: Decision makers probabilistically select between a non-compensatory lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). To prevent extreme over-sensitivity when feature differences are large, the compensatory Tallying component uses a simple proportional (linear) rule rather than an exponential softmax.

**Rationale:** Following the latest feedback, I reverted to the successful Iteration 1 base (a fixed-probability mixture of Take The Best and Tallying) but replaced the exponential softmax in the Tallying component with a linear proportional rule (`scores / sum(scores)`). The beta parameter has been removed entirely. This linear scaling prevents the Tallying component from confidently overpowering the TTB component when feature differences are large, which dampens the extreme compensatory pull in Experiment 4 and should push its metric closer to the observed flat response without breaking the fits in Experiments 1 and 2.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(validities)[::-1]
    chosen_ttb = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen_ttb = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen_ttb = 1
            break
            
    if chosen_ttb == 0:
        p_ttb_core = np.array([1.0, 0.0])
    elif chosen_ttb == 1:
        p_ttb_core = np.array([0.0, 1.0])
    else:
        p_ttb_core = np.array([0.5, 0.5])
        
    # Tallying prediction (linear proportional rule)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    scores = np.array([a_wins, b_wins])
    
    if np.sum(scores) > 0:
        p_tally_core = scores / np.sum(scores)
    else:
        p_tally_core = np.array([0.5, 0.5])
    
    # Strategy mixture
    p_ttb = float(parameters["p_ttb"])
    p_mixed = p_ttb * p_ttb_core + (1.0 - p_ttb) * p_tally_core
    
    # Response noise (lapse rate)
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
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


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Subjective Take The Best (Noisy-Validity TTB). Decision makers strictly follow the non-compensatory Take The Best (TTB) heuristic on any given trial, meaning they base their choice solely on the first discriminating cue they consider and ignore all others. However, their internal ranking of cue validities is noisy. This is modeled by sampling the primary discriminating cue via a softmax distribution over the objective validities of the cues that differ between the options. The inverse temperature parameter beta controls the noise in the validity ranking: as beta increases, the model converges to deterministic TTB, whereas lower beta values allow sub-optimal cues to occasionally be evaluated first. Because decisions rely on a single cue, the model captures the flat sensitivity curves to supporting cue quantities (Exps 1, 2, 6). Meanwhile, the stochastic cue selection suppresses the overall agreement with the objective TTB predictions, matching the lower empirical performance in Exps 3, 4, and 5. Response noise is included via a simple lapse rate (epsilon).

**Rationale:** Following the arbiter's feedback, this model implements Noisy-Validity TTB. Subjects use a strictly non-compensatory rule on each trial (evaluating only one cue), but the cue they select as 'best' is noisy. By exploiting the properties of the Plackett-Luce model, the probability that a cue is the first discriminating cue considered is exactly the softmax over the validities of the discriminating cues. This elegantly captures both the flat sensitivity to cue quantities (since only one cue is used per trial) and the suppressed agreement with objective TTB (due to the stochastic selection of the primary cue).

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating cues
    diff = stim[0] - stim[1]
    discrim_idx = np.where(diff != 0)[0]
    
    if len(discrim_idx) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Softmax over validities of discriminating cues
        v_discrim = validities[discrim_idx]
        # Numerically stable softmax
        z = beta * v_discrim
        z -= np.max(z)
        w = np.exp(z)
        p_discrim = w / np.sum(w)
        
        p_a = 0.0
        p_b = 0.0
        for i, idx in enumerate(discrim_idx):
            if diff[idx] > 0:
                p_a += p_discrim[i]
            else:
                p_b += p_discrim[i]
                
        p_core = np.array([p_a, p_b])
        
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
