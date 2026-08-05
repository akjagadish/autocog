# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

**Rationale:** Following the arbiter's recommendation, this model implements the 'Take-The-Best' (TTB) heuristic. Tallying and WADD failed to capture the empirical data because subjects strongly favor the option that wins on the highest-validity cue, even if the alternative has a greater total number of positive features. By searching cues in descending order of validity and stopping at the first discriminating cue, TTB perfectly captures the non-compensatory nature of human choice in these experiments. Noise is modeled simply as a lapse rate (epsilon) as requested, replacing the softmax mechanism.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Decision-makers use a 'Noisy-Perception Take-The-Best' heuristic. They search through cues in descending order of validity, stopping at the first cue that appears to discriminate between the options. However, their perception of each option's feature value is subject to independent noise: with probability p_error, a feature's value (0 or 1) is misread. Consequently, multiple lower-validity cues can cumulatively influence the choice if earlier high-validity cues are misperceived as tied. The perception error is restricted to be very low, ensuring the model remains predominantly non-compensatory, while a decoupled lapse rate accounts for baseline errors.

**Rationale:** Following the critic's advice, the parameter ranges for p_error and epsilon have been decoupled. We maintain p_error in [0.0, 0.1] to preserve the strict non-compensatory structure needed for Exps 3, 4, 5, and 6, while relaxing the upper bound of epsilon to [0.0, 0.25] to better capture the baseline lapse rates observed in Exps 1 and 2.

**Parameters:**
  - `p_error`: `[0.0, 0.1]`
  - `epsilon`: `[0.0, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    p_e = float(parameters["p_error"])
    epsilon = float(parameters["epsilon"])
    
    P_A = 0.0
    P_B = 0.0
    P_rem = 1.0
    
    # Search through features in descending order of validity
    for idx in order:
        val_a = a[idx]
        val_b = b[idx]
        
        # Probabilities of perceived cue values given true cue values
        if val_a > val_b:
            p_a_wins = (1.0 - p_e) ** 2
            p_b_wins = p_e ** 2
        elif val_b > val_a:
            p_a_wins = p_e ** 2
            p_b_wins = (1.0 - p_e) ** 2
        else:  # val_a == val_b
            p_a_wins = p_e * (1.0 - p_e)
            p_b_wins = p_e * (1.0 - p_e)
            
        p_tie = 1.0 - p_a_wins - p_b_wins
        
        P_A += P_rem * p_a_wins
        P_B += P_rem * p_b_wins
        P_rem *= p_tie
        
    # If all features tie, guess randomly
    P_A += P_rem * 0.5
    P_B += P_rem * 0.5
    
    p_core = np.array([P_A, P_B])
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Decision-makers in multi-attribute choice are heterogeneous in their strategy use. A large majority of the population relies on the non-compensatory 'Take-The-Best' (TTB) heuristic, which sequentially searches cues by validity and stops at the first discriminating feature. However, a small subset of the population uses a compensatory 'Tallying' strategy, integrating information by simply counting the number of winning features for each option. This mixture preserves the predominantly non-compensatory nature of the population's choices (keeping conflict-vs-alignment effects near zero) while probabilistically accounting for the slight elevation in compensatory choices observed in specific trial types. Both strategies are subject to a uniform lapse rate (epsilon).

**Rationale:** Following the critic's advice, we replace the discrete threshold parameter with a continuous probability `p_tally`. This implements the strategy mixture probabilistically at the prediction level, smoothing the loss landscape for the continuous optimizer. By constraining `p_tally` to `[0.0, 0.15]`, we allow enough compensatory choices to capture Experiments 2 and 4 without disrupting the strong TTB dominance required for Experiments 5-7, resolving the optimization failure of the previous iteration.

**Parameters:**
  - `p_tally`: `[0.0, 0.15]`
  - `epsilon`: `[0.0, 0.3]`
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
    
    # Take-The-Best (TTB) Prediction
    order = np.argsort(-validities)
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_pred = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_pred = np.array([0.0, 1.0])
    else:
        p_tally_pred = np.array([0.5, 0.5])
        
    # Probabilistic mixture to smooth the optimization landscape
    p_tally = float(parameters["p_tally"])
    p_core = (1.0 - p_tally) * p_ttb + p_tally * p_tally_pred
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
