# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Sequential Cue Integration with Tallying Fallback (Compensatory Shift)

**Rationale:** By increasing the threshold bounds from [0.0, 5.0] to [2.0, 10.0], we force the model to integrate more cues before terminating. This increases the model's compensatory behavior, moving it away from the purely non-compensatory Take-The-Best reference probabilities and better capturing the human variance observed in the experiments.

**Parameters:**
  - `threshold`: `[2.0, 10.0]`
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

    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    diff = 0.0
    crossed = False
    # Accumulate evidence sequentially
    for j in cue_order:
        diff += val[j] * (a[j] - b[j])
        if abs(diff) >= threshold:
            crossed = True
            break

    # If all cues are exhausted without crossing the threshold, fall back to tallying
    if not crossed:
        diff = float(np.sum(a > b) - np.sum(b > a))

    scores = np.array([diff, 0.0])
    
    z = beta * scores
    z -= np.max(z)  # numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)

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


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** People evaluate options by computing a weighted sum of their feature values, where the weights correspond to the subjective or objective validities of the cues (Weighted Additive rule, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), this compensatory strategy allows multiple lower-validity cues to jointly outweigh a single high-validity cue. Choices are made probabilistically via a softmax over the options' weighted sums, along with a lapse rate to account for random errors.

**Rationale:** Following the arbiter's instructions, this model implements the Weighted Additive (WADD) heuristic. Unlike Take The Best (which relies on a single cue) or Tallying (which weights all cues equally), WADD multiplies each feature by its validity and sums them to evaluate each option. This compensatory mechanism is capable of explaining behavior where participants trade off multiple weak cues against a single strong cue, which is often necessary to explain intermediate choice patterns observed across different experiments.

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
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Probabilistic Take-The-Best (TTB) with Cue Misordering: Decision makers primarily rely on a non-compensatory one-reason heuristic, evaluating features strictly in descending order of their validities and choosing based solely on the first discriminating cue. However, due to cognitive noise or memory retrieval failures, there is a probability that the cue hierarchy is misordered (effectively randomizing the cue search order). When the order is randomized, the first discriminating cue encountered is uniformly distributed among all available discriminating cues. This provides a stark, non-compensatory alternative to WADD, where apparent 'tallying' behavior naturally emerges from random cue misordering rather than a distinct compensatory calculation. Response noise is handled via a softmax temperature on the final cue and an independent uniform lapse rate.

**Rationale:** Following the critic's advice, we revert the `beta` range to its original [0.1, 20.0] and restrict the upper bound of the lapse rate `epsilon` to 0.2 (changing the range to [0.0, 0.2]). High lapse rates may have been homogenizing the simulated population by pulling all subjects toward a 0.5 probability, thereby collapsing between-subject variance. Restricting `epsilon` to a lower range allows the distinct idiosyncratic strategies (strict TTB vs. randomized TTB) to dominate the response probabilities, which should yield higher between-subject variance similar to the empirical data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `p_random_order`: `[0.0, 1.0]`
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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_random = float(parameters["p_random_order"])

    # --- Strict TTB (Validities Order) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_strict = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_strict = 0
            break
        elif b[j] > a[j]:
            winner_strict = 1
            break

    p_strict = np.array([0.5, 0.5])
    if winner_strict is not None:
        scores = np.array([1.0, 0.0]) if winner_strict == 0 else np.array([0.0, 1.0])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_strict = e / np.sum(e)

    # --- Random Order TTB (Misordered Cues) ---
    # If the cue order is fully randomized, the first discriminating cue found
    # is uniformly selected from the set of all discriminating cues.
    discriminating_cues = []
    for j in range(len(val)):
        if a[j] > b[j]:
            discriminating_cues.append(0)
        elif b[j] > a[j]:
            discriminating_cues.append(1)

    p_rand = np.array([0.5, 0.5])
    if len(discriminating_cues) > 0:
        count_0 = sum(1 for w in discriminating_cues if w == 0)
        count_1 = sum(1 for w in discriminating_cues if w == 1)
        prob_0_wins = count_0 / len(discriminating_cues)
        prob_1_wins = count_1 / len(discriminating_cues)
        
        # Softmax probabilities if a cue favoring option 0 is found first
        scores_0 = np.array([1.0, 0.0])
        z_0 = beta * (scores_0 - np.max(scores_0))
        e_0 = np.exp(z_0)
        p_win_0 = e_0 / np.sum(e_0)
        
        # Softmax probabilities if a cue favoring option 1 is found first
        scores_1 = np.array([0.0, 1.0])
        z_1 = beta * (scores_1 - np.max(scores_1))
        e_1 = np.exp(z_1)
        p_win_1 = e_1 / np.sum(e_1)
        
        # Expected probability under random cue misordering
        p_rand = prob_0_wins * p_win_0 + prob_1_wins * p_win_1

    # --- Mixture and Lapse ---
    p_core = (1.0 - p_random) * p_strict + p_random * p_rand
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
