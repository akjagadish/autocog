# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_10` — KILLED ✗

**Description:** Heuristic Mixture Theory: Decision-makers evaluate options by probabilistically choosing between two distinct non-compensatory and compensatory heuristics on any given trial. A proportion of the time, governed by the individual parameter 'p_ttb', the decision-maker uses Take-The-Best (TTB), which bases the decision entirely on the single most valid discriminating cue. The rest of the time, they use Pure Tallying, which counts the total number of winning features for each option irrespective of their validities. The final choice probability is a mixture of the softmax probabilities derived from the chosen heuristic's evidence, further subject to random lapse noise.

**Rationale:** Following the critic's advice, we revert to the Iteration 1 structure which was the most robust and accepted by the gate. To address the remaining misfit on directional experiments, we widen the `beta` parameter range to `[0.1, 50.0]` to allow the softmax to approach a deterministic step function when needed. We also explicitly initialize the TTB evidence to `[0.5, 0.5]` so that it symmetrically handles cases where no cues discriminate.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `p_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Take-The-Best (TTB) evaluation
    order = np.argsort(-val, kind="stable")
    ttb_ev = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_ev = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_ev = np.array([0.0, 1.0])
            break
            
    # Pure Tallying evaluation
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_ev = np.array([a_wins, b_wins])
    
    # Apply softmax response noise to TTB output
    z_ttb = beta * (ttb_ev - np.max(ttb_ev))
    e_ttb = np.exp(z_ttb)
    p_ttb_choice = e_ttb / np.sum(e_ttb)
    
    # Apply softmax response noise to Tallying output
    z_tally = beta * (tally_ev - np.max(tally_ev))
    e_tally = np.exp(z_tally)
    p_tally_choice = e_tally / np.sum(e_tally)
    
    # Mixture of the two heuristics' choice probabilities
    p_core = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Apply lapse rate
    n_opts = len(p_core)
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


### slot 2 — `pi_2` — SURVIVED ✓

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_11` → slot 1 (via `new_theory`)

**Description:** Weighted Additive (WADD) Theory with Dampened Log-Odds: Decision-makers evaluate options by computing a weighted sum of their features. The subjective weight of each feature interpolates between a flat weight (Tallying) and the normative log-odds of its cue validity. To prevent extreme validities from completely overpowering the compensatory tallying behavior, the log-odds are dampened using a squashing function (tanh). This allows the strategy to capture both tally-like behavior and validity-driven normative evidence integration without creating extreme, non-compensatory weights.

**Rationale:** Following the critic's advice, we reverted to the successful log-odds weighting scheme from Iteration 2 but introduced a `tanh` dampening function to the log-odds before mixing. This squashing function bounds the log-odds values, preventing high-validity features from creating extreme weights that entirely overpower the flat (Tallying) component on critical conflict trials. This minimal edit preserves the normative evidence-accumulation properties of the log-odds transformation while ensuring the model can still capture the strong compensatory tallying preferences observed in the human data.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into log-odds, clipping to avoid infinities
    val_clipped = np.clip(val, 0.001, 0.999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    # Dampen the log-odds using tanh to prevent extreme values from overpowering tallying
    dampened_log_odds = np.tanh(log_odds)
    
    # Interpolate between flat weighting (Tallying, gamma=0) and dampened log-odds (WADD, gamma=1)
    w = (1.0 - gamma) * 1.0 + gamma * dampened_log_odds
    
    # Compute the weighted additive value for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
