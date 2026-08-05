# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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


### slot 2 — `pi_10` — KILLED ✗

**Description:** Decision-makers do not integrate all cues on every trial. Instead, they maintain a repertoire of distinct heuristics—specifically Take The Best (TTB) and Tallying—and select one to strictly apply on each decision. On any given trial, an individual either completely relies on TTB (basing their choice solely on the single most valid discriminating cue), falls back to Tallying (counting the number of positive features for each option while ignoring validities), or simply guesses randomly due to lapses in attention. This discrete strategy selection captures bimodal response patterns without predicting the systematic compensatory shifts that cue-integration models enforce.

**Rationale:** Following the arbiter's feedback, this theory abandons cue integration (like the exponential decay in Theory 2) in favor of a strict discrete mixture of heuristics. On any given trial, the subject probabilistically selects a strategy: strictly applying Take The Best (TTB), strictly applying Tallying, or guessing randomly. When a heuristic is chosen, it is applied deterministically without any compensatory leakage or softmax noise. This correctly models human behavior as a mixture of distinct, non-compensatory and unweighted-compensatory decision rules, capturing the bimodal nature of responses and avoiding the monotonic compensatory shifts predicted by continuous integration models.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `w_tally_rel`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take The Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner == 0:
        ttb_probs = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        ttb_probs = np.array([0.0, 1.0])
    else:
        ttb_probs = np.array([0.5, 0.5])
        
    # 2. Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_probs = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_probs = np.array([0.0, 1.0])
    else:
        tally_probs = np.array([0.5, 0.5])
        
    # 3. Strategy mixture weights
    w_ttb = float(parameters["w_ttb"])
    w_tally_rel = float(parameters["w_tally_rel"])
    
    w_tally = (1.0 - w_ttb) * w_tally_rel
    w_rand = (1.0 - w_ttb) * (1.0 - w_tally_rel)
    
    # Final probability is a discrete mixture of the strategies
    p = w_ttb * ttb_probs + w_tally * tally_probs + w_rand * np.array([0.5, 0.5])
    
    return p
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

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a Sequential Evidence Accumulation strategy. They sample cues in descending order of validity and maintain a running sum of evidence (accumulating the differences between options). At each discriminating cue, there is a probabilistic decision to stop sampling and make a choice based on the accumulated evidence so far. A high stopping probability mimics Take The Best (TTB) by usually halting at the first discriminating cue, but the continuous probability of sampling further allows for occasional compensatory effects (Tallying-like behavior) when multiple lower-validity cues overturn the initial evidence.

**Rationale:** Following the arbiter's suggestion, this model instantiates a Sequential Evidence Accumulation process. Instead of forcing a discrete bimodal mixture of TTB and Tallying, it unifies them into a single sequential process. Cues are sampled in descending order of validity, and evidence is accumulated. At each discriminating cue, the decision-maker stops with probability `gamma`. If they stop early, the choice is driven by the strongest cues (mimicking TTB). If they continue, evidence accumulates unweighted, allowing multiple weaker cues to overturn the first cue (mimicking Tallying). This probabilistic stopping mechanism naturally accommodates the near-zero differences in flat-line experiments while allowing slight deviations for compensatory behavior.

**Parameters:**
  - `gamma`: `[0.1, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.special import expit
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sample cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    p_A_total = 0.0
    weight_remaining = 1.0
    
    for j in cue_order:
        delta = a[j] - b[j]
        E += delta
        # Only consider stopping if the cue discriminates
        if delta != 0:
            p_stop = gamma
            # Probability of choosing A if we stop here
            p_A_given_stop = expit(beta * E)
            
            p_A_total += weight_remaining * p_stop * p_A_given_stop
            weight_remaining *= (1.0 - p_stop)
            
    # If the decision-maker evaluates all cues without stopping, decide based on final evidence
    p_A_given_stop = expit(beta * E)
    p_A_total += weight_remaining * p_A_given_stop
    
    # Apply lapse rate
    p_A_final = (1.0 - epsilon) * p_A_total + epsilon * 0.5
    
    return np.array([p_A_final, 1.0 - p_A_final])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
