# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People are modeled as running three classic heuristics in parallel — Take The Best (one-reason, top-validity cue), Tallying (count of feature-wise wins), and Weighted Additive (validity-weighted sum) — reading off the single option each heuristic deterministically prefers, and then leaning toward whichever option the majority of the three did NOT prefer. With three binary voters at least two always agree, so a majority option always exists; the model targets its complement. Response noise enters only at the final, ensemble level: a softmax over the binary "flipped-majority" score with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options. The individual heuristics contribute only their hard preference (their argmax) to the vote — they carry no internal noise — so beta and epsilon describe noise in the contrarian response itself, not in the constituent heuristics.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Anti-majority-vote ensemble. Each of Tallying / TTB / WADD
    # contributes a single hard preference (the argmax of its
    # deterministic score — no per-model softmax or lapse). We take the
    # majority vote across the three binary votes and flip it: when at
    # least two models agree on an option, we target the other one.
    # Choice noise (softmax over beta, lapse over epsilon) is applied
    # ONCE here, to the flipped-majority decision.

    def _pref_tallying(stimulus):
        # Tallying (Dawes 1979; Gigerenzer & Goldstein 1999): count
        # strict feature-wise wins; prefer the option with more wins.
        # Ties (equal win counts) fall to option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        a, b = stim[0], stim[1]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        return int(np.argmax(np.array([a_wins, b_wins])))

    def _pref_ttb(parameters, stimulus):
        # Take The Best (Gigerenzer & Goldstein 1996): consult cues in
        # descending validity; the first discriminating cue decides.
        # If no cue discriminates, the score is uniform and argmax
        # falls to option A.
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
            scores = np.ones(2) / 2.0
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        return int(np.argmax(scores))

    def _pref_wadd(parameters, stimulus):
        # Weighted Additive (Payne, Bettman & Johnson 1993): prefer the
        # option with the higher validity-weighted sum. Ties fall to
        # option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        n_features = stim.shape[1]
        validities = np.asarray(parameters["validities"], dtype=float)
        w = np.asarray(parameters["weights"], dtype=float)
        if w.shape[0] != n_features:
            raise ValueError(
                f"weights length {w.shape[0]} != n_features {n_features}."
            )
        scores = stim @ (validities * w)
        return int(np.argmax(scores))

    # Each model's hard preference (no noise inside the models).
    c_tallying = _pref_tallying(stimulus)
    c_ttb = _pref_ttb(parameters, stimulus)
    c_wadd = _pref_wadd(parameters, stimulus)

    # Majority vote across the three binary votes, then flip it.
    votes = [c_tallying, c_ttb, c_wadd]
    majority = 0 if votes.count(0) >= 2 else 1
    flipped = 1 - majority

    # Binary score for the flipped-majority choice (winner = 1, loser = 0),
    # then ONE softmax + lapse over the two options.
    scores = np.array([1.0, 0.0]) if flipped == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # flipped score this collapses to sigmoid(beta) for the targeted
    # option (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic flip).
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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Reverse Weighted Additive (Negative Utility) Theory: Subjects systematically interpret the features or validities as negative signals (such as costs, penalties, or negative reviews) rather than positive attributes. Consequently, they compute a standard Weighted Additive (WADD) score—the sum of feature values weighted by their respective validities—but aim to minimize this score. Choice behavior is modeled as a softmax over the negative WADD scores, offering an elegant, single-process cognitive explanation for the choice inversions observed across experiments. The noise parameters are tuned to reflect a higher degree of stochasticity observed in human data.

**Rationale:** Following the critic's feedback, the core Reverse WADD mechanism is retained, but the range of the inverse temperature parameter `beta` has been shifted from [0.1, 20.0] to [0.01, 5.0]. This minimal edit increases the baseline stochasticity of the choice process, pulling the extreme deterministic predictions closer to 0.5 to better match the variance and noise levels observed in the human data across all experiments.

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Reverse WADD expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute standard WADD scores (sum of features weighted by validities)
    wadd_scores = stim @ validities
    
    # Reverse the scores to model negative utility (minimization)
    scores = -wadd_scores
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the negative WADD scores with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Strict Anti-TTB (Contrarian Lexicographic) Theory with Simple Lapse Noise

**Rationale:** Following the critic's advice, the noise model has been simplified by removing the redundant softmax inverse temperature (`beta`). Since the core heuristic yields deterministic binary preferences (1.0 and 0.0), applying both a softmax and a lapse rate created an unidentifiable parameter space. We now directly apply a uniform lapse rate (`epsilon`) to the deterministic choice, and have widened its range to [0.0, 1.0] to give the optimizer full flexibility to capture varying levels of stochasticity across experiments.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Anti-TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    cue_order = np.argsort(-validities, kind="stable").tolist()
    
    winner = None
    # Find the first discriminating cue
    for j in cue_order:
        if a[j] > b[j]:
            # a has 1, b has 0. Anti-TTB prefers the option with 0, which is b.
            winner = 1
            break
        elif b[j] > a[j]:
            # b has 1, a has 0. Anti-TTB prefers the option with 0, which is a.
            winner = 0
            break
            
    if winner is None:
        p_core = np.ones(2) / 2.0
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    epsilon = float(parameters["epsilon"])
    
    n_opts = p_core.shape[0]
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
