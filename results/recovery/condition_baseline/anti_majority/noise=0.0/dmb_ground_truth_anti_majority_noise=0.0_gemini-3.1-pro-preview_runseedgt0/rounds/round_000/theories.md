# Round 0 — Theories

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


### slot 2 — `pi_2` — KILLED ✗

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

### `pi_3` → slot 2 (via `new_theory`)

**Description:** Anti-Tallying: Subjects systematically misinterpret the expert ratings, treating a rating of 0 as superior to 1. They tally the number of features where an option 'wins' under this reversed interpretation (i.e., has a 0 while the other has a 1) and prefer the option with more reversed wins. Choice noise is modeled via a softmax over these anti-tallies and a uniform lapse rate.

**Rationale:** Following the arbiter's feedback, the data strongly suggests a contrarian pattern where subjects systematically prefer the option with fewer traditional 'wins' or lower cue validities. Instead of a complex flipped-majority ensemble, this model implements a much simpler 'Anti-Tallying' heuristic. It operates under the hypothesis that subjects systematically misinterpret the binary expert ratings, treating 0 as a better rank than 1. The model tallies these 'reversed wins' and applies standard softmax and lapse noise. This provides a direct, parsimonious explanation for the contrarian behavior observed across both experiments.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Anti-Tallying expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Anti-Tallying: subjects treat 0 as better than 1.
    # Thus, option A wins a feature if a < b (i.e., a=0, b=1).
    a_anti_wins = float(np.sum(a < b))
    b_anti_wins = float(np.sum(b < a))
    
    scores = np.array([a_anti_wins, b_anti_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the anti-tallies with numerical stability
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
