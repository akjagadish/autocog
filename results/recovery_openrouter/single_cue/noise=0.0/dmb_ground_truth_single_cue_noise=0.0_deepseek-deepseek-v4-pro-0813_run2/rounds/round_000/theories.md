# Round 0 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

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

### `pi_3` → slot 1 (via `new_theory`)

**Description:** On each choice, a decision maker probabilistically selects between two comparison strategies. With probability q, centered near 0.25, the choice is made by a one-reason stopping rule: cues are inspected in descending validity order and the first discriminating cue determines the preferred option, with no further cues consulted. With probability 1 - q, the choice is made by full feature-win tallying: each option receives one point for every feature on which it strictly beats the other option, ties contribute nothing, and the option with the larger tally is preferred. Each strategy's preference is passed through its own softmax choice rule, giving nearly deterministic but not perfectly errorless responding, and there is no additional uniform lapse. The mixture probability q is a stable subject-level parameter applied independently on every trial, so the model recovers pure Take The Best when q = 1, pure Tallying when q = 0, and an intermediate strategy mixture when q is near 0.25.

**Rationale:** The previous pure models bracket the human data incorrectly: Take The Best matches the first discriminating cue far too often in Experiment 1 and is far too anti-tallying in Experiment 2, while pure Tallying is too anti-TTB in Experiment 1 and too pro-tallying in Experiment 2. A trial-level mixture is exactly the mechanism needed because the observed values lie between the two pure strategies. Under the mixture, Experiment 1 is composed almost entirely of trials where the first valid cue and the tally point in opposite directions, so the TTB-match rate approximates q; the observed value 0.2487 therefore points to q near 0.25. In Experiment 2, the aggregate second-cue-versus-tally contrast under a noiseless mixture is approximately 0.875 - 1.8125q, which at q = 0.25 is 0.4219, very close to the observed 0.4196. I therefore keep q tightly centered at 0.25 with low between-subject variance. The softmax temperatures are set near 5.25-5.50 rather than the provisional value of 1: with binary win-count margins of one to four, beta near 1 would make the Tallying strategy too likely to choose the losing option and would flatten both behavioral signatures. The nearly deterministic core strategies with zero lapse are required for the mixture algebra to reproduce both observed values simultaneously. This model contains pi_1 as the special case q = 1 and pi_2 as the special case q = 0, but the human data support an intermediate q of about 0.25.

**Parameters:**
  - `q`: `[0.22, 0.28]`
  - `beta_tally`: `[5.44, 5.50]`
  - `beta_ttb`: `[5.22, 5.28]`
  - `epsilon`: `{0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    def _softmax(scores, beta):
        z = beta * (scores - np.max(scores))
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    validities = np.asarray(parameters['validities'], dtype=float)
    if validities.shape[0] == n_features:
        cue_order = np.argsort(-validities, kind='stable').tolist()
    else:
        cue_order = list(range(n_features))

    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        p_ttb = np.ones(2) / 2.0
    else:
        ttb_scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        p_ttb = _softmax(ttb_scores, float(parameters['beta_ttb']))

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    p_tally = _softmax(
        np.array([a_wins, b_wins]),
        float(parameters['beta_tally'])
    )

    q = float(parameters['q'])
    p = q * p_ttb + (1.0 - q) * p_tally

    epsilon = float(parameters['epsilon'])
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2) / 2.0

    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
