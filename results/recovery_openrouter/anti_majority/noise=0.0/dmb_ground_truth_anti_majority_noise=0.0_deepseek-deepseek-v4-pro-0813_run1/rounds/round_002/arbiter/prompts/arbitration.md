# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People execute a display-order Take The Best rule: they scan the physical feature vector left-to-right and stop at the first feature that discriminates between the two options. That first displayed discriminator determines the choice, and later features are ignored. If no feature discriminates, the person guesses. Stated validities do not reorder the scan, but they weakly modulate salience: a first discriminator with lower stated validity is used less decisively than one with higher stated validity. Response noise enters through softmax with inverse temperature beta plus a small lapse probability epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Display-order TTB expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    # Optional validities; they affect decisiveness, not cue order.
    if 'validities' in parameters and parameters['validities'] is not None:
        val = np.asarray(parameters['validities'], dtype=float)
        if val.shape[0] != n_features:
            raise ValueError(
                f'validities length {val.shape[0]} != n_features {n_features}.'
            )
    else:
        val = np.ones(n_features, dtype=float)

    # Scan features in physical display order: 0, 1, 2, ..., n-1.
    winner = None
    discriminator = None
    for j in range(n_features):
        if a[j] > b[j]:
            winner = 0
            discriminator = j
            break
        if b[j] > a[j]:
            winner = 1
            discriminator = j
            break

    # Complete tie: no discriminator, so guess.
    if winner is None:
        return np.ones(2, dtype=float) / 2.0

    beta = float(parameters['beta'])
    lam = float(parameters['lambda'])
    epsilon = float(parameters['epsilon'])

    # The first displayed discriminator is the only cue used for the choice.
    # Its stated validity weakly scales the effective inverse temperature.
    v_j = float(val[discriminator])
    salience = 1.0 + lam * max(0.0, v_j - 0.5)
    beta_eff = beta * salience

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax.
    z = beta_eff * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts, dtype=float) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
People scan the displayed features from left to right and accumulate a signed preference signal, without ever committing to a hard lexicographic stop. Each discriminating feature contributes an increment whose weight is a continuous function of serial position, stated validity, and list length. For short lists the position weight is a steep exponential primacy gradient, so the first discriminator dominates probabilistically rather than absolutely. As list length exceeds working-memory capacity the gradient collapses toward a flat weighting function, making the rule behave like a tallying heuristic on long lists. The accumulated score is passed through a softmax with inverse temperature beta, and an independent lapse probability permits occasional intention-independent errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Capacity-sensitive evidence accumulation expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Stated validities weakly modulate feature weight.
    if 'validities' in parameters and parameters['validities'] is not None:
        validities = np.asarray(parameters['validities'], dtype=float)
        if validities.shape[0] != n_features:
            validities = np.ones(n_features, dtype=float)
    else:
        validities = np.ones(n_features, dtype=float)

    lambda_5 = float(parameters['lambda_5'])
    capacity_power = float(parameters['capacity_power'])
    validity_power = float(parameters['validity_power'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Capacity-sensitive flattening. For n_features = 5 this factor is 1,
    # preserving a strong exponential primacy gradient. For longer lists it
    # shrinks rapidly, approximating equal tallying weights.
    capacity_factor = (5.0 / n_features) ** capacity_power

    lam = lambda_5 * capacity_factor
    nu = validity_power * capacity_factor

    positions = np.arange(n_features, dtype=float)
    weights = np.exp(-lam * positions) * (validities ** nu)

    diff = a - b
    score_a = float(np.sum(weights[diff > 0]))
    score_b = float(np.sum(weights[diff < 0]))
    relative_score = score_a - score_b

    # Stable softmax over the accumulated relative evidence.
    logits = np.array([beta * relative_score, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.55, 0.85, 0.85, 0.85, 0.85]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** Five experts with validities [0.55, 0.85, 0.85, 0.85, 0.85]. The low-validity expert occupies display position 0; positions 1-3 are high-validity; position 4 is a high-validity filler that never discriminates. Every unique trial has exactly one discriminating feature, placed at display position p=0,1,2,3 by leading ties, with all later features tied. Thus the tally margin is always +/-1, so any flat-weight tallying process predicts identical P(A) across positions. The advocated display-order TTB predicts a flat P(A) across p=1-3 (same first-discriminator validity 0.85; about 0.93 for A-first and 0.07 for B-first) and a less decisive P(A) at p=0 (validity 0.55; about 0.74 for A-first and 0.26 for B-first). The competing primacy-gradient accumulator predicts a steep position decline: A-first P(A) about 0.94 at p=0, 0.86 at p=1, 0.73 at p=2, and 0.62 at p=3, with mirrored B-first values. This produces a crossover at p=0 versus p=1 and a flat-versus-declining dissociation across p=1-3. Eight unique pairs repeated 12 times give 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
For each trial, identify the first displayed feature that discriminates between the two options and code whether the response matched the side favored by that discriminating feature. Let c0 be the proportion of position-0 trials with such consistency, and c123 be the mean proportion for positions 1-3. The metric returns c123 - c0. Display-order TTB should produce a positive value because position 0 has low stated validity (0.55), making choices at that position less decisive than at high-validity positions 1-3 (0.85). The primacy-gradient accumulator should produce a negative value because its steep exponential position weights make position 0 dominate despite its lower validity, yielding the highest consistency at position 0 and declining consistency at later positions. This contrast uses both directions of each position and all trials, giving a large mean separation with small per-subject binomial noise.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    low = []
    high = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        nz = np.flatnonzero(diff)
        if nz.size == 0:
            continue
        pos = int(nz[0])
        favored = 0 if diff[pos] > 0 else 1
        consistent = 1.0 if int(row.response) == favored else 0.0
        if pos == 0:
            low.append(consistent)
        elif pos in (1, 2, 3):
            high.append(consistent)
    if not low or not high:
        return float('nan')
    return float(np.mean(high) - np.mean(low))

### RESULTS
- Predicted under pi_3 (simulated): 0.1878 (var=0.0101)
- Predicted under pi_4 (simulated): -0.2019 (var=0.0042)
- Observed on real data: -0.0222 (var=0.0067)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=12):** [0.55, 0.9, 0.65, 0.8, 0.5, 0.95, 0.7, 0.85, 0.6, 0.75, 0.92, 0.58]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design uses a long 12-feature list to push the advocated capacity-sensitive accumulator into its flat-weight regime. With n_features=12, the advocated capacity_factor is (5/12)^14 ≈ 5e-6, so the exponential position gradient and the validity_power modulation are both effectively zero: the model becomes a flat tally of discriminating features with beta≈4.5 and epsilon≈0.1. The competing display-order TTB, by contrast, always stops at the first displayed discriminator and ignores all later evidence, with validity weakly modulating decisiveness. The 12 unique pairs form a 2 (first-discriminator validity: 0.55 at position 0 vs 0.90 at position 1) × 3 (full tally margin) × 2 (first-discriminator direction) factorial. Crucially, for A-first trials the full tally is negative or zero, and for B-first trials the full tally is positive or zero, so the two theories never make the same directional prediction. The advocated model predicts P(A) ≈ 0.05, 0.06, 0.50, 0.94, 0.95 for d = -2, -1, 0, +1, +2, with no validity effect. Display-order TTB predicts P(A) ≈ 0.72-0.77 for low-validity A-first, ≈ 0.93-0.97 for high-validity A-first, and mirrored values for B-first, with no tally-margin effect. This yields both qualitative directional dissociations and quantitative validity-modulation versus tally-margin dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where feature index 1 is the first discriminating feature. Feature 1 has stated validity 0.90, so the display-order TTB theory predicts that it should almost fully determine choice: when A has the high-validity cue, B-response rate is near 0.05; when B has it, B-response rate is near 0.95, producing a difference of roughly -0.90. The advocated capacity-sensitive accumulator has nearly flat weights for 12 features, so the same trials are decided by the accumulated tally. For the A-first index-1 trials the tally is negative or zero and for the B-first index-1 trials it is positive or zero, giving a B-rate difference near +0.59. The two theories therefore produce opposite signs and a large mean separation. Each category contains 24 trials per subject, so the metric also has low per-subject binomial variability.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hi_a = []
    hi_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'])
        b = np.asarray(row['option_b_ratings'])
        diff = a != b
        if not np.any(diff):
            continue
        first_diff = int(np.argmax(diff))
        if first_diff != 1:
            continue
        if a[1] > b[1]:
            hi_a.append(float(row['response']))
        else:
            hi_b.append(float(row['response']))
    if not hi_a or not hi_b:
        return 0.0
    return float(np.mean(hi_a) - np.mean(hi_b))

### RESULTS
- Predicted under pi_3 (simulated): -0.8958 (var=0.0049)
- Predicted under pi_4 (simulated): 0.6033 (var=0.0124)
- Observed on real data: -0.0633 (var=0.1971)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=int)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=int)
    r = data['response'].to_numpy(dtype=float)

    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    tie = a_wins == b_wins

    # Among tally ties, the high-validity cue at index 2 discriminates.
    cue_favors_b = tie & (b[:, 2] > a[:, 2])
    cue_favors_a = tie & (a[:, 2] > b[:, 2])

    if not np.any(cue_favors_b) or not np.any(cue_favors_a):
        return 0.0

    p_b_given_b_cue = float(r[cue_favors_b].mean())
    p_b_given_a_cue = float(r[cue_favors_a].mean())

    return p_b_given_b_cue - p_b_given_a_cue

```

**Observed (real) value:** -0.4400 (var=0.1503)
**Predicted under pi_3:** -0.6500 (var=0.0231)
**Predicted under pi_4:** -0.4933 (var=0.0313)

### Experiment 4
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
import numpy as np

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.78, 0.55, 0.91, 0.50, 0.65], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    a = np.vstack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.vstack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b

    ordered_diff = diff[:, cue_order]
    nonzeros = ordered_diff != 0
    has_discriminator = nonzeros.any(axis=1)
    first_pos = nonzeros.argmax(axis=1)
    first_values = ordered_diff[np.arange(ordered_diff.shape[0]), first_pos]

    choose_a = (data['response'].to_numpy() == 0).astype(float)
    first_cue_favors_a = has_discriminator & (first_values > 0)
    first_cue_favors_b = has_discriminator & (first_values < 0)

    if not (first_cue_favors_a.any() and first_cue_favors_b.any()):
        return 0.0

    p_a_given_cue_a = choose_a[first_cue_favors_a].mean()
    p_a_given_cue_b = choose_a[first_cue_favors_b].mean()
    return float(p_a_given_cue_a - p_a_given_cue_b)
```

**Observed (real) value:** -0.3750 (var=0.0321)
**Predicted under pi_3:** -0.3296 (var=0.0048)
**Predicted under pi_4:** -0.3458 (var=0.0038)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    matches = []
    for a_ratings, b_ratings, response in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        winner = None
        for a_val, b_val in zip(a_ratings, b_ratings):
            if a_val > b_val:
                winner = 0
                break
            if a_val < b_val:
                winner = 1
                break
        if winner is not None:
            matches.append(1.0 if response == winner else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6246 (var=0.0093)
**Predicted under pi_3:** 0.8567 (var=0.0014)
**Predicted under pi_4:** 0.6208 (var=0.0020)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b_vals = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    a_wins = np.sum(a_vals > b_vals, axis=1)
    b_wins = np.sum(b_vals > a_vals, axis=1)
    tally_tie = a_wins == b_wins

    differ = a_vals != b_vals
    has_disc = differ.any(axis=1)
    first_disc = np.argmax(differ.astype(int), axis=1)

    row_idx = np.arange(a_vals.shape[0])
    first_a_wins = a_vals[row_idx, first_disc] > b_vals[row_idx, first_disc]

    response = data['response'].to_numpy(dtype=float)

    selected = tally_tie & has_disc
    if not np.any(selected):
        return 0.0

    disc_positions = np.unique(first_disc[selected])
    per_pos_diff = []
    for pos in disc_positions:
        in_pos = selected & (first_disc == pos)
        b_first = in_pos & ~first_a_wins
        a_first = in_pos & first_a_wins
        if np.sum(b_first) == 0 or np.sum(a_first) == 0:
            continue
        per_pos_diff.append(
            float(np.mean(response[b_first]) - np.mean(response[a_first]))
        )

    if not per_pos_diff:
        return 0.0
    return float(np.mean(per_pos_diff))
```

**Observed (real) value:** -0.1173 (var=0.0257)
**Predicted under pi_3:** 0.7160 (var=0.0149)
**Predicted under pi_4:** 0.0280 (var=0.0391)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_4 is the stronger theory. It is close to the observed data on Experiment 3 (-0.493 vs -0.440), Experiment 4 (-0.346 vs -0.375), Experiment 5 (0.621 vs 0.625), and Experiment 6 (0.028 vs -0.117), and it at least has the correct sign on Experiment 1 (-0.202 vs -0.022). By contrast, pi_3 is qualitatively falsified: it predicts the wrong sign on Experiment 1 (+0.188 vs -0.022) and Experiment 6 (+0.716 vs -0.117), strongly overpredicts first-displayed-cue consistency on Experiment 5 (0.857 vs 0.625), and predicts nearly deterministic high-validity cue following on Experiment 2 (-0.896 vs -0.063). The observed behavior is therefore incompatible with a hard display-order take-the-best rule, even with softmax noise and a lapse term. Pi_4 is not perfect either: its capacity factor collapses to an almost pure tally on the 12-feature Experiment 2 list, producing the wrong sign there (+0.603 vs -0.063). However, that is a parameterization problem rather than a fundamental failure of the accumulation idea. The most useful move is to replace the degenerate pi_3 theory with a new theory that combines moderate primacy, validity sensitivity, stochastic sampling, and a bounded capacity effect, and which can serve as a stronger competitor to pi_4.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 with a stochastic bounded-sampling evidence-accumulation theory. The new theory should not use a hard stop at the first displayed discriminator. Instead, discriminating cues should be sampled stochastically, with sampling probability determined by both serial position and stated validity. Serial position should produce only a moderate primacy gradient on short lists and a weak or partially flattened gradient on long lists, rather than the extreme exponential gradient that pi_4 currently uses. Stated validity should remain influential on all list lengths, but with a compressive rather than all-or-none effect, so a 0.90 validity cue can compete with opposing tally evidence and produce near-zero direction effects as in Experiment 2. A bounded capacity function such as f(n) = 1 / (1 + kappa * (n - 5)) or f(n) = (5 / n)^gamma with small gamma around 0.3 to 1.0 should be used, so longer lists flatten attention only partially, never to a pure equal-weight tally. Decision noise should be nontrivial: inverse temperature around 1 to 3 and lapse epsilon around 0.1 to 0.2, matching the modest first-cue consistency of 0.62 in Experiment 5 and the moderate validity effects in Experiments 3 and 4. The new theory also needs to allow subject-level heterogeneity in cue weights, because Experiment 2 shows high between-subject variance (var=0.197). This theory differs from pi_3 by abandoning deterministic first-discriminator commitment, and differs from pi_4 by making cue weighting stochastic and bounded, with validity effects that do not disappear on long lists."
}
```

## Usage

```json
{
  "input_tokens": 9626,
  "output_tokens": 13120,
  "reasoning_tokens": 12420,
  "provider": "Baidu"
}
```
