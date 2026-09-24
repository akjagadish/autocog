# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.55, 0.9, 0.8, 0.65, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 12: A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Rationale:** This design dissociates display-order Take The Best (TTB) from Tallying by making the first left-to-right discriminator favor one option while the full feature tally favors the other option, or is exactly balanced. The five validities are non-monotonic across display positions ([0.55, 0.90, 0.80, 0.65, 0.60]), so the first discriminator's validity varies across trials. In pairs 1-6 and 9-10, the first discriminating feature favors one option while the tally margin has the opposite sign, producing opposite directional predictions. In pairs 7, 8, 11, and 12, the tally is exactly balanced (d = 0), so Tallying must predict chance regardless of beta or epsilon, whereas TTB predicts a directional choice based on the first discriminator. Pairs 5 vs 9 and 6 vs 10 hold the tally margin constant (±1) but move the first discriminator from a high-validity feature (0.80) to a low-validity feature (0.55), so TTB predicts different choice probabilities while Tallying predicts identical probabilities. This yields both qualitative directional dissociations and quantitative validity-modulation dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures first-cue concordance: for each trial it finds the leftmost displayed feature on which options A and B differ, and codes 1 if the chosen option is the one favored by that feature. Display-order Take The Best predicts high concordance because the first displayed discriminator determines choice, with only weak validity-based modulation. Tallying predicts low concordance in this design: in the eight non-tied pairs the tally favors the opposite option, and in the four tally-balanced pairs it predicts 0.5. Averaging this code over all 96 trials produces a large mean separation and low per-subject variance, so the metric is stable with small human samples.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.8567 (var=0.0014)
- Predicted under pi_2 (simulated): 0.2706 (var=0.0057)
- Observed on real data: 0.6246 (var=0.0093)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 17: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 18: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Rationale:** Eight experts with non-monotonic validities [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]. The first three display positions form a first-discriminator validity staircase (0.53, 0.91, 0.82). Leading ties place the first discriminator at position 0, 1, or 2. The design is a 3 (first-discriminator validity) x 3 (tally margin d = +4/+2/0 for B-first; -4/-2/0 for A-first) x 2 (first-discriminator direction) factorial. Under Tallying, P(A) depends only on d: high for d=+4, above chance for d=+2, exactly chance for d=0, below chance for d=-2, and low for d=-4; it is identical across the three validity levels for a given d. Under display-order TTB, P(A) depends only on the first displayed discriminator's direction and validity: low for B-first (lowest for v=0.91, highest for v=0.53) and high for A-first, and it is flat across d. Thus all 18 pairs are dissociative: d!=0 trials give opposite directional predictions, and d=0 trials give chance (Tallying) versus directional (TTB). The factorial also provides quantitative tests: Tallying predicts a d effect with no validity effect, while TTB predicts a validity effect with no d effect. The 18 unique pairs are repeated 5 times for 90 trials, close to the 96-trial target.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates display-order directional responding on exactly the trials where Tallying has no preference. It selects tally-tie trials, where the feature-wise win counts for A and B are equal. For each first displayed discriminator position, it computes the average B choice rate when the B option is the first discriminator minus the average B choice rate when the A option is the first discriminator, then averages across those positions. Under Tallying, tally ties are uninformative and choice probability is 0.5 in both directions for every position, so this signed difference is close to 0. Under display-order Take The Best, the first discriminator controls choice: B-first trials produce mostly B responses and A-first trials produce mostly A responses. With the design validities 0.53, 0.91, 0.82 and the competing theory's beta/lambda ranges, the per-position differences are approximately 0.4 to 0.9, yielding a pooled value near 0.7. The within-subject signed contrast uses all 30 tally-tie trials per subject, keeping binomial estimation noise small, so the between-theory gap is large relative to between-subject variance.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.7160 (var=0.0149)
- Predicted under pi_2 (simulated): -0.0147 (var=0.0280)
- Observed on real data: -0.1173 (var=0.0257)

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
**Predicted under pi_2:** 0.0383 (var=0.0359)

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
**Predicted under pi_2:** -0.0029 (var=0.0086)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The four experiments do not support either theory as a task-invariant account. pi_3, the display-order Take The Best theory, is directionally correct in three of four cases: Experiment 1 observed 0.6246 first-cue concordance versus pi_3 at 0.8567 and pi_2 at 0.2706; Experiment 3 observed -0.4400 versus pi_3 at -0.6500 and pi_2 at 0.0383; Experiment 4 observed -0.3750 versus pi_3 at -0.3296 and pi_2 at -0.0029. In all three, pi_3 captures the sign and rough magnitude of the effect. However, pi_3 fails badly on Experiment 2, where it predicts a strong positive first-discriminator contrast of 0.7160 but the observed value is -0.1173, essentially chance-like. pi_2, the equal-count tallying theory, is the better account only in Experiment 2, where its prediction of -0.0147 is close to the observed -0.1173, but it is flatly wrong in Experiments 1, 3, and 4, especially because it cannot produce directional responding on tally-tie trials. The critical pattern is that TTB-like behavior appears with five features, while the eight-feature experiment produces tally-like chance behavior on tie trials. That suggests a capacity-sensitive or working-memory-dependent comparison process rather than either pure disappearing order. Neither current model can be fixed by parameter changes alone: pi_3's description requires the first displayed discriminator to dominate, so it cannot produce the near-zero Experiment 2 metric; pi_2's description requires equal cue weights and tie guessing, so it cannot produce the directional tie effects in Experiments 3 and 4.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace pi_2. The new theory should be a capacity-sensitive, order-weighted evidence accumulation account. People scan features left-to-right and accumulate a preference score: each discriminating feature j contributes a signed increment weighted by display position and stated validity. Early positions receive stronger weight, producing take-the-best-like choices for short feature lists, while the position gradient flattens as feature count exceeds working-memory capacity, producing tallying-like behavior for eight-feature problems. Choice probability comes from a softmax over the accumulated score plus a lapse probability. This would explain Experiment 1 as partial, not absolute, first-cue dominance; Experiments 3 and 4 as display-order directional choices on tally ties; and Experiment 2 as near-chance responding when the long feature list flattens the weights and the sign counts balance. Avoid a hard lexicographic stop and avoid equal weighting across all cues; instead estimate a continuous weight function that depends on feature count, display position, and stated validity."
}
```

## Usage

```json
{
  "input_tokens": 7739,
  "output_tokens": 23536,
  "reasoning_tokens": 22914,
  "provider": "Baidu"
}
```
