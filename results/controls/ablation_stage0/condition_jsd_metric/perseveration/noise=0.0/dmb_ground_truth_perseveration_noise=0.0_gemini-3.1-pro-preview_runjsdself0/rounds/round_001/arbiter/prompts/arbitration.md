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
People make choices by computing a weighted sum of the features for each option (the Weighted Additive, or WADD, heuristic). Unlike Take The Best or Tallying, WADD scales the contribution of each feature by its validity. Furthermore, features are processed as bipolar evidence: the presence of a feature adds to the option's value proportionally to its validity, while its absence actively penalizes the option. The resulting option scores are translated into choice probabilities via a softmax function, with an independent lapse rate accounting for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # WADD (Weighted Additive) heuristic.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    
    # Center the stimulus features to -1 and 1 so absence of a feature penalizes
    stim_centered = stim * 2.0 - 1.0
    
    # Compute the weighted sum of features for each option.
    # The weights are directly proportional to the cue validities.
    scores = np.dot(stim_centered, val)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


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
**Validities (n_features=5):** [0.95, 0.85, 0.55, 0.5, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To quantitatively dissociate WADD from Tallying, this design uses 5 features with a steep drop-off in validities. WADD computes a weighted sum of the features based on their validities, whereas Tallying simply counts the number of features favoring each option, ignoring validities. By pitting a small number of high-validity features against a larger number of low-validity features, we create clear conflicts. For instance, in trials where Option A possesses the top two features and Option B possesses the bottom three, WADD strongly favors Option A while Tallying favors Option B. We also include trials where Tallying predicts a tie (equal number of wins) but WADD makes a strong deterministic prediction, and baseline trials where both heuristics agree to ensure data quality.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8303078137332282, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8450802512212142, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8291666666666667, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.819047619047619, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13747054202670855, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13043478260869565, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.8546296296296296, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.8617283950617284, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.7611386138613861, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.7583025830258303, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.2328042328042328, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.22727272727272727, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.16793168880455409, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17618469015795868, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15034965034965034, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.13228346456692913, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|0': 0.13085234093637454, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|1': 0.1769825918762089, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8602693602693603, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8611111111111112}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0


### RESULTS
- Predicted under pi_3 (simulated): 0.0008 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0579 (var=0.0005)
- Observed on real data: 0.2566 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Tallying from WADD using 6 features. Tallying counts unweighted feature-wise wins, ignoring validities and ties. WADD computes a validity-weighted sum where the absence of a feature actively penalizes the option's score. The trials are constructed to create specific dissociations: (1) WADD Tie / Tallying Strong: Option A has the two most valid features, while Option B has three lower-validity features whose combined weights exactly equal A's, resulting in a perfect tie for WADD but a clear win for Tallying. (2) Opposite Predictions: Option A has the top two features, while Option B has the bottom three. The penalty for missing features makes WADD favor A, while Tallying favors B's sheer quantity of wins. (3) Tallying Tie / WADD Strong: Both options win on an equal number of features, leading Tallying to guess, but WADD strongly prefers the option with higher-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 1))|0': 0.8378978534418949, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 1))|1': 0.8474877723432637, '((0, 0, 0, 1, 0, 0), (1, 1, 1, 0, 0, 0))|0': 0.8636604774535809, '((0, 0, 0, 1, 0, 0), (1, 1, 1, 0, 0, 0))|1': 0.8693877551020408, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.15860517435320584, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.15806805708013172, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.8434393638170974, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8457178841309824, '((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 0, 0))|0': 0.5163297045101088, '((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 0, 0))|1': 0.5230400957510473, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|0': 0.5212169735788631, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|1': 0.49829351535836175, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0))|0': 0.1387252276379218, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0))|1': 0.13791113675706868, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.14910262310170272, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.16398037841625787}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0


### RESULTS
- Predicted under pi_3 (simulated): 0.0874 (var=0.0014)
- Predicted under pi_2 (simulated): 0.0025 (var=0.0001)
- Observed on real data: 0.2497 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.12373831775700934, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.16864864864864865, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8707455429497569, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.842756183745583, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13971830985915493, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13315068493150686, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.14070162202942285, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.15911485774499473, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.1434994379917572, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.1439312567132116, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.862147406733394, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8509272467902995, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14093959731543623, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.15824175824175823, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.12565812565812565, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17738359201773837}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2326 (var=0.0093)
**Predicted under pi_3:** 0.2387 (var=0.0045)
**Predicted under pi_2:** 0.2322 (var=0.0071)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.15132924335378323, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.15049226441631505, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8864696734059098, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8513513513513513, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.1476923076923077, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.1409090909090909, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.5088640840446487, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.5267958950969214, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.14424173318129987, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.16408668730650156, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.8602006688963211, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.8486187845303867, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8741418764302059, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8512396694214877, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5034550839091807, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.4830569574621485, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.12566641279512566, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.15547378104875806, '((1, 1, 0, 0, 1), (1, 0, 1, 1, 0))|0': 0.504950495049505, '((1, 1, 0, 0, 1), (1, 0, 1, 1, 0))|1': 0.49954001839926404, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.4824976348155156, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.47282204020848845, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.13087248322147652, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.13186813186813187}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2312 (var=0.0009)
**Predicted under pi_3:** 0.0006 (var=0.0001)
**Predicted under pi_2:** 0.0009 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, both WADD (THEORY 1) and Tallying (THEORY 2) fail to consistently capture the observed human behavior. In Experiments 1, 2, and 4, the observed Jensen-Shannon Divergence (JSD) from the reference models remains consistently around 0.23-0.25. However, both theories produce simulated JSD values that wildly fluctuate depending on the specific trial design (often predicting values near 0.00, meaning they expect the data to perfectly match their deterministic/softmax predictions, which it does not). The consistent ~0.24 divergence suggests humans are using a different strategy entirely, one that is not captured by either a fully compensatory weighted sum (WADD) or an unweighted count (Tallying).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Tallying theory with the 'Take-The-Best' (TTB) heuristic. Under TTB, decision-makers do not integrate all available information. Instead, they search through the features in descending order of their validities. The search stops at the first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0), and the decision-maker chooses the option favored by that feature. If no features discriminate, they guess randomly. This non-compensatory, sequential search mechanism provides a strong alternative to the exhaustive integration mechanisms of both WADD and Tallying, and may better explain the robust behavioral deviations observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 10401,
  "candidates_token_count": 350,
  "total_token_count": 11484
}
```
