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
Decision-makers use a Weighted Additive (WADD) strategy, evaluating each option by summing its features weighted by their respective validities, and choose the option with the highest weighted sum. Choices are subject to softmax noise and independent lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted additive sums
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


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
**Validities (n_features=5):** [0.95, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities. Tallying simply counts the number of features each option wins, ignoring the validities entirely. WADD weights each feature by its validity. We include trials where WADD strongly prefers an option with fewer winning features because it wins on the highest validity cues (e.g., A wins on the top 2 features, B wins on the bottom 3 features). We also include trials where Tallying predicts indifference (a 2-vs-2 tie in feature wins), but WADD has a strong preference due to the specific validities of the winning features. This ensures a broad range of predicted choice probabilities that sharply separate the two models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.26369728025226646, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.29727187206020694, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.6797778806108283, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.6608756080611535, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.22816114878340646, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.2598353156450137, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.32966136270909835, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.35422106179286333, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1391884951206985, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1355111917725348, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.3110687022900763, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.3377551020408163, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.18157181571815717, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.2274401473296501, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.600963644327639, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.5854214123006833}
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
- Predicted under pi_3 (simulated): 0.0003 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0492 (var=0.0006)
- Observed on real data: 0.1453 (var=0.0027)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [1.0, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from Weighted Additive (WADD), we use a 5-feature design with an extreme drop-off in validities (one cue at 1.0, and the rest at 0.5). Tallying ignores validities entirely and simply counts the number of features each option wins. WADD evaluates options by their validity-weighted sums. This allows us to create trials where Tallying strictly prefers an option (because it wins on three low-validity cues) while WADD strongly prefers the other (because it wins on the single highest-validity cue plus one low-validity cue). We also include trials where WADD is indifferent but Tallying has a strong preference, and vice versa. This full spectrum of dissociations provides a sharp contrast between the two models' predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8586609989373007, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8712757830404889, '((0, 1, 0, 0, 0), (1, 0, 1, 0, 0))|0': 0.8609756097560975, '((0, 1, 0, 0, 0), (1, 0, 1, 0, 0))|1': 0.8570909090909091, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.15002239140170176, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.14257888585897935, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|0': 0.5017473789316026, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|1': 0.49553092599213444, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.49635036496350365, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.4961554026709834, '((0, 1, 1, 1, 0), (1, 0, 0, 1, 1))|0': 0.4830261881668283, '((0, 1, 1, 1, 0), (1, 0, 0, 1, 1))|1': 0.49488677867056247}
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
- Predicted under pi_3 (simulated): 0.0559 (var=0.0009)
- Predicted under pi_2 (simulated): 0.0003 (var=0.0001)
- Observed on real data: 0.0896 (var=0.0017)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8565906455353691, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8373248983280615, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.15898767034393252, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14881865602945687, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1520648398301814, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16161158895427796, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8453125, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8523255813953489, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.15064102564102563, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.1757028112449799, '((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.8585072515033605, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.8393309680689306}
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

**Observed (real) value:** 0.2566 (var=0.0090)
**Predicted under pi_3:** 0.1758 (var=0.0032)
**Predicted under pi_2:** 0.2040 (var=0.0039)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.1402002861230329, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.1497695852534562, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.5043969849246231, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.48375451263537905, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.5103884372177055, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.5065913370998116, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8472344161545216, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8648302370275465, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.16136919315403422, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1365568544102019, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.847210994341148, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8530416951469583, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.12895927601809956, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.1579330422125182, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.494475138121547, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.4640449438202247, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.8562322946175638, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.854251012145749, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.5150684931506849, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.4637096774193548}
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

**Observed (real) value:** 0.0866 (var=0.0009)
**Predicted under pi_3:** 0.0192 (var=0.0003)
**Predicted under pi_2:** 0.0003 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither the Weighted Additive (WADD, Theory 1) nor the Tallying (Theory 2) model fully captures the observed data. In Experiment 1, the real data deviates substantially from WADD's predictions (JSD to self-reference = 0.1453 vs. WADD's 0.0003). In Experiment 2 and 4, the real data deviates from Tallying's predictions as well. The systematic deviations suggest that human decision-makers are not purely compensatory (as in WADD) nor purely unweighted (as in Tallying). A classic alternative in this domain is a non-compensatory lexicographic strategy, which could better explain the specific choice probabilities observed when validities drop off steeply.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Take The Best' (Lexicographic) theory. In this theory, decision-makers inspect features sequentially in descending order of their validities. They stop at the first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0) and choose the option favored by that cue. If no cues discriminate, they guess randomly. This non-compensatory heuristic provides a stark structural contrast to both WADD (compensatory, weighted) and Tallying (compensatory, unweighted), and is well-supported by literature on bounded rationality."
}
```

## Usage

```json
{
  "prompt_token_count": 8780,
  "candidates_token_count": 328,
  "total_token_count": 9918
}
```
