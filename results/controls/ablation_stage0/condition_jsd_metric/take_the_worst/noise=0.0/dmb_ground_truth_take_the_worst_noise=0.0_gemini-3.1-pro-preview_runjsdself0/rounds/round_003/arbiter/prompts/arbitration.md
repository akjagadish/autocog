# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Decision-makers use a Compensatory Weighted Additive (WADD) strategy where they integrate all available cues, weighting each by its validity centered around chance (validity - 0.5). This linear integration allows multiple weaker cues to appropriately accumulate and potentially override a single strong cue, avoiding the extreme dominance that log-odds weighting can cause. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Extract validities
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Linear weighting centered around chance (0.5) to prevent extreme weights
    weights = val - 0.5
    
    # Weighted additive sums
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) from Tallying, we use a 5-feature design with linearly decreasing validities. WADD centers validities around 0.5 to compute weights (0.4, 0.3, 0.2, 0.1, 0.0), whereas Tallying simply counts the number of winning features for each option, completely ignoring validities. We design trials where WADD predicts indifference but Tallying strongly prefers one option (e.g., A wins on the highest validity cue while B wins on three lower validity cues). We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD strongly prefers one option due to higher validity weights. Finally, we include trials where the models make strictly opposing predictions, creating a robust quantitative double dissociation between the compensatory validity-weighted integration of WADD and the unweighted feature counting of Tallying.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.17096165933375235, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.21396054628224584, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.49543639475185397, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.48299845440494593, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.49304377332880894, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.5013491635186185, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.22198603097479502, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.26808228268082285, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.2222222222222222, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.25272727272727274, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.30790872147546106, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.31667707682698315}
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
- Predicted under pi_5 (simulated): 0.0006 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0936 (var=0.0019)
- Observed on real data: 0.1608 (var=0.0034)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]

**Rationale:** To quantitatively dissociate Tallying from WADD, we use a 6-feature design with two highly valid cues and four weakly valid cues. WADD translates these into weights (val - 0.5): [0.5, 0.4, 0.1, 0.1, 0.1, 0.1]. Tallying simply counts the number of winning features, completely ignoring the validities. We design trials where WADD strongly prefers an option because it wins on the top two features, while Tallying prefers the other option because it wins on the four weaker features. We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD has a strict preference, and trials where WADD predicts a tie (equal weighted sum) but Tallying has a strict preference. This ensures a robust double dissociation between unweighted feature counting and compensatory validity-weighted integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|0': 0.8675363773206222, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|1': 0.8643434971997511, '((0, 1, 1, 1, 1, 0), (1, 0, 0, 0, 0, 1))|0': 0.1512301013024602, '((0, 1, 1, 1, 1, 0), (1, 0, 0, 0, 0, 1))|1': 0.13210099188458072, '((1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 0))|0': 0.8493231312536786, '((1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 0))|1': 0.8616517622304051, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.8547832071576049, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.8695854680950164, '((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1))|0': 0.506993006993007, '((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1))|1': 0.48770053475935826, '((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0))|0': 0.14600409836065573, '((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0))|1': 0.14381067961165048, '((1, 1, 0, 0, 1, 0), (0, 0, 1, 1, 0, 1))|0': 0.5154083204930663, '((1, 1, 0, 0, 1, 0), (0, 0, 1, 1, 0, 1))|1': 0.5013032145960035, '((1, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1))|0': 0.5055849500293945, '((1, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1))|1': 0.49025803054239075}
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
- Predicted under pi_5 (simulated): 0.0622 (var=0.0007)
- Predicted under pi_2 (simulated): 0.0003 (var=0.0001)
- Observed on real data: 0.0624 (var=0.0010)

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
**Predicted under pi_5:** 0.0905 (var=0.0010)
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
**Predicted under pi_5:** 0.0900 (var=0.0022)
**Predicted under pi_2:** 0.0003 (var=0.0001)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.1453 (var=0.0027)
**Predicted under pi_5:** 0.0222 (var=0.0003)
**Predicted under pi_2:** 0.0492 (var=0.0006)

### Experiment 6
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0896 (var=0.0017)
**Predicted under pi_5:** 0.1143 (var=0.0037)
**Predicted under pi_2:** 0.0003 (var=0.0001)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.14216867469879518, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.15532646048109966, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.15469613259668508, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.15073272854152128, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8447457627118644, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8497959183673469, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8611491108071135, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.8594507269789984, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.15014436958614052, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.14148103552077063, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.8241758241758241, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.8639021796916534, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14147018030513175, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.14308426073131955, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.8389261744966443, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8534768211920529, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.14962251201098148, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.1568785197103781, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8593523441275979, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.8193343898573693}
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

**Observed (real) value:** 0.2225 (var=0.0052)
**Predicted under pi_5:** 0.1368 (var=0.0019)
**Predicted under pi_2:** 0.2370 (var=0.0044)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0, 1, 1), (0, 1, 0, 1, 0, 0))|0': 0.12998137802607077, '((1, 1, 1, 0, 1, 1), (0, 1, 0, 1, 0, 0))|1': 0.13947990543735225, '((0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 0, 0))|0': 0.4952642835319279, '((0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 0, 0))|1': 0.5225933202357563, '((1, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0))|0': 0.14586799778147533, '((1, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0))|1': 0.13480146813480146, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1))|0': 0.8642156862745098, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1))|1': 0.8666666666666667, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|0': 0.4955338153977031, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|1': 0.4932526756630991, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.8539269634817409, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.8593359514459121}
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

**Observed (real) value:** 0.0192 (var=0.0004)
**Predicted under pi_5:** 0.0826 (var=0.0009)
**Predicted under pi_2:** 0.0004 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither Theory 1 (WADD with linear weights) nor Theory 2 (Tallying) consistently explains the observed data. In some experiments (like Experiment 2 and 4), the observed data closely matches the predictions of Theory 1, where subjects seem to weight cues by their validities rather than just counting them. In other experiments (like Experiment 8), the observed data is much closer to Theory 2, suggesting subjects might be ignoring validities and relying on a simple tally of feature wins. This divergence indicates that a single fixed heuristic (either strictly compensatory or strictly unweighted) is insufficient. However, Theory 2's strict unweighted tallying fails dramatically in environments with highly skewed validities (like Exp 2, where validities are 1.0 and 0.9 vs 0.6). A more robust non-compensatory heuristic is 'Take-The-Best' (TTB), which relies on validities to order cues but does not integrate them, potentially bridging the gap by acting like WADD when top validities dominate, and acting differently when they don't.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Tallying theory with the 'Take-The-Best' (TTB) heuristic. In TTB, decision-makers do not integrate all features (unlike WADD) nor do they treat all features equally (unlike Tallying). Instead, they search through features in descending order of their validities. The first feature that discriminates between the two options determines the choice, and all subsequent features are ignored. If no features discriminate, the subject guesses. Response noise can be incorporated as an overall lapse rate (epsilon) where subjects guess randomly. This introduces a lexigraphic decision rule that strongly contrasts with both WADD and Tallying, while remaining a classic, psychologically plausible heuristic."
}
```

## Usage

```json
{
  "prompt_token_count": 15360,
  "candidates_token_count": 411,
  "total_token_count": 17457
}
```
