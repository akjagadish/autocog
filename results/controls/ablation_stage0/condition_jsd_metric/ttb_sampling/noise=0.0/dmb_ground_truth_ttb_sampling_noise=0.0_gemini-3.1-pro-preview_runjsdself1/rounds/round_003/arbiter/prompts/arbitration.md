# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) heuristic: Individuals make decisions by sequentially searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the choice is based entirely on that single feature, ignoring all others. If no feature discriminates, a random guess is made.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            scores = np.array([0.0, 1.0])
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_5
Weighted Additive (WADD) Model with Log-Odds: Decision-makers evaluate options using a purely compensatory strategy. They compute an overall value for each option by summing the log-odds of the validities for all features where the option has a positive cue. A choice is then made probabilistically based on the difference in these overall values, using a softmax rule combined with a uniform lapse rate to account for decision noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities into log-odds weights
    v = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v / (1.0 - v))
    
    # Compute the overall value (weighted sum) for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.75, 0.92, 0.85, 0.88, 0.7]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) Model, we use a 5-feature design with a non-monotonic validity array. The validities are chosen such that their log-odds follow a specific structure: the log-odds of any single feature is strictly less than the sum of the log-odds of all features with lower validities. This allows us to create a series of 'compensatory crossover' trials where TTB (which stops at the first discriminating feature in descending validity order) strictly prefers Option A, while WADD (which sums all log-odds weights) strictly prefers Option B. By varying the depth at which the first discriminating feature occurs, we can test the models' divergence across different levels of the validity hierarchy.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.14928732183045762, '((0, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.23845193508114856, '((0, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.149069884364002, '((0, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.20559610705596107, '((0, 1, 1, 1, 0), (1, 1, 0, 1, 1))|0': 0.16132215096201283, '((0, 1, 1, 1, 0), (1, 1, 0, 1, 1))|1': 0.20107238605898123, '((1, 1, 1, 1, 0), (0, 1, 1, 1, 1))|0': 0.15232495991448422, '((1, 1, 1, 1, 0), (0, 1, 1, 1, 1))|1': 0.23087071240105542, '((0, 1, 0, 0, 1), (1, 0, 1, 1, 0))|0': 0.14995083579154375, '((0, 1, 0, 0, 1), (1, 0, 1, 1, 0))|1': 0.23633879781420766, '((0, 1, 0, 1, 0), (1, 1, 1, 0, 1))|0': 0.14196762141967623, '((0, 1, 0, 1, 0), (1, 1, 1, 0, 1))|1': 0.2394904458598726}
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
- Predicted under pi_5 (simulated): 0.1587 (var=0.0025)
- Observed on real data: 0.0005 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) Model from the Take-The-Best (TTB) heuristic, we utilize a 6-feature design with a carefully calibrated validity gradient. The validities are chosen such that their log-odds follow a specific sub-additive structure: the log-odds of any single feature is strictly less than the sum of the log-odds of the next two immediately following features. This allows us to construct a series of 'compensatory crossover' trials where TTB (which strictly follows the first discriminating feature in descending validity order) consistently prefers Option A, while WADD (which sums all log-odds weights) consistently prefers Option B, because the combined weight of two or more lower-validity features outweighs the single highest-validity feature. By varying the position of the first discriminating feature across trials, we can test this divergence at every level of the validity hierarchy.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.7506666666666667, '((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8168421052631579, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.764763779527559, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.8215944272445821, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|0': 0.7758186397984886, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|1': 0.849607982893799, '((0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))|0': 0.49940968122786306, '((0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))|1': 0.5266981474754813, '((1, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|0': 0.7324613555291319, '((1, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|1': 0.7934119560797072, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0))|0': 0.6412556053811659, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0))|1': 0.6528803545051699, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.7929125138427464, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.8413051538746755, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0))|0': 0.8025974025974026, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0))|1': 0.8706713780918728}
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
- Predicted under pi_3 (simulated): 0.1877 (var=0.0044)
- Predicted under pi_5 (simulated): 0.0005 (var=0.0001)
- Observed on real data: 0.2045 (var=0.0034)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5089498806682577, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5178571428571429, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.17070484581497797, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.14742152466367714, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.15504358655043587, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.15195586760280844, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.48748293126991354, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.4925160370634355, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.4931506849315068, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.4925986842105263, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8503787878787878, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8474462365591398, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.5023847376788553, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.4997081144191477, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8523111612175873, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.848302300109529}
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

**Observed (real) value:** 0.1872 (var=0.0074)
**Predicted under pi_3:** 0.1725 (var=0.0055)
**Predicted under pi_5:** 0.1843 (var=0.0034)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.6349115801483172, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.6096372495939361, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|0': 0.3445139283683911, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|1': 0.3454644215100489, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.4446428571428571, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.5481481481481482, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.5600208768267223, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.42755344418052255, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.6502320185614849, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.5591684434968017, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.3051813471502591, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.49101796407185627, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.44503862150920975, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4282733437663015, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.6614420062695925, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.5}
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

**Observed (real) value:** 0.0567 (var=0.0012)
**Predicted under pi_3:** 0.0569 (var=0.0021)
**Predicted under pi_5:** 0.0548 (var=0.0014)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 1, 1, 0))|0': 0.14050822122571002, '((1, 0, 0, 1), (0, 1, 1, 0))|1': 0.16446955430006277, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.17253774263120059, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.15436849253055682, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.835639180962922, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.8326826547685443, '((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.85, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.8391089108910891, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8409255242227043, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8592692828146143, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.14977755808205634, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.14774889029803423, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8511904761904762, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8495762711864406, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.15014005602240896, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.15973597359735975}
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

**Observed (real) value:** 0.0015 (var=0.0001)
**Predicted under pi_3:** 0.0003 (var=0.0001)
**Predicted under pi_5:** 0.0836 (var=0.0006)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.45472703062583225, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.629297075423294, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.7783191230207065, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8400689995071464, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.5674967234600262, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.6655486344034499, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.7061978545887961, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.7849328692192938, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.7192262602579133, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.7954431647471207}
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

**Observed (real) value:** 0.1506 (var=0.0048)
**Predicted under pi_3:** 0.1473 (var=0.0034)
**Predicted under pi_5:** 0.0780 (var=0.0027)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|0': 0.1345585822120118, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|1': 0.1735985533453888, '((0, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|0': 0.13043478260869565, '((0, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|1': 0.20147874306839186, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|0': 0.13251454427925016, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|1': 0.16798418972332016, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.12548387096774194, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.194, '((1, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|0': 0.14308733483725428, '((1, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|1': 0.22334004024144868, '((1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 1, 1))|0': 0.14262295081967213, '((1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 1, 1))|1': 0.22181818181818183, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13987252124645894, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.21638655462184875, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13222079589216945, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.19421487603305784}
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

**Observed (real) value:** 0.0024 (var=0.0002)
**Predicted under pi_3:** 0.0015 (var=0.0002)
**Predicted under pi_5:** 0.0542 (var=0.0005)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 0, 0))|0': 0.5708358032009484, '((0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 0, 0))|1': 0.6063774176685833, '((1, 1, 1, 0, 0, 0), (1, 1, 0, 1, 1, 1))|0': 0.4607361963190184, '((1, 1, 1, 0, 0, 0), (1, 1, 0, 1, 1, 1))|1': 0.4025380710659898, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|0': 0.6022727272727273, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|1': 0.5747716894977168, '((0, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.8103298611111112, '((0, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.7932098765432098, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.20738820479585224, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.1973748176956733, '((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0))|0': 0.14415862808145766, '((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0))|1': 0.14705882352941177, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|0': 0.8484848484848485, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|1': 0.8550505050505051, '((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|0': 0.5021186440677966, '((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|1': 0.6478937728937729}
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

**Observed (real) value:** 0.0363 (var=0.0007)
**Predicted under pi_3:** 0.0311 (var=0.0008)
**Predicted under pi_5:** 0.0208 (var=0.0003)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across almost all experiments, Take-The-Best (THEORY 1, pi_3) provides a significantly better fit to the observed human data than the Weighted Additive Model (THEORY 2, pi_5). In Experiments 1, 5, 6, and 7, the observed data's divergence perfectly aligns with pi_3's predictions, while pi_5 predicts massive deviations. Most notably, even in Experiment 2 (which was designed by pi_5 to showcase compensatory behavior), the observed data strongly deviates from pi_5's prediction (0.2045 vs 0.0005) and instead closely matches pi_3's prediction (0.1877). This decisively indicates that subjects are employing non-compensatory, lexicographic decision rules rather than integrating all cues via weighted sums.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since the fully compensatory WADD model (THEORY 2) has been empirically defeated by the lexicographic TTB heuristic across multiple experiments, THEORY 2 should be completely replaced by another boundedly rational or heuristic-based theory. A strong competitor to strict TTB would be the Tallying heuristic (Unit-Weight Additive Model), which simply counts the number of positive features for each option while ignoring their validities. Alternatively, a Probabilistic Take-The-Best model, where the feature search order is stochastic but biased by validity (rather than strictly deterministic), could serve as a highly competitive alternative to account for the noise and slight deviations from strict TTB seen in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 15155,
  "candidates_token_count": 354,
  "total_token_count": 16954
}
```
