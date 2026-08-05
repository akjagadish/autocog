# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_6
Probabilistic Take-The-Best (PTTB): Decision-makers search through features sequentially to find the first one that discriminates between options. However, rather than searching in a strict deterministic order based on validities, the search order is stochastic. Features are sampled with probabilities proportional to an exponential function of their validities, allowing for noise and slight deviations from strict TTB. The search stops at the first discriminating feature.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating features
    diff = a - b
    D = diff != 0
    
    if not np.any(D):
        p_core = np.array([0.5, 0.5])
    else:
        # Under a Plackett-Luce sequential sampling model where items are sampled 
        # proportional to w_i, the probability that a specific item in D is sampled 
        # *first* among all items in D is exactly w_i / sum(w_j for j in D).
        v_D = validities[D]
        
        # Max-subtraction for numerical stability
        w = np.exp(gamma * (v_D - np.max(v_D)))
        w_sum = np.sum(w)
        
        # Probability of choosing A is the sum of weights of features favoring A, divided by w_sum
        p_A = np.sum(w[diff[D] > 0]) / w_sum
        p_B = np.sum(w[diff[D] < 0]) / w_sum
        p_core = np.array([p_A, p_B])
        
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate strict Take-The-Best (TTB) from Probabilistic Take-The-Best (PTTB), we exploit PTTB's stochastic sampling of discriminating features. TTB strictly terminates search at the single highest-validity discriminating feature, choosing the option it favors deterministically. In contrast, PTTB samples features proportionally to an exponential function of their validities. By creating trials where Option A is favored by the single highest-validity discriminating feature, but Option B is favored by multiple slightly lower-validity features, TTB will deterministically choose Option A. However, PTTB will assign a significant probability to sampling one of the multiple lower-validity features first, leading to a probabilistic preference that can even favor Option B depending on the noise parameter (gamma). We vary the number and validities of the opposing features across trials to trace the divergence between deterministic one-reason decision making and stochastic multi-feature sampling.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|0': 0.1356390977443609, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|1': 0.22608695652173913, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|0': 0.13323353293413173, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|1': 0.225, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.1313821138211382, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.2, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.13957649865791827, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.19378427787934185, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.1411800120409392, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.21453287197231835, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.1379519902765117, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.20689655172413793, '((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13615870153291254, '((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.20593368237347295}
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
- Predicted under pi_3 (simulated): 0.0006 (var=0.0001)
- Predicted under pi_6 (simulated): 0.0200 (var=0.0014)
- Observed on real data: 0.0006 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.88, 0.86, 0.84, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Probabilistic Take-The-Best (PTTB) from strict Take-The-Best (TTB), we exploit PTTB's stochastic sampling mechanism, which is proportional to an exponential function of the feature validities. Strict TTB deterministically selects the option favored by the single highest-validity discriminating feature. By using a set of validities with a very shallow gradient at the top (e.g., 0.90, 0.88, 0.86, 0.84) and a drop-off at the bottom, we can construct 'stochastic crossover' trials. In these trials, Option A is favored by the single highest-validity feature (e.g., 0.90), while Option B is favored by multiple features with only slightly lower validities (e.g., 0.88, 0.86, 0.84). TTB will deterministically predict Option A. However, because the validities are so close, PTTB will assign significant sampling weight to the features favoring Option B. For a wide range of its noise parameter (gamma), the sum of the weights for Option B's features will exceed the weight for Option A's single best feature, leading PTTB to predict a preference for Option B. We vary the number of opposing features to map out the divergence between deterministic single-feature reliance and probabilistic multi-feature sampling.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.5533216783216783, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.5788216560509554, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.46537530266343824, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.4625228519195612, '((0, 0, 1, 0, 1), (1, 1, 0, 1, 0))|0': 0.7387571545380213, '((0, 0, 1, 0, 1), (1, 1, 0, 1, 0))|1': 0.7353440951571792, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.35082393755420643, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.37289494787489974, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.4914645103324349, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.5017590149516271, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.5122059476253884, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.5347467608951708}
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
- Predicted under pi_3 (simulated): 0.0513 (var=0.0016)
- Predicted under pi_6 (simulated): 0.0004 (var=0.0001)
- Observed on real data: 0.0491 (var=0.0020)

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
**Predicted under pi_6:** 0.1041 (var=0.0028)

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
**Predicted under pi_6:** 0.0233 (var=0.0009)

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
**Predicted under pi_6:** 0.0042 (var=0.0006)

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
**Predicted under pi_6:** 0.0823 (var=0.0044)

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
**Predicted under pi_6:** 0.0040 (var=0.0009)

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
**Predicted under pi_6:** 0.0140 (var=0.0006)

### Experiment 9
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0005 (var=0.0001)
**Predicted under pi_3:** 0.0008 (var=0.0001)
**Predicted under pi_6:** 0.0245 (var=0.0007)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.2045 (var=0.0034)
**Predicted under pi_3:** 0.1877 (var=0.0044)
**Predicted under pi_6:** 0.1041 (var=0.0032)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 10 experiments, the observed data aligns remarkably well with the predictions of Theory 1 (Strict Take-The-Best, pi_3). In Experiment 1 (designed to favor pi_3), the observed metric is 0.0006, exactly matching pi_3's prediction. In Experiment 2 (designed to favor pi_6), the observed metric is 0.0491, which is very close to pi_3's prediction of 0.0513 and far from pi_6's prediction of 0.0004. This pattern holds across all other experiments, where pi_3's predicted JSD consistently mirrors the observed data, while pi_6 (Probabilistic TTB) consistently fails to capture the true distribution. The data strongly supports a strict, non-compensatory decision rule over a probabilistic feature-sampling approach.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the degenerate Probabilistic TTB theory with a fully compensatory Weighted Additive (WADD) model. While TTB is strictly non-compensatory (stopping at the first discriminating feature), WADD integrates all available information by computing a weighted sum of feature values multiplied by their validities. Choices are then generated by passing the integrated values through a softmax function. Proposing WADD will provide a strong, classic compensatory baseline to contrast against the highly successful non-compensatory TTB model."
}
```

## Usage

```json
{
  "prompt_token_count": 18366,
  "candidates_token_count": 325,
  "total_token_count": 19979
}
```
