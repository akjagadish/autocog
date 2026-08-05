# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
People make decisions by sequentially accumulating evidence from cues in order of their validity. Evidence is weighted by the log-odds of each cue's validity. The accumulation stops and a choice is made as soon as the absolute difference in accumulated evidence between the two options exceeds an internal threshold. A low threshold leads to frugal 'Take The Best' behavior, while a high threshold leads to compensatory 'Weighted Additive' behavior. This unified mechanism accounts for varying degrees of cue integration depending on task context and individual differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Monotonic transformation of validities to log-odds
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Cues are consulted in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    a, b = stim[0], stim[1]
    score_a = 0.0
    score_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        score_a += weights[j] * a[j]
        score_b += weights[j] * b[j]
        diff = abs(score_a - score_b)
        # Stop if the accumulated difference exceeds the threshold
        # (and ensure we don't stop on a zero difference if threshold is 0)
        if diff >= threshold and diff > 1e-6:
            break

    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_3
People integrate all available evidence by weighting each feature according to its validity. Specifically, they compute a weighted sum of the features for each option, where the weights are the log-odds of the cue validities. This allows for compensatory decision making, where multiple weaker cues can jointly override a single stronger cue. Choice probabilities are generated via a softmax function over these weighted sums, accommodating response noise, along with an independent lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Monotonic transformation of validities to log-odds
    # Clip to avoid division by zero or infinite weights
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Calculate weighted additive score for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.88, 0.82, 0.76, 0.71, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the sequential evidence accumulation model from the Weighted Additive (WADD) model, we use 5 features with validities designed such that the log-odds weight of the best cue is smaller than the sum of the remaining cues' weights, and similarly for subsequent cues. Critical trials contrast an option favored by the single best available discriminating cue against an option favored by a combination of multiple lower-validity cues. Because WADD strictly integrates all cues, it predicts the option with the higher weighted sum of all features. The advocated sequential model, however, can stop accumulating evidence once an internal threshold is reached. By fitting a threshold, the sequential model can predict choices aligned with early stopping (non-compensatory behavior) that WADD inherently cannot capture, as well as varying degrees of partial integration. The inclusion of agreement trials ensures robust estimation of baseline noise and threshold parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.7100725952813067, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.7435530085959885, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.25595601710445937, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.32603158430973, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.7371571072319202, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.7088803088803088, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.14108187134502925, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.17939814814814814, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.14198557958957295, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15247634947134112, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|0': 0.1446099912357581, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|1': 0.1646433990895296, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.7527333894028595, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.7945990180032734, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.3818286371477861, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.23911875335840946}
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
- Predicted under pi_4 (simulated): 0.0004 (var=0.0013)
- Predicted under pi_3 (simulated): 0.0056 (var=0.0002)
- Observed on real data: 0.1447 (var=0.0046)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.92, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To demonstrate that decision-making relies on compensatory weighted addition (WADD) rather than sequential early-stopping, we construct trial pairs where the difference in evidence from the most valid cues strongly favors Option A, but the evidence from lower-validity cues systematically varies in support of Option B. A sequential model with a finite threshold predicts that once the evidence difference threshold is crossed by the top cues, the remaining lower-validity cues are ignored, resulting in identical choice probabilities across these variations (a step-function response). WADD, however, predicts that every cue continuously contributes to the final weighted sum, producing a smooth, graded shift in choice probabilities. By observing sensitivity to the lowest-validity cues even when the top cues provide strong evidence, we can rule out the threshold-based stopping rule and support continuous integration.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|0': 0.13136574074074073, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|1': 0.1337448559670782, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.13305395495023573, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.11378002528445007, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.7671232876712328, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.7616033755274262, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.13731495661051557, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.17273954116059378, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.12041884816753927, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17959183673469387, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8181390977443609, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.7814685314685315, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.15607524148449414, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.17871759890859482, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.13064030941125912, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.2037533512064343, '((1, 1, 0, 0, 0), (0, 0, 1, 0, 0))|0': 0.1125765426283561, '((1, 1, 0, 0, 0), (0, 0, 1, 0, 0))|1': 0.13518197573656845, '((1, 1, 0, 0, 0), (0, 0, 0, 0, 0))|0': 0.12721049629207073, '((1, 1, 0, 0, 0), (0, 0, 0, 0, 0))|1': 0.13621964097148892}
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
- Predicted under pi_4 (simulated): 0.0033 (var=0.0004)
- Predicted under pi_3 (simulated): 0.0003 (var=0.0001)
- Observed on real data: 0.2003 (var=0.0050)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.1458546571136131, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.1589895988112927, '((1, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8495702005730659, '((1, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8625226860254084, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.16703296703296702, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.14675615212527965, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.8453101361573374, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.8483263598326359, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8259762308998302, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8323699421965318, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.8504672897196262, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8556990454800674, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.1448481831757093, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.1583909490886235, '((1, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.16117764471057885, '((1, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.13972431077694236}
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

**Observed (real) value:** 0.0720 (var=0.0033)
**Predicted under pi_4:** 0.0657 (var=0.0032)
**Predicted under pi_3:** 0.1052 (var=0.0021)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.13289658399625642, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.1367053998632946, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.482837528604119, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.49584971603320227, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.5096097845078625, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.49814126394052044, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.8581730769230769, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8388429752066116, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.8467995802728226, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.8524203069657615, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.15416451112260735, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.14697060587882424, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8478802992518704, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8612224448897795, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.144905273937532, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.14179658500371195}
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

**Observed (real) value:** 0.1803 (var=0.0062)
**Predicted under pi_4:** 0.1152 (var=0.0041)
**Predicted under pi_3:** 0.0978 (var=0.0010)

### Experiment 5
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8371659415786202, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.859447567831826, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8092909535452323, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.855739276300024, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.1607806691449814, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.14261555806087936, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8188010899182562, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.8620037807183365, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8426698450536353, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8707110890104426}
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

**Observed (real) value:** 0.1151 (var=0.0017)
**Predicted under pi_4:** 0.1002 (var=0.0035)
**Predicted under pi_3:** 0.1371 (var=0.0024)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.16258919469928645, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.15760441292356187, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8594682582745523, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8654041258031789, '((1, 0, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8035264483627204, '((1, 0, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8275217932752179, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.5700483091787439, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5649402390438247, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8424556213017751, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8700696055684455, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8271474019088016, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8685506870624838}
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

**Observed (real) value:** 0.2052 (var=0.0083)
**Predicted under pi_4:** 0.0167 (var=0.0121)
**Predicted under pi_3:** 0.0006 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, both Theory 1 (sequential evidence accumulation with a threshold) and Theory 2 (weighted additive) consistently underpredict the divergence (JSD) observed in the real data. The real data shows significantly more variance and deviation from the reference profiles than either model predicts, particularly in Experiments 1, 2, 4, and 6. Theory 1 performs slightly better on Experiments 3 and 5, but neither theory adequately captures the true complexity of human choices. The sequence-aware JSD metric indicates there might be trial-to-trial dependencies or a mixture of strategies that neither a pure WADD nor a static-threshold sequential model captures.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that incorporates a mixture of decision strategies or trial-to-trial variability in strategy execution. For example, a 'Strategy Selection' theory where subjects probabilistically switch between a compensatory strategy (like WADD or Tallying) and a non-compensatory strategy (like Take-The-Best) depending on the difficulty of the trial or previous outcomes. Alternatively, introduce a noisy evidence accumulation model (like a Drift Diffusion Model for discrete cues) where the accumulation process itself is subject to moment-to-moment noise, naturally producing the higher variance and sequence-aware divergence seen in the real data."
}
```

## Usage

```json
{
  "prompt_token_count": 11964,
  "candidates_token_count": 301,
  "total_token_count": 12945
}
```
