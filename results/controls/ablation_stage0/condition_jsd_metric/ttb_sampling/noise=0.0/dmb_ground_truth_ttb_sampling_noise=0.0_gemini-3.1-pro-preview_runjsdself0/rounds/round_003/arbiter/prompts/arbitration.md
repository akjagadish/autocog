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
Take-The-Best (TTB) heuristic: People employ a non-compensatory, lexicographic decision process. They evaluate features sequentially in descending order of their subjective validity. The very first feature that discriminates between the two options (i.e., one option has a positive feature value while the other does not) determines the choice, and all lower-validity features are strictly ignored. If no feature discriminates, they guess. Response noise is modeled via a softmax over the resulting binary preference and an independent random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Evaluate features sequentially
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to allow for noise
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_5
Weighted Additive (WADD) with Validity-Proportional Weights: Subjects make decisions by computing a compensatory weighted sum of features for each option. Unlike a naive tallying strategy, features are weighted by their subjective importance, which is directly derived from their known validities (e.g., raw validity, validity above chance, or log-odds). This allows multiple weak cues to overcome a single strong cue, contrasting with non-compensatory heuristics like Take-The-Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    validities = np.asarray(parameters["validities"], dtype=float)
    # Clip to avoid division by zero or log(0) for validities of 1.0 or 0.5
    v = np.clip(validities, 0.5001, 0.9999)
    
    wt = parameters["weight_type"]
    if wt == "log_odds":
        w = np.log(v / (1.0 - v))
    elif wt == "validity":
        w = v
    else:  # validity_minus_half
        w = v - 0.5
        
    # Compute weighted sum for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.8, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Take-The-Best (TTB) from Weighted Additive (WADD) using a 5-feature task with a gentle linear decrease in validities. TTB decides solely based on the first discriminating feature, ignoring all others. WADD, however, integrates all features weighted by their validity. The core dissociation trials pit the single most valid cue against a coalition of two or three slightly less valid cues. Because the validity drop-off is gentle, the sum of the weights of the lower-validity cues easily outweighs the single highest-validity cue for WADD, whereas TTB inflexibly follows the single best cue. Agreement trials are also included to measure baseline accuracy and noise.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1465648854961832, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.16428571428571428, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.14468503937007873, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.1485969387755102, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.15037593984962405, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.16276595744680852, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.866234756097561, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.8237704918032787, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1322049405306496, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17055655296229802, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.13901147396293026, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.1679160419790105, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.1448247078464107, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.1553156146179402, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.8531844499586435, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.8240270727580372}
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
- Predicted under pi_3 (simulated): 0.0004 (var=0.0001)
- Predicted under pi_5 (simulated): 0.1803 (var=0.0056)
- Observed on real data: 0.0041 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) theory from the Take-The-Best (TTB) heuristic by pitting the single most valid feature against a coalition of less valid features. The validities are chosen such that the highest validity feature is strong, but can be mathematically outweighed by the sum of the remaining features across all WADD weighting schemes (log-odds, raw validity, and validity-minus-half). TTB, being strictly non-compensatory, will invariably choose the option favored by the single most valid discriminating feature, whereas WADD will integrate the information and choose the option favored by the coalition of lower-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8193069306930693, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8622278056951423, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8321167883211679, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8466413181242078, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.7625621203754832, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.7730575740637228, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.14979195561719832, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.15477293790546803, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8448362720403023, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8357414448669201, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.17861885790172644, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.15090735434574976, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.160196828868234, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.16431394692264256, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8098891730605285, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8355995055624228}
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
- Predicted under pi_3 (simulated): 0.1761 (var=0.0040)
- Predicted under pi_5 (simulated): 0.0008 (var=0.0002)
- Observed on real data: 0.1572 (var=0.0035)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8623693379790941, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8461128860489883, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.505, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.50125, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.1423611111111111, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.14155982905982906, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.141643059490085, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.14114441416893733, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.845903165735568, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8489583333333334, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|0': 0.49377415199656505, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|1': 0.5137686860739575, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.1483957219251337, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.13878326996197718, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8460743801652892, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8719604863221885}
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

**Observed (real) value:** 0.1667 (var=0.0035)
**Predicted under pi_3:** 0.1580 (var=0.0031)
**Predicted under pi_5:** 0.0867 (var=0.0034)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.5470953186689227, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.6162409954158481, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.6204819277108434, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.5631528964862298, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.42092574734811955, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.40629095674967236, '((0, 0, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.5720338983050848, '((0, 0, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.5940420560747663, '((1, 0, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.4491150442477876, '((1, 0, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.36551339285714285, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|0': 0.15894039735099338, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|1': 0.16142341413099537, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|0': 0.8586224754013465, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|1': 0.8538046734571599, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.37244897959183676, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4596456692913386}
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

**Observed (real) value:** 0.0482 (var=0.0008)
**Predicted under pi_3:** 0.0573 (var=0.0013)
**Predicted under pi_5:** 0.0182 (var=0.0012)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.13922155688622753, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.1411637931034483, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13836239575435935, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.15904365904365905, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.13457556935817805, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.18233618233618235, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.86254851228978, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.7893700787401575, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.12878524190741386, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.15130674002751032, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.14434861552428507, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.16226071103008205, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.1320754716981132, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.21138211382113822, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.13057103064066852, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.17032967032967034}
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

**Observed (real) value:** 0.0022 (var=0.0001)
**Predicted under pi_3:** 0.0010 (var=0.0002)
**Predicted under pi_5:** 0.1457 (var=0.0079)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.511085626911315, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.5613553113553114, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.2484560570071259, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.21335807050092764, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.33142201834862384, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.35302197802197804, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.23064043532858935, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.23392783077561177, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8160059835452506, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8146754468485419, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.7480383609415867, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.7937443336355394}
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

**Observed (real) value:** 0.1305 (var=0.0041)
**Predicted under pi_3:** 0.1288 (var=0.0046)
**Predicted under pi_5:** 0.0094 (var=0.0006)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8606185567010309, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.8348936170212766, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.15697036223929747, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.14116985376827895, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.1360144600090375, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.16149963950973323, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|0': 0.14196502914238135, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|1': 0.15924276169265034, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|0': 0.14955640050697086, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|1': 0.14737883283877348, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8422504136789851, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.8522663682148853, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.15792349726775956, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13954802259887006, '((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8529188779378317, '((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8347193347193347}
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

**Observed (real) value:** 0.0003 (var=0.0002)
**Predicted under pi_3:** 0.0003 (var=0.0001)
**Predicted under pi_5:** 0.1147 (var=0.0022)

### Experiment 8
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.14180206794682423, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.14850223072020396, '((1, 0, 0, 1, 1), (1, 1, 0, 0, 0))|0': 0.15373493975903615, '((1, 0, 0, 1, 1), (1, 1, 0, 0, 0))|1': 0.14098360655737704, '((0, 0, 0, 1, 0), (0, 0, 0, 0, 1))|0': 0.49203314212874444, '((0, 0, 0, 1, 0), (0, 0, 0, 0, 1))|1': 0.5076317085179715, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8754696725711218, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8503166378814048, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 1))|0': 0.4786096256684492, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 1))|1': 0.49248554913294795, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.16560913705583757, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.15316205533596838, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8678474114441417, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8684497816593887, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8739545997610514, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8618899273104881}
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

**Observed (real) value:** 0.2299 (var=0.0072)
**Predicted under pi_3:** 0.2336 (var=0.0066)
**Predicted under pi_5:** 0.0427 (var=0.0017)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Take-The-Best, pi_3) consistently and accurately predicts the observed metric values. In contrast, Theory 2 (Weighted Additive, pi_5) systematically fails to match the observed data. The metrics measure the Jensen-Shannon Divergence between the data and a reference profile. In experiments proposed by pi_5 (where pi_5 predicts a JSD near 0), the observed JSD is consistently high and closely matches the high JSD predicted by pi_3. Conversely, in experiments proposed by pi_3, the observed JSD is near 0, matching pi_3's predictions. This strongly indicates that subjects are employing a non-compensatory lexicographic strategy rather than a compensatory weighted additive one.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since Take-The-Best (Theory 1) is highly successful, the Weighted Additive theory (Theory 2) should be completely replaced. Propose a new theory that represents a stronger competitor in the non-compensatory or bounded-rationality family. For instance, a 'Lexicographic Semiorder' heuristic (which acts like TTB but requires a minimum threshold of difference to discriminate), or a 'Tallying' heuristic (which counts the number of positive features ignoring validity weights), or a two-stage model that only uses WADD if TTB fails to yield a confident choice. This will provide a more nuanced contrast to the strict TTB model."
}
```

## Usage

```json
{
  "prompt_token_count": 15356,
  "candidates_token_count": 332,
  "total_token_count": 16683
}
```
