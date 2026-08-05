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
People use the 'Take The Best' (TTB) heuristic to make binary choices. They search through features in descending order of their subjective or stated validities. The first feature that discriminates between the two options (i.e., one option has a positive feature and the other does not) strictly determines the choice, ignoring all other features. If no features discriminate, they guess uniformly. Response noise is modeled via a simple lapse rate where the individual occasionally guesses randomly instead of following the deterministic rule.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


## THEORY 2 — pi_6
Take The Best with Validity-Dependent Confidence

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] != b[idx]:
            # Convert validity to log-odds weight
            v = np.clip(validities[idx], 0.501, 0.999)
            w = np.log(v / (1.0 - v))
            
            # Assign weight to the option that has the feature
            scores = np.zeros(2)
            if a[idx] > b[idx]:
                scores[0] = w
            else:
                scores[1] = w
                
            # Softmax to convert to probability
            z = beta * scores
            z -= np.max(z)
            e = np.exp(z)
            p_core = e / np.sum(e)
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.64, 0.73, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.896, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.8854054054054054, '((1, 1, 0, 1), (0, 0, 1, 0))|0': 0.12173263629574309, '((1, 1, 0, 1), (0, 0, 1, 0))|1': 0.15184381778741865, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.11198208286674133, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.11686879823594266, '((0, 1, 1, 1), (1, 0, 0, 1))|0': 0.890497737556561, '((0, 1, 1, 1), (1, 0, 0, 1))|1': 0.8633093525179856, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.13507377979568672, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.13166485310119697, '((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.11904761904761904, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.13513513513513514, '((1, 1, 1, 0), (1, 1, 0, 1))|0': 0.1497695852534562, '((1, 1, 1, 0), (1, 1, 0, 1))|1': 0.1376281112737921, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8603603603603603, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8809523809523809, '((0, 0, 0, 0), (1, 0, 1, 0))|0': 0.8689616879511383, '((0, 0, 0, 0), (1, 0, 1, 0))|1': 0.8854919399666481, '((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8782098312545855, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8627002288329519, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.12518628912071536, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.12008733624454149, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.8905742145178764, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.895096921322691, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.11517857142857142, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.125, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.128, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.14074074074074075, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.12944983818770225, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.12131519274376418}
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
- Predicted under pi_3 (simulated): 0.0010 (var=0.0001)
- Predicted under pi_6 (simulated): 0.0021 (var=0.0002)
- Observed on real data: 0.0021 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.73, 0.56, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 1), (1, 0, 1, 0))|0': 0.15174363807728558, '((1, 1, 0, 1), (1, 0, 1, 0))|1': 0.14479025710419485, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.11649580615097857, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.13892709766162312, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8733986435568952, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8583509513742071, '((0, 1, 1, 0), (1, 0, 1, 1))|0': 0.8860898138006572, '((0, 1, 1, 0), (1, 0, 1, 1))|1': 0.8669673055242391, '((1, 0, 1, 0), (1, 1, 1, 0))|0': 0.8724226804123711, '((1, 0, 1, 0), (1, 1, 1, 0))|1': 0.85546875, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8840579710144928, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8676900584795322, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.25306748466257667, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.22822299651567945, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.8912386706948641, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.8760842627013631, '((1, 1, 1, 0), (1, 1, 0, 1))|0': 0.23638470451911936, '((1, 1, 1, 0), (1, 1, 0, 1))|1': 0.2390608324439701, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.78159757330637, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.750924784217016, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22424242424242424, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.21666666666666667, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.2128146453089245, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.2451403887688985, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.14312796208530806, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.15704697986577182, '((1, 1, 0, 1), (0, 0, 1, 0))|0': 0.11736526946107785, '((1, 1, 0, 1), (0, 0, 1, 0))|1': 0.12020725388601036, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.21745788667687596, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.24062772449869224}
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
- Predicted under pi_3 (simulated): 0.0040 (var=0.0001)
- Predicted under pi_6 (simulated): 0.0011 (var=0.0003)
- Observed on real data: 0.0036 (var=0.0001)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.43703703703703706, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.5026143790849673, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.8568872987477638, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.8424336973478939, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.15728155339805824, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.17142857142857143, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.1486784140969163, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.1547085201793722, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.125, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1565217391304348, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8710166919575114, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8692033293697978, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8631578947368421, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8813008130081301, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8595360824742269, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8349609375, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.11976047904191617, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.1157243816254417, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.5056980056980057, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.5151515151515151, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.527147087857848, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4714104193138501, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8561643835616438, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8524229074889867, '((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8854961832061069, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8607021517553793, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.17066666666666666, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.1368421052631579, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8222222222222222, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8921810699588477}
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

**Observed (real) value:** 0.0748 (var=0.0005)
**Predicted under pi_3:** 0.0708 (var=0.0006)
**Predicted under pi_6:** 0.0689 (var=0.0006)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8401037165082109, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.80248833592535, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.6113207547169811, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.6108108108108108, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.34727061556329847, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.3184238551650692, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8583850931677018, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8170854271356784, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22069317023445464, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.23318385650224216, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.38506224066390043, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.3277310924369748, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.15520833333333334, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.17023809523809524, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.760061919504644, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.7381889763779528, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.1650114591291062, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.1955193482688391, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.14410134600158353, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.19366852886405958, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.7028301886792453, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.6401515151515151, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.20399666944213155, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.28046744574290483, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.3624242424242424, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.37333333333333335, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.148636763412489, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.16289592760180996}
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

**Observed (real) value:** 0.0238 (var=0.0003)
**Predicted under pi_3:** 0.0238 (var=0.0003)
**Predicted under pi_6:** 0.0209 (var=0.0002)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12193588937774984, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.18181818181818182, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8410087719298246, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8862612612612613, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8716814159292036, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8671875, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8713745271122321, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8917576961271102, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.11723329425556858, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.14064914992272023, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8592017738359202, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8775055679287305, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.1415929203539823, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.12388724035608309, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.11632100991884581, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.12590448625180897, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1254125412541254, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.10998877665544332, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.12875536480686695, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.1191904047976012, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.09761549925484352, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.13100436681222707, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8807649043869517, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8858397365532382, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.14576271186440679, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.1366120218579235, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8717948717948718, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8830255057167986, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.9044117647058824, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.8616071428571429}
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

**Observed (real) value:** 0.0037 (var=0.0003)
**Predicted under pi_3:** 0.0009 (var=0.0001)
**Predicted under pi_6:** 0.0019 (var=0.0001)

### Experiment 6
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8526829268292683, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8296774193548387, '((1, 1, 0, 0), (1, 0, 0, 1))|0': 0.40131578947368424, '((1, 1, 0, 0), (1, 0, 0, 1))|1': 0.40673076923076923, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.42889137737961924, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.38588754134509373, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.6492082825822169, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.5985699693564862, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.2508710801393728, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.234341252699784, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5834127740705434, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5845539280958721, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.25029655990510086, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.2251828631138976, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.19047619047619047, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.20503597122302158, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.6282051282051282, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.617231638418079, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8502512562814071, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8285714285714286, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4478114478114478, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.45324532453245325, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.826625386996904, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8303249097472925, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.5434782608695652, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.5343811394891945, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.2383025027203482, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.21112372304199772, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.7726063829787234, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.7958015267175572}
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

**Observed (real) value:** 0.0221 (var=0.0005)
**Predicted under pi_3:** 0.0330 (var=0.0004)
**Predicted under pi_6:** 0.0351 (var=0.0004)

### Experiment 7
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8621700879765396, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8805903398926654, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.11504424778761062, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.10695187165775401, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.8700440528634361, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.8845291479820628, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8587731811697575, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8789808917197452, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8701594533029613, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8765613519470977, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.864406779661017, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.9080036798528058, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.12398190045248869, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1223021582733813, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8806306306306306, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8892543859649122, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8705357142857143, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8809171597633136, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.11958146487294469, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.10698496905393456, '((0, 1, 1, 1), (0, 0, 1, 1))|0': 0.1348973607038123, '((0, 1, 1, 1), (0, 0, 1, 1))|1': 0.11001788908765653, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8956228956228957, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8767876787678768, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8747795414462081, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8738738738738738, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.13157894736842105, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.11469534050179211, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.12855007473841554, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.12024756852343059}
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

**Observed (real) value:** 0.0007 (var=0.0001)
**Predicted under pi_3:** 0.0010 (var=0.0001)
**Predicted under pi_6:** 0.0054 (var=0.0004)

### Experiment 8
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.4839572192513369, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.5056100981767181, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8486725663716814, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8488479262672811, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8265486725663717, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8720647773279352, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8492537313432836, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8469026548672567, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8308702791461412, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8547439126784215, '((0, 0, 1, 0), (0, 1, 1, 0))|0': 0.8071428571428572, '((0, 0, 1, 0), (0, 1, 1, 0))|1': 0.8763157894736842, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.5056179775280899, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.5016307893020222, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.4763779527559055, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.49489795918367346, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.19220055710306408, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.14989590562109645, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.14414414414414414, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.1826086956521739, '((1, 0, 0, 0), (0, 1, 0, 1))|0': 0.8669673055242391, '((1, 0, 0, 0), (0, 1, 0, 1))|1': 0.8466593647316539, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.5126903553299492, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.46732837055417703, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.844106463878327, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8437190900098912, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8591954022988506, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8768115942028986, '((1, 0, 0, 0), (0, 0, 1, 0))|0': 0.4816112084063047, '((1, 0, 0, 0), (0, 0, 1, 0))|1': 0.515052888527258}
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

**Observed (real) value:** 0.0550 (var=0.0003)
**Predicted under pi_3:** 0.0495 (var=0.0004)
**Predicted under pi_6:** 0.0462 (var=0.0003)

### Experiment 9
**Design**
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.132890365448505, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.12709030100334448, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12929061784897025, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11661341853035144, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8674377224199288, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.863905325443787, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.13922651933701657, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.13631284916201117, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8639269406392694, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8780141843971632, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.11243484735666419, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12253829321663019, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1341991341991342, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.12557077625570776, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8754171301446051, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8690344062153164, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.15171650055370986, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12374581939799331, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8614457831325302, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8785211267605634, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8621987951807228, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8326271186440678, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8367952522255193, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8738898756660746, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.1319910514541387, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1390728476821192, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.8669064748201439, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.8677325581395349, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.14099783080260303, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1366742596810934, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.1492007104795737, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.13056379821958458}
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

**Observed (real) value:** 0.0039 (var=0.0001)
**Predicted under pi_3:** 0.0005 (var=0.0001)
**Predicted under pi_6:** 0.0020 (var=0.0003)

### Experiment 10
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8622305529522024, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8703956343792633, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.8287153652392947, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.872416250890948, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.24222222222222223, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.23407407407407407, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.8309859154929577, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.875968992248062, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.8403361344537815, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8640483383685801, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8552631578947368, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8661825726141079, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8056206088992974, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8361252731245448, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.840958605664488, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8419090231170768, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.2574626865671642, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.2519582245430809, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.15707620528771385, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.15519253208868145, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8338164251207729, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.869281045751634, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8418604651162791, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8476190476190476, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.24970131421744324, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.2523364485981308, '((1, 0, 1, 1), (0, 1, 1, 0))|0': 0.14170692431561996, '((1, 0, 1, 1), (0, 1, 1, 0))|1': 0.1662425784563189, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8603238866396761, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8591117917304747, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.8641390205371248, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.882604970008569}
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

**Observed (real) value:** 0.0056 (var=0.0001)
**Predicted under pi_3:** 0.0038 (var=0.0001)
**Predicted under pi_6:** 0.0008 (var=0.0002)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Take The Best, pi_3) often provides predictions that are remarkably close to the observed data, sometimes outperforming Theory 2 (Take The Best with Validity-Dependent Confidence, pi_6) (e.g., in Experiments 2, 3, 4, 6, 7, 8, and 10). Theory 2's attempt to map the first discriminating feature's validity to a softmax confidence score does not yield a consistent improvement and sometimes heavily misses the observed metric (like in Experiment 7). However, both theories are purely non-compensatory (they only consider the single best discriminating feature). Given that TTB is a strong baseline, instead of tweaking the confidence of TTB, we should test a fundamentally different decision-making mechanism: a compensatory model.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a fully compensatory Weighted Additive (WADD) theory. In this theory, subjects consider all features, weighting each by its stated validity (or the log-odds of the validity). The overall value of each option is the sum of its weighted features, and the decision is made probabilistically using a softmax function over the options' values, along with a lapse rate. This will provide a strong compensatory contrast to the non-compensatory Take The Best theory."
}
```

## Usage

```json
{
  "prompt_token_count": 26217,
  "candidates_token_count": 310,
  "total_token_count": 27625
}
```
