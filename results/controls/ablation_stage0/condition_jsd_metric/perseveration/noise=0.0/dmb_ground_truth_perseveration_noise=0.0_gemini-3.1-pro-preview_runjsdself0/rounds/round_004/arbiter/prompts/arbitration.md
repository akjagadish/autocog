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


## THEORY 2 — pi_6
Weighted Additive with Choice Inertia: Decision-makers evaluate options by taking a weighted sum of their features, where the weights correspond to cue validities. However, choices are not made independently across trials; they are subject to choice inertia. A stickiness parameter biases the current choice toward the option selected in the immediately preceding trial. The final decision is made probabilistically via a softmax function over the adjusted option values, with an additional lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted addition of features
    v_a = np.sum(val * a)
    v_b = np.sum(val * b)
    
    # Apply choice inertia (stickiness) if there is a previous trial
    stickiness = float(parameters['stickiness'])
    if len(history['response']) > 0:
        prev_response = history['response'][-1]
        if prev_response == 0:
            v_a += stickiness
        elif prev_response == 1:
            v_b += stickiness
            
    scores = np.array([v_a, v_b])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Rationale:** This design explicitly targets the Choice Inertia (stickiness) mechanism present in the Competing theory but absent in the Advocated theory. By including 'catch' trials where the options are either perfectly identical (e.g., both options have the same features) or have extremely small net validity differences, the deterministic feature-based score difference becomes zero or negligible. In these specific trials, the Advocated theory predicts pure 50/50 guessing (plus random lapse), completely independent of the trial sequence. In contrast, the Competing theory predicts that choices on these trials will be strongly determined by the choice made on the immediately preceding trial due to the stickiness parameter. Comparing the sequential dependence of choices on these zero- and low-difference trials provides a stark quantitative dissociation between the two models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0), (1, 1, 0, 0))|0': 0.49550502379693284, '((1, 1, 0, 0), (1, 1, 0, 0))|1': 0.49151550614394385, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.2635102821616451, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.2783300198807157, '((0, 0, 1, 1), (0, 0, 1, 1))|0': 0.4987684729064039, '((0, 0, 1, 1), (0, 0, 1, 1))|1': 0.48785425101214575, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.13649564375605033, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.14667535853976532, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8082103912764592, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.7961783439490446, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1878144214645053, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.1932633903920486, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.865625, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8456521739130435, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.502724795640327, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.5107296137339056}
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
- Predicted under pi_6 (simulated): 0.0293 (var=0.0028)
- Observed on real data: 0.2398 (var=0.0004)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated theory (WADD with Choice Inertia) from the Competing theory (WADD without Choice Inertia), this design provides a full spectrum of net validity differences rather than relying solely on zero-difference 'catch' trials. By systematically varying the WADD score difference from zero to large values, we can map the entire psychometric choice curve. The Competing theory predicts a symmetric sigmoid curve centered exactly at zero difference. In contrast, the Advocated theory predicts that the entire psychometric curve will shift left or right on any given trial depending on the subject's choice in the immediately preceding trial. Estimating this horizontal shift across a wide range of evidence strengths provides a robust, curve-wide quantitative test of the stickiness parameter.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.4691035683202785, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.5507593924860112, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|0': 0.49821109123434704, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|1': 0.5655226209048362, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.4782958199356913, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.5268166089965398, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.49784668389319553, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.5520581113801453, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.47906602254428343, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.5267702936096719, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.47961838681699914, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.5212510024057738, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.5942519019442096, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.6458504519309778, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.4825306893295562, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.5513928914505284, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.47809878844361603, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5357950263752826, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.47058823529411764, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5224123879380603, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.36980491942324, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.4430794430794431, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.4470588235294118, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.4991735537190083}
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
- Predicted under pi_3 (simulated): 0.0409 (var=0.0012)
- Predicted under pi_6 (simulated): 0.0006 (var=0.0022)
- Observed on real data: 0.2003 (var=0.0000)

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
**Predicted under pi_6:** 0.0992 (var=0.0038)

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
**Predicted under pi_6:** 0.0384 (var=0.0043)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.2566 (var=0.0000)
**Predicted under pi_3:** 0.0008 (var=0.0001)
**Predicted under pi_6:** 0.0544 (var=0.0038)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.2497 (var=0.0000)
**Predicted under pi_3:** 0.0874 (var=0.0014)
**Predicted under pi_6:** 0.0426 (var=0.0027)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8727025187202179, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8732737611697806, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|0': 0.13749114103472715, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|1': 0.15267947421638017, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|0': 0.8467741935483871, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|1': 0.8539603960396039, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.12545018007202882, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.14216634429400388, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|0': 0.8807600950118765, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|1': 0.8521008403361344, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15547263681592038, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.1371308016877637, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8691049085659288, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8585189644792294, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.11428571428571428, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.13272727272727272, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|0': 0.1178082191780822, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|1': 0.15080645161290324, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8625877632898696, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8526130358191427}
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

**Observed (real) value:** 0.2658 (var=0.0000)
**Predicted under pi_3:** 0.0008 (var=0.0001)
**Predicted under pi_6:** 0.0391 (var=0.0042)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.22847522847522847, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.30434782608695654, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0))|0': 0.2122549019607843, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0))|1': 0.33484848484848484, '((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0))|0': 0.22483221476510068, '((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0))|1': 0.30781758957654726, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|0': 0.20794958797867183, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|1': 0.3108320251177394, '((1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0))|0': 0.19950980392156864, '((1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0))|1': 0.3303030303030303, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.2198751800288046, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.3338735818476499, '((1, 0, 0, 1, 0, 0), (0, 1, 1, 0, 1, 1))|0': 0.21141439205955334, '((1, 0, 0, 1, 0, 0), (0, 1, 1, 0, 1, 1))|1': 0.33138686131386863, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.20784883720930233, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.3333333333333333, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|0': 0.2015376166941241, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|1': 0.3229706390328152, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.19639376218323587, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.32098765432098764}
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

**Observed (real) value:** 0.2074 (var=0.0151)
**Predicted under pi_3:** 0.1229 (var=0.0018)
**Predicted under pi_6:** 0.0479 (var=0.0036)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8195797027165556, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.7909562638991846, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.20248166948674562, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.18555008210180624, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8475336322869955, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8538499264345267, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13093196112064037, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1388438681793625, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8201219512195121, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8131127450980392, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.8556298773690078, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.8815060908084164, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.26512543039842595, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.24377791959157627, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.1711273317112733, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.13772708069286016}
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

**Observed (real) value:** 0.2587 (var=0.0000)
**Predicted under pi_3:** 0.0008 (var=0.0001)
**Predicted under pi_6:** 0.0575 (var=0.0030)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 1, 1, 0))|0': 0.7085889570552147, '((0, 1, 0, 0, 0), (1, 0, 1, 1, 0))|1': 0.753949678174371, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.6917030567685589, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.7258064516129032, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6758156395649922, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.732199481118478, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.28841169937060346, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.27575858619539845, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.49389747762408465, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5027760641579272}
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

**Observed (real) value:** 0.2302 (var=0.0019)
**Predicted under pi_3:** 0.0430 (var=0.0009)
**Predicted under pi_6:** 0.0153 (var=0.0024)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both Theory 1 (WADD) and Theory 2 (WADD with Choice Inertia) consistently fail to capture the observed data across all experiments. The metrics used are sequence-aware Jensen-Shannon Divergences between the data and the proposing theory's reference probabilities. In almost every experiment, the observed JSD is between 0.20 and 0.26, while the models predict much lower values (often close to 0). This massive discrepancy indicates that the subjects' choice profiles, conditional on previous responses and current stimuli, are fundamentally different from both a standard compensatory WADD model and a WADD model with simple choice inertia. The consistent failure of both models suggests that humans might be using a completely different, likely non-compensatory decision strategy, such as a lexicographic heuristic, rather than any variant of weighted additive utility.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory based on a non-compensatory heuristic, such as Take-The-Best (TTB) or a Lexicographic strategy. Instead of computing a weighted sum of all features, the decision-maker compares the options sequentially based on cue validities, stopping at the first cue that discriminates between the two options. This would produce a completely different choice probability profile (often more deterministic on certain trials) and could better explain the high divergence from the WADD-based predictions."
}
```

## Usage

```json
{
  "prompt_token_count": 20689,
  "candidates_token_count": 315,
  "total_token_count": 21780
}
```
