# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Weighted Additive (WADD) theory posits that decision-makers integrate all available cues by weighting them proportionally to their validities. The overall value for each option is the validity-weighted sum of its features. Choices are made probabilistically via a softmax over these aggregated values, incorporating response noise through an inverse temperature parameter and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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


## THEORY 2 — pi_4
Probabilistic-Stopping Take-The-Best (TTB)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Descending validity; argsort is stable so validity ties break toward earlier feature index
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    for j in cue_order:
        if a[j] > b[j]:
            p_A += p_continue * alpha
            p_continue *= (1.0 - alpha)
        elif b[j] > a[j]:
            p_B += p_continue * alpha
            p_continue *= (1.0 - alpha)
            
    # If search exhausts all cues or fails to stop, guess uniformly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.7, 0.6, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) theory from Probabilistic-Stopping Take-The-Best (TTB) theory, we use 5 features with a prominent highest-validity cue and a set of lower-validity cues whose combined weight can exceed the highest one. TTB makes decisions based primarily on the highest-validity discriminating cue and stops search probabilistically, often ignoring the remaining cues. WADD integrates all cues weighted by their validities. We design 'compensatory' trials where Option A possesses the single most valid cue (strongly favored by TTB) but Option B possesses multiple lower-validity cues whose sum outweighs Option A's single cue (strongly favored by WADD). We also include trials where both theories agree, to serve as a baseline for noise and overall performance.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.15749525616698293, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.1589895988112927, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14057507987220447, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1542997542997543, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8291316526610645, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8264746227709191, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.3562231759656652, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.3474114441416894, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7488196411709159, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.7942882641677822, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6647093364650617, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.6510279388508171, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.7594123048668503, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8104340900039825, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8479498861047836, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8378524945770065}
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
- Predicted under pi_3 (simulated): 0.0010 (var=0.0002)
- Predicted under pi_4 (simulated): 0.0569 (var=0.0026)
- Observed on real data: 0.0081 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** To quantitatively dissociate Probabilistic-Stopping Take-The-Best (TTB) from Weighted Additive (WADD) theory, we use 5 features with linearly decreasing validities. TTB evaluates cues in descending order of validity and stops probabilistically upon finding a discriminating cue, meaning it generally favors the option with the highest-validity discriminating feature. WADD, however, computes a compensatory sum of all features weighted by their validities. We design a set of trials where Option A possesses the single most valid discriminating cue (which TTB will heavily favor), while Option B possesses multiple lower-validity cues whose combined weight strongly exceeds Option A's sum (which WADD will heavily favor). By varying which cue is the highest valid discriminator (e.g., the first vs. the second cue), we can also probe the probabilistic stopping mechanism of TTB, as its preference strength will degrade if it fails to stop at earlier cues.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.405341446923597, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.4815418023887079, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.34856351678781583, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.41601255886970173, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.3780487804878049, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.45619158878504673, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.27576503693281745, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.3546244251405212, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.3910427807486631, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.45741150442477874, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.3926247288503254, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4783677482792527}
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
- Predicted under pi_3 (simulated): 0.0781 (var=0.0029)
- Predicted under pi_4 (simulated): 0.0005 (var=0.0005)
- Observed on real data: 0.1147 (var=0.0024)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (0, 0, 1, 1))|0': 0.1259320629660315, '((1, 1, 0, 0), (0, 0, 1, 1))|1': 0.14502529510961215, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1400454201362604, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.12531581606872158, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8697047496790757, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8687561214495593, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.1400296882731321, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14186193793540217, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.1333997013439522, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.12696417347580138, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8359240069084629, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8751023751023751, '((0, 1, 0, 1), (1, 0, 0, 0))|0': 0.8776622090143635, '((0, 1, 0, 1), (1, 0, 0, 0))|1': 0.8621125869702719, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.85957213384531, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.8773213280810355}
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

**Observed (real) value:** 0.2429 (var=0.0060)
**Predicted under pi_3:** 0.1594 (var=0.0017)
**Predicted under pi_4:** 0.0486 (var=0.0012)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.15036743923120408, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1583833970507919, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.12419070041200707, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.13992635455023672, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8402915838303512, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8589191774270684, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8630310716550412, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8645575877409788, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.16549069916613213, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.14061734443900048, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.4925219185146983, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.473208910295003, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8657289002557544, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.861984282907662, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.855036855036855, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8638768638768639}
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

**Observed (real) value:** 0.0405 (var=0.0004)
**Predicted under pi_3:** 0.0042 (var=0.0001)
**Predicted under pi_4:** 0.1145 (var=0.0055)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.415614773258532, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.39308530627583615, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.6572261557684298, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.6523551479783243, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.333597150771666, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.354157501099868, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.5937165298107818, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5707853926963482, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.2462406015037594, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.2575885328836425, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.74, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.7665306122448979}
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

**Observed (real) value:** 0.1644 (var=0.0058)
**Predicted under pi_3:** 0.0004 (var=0.0002)
**Predicted under pi_4:** 0.0086 (var=0.0001)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5014989293361884, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5281947261663286, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15499070055796652, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.1492899203325251, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8430979133226324, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8453206239168111, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.13958060288335516, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.15560640732265446, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8514970059880239, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8496732026143791, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.48419721871049304, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.49814585908529047}
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

**Observed (real) value:** 0.0327 (var=0.0005)
**Predicted under pi_3:** 0.0430 (var=0.0005)
**Predicted under pi_4:** 0.0839 (var=0.0035)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple experiments, Weighted Additive (WADD, Theory 1) generally outperforms Probabilistic-Stopping Take-The-Best (TTB, Theory 2). In Experiment 1, 3, 4, and 6, the predictions of WADD are noticeably closer to the observed data than those of TTB. Even in Experiment 2, which was designed by TTB to favor itself, the observed metric (0.1147) was much closer to WADD's prediction (0.0781) than to TTB's prediction (0.0005). However, in some experiments (like Experiment 3 and 5), both theories fail to perfectly capture the observed metric, suggesting that pure WADD is also incomplete. Since TTB systematically fails to capture the data where it diverges from WADD, it should be discarded in favor of a new theory.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Mixture Strategy Theory' or 'Tallying / Validity-Threshold Theory'. A Mixture Strategy Theory would posit that decision-makers are heterogenous or trial-by-trial variable: they probabilistically employ WADD on some trials/subjects and a heuristic like TTB or Tallying on others. Alternatively, a 'Validity-Threshold Tallying Theory' would posit that subjects only consider features whose validities exceed a certain threshold, and simply tally the number of winning features among that subset rather than strictly weighting them by exact validity. This could bridge the gap between the purely compensatory WADD and the purely non-compensatory TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 11296,
  "candidates_token_count": 358,
  "total_token_count": 12526
}
```
