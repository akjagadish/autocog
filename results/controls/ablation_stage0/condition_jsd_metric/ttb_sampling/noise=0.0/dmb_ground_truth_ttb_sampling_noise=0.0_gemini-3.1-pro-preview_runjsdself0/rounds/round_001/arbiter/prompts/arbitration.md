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

## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates Take-The-Best (TTB) from the Weighted Additive (WADD) model. We use a 5-feature design with a spread of validities. The core of the dissociation lies in 'compensatory' trials where the single most valid discriminating feature favors one option (dictating the TTB choice), but a coalition of multiple lower-validity features favors the other option (dictating the WADD choice). By varying which feature is the most valid discriminator and how many lower-validity features oppose it, we can firmly separate lexicographic single-reason decision making from compensatory weighted summation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_3 (simulated): 0.0010 (var=0.0002)
- Predicted under pi_2 (simulated): 0.1558 (var=0.0057)
- Observed on real data: 0.0022 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) from Take-The-Best (TTB), we employ a 5-feature design with linearly decreasing validities. The core dissociation relies on 'compensatory' trials where the single most valid discriminating feature favors one option (dictating the TTB choice), but a coalition of multiple lower-validity features favors the other option (dictating the WADD choice). We vary which feature is the most valid discriminator by tying the higher-validity features on some trials, ensuring that TTB's lexicographic stopping rule is tested at multiple points in the hierarchy, while WADD consistently integrates all available information.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_3 (simulated): 0.1288 (var=0.0046)
- Predicted under pi_2 (simulated): 0.0016 (var=0.0008)
- Observed on real data: 0.1305 (var=0.0041)

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
**Predicted under pi_2:** 0.0329 (var=0.0033)

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
**Predicted under pi_2:** 0.0015 (var=0.0015)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Take-The-Best, pi_3) consistently and accurately predicts the observed data, whereas Theory 2 (Weighted Additive, pi_2) fails entirely. Even in Experiment 2, which was explicitly designed by Theory 2 to showcase compensatory behavior, the observed metric (0.1305) closely matches the prediction of Take-The-Best (0.1288) rather than Weighted Additive (0.0016). The same pattern holds for Experiments 3 and 4. This indicates that participants are employing a non-compensatory, lexicographic heuristic rather than computing weighted sums. Since Theory 1 is performing exceptionally well and Theory 2 is consistently failing to capture human behavior, Theory 2 should be replaced with a new competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Tallying' (Equal Weight) theory. In this theory, decision-makers simply count the number of positive features for each option and choose the one with the higher count, entirely ignoring the cue validities. If the counts are tied, they guess. This provides a strong alternative to Take-The-Best: where TTB uses validities but only considers one feature, Tallying uses all features but ignores validities. This will test whether any compensatory behavior exists when validities are stripped away."
}
```

## Usage

```json
{
  "prompt_token_count": 9058,
  "candidates_token_count": 313,
  "total_token_count": 10124
}
```
