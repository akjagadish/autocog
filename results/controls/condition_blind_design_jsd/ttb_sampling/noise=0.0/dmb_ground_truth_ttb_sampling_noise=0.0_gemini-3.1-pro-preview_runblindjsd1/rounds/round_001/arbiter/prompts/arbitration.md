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
Take-The-Best (TTB) is a non-compensatory fast-and-frugal heuristic. People rank features by their subjective validity and compare options sequentially. The decision is based entirely on the first feature that discriminates between the options, ignoring all remaining features. If all features tie, the decision maker guesses randomly. Response noise is captured by a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
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
**Validities (n_features=4):** [0.95, 0.6, 0.87, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8840579710144928, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8748615725359912, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.11559139784946236, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12134502923976608, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.1378692927484333, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.14641288433382138, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.10933333333333334, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1362962962962963, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.8860648553900088, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.874051593323217, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.11531531531531532, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.14202898550724638, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1309823677581864, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1650943396226415, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.1084070796460177, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.11160714285714286, '((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.10942441492726122, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.2146118721461187, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.860832137733142, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8712601994560291, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.11829134720700986, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.11161217587373168, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14411764705882352, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.12142857142857143, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8772378516624041, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8177966101694916, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.10836501901140684, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.1891891891891892, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.11185682326621924, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16993464052287582, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8882733148661126, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8729016786570744}
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
- Predicted under pi_3 (simulated): 0.0011 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0207 (var=0.0006)
- Observed on real data: 0.0038 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.57, 0.58, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.6834677419354839, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.75, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.2972972972972973, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.3210930828351836, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.5884861407249466, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.525522041763341, '((0, 0, 1, 0), (1, 1, 1, 1))|0': 0.8384074941451991, '((0, 0, 1, 0), (1, 1, 1, 1))|1': 0.8456659619450317, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.23440453686200377, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.2749326145552561, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.15806451612903225, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.1875, '((0, 1, 1, 0), (0, 0, 1, 1))|0': 0.4133489461358314, '((0, 1, 1, 0), (0, 0, 1, 1))|1': 0.5232558139534884, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.6568627450980392, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.7034990791896869, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.48372781065088755, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.5204626334519573, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.3069544364508393, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.3115942028985507, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8237082066869301, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8081180811808119, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.823943661971831, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8312236286919831, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.376425855513308, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.44919786096256686, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.721120186697783, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.7592788971367974, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.5577156743620899, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.646878198567042, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.26953748006379585, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.23870417732310314}
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
- Predicted under pi_3 (simulated): 0.0278 (var=0.0004)
- Predicted under pi_2 (simulated): 0.0017 (var=0.0006)
- Observed on real data: 0.0335 (var=0.0005)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8471337579617835, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8368298368298368, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.17772215269086358, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.16583416583416583, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.15309734513274337, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.1537313432835821, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8552522746071133, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8324873096446701, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.16329479768786126, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1299638989169675, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8578052550231839, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8640093786635404, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8636779505946935, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8458274398868458, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.11648568608094768, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.16010165184243966, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.49140271493212667, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.5179856115107914, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.5100502512562815, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.49302788844621515, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.10619469026548672, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.11892963330029732, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.512781954887218, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.474468085106383, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12003311258278146, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.13175675675675674, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.14905814905814907, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.15889464594127806, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.508296943231441, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.4900763358778626, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8048780487804879, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8489451476793249}
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

**Observed (real) value:** 0.0205 (var=0.0002)
**Predicted under pi_3:** 0.0249 (var=0.0002)
**Predicted under pi_2:** 0.0071 (var=0.0004)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.7913950456323338, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.7831558567279767, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.2125, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.17314814814814813, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.3384201077199282, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.35276967930029157, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.8560700876095119, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8271728271728271, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.22364217252396165, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.206984667802385, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.2425249169435216, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.21962095875139354, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7389858686616791, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.6549413735343383, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.2998805256869773, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.3052959501557632, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.45555555555555555, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.45308641975308644, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8474387527839644, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.835920177383592, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8377125193199382, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8475967174677609, '((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.16120906801007556, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.17693836978131214, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.7583444592790387, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.7916270218839201, '((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.8501026694045175, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8317191283292978, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.2862669245647969, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.24551831644583008, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.4927675988428158, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.583224115334207}
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

**Observed (real) value:** 0.0172 (var=0.0002)
**Predicted under pi_3:** 0.0258 (var=0.0002)
**Predicted under pi_2:** 0.0015 (var=0.0003)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data consistently diverges from the predictions of the Weighted Additive (WADD) model (THEORY 2) and closely aligns with the predictions of Take-The-Best (TTB) (THEORY 1). In Experiment 1, the observed JSD to TTB's reference profile is 0.0038, nearly matching TTB's own 0.0011 and far from WADD's 0.0207. In Experiments 2, 3, and 4 (which use WADD's reference profiles), the observed data's distance from WADD is large (0.0335, 0.0205, 0.0172) and tracks TTB's distance to WADD (0.0278, 0.0249, 0.0258). This suggests that WADD is fundamentally mischaracterizing the subjects' decision process, while TTB captures the non-compensatory nature of their choices.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since the fully compensatory WADD model is consistently failing and TTB is performing very well, THEORY 2 should be replaced with a brand-new theory that explores a different bounded-rationality heuristic. Propose a 'Tallying' (Equal Weight) heuristic, which simply counts the number of positive cues for each option regardless of their validities, or a 'Two-Stage/Hybrid' model that uses TTB if the highest validity cue discriminates strongly, but falls back to Tallying if the top cues are tied or their validities are too similar. This will provide a stronger, more plausible heuristic competitor to pure TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 12508,
  "candidates_token_count": 393,
  "total_token_count": 14082
}
```
