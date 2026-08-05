# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
Decision-makers use a Weighted Additive (WADD) strategy, evaluating each option by computing a weighted sum of its features, where the weights are directly proportional to the provided cue validities. The option with the higher weighted sum is favored. Response noise is modeled via a softmax over these weighted sums with an inverse temperature, along with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])

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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.88, 0.77, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 5: A=[0, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 14: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 15: A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 16: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8241525423728814, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8621987951807228, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.1432408236347359, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1683748169838946, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.1524024024024024, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.17094017094017094, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.13938053097345132, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.14397321428571427, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.1352154531946508, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.16519823788546256, '((0, 1, 1, 0), (0, 1, 1, 1))|0': 0.831140350877193, '((0, 1, 1, 0), (0, 1, 1, 1))|1': 0.847972972972973, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8285302593659942, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8318264014466547, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8353863381858903, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8412348401323043, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.18046709129511676, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.13393528969149737, '((0, 1, 1, 1), (1, 1, 1, 1))|0': 0.8490909090909091, '((0, 1, 1, 1), (1, 1, 1, 1))|1': 0.8514285714285714, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.14977973568281938, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.16704035874439463, '((0, 0, 0, 0), (0, 1, 1, 1))|0': 0.8558382257012394, '((0, 0, 0, 0), (0, 1, 1, 1))|1': 0.8239700374531835, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.14167294649585532, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.17758985200845667, '((0, 0, 1, 0), (0, 0, 0, 0))|0': 0.16057585825027684, '((0, 0, 1, 0), (0, 0, 0, 0))|1': 0.13154960981047936, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.16817359855334538, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.16714697406340057, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8523076923076923, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8458823529411764}
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
- Predicted under pi_1 (simulated): 0.0013 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0160 (var=0.0003)
- Observed on real data: 0.2601 (var=0.0008)

## EXPERIMENT 2 (proposed by pi_3)

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
P_REF = {'((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.13957759412304868, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.180028129395218, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8734723220704529, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8337408312958435, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14118629908103592, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.15754560530679934, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8639705882352942, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8455056179775281, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8453038674033149, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8277310924369747, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.16892502258355918, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.19336219336219337, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.14307342922028765, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.18997912317327767, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.1967020023557126, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.23809523809523808, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.130348913759052, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.2491103202846975, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1337099811676083, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.14769647696476965, '((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.13020833333333334, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.15432098765432098, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.6285452881976212, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.6025459688826026, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8790072388831437, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.885954381752701, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.1439312567132116, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.11392405063291139, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.1753902662993572, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.19127988748241911, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.13533284564740308, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.18475750577367206}
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
- Predicted under pi_1 (simulated): 0.0048 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0010 (var=0.0003)
- Observed on real data: 0.2320 (var=0.0050)

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
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.126044039483675, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.15734989648033126, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8735119047619048, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8728070175438597, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.16129032258064516, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.1453940066592675, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.85997171145686, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.8664226898444648, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8376068376068376, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8716216216216216, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8649262202043133, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8324265505984766, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.16248153618906944, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.12911843276936777, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.1632047477744807, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1483126110124334, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.13676731793960922, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.13501483679525222, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.12455516014234876, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15680473372781065, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8619676945668135, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8543342269883825, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8610855565777369, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8401122019635343, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8388520971302428, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8893838158871566, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.14516129032258066, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.1252236135957066, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.13047445255474452, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.15346534653465346, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8712328767123287, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8567375886524823}
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
**Predicted under pi_1:** 0.0009 (var=0.0001)
**Predicted under pi_3:** 0.0020 (var=0.0002)

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
P_REF = {'((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.862453531598513, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8751258811681772, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.47289156626506024, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.4876760563380282, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.13242574257425743, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.14818548387096775, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8419282511210763, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8601321585903083, '((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.14113785557986872, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.14672686230248308, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.15040183696900114, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1496232508073197, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.8631346578366446, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8534675615212528, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8433179723502304, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8630393996247655, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.4905814905814906, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.5077720207253886, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.13493377483443708, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.13682432432432431, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1644325290437891, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.17033773861967694, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.14156626506024098, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.14925373134328357, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8377483443708609, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8534675615212528, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8529048207663782, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8425832492431887, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8533627342888643, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.8779395296752519, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.14985163204747776, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.15008880994671403}
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

**Observed (real) value:** 0.2554 (var=0.0000)
**Predicted under pi_1:** 0.0281 (var=0.0001)
**Predicted under pi_3:** 0.0040 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, both Theory 1 (Take The Best) and Theory 2 (Weighted Additive) severely underestimate the sequence-aware Jensen-Shannon Divergence metric. The simulated predictions for both theories yield JSD values close to 0, whereas the observed data consistently shows a JSD of around 0.23 to 0.26. This large discrepancy indicates that the real human data deviates significantly from the static, trial-independent choice probabilities predicted by both TTB and WADD. Crucially, the metric conditions on the previous response (`r[t-1]`), suggesting that sequential effects\u2014such as choice inertia, hysteresis, or win-stay/lose-shift dynamics\u2014play a significant role in human decision-making in this task. Because neither theory accounts for trial-to-trial history, both fail to capture the observed behavioral patterns.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory that explicitly incorporates sequential dependencies. The new theory should model how previous choices influence the current decision, for instance through 'choice inertia' (a baseline bias to repeat the previous response) or an evidence-accumulation process that carries over between trials. This will differentiate it from the purely stimulus-driven, memoryless mechanisms of both TTB and WADD, directly addressing the sequence-aware metric's findings."
}
```

## Usage

```json
{
  "prompt_token_count": 12688,
  "candidates_token_count": 304,
  "total_token_count": 13689
}
```
