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
Weighted Additive (WADD) Model: People integrate all available information by computing a weighted sum of feature values for each option. The weights are proportional to the log-odds of the cue validities. This allows for compensatory decision-making, where multiple weakly predictive features can jointly override a single highly predictive feature. Choice probabilities are determined by applying a softmax function to the weighted sums, with an independent lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Extract validities and clip to avoid log(0) or division by zero
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-4, 1.0 - 1e-4)
    
    # Transform validities to weights using log-odds
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.74, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 14: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.1453900709219858, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.15476190476190477, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8635863586358636, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8630751964085297, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1334056399132321, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.16970387243735763, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.821656050955414, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8660647103085026, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8529672276350753, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8464977645305514, '((1, 0, 0, 1), (0, 1, 0, 1))|0': 0.13004484304932734, '((1, 0, 0, 1), (0, 1, 0, 1))|1': 0.14317180616740088, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8511560693641619, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8709386281588448, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.16025641025641027, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.14414414414414414, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.1478770131771596, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.1611459265890779, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8492159227985525, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8524590163934426, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.15625, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1340909090909091, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8645990922844176, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8451882845188284, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8070175438596491, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8415178571428571, '((0, 0, 0, 0), (1, 0, 1, 0))|0': 0.8527827648114902, '((0, 0, 0, 0), (1, 0, 1, 0))|1': 0.8542274052478134, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.15870786516853932, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.12867647058823528, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8316546762589928, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8552036199095022}
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
- Predicted under pi_1 (simulated): 0.0009 (var=0.0004)
- Predicted under pi_3 (simulated): 0.0017 (var=0.0001)
- Observed on real data: 0.2529 (var=0.0004)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.77, 0.8, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 15: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.8734177215189873, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.869815668202765, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8942891859052248, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8751279426816786, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.1459227467811159, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.1407811080835604, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.16621004566210046, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.16453900709219857, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.875, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8569819819819819, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8639028475711893, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8432343234323433, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.1772853185595568, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.14007421150278293, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.1342434584755404, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11834961997828447, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.24841915085817526, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.2784992784992785, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1471264367816092, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.16767676767676767, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8655172413793103, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.867741935483871, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.15876777251184834, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.12481857764876633, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.7990919409761634, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.779107725788901, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.14925373134328357, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.14274981217129978, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8751369112814896, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.859075535512965}
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
- Predicted under pi_1 (simulated): 0.0015 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0008 (var=0.0001)
- Observed on real data: 0.2659 (var=0.0000)

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
P_REF = {'((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8443465491923642, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8437118437118437, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8369098712446352, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8530734632683659, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8735177865612648, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8802163833075735, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8077753779697624, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8526551982049364, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.16240266963292546, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.13873473917869034, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.14475138121546963, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.15083798882681565, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.8528301886792453, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.8042105263157895, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.13489208633093525, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.15843023255813954, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8515625, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8517699115044248, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.16393442622950818, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.14526129317980513, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8734756097560976, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8483606557377049, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.14634146341463414, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.1402936378466558, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.17346938775510204, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.15798922800718132, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8537020517395183, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8527245949926362, '((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.8435754189944135, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.8475138121546961}
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

**Observed (real) value:** 0.2701 (var=0.0005)
**Predicted under pi_1:** 0.0010 (var=0.0001)
**Predicted under pi_3:** 0.0021 (var=0.0002)

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
P_REF = {'((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.1539301310043668, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.1414027149321267, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.14615829000577701, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.13964686998394862, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8874259381171824, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.7758007117437722, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.8291015625, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8260309278350515, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.15993907083015993, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.1868583162217659, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.15517241379310345, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.15759312320916904, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.48265460030165913, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.5527426160337553, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.12248865845755022, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.22957198443579765, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8393316195372751, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8405088062622309, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1339366515837104, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.16115107913669063, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.154006968641115, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.18904109589041096, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.15240384615384617, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.16475409836065574, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.5254942767950052, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.4803337306317044, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8481012658227848, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8198847262247838}
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

**Observed (real) value:** 0.2327 (var=0.0048)
**Predicted under pi_1:** 0.0262 (var=0.0001)
**Predicted under pi_3:** 0.0302 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, both Take The Best (THEORY 1) and Weighted Additive (THEORY 2) consistently fail to capture the observed human behavior. The observed Jensen-Shannon Divergence (JSD) from the simulated reference probabilities is consistently around 0.23 to 0.27, whereas both models predict near-zero JSD for their own generated data. The failure of both a strictly non-compensatory, validity-ordered heuristic (TTB) and a fully compensatory, validity-weighted model (WADD) suggests that participants might be ignoring the provided cue validities altogether and using a simpler integration strategy. Neither theory currently explains the data well, which warrants introducing a completely new theory.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying' (Equal Weights) theory. In this theory, decision-makers ignore the varying expert validities entirely. Instead, they simply count the total number of positive features (1s) for each option and choose the option with the higher tally. If the tallies are tied, they guess randomly. This represents a different class of boundedly rational heuristics that is compensatory but unweighted, distinguishing it clearly from both TTB (which relies heavily on validity rankings) and WADD (which relies on precise validity weighting)."
}
```

## Usage

```json
{
  "prompt_token_count": 12361,
  "candidates_token_count": 296,
  "total_token_count": 13253
}
```
