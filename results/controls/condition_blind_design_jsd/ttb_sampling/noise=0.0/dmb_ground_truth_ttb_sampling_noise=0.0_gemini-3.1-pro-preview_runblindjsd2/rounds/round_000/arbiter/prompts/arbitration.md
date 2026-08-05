# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.82, 0.84, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 2: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 9: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 15: A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 16: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.0010 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0115 (var=0.0007)
- Observed on real data: 0.0748 (var=0.0005)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.78, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 11: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 13: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 15: A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 16: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.0090 (var=0.0002)
- Predicted under pi_2 (simulated): 0.0008 (var=0.0002)
- Observed on real data: 0.0238 (var=0.0003)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
(no other experiments)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both Tallying (pi_1) and Weighted Additive (pi_2) significantly underestimate the sequence-aware Jensen-Shannon divergence observed in the real data across both experiments. In Experiment 1, the observed JSD (0.0748) is much higher than predicted by either pi_1 (0.0010) or pi_2 (0.0115). In Experiment 2, the observed JSD (0.0238) is also substantially higher than predicted by pi_1 (0.0090) and pi_2 (0.0008). This suggests that human decision-makers are systematically deviating from both simple equal-weight tallying and fully compensatory weighted integration. The data might be better explained by a non-compensatory, lexicographic heuristic that heavily privileges the most valid cues.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic. Under TTB, individuals search through features in descending order of their validity. The first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0) strictly determines the choice. If no features discriminate, the subject guesses uniformly. This differs from Tallying (which counts all discriminating features equally regardless of validity) and WADD (which compensates for deficits on highly valid cues with advantages on multiple less valid cues)."
}
```

## Usage

```json
{
  "prompt_token_count": 7474,
  "candidates_token_count": 329,
  "total_token_count": 8459
}
```
