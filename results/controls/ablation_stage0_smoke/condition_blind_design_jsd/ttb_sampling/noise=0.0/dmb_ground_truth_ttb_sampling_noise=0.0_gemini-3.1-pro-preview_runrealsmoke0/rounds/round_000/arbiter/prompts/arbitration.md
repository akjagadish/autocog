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
**Validities (n_features=4):** [0.95, 0.76, 0.92, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 10: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 11: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 12: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 14: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 15: A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8431952662721893, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8665480427046264, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8254649499284692, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8592188919164396, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.8591772151898734, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.8553082191780822, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.1513859275053305, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.1585274229902329, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.10526315789473684, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.10247747747747747, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.5180327868852459, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.519774011299435, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8691335740072202, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8265895953757225, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.16736401673640167, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.12934947049924356, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8623853211009175, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8492957746478873, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8579710144927536, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8621621621621621, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8696604600219058, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8680947012401353, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.17297297297297298, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1614457831325301, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.1541095890410959, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.13240131578947367, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8710407239819005, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8565400843881856, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8608534322820037, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8751300728407908, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.157439446366782, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12520458265139117}
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
- Predicted under pi_1 (simulated): 0.0015 (var=0.0002)
- Predicted under pi_2 (simulated): 0.0091 (var=0.0004)
- Observed on real data: 0.0197 (var=0.0002)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.89, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 14: A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 16: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.707843137254902, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.7930232558139535, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.20941558441558442, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.16385135135135134, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.7039848197343453, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.7148468185388845, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.691764705882353, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.7185454545454546, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8456973293768546, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8374777975133215, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5599078341013825, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6317715959004392, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.829686013320647, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8277703604806409, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.6508728179551122, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.6769790718835305, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.23106796116504855, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.20311284046692607, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.512743628185907, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.47572815533980584, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.835990888382688, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.8394793926247288, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.7595419847328244, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.7830374753451677, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.7261698440207972, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.7375306623058054, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.193390452876377, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.23194303153611392, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8232611174458381, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8418201516793066, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.7265372168284789, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.7986463620981388}
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
- Predicted under pi_1 (simulated): 0.0081 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0010 (var=0.0002)
- Observed on real data: 0.0203 (var=0.0002)

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
  "interpretation": "Both Tallying (Theory 1) and WADD (Theory 2) fail to capture the real data. In Experiment 1, the observed JSD to Theory 1's reference (0.0197) is significantly higher than both Theory 1's self-prediction (0.0015) and Theory 2's cross-prediction (0.0091). Similarly, in Experiment 2, the observed JSD to Theory 2's reference (0.0203) vastly exceeds Theory 2's self-prediction (0.0010) and Theory 1's cross-prediction (0.0081). Because the observed data diverges heavily from the predictions of both compensatory strategies (unweighted and weighted), it is highly likely that subjects are using a fundamentally different, non-compensatory decision process.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic. Unlike Tallying (which equally weighs all features) and WADD (which computes a validity-weighted sum across all features), TTB is a non-compensatory lexicographic strategy. It assumes people rank features by their validity, compare the options on the most valid feature first, and if one option is strictly better, they choose it immediately without considering any other features. If there is a tie, they move to the next most valid feature, and so on. This introduces a fundamentally different mechanism for combining binary features that may better explain the observed choice patterns."
}
```

## Usage

```json
{
  "prompt_token_count": 7774,
  "candidates_token_count": 351,
  "total_token_count": 8935
}
```
