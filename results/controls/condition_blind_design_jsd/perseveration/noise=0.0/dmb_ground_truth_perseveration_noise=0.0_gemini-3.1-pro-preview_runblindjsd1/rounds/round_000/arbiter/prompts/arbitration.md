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


## THEORY 2 — pi_2
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.72, 0.73, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 6: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 14: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 15: A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.0009 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0194 (var=0.0002)
- Observed on real data: 0.2658 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.8, 0.85, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 15: A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 16: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.0281 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0009 (var=0.0001)
- Observed on real data: 0.2554 (var=0.0000)

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
  "interpretation": "Both Take The Best (Theory 1) and Tallying (Theory 2) fail to accurately capture the real data. In Experiment 1, the observed JSD from Theory 1's predictions is 0.2658, whereas Theory 1 predicted ~0.0009. Similarly, in Experiment 2, the observed JSD from Theory 2's predictions is 0.2554, compared to its own prediction of ~0.0009. This large divergence indicates that human subjects are neither using a strict one-reason decision rule (TTB) nor completely ignoring cue validities (Tallying). A compensatory strategy that incorporates both all available features and their respective validities is likely needed.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) theory. In this theory, decision-makers evaluate each option by computing a weighted sum of its features, where the weights are directly proportional to the provided cue validities. The option with the higher weighted sum is chosen. This differs from TTB (which is non-compensatory and stops at the first discriminating cue) and Tallying (which weighs all features equally regardless of validity)."
}
```

## Usage

```json
{
  "prompt_token_count": 7935,
  "candidates_token_count": 282,
  "total_token_count": 9019
}
```
