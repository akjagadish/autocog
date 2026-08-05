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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Rationale:** To dissociate Take The Best (TTB) from Tallying, we design trials where the option favored by the most valid cue is opposed by the majority of lower-validity cues. TTB will strictly follow the highest-validity discriminating cue, ignoring the total count of winning cues. In contrast, Tallying ignores validities entirely and simply counts the number of cues each option wins. We include trials where TTB chooses one option while Tallying chooses the other, as well as trials where Tallying is indifferent (predicts a 50/50 guess) but TTB has a strong preference.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8565906455353691, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8373248983280615, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.15898767034393252, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14881865602945687, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1520648398301814, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16161158895427796, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8453125, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8523255813953489, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.15064102564102563, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.1757028112449799, '((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.8585072515033605, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.8393309680689306}
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
- Predicted under pi_1 (simulated): 0.0005 (var=0.0002)
- Predicted under pi_2 (simulated): 0.2040 (var=0.0039)
- Observed on real data: 0.2566 (var=0.0090)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tallying (which gives equal weight to all cues and chooses the option with the most feature wins) from Take The Best (which relies solely on the most valid discriminating cue), we design trials where the two models make completely opposed predictions or where one model predicts a tie while the other strongly prefers one option. By using 5 features with cascading validities, we construct cases where the highest-validity cue points to one option, but a majority of lower-validity cues point to the other. Additionally, we include trials where the top cues tie, forcing TTB to rely on a mid-validity cue, while Tallying counts the remaining features to favor the opposite option. This creates a clear double dissociation in choice probabilities.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Auto-generated JSD-to-self metric (jsd_metric control): sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) between the dataset's conditional choice profile and the proposing theory's, over (trial content, previous response) states. 0 means the data behaves exactly like the proposing theory; ln 2 means maximally different.

Source:
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.1402002861230329, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.1497695852534562, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.5043969849246231, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.48375451263537905, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.5103884372177055, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.5065913370998116, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8472344161545216, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8648302370275465, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.16136919315403422, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1365568544102019, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.847210994341148, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8530416951469583, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.12895927601809956, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.1579330422125182, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.494475138121547, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.4640449438202247, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.8562322946175638, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.854251012145749, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.5150684931506849, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.4637096774193548}
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
- Predicted under pi_1 (simulated): 0.1925 (var=0.0062)
- Predicted under pi_2 (simulated): 0.0003 (var=0.0001)
- Observed on real data: 0.0866 (var=0.0009)

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
  "interpretation": "Across both experiments, Theory 2 (Tallying) provides a better account of the data than Theory 1 (Take The Best). In Experiment 1, the observed JSD to Theory 1's predictions was very high (0.2566), matching or exceeding the divergence predicted by Theory 2 (0.2040), indicating that subjects are not strictly following the highest-validity cue. In Experiment 2, the data's divergence from Theory 2 was moderate (0.0866), falling between the predictions of pure Tallying and pure Take The Best, but closer to Tallying. This suggests that while subjects integrate multiple features, they do not ignore validities entirely (as pure Tallying would), nor do they rely on a single cue (as TTB would). A compensatory model that weights features by their validities would likely perform better than both.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Take The Best (Theory 1) with a Weighted Additive (WADD) theory. In this new theory, decision-makers evaluate each option by taking the sum of its features weighted by their respective validities. The option with the higher weighted sum is chosen. This provides a compensatory mechanism that accounts for both the number of winning features (like Tallying) and the relative importance of those features (like Take The Best), bridging the gap between the two current models and likely capturing the observed intermediate choice probabilities."
}
```

## Usage

```json
{
  "prompt_token_count": 6286,
  "candidates_token_count": 332,
  "total_token_count": 7354
}
```
