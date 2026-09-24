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
**Validities (n_features=6):** [0.91, 0.83, 0.74, 0.65, 0.6, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Rationale:** This design uses six cues with one clearly highest-validity cue (0.91) that discriminates on every trial, while the five lower-validity cues are varied so their tally advantage runs from strongly pro-A to strongly pro-B. In trials 1-4 the highest-validity cue favors A; in trials 5-8 it favors B. Take The Best stops at that highest cue, so it predicts one constant A-choice probability for trials 1-4 and one constant complementary A-choice probability for trials 5-8, with no effect of the lower-cue tallies. Tallying ignores validity and uses the net feature-win count, so it predicts a monotone sigmoid in the tally difference within each block: across trials 1-4 the net advantage changes from +4 to +2 to 0 to -2, and across trials 5-8 from +2 to 0 to -2 to -4. The +4 and +2 top-A conditions separate the theories quantitatively even when both prefer A, while the tie and negative-tally conditions produce neutral or opposite predictions. This creates a flat-vs-monotone signature rather than a single ambiguous pair. Each unique pair is repeated 12 times because 96 // 8 = 12.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric separates trials by the top-validity cue, which is feature index 0 in this design. Within each side of that top-cue split, it computes the relation between the lower-cue tally advantage for A and actual A choices: mean tally advantage on A-choice trials minus mean tally advantage on B-choice trials, normalized by the tally range of 6. Take The Best predicts this relation is zero on both sides, because the top cue discriminates on every trial and all lower cues are ignored. Tallying predicts a positive relation on both sides, because larger tally advantage for A makes an A choice more likely. The two side-wise values are averaged, and a minimum response-class size of 12 avoids unstable undefined estimates while keeping the metric tight within subjects. Under Take The Best the expected value is near zero; under Tallying it is clearly positive.

Source:
def metric(data):
    import numpy as np
    min_class = 12
    groups = {0: [], 1: []}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        if a[0] > b[0]:
            grp = 0
        elif b[0] > a[0]:
            grp = 1
        else:
            continue
        tally = int(np.sum(a > b) - np.sum(b > a))
        chose_a = 1.0 if int(row['response']) == 0 else 0.0
        groups[grp].append((tally, chose_a))
    parts = []
    for grp in (0, 1):
        vals = groups[grp]
        if not vals:
            parts.append(0.0)
            continue
        d_a = [d for d, y in vals if y == 1.0]
        d_b = [d for d, y in vals if y == 0.0]
        if len(d_a) < min_class or len(d_b) < min_class:
            parts.append(0.0)
        else:
            parts.append((np.mean(d_a) - np.mean(d_b)) / 6.0)
    return float(np.mean(parts))

### RESULTS
- Predicted under pi_1 (simulated): 0.0157 (var=0.0016)
- Predicted under pi_2 (simulated): 0.4517 (var=0.0090)
- Observed on real data: -0.4323 (var=0.0402)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.65, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 11: A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** This design contrasts Tallying's use of the net feature-win count with TTB's use of only the first discriminating cue. Five cues have spread validities (0.90, 0.80, 0.65, 0.55, 0.50). The 12 unique pairs are grouped by net tally d = #(A>B) - #(B>A): two pairs each at d=+2, -2, +1, -1, and four at d=0. Within each same-d group, Tallying predicts identical A-choice probabilities (sigmoid(beta*d)), whereas TTB predicts opposite directions because the first discriminating cue changes: in one member the top cue discriminates; in the other the top cue ties and the second cue discriminates in the opposite direction. For example, trials 1 and 2 both have d=+2; Tallying predicts strong A for both, but TTB predicts A on trial 1 and B on trial 2. Across d, Tallying predicts a monotone sigmoid, while TTB predicts a flat probability depending only on the first-cue winner. This yields a double dissociation and also quantitative differences on same-direction trials (e.g., d=+2 gives Tallying sigmoid(2*beta) versus TTB sigmoid(beta)). Tie values on non-discriminating features are varied for display and do not affect either model.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric is a crossover contrast using the design's matched pairs. It first computes, for every trial, which option Take The Best would select by scanning cues in descending validity order. It then subtracts the observed A-choice rate on trials where TTB selects B from the A-choice rate on trials where TTB selects A. Under Tallying, the two TTB-predicted trial sets are balanced in signed tally margin d: each has one pair at d=+2, one at d=-2, one at d=+1, one at d=-1, and two at d=0. Since Tallying predicts p(A)=sigmoid(beta*d) after noise, the positive-d and negative-d contributions cancel exactly, so the expected value of this metric is zero for every beta and epsilon, leaving only binomial sampling variance. Under TTB, every trial in the TTB-selects-A set has A-choice rate p and every trial in the TTB-selects-B set has A-choice rate 1-p, so the metric equals 2*sigmoid(beta)-1 shrunk by 1-epsilon. Each subject contributes 48 trials to each side, producing a large mean separation with low per-subject variance.

Source:
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.65, 0.55, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    a_when_ttb_a = []
    a_when_ttb_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner == -1:
            continue
        chose_a = 1 if int(row['response']) == 0 else 0
        if ttb_winner == 0:
            a_when_ttb_a.append(chose_a)
        else:
            a_when_ttb_b.append(chose_a)
    if not a_when_ttb_a or not a_when_ttb_b:
        return 0.0
    return float(np.mean(a_when_ttb_a) - np.mean(a_when_ttb_b))

### RESULTS
- Predicted under pi_1 (simulated): 0.6883 (var=0.0485)
- Predicted under pi_2 (simulated): -0.0083 (var=0.0072)
- Observed on real data: -0.2650 (var=0.0713)

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
  "interpretation": "Experiment 1's metric isolates the relation between lower-cue tally advantage and A choice within each side of the top-validity cue. TTB predicts approximately zero because the top cue discriminates on every trial and all lower cues are ignored; Tallying predicts a clearly positive value because larger lower-cue tally should favor A. The observed value is strongly negative (-0.432), meaning subjects chose A more when the lower-cue tally favored B. This is opposite to Tallying and directly violates TTB's one-reason claim. Experiment 2's crossover contrast is +0.69 under TTB and approximately 0 under Tallying by design, but the observed value is -0.265: subjects chose A less on TTB-A trials than on TTB-B trials. This is opposite to TTB and not the null Tallying prediction. Together, the two experiments indicate that feature advantages are used negatively, with the highest-validity cue carrying a negative weight. A single negative validity-weighted additive rule can produce both patterns: in Experiment 1 it yields a negative lower-cue tally relation within top-cue groups; in Experiment 2, TTB-A trials have the top cue favoring A but lower cues favoring B, so a more negative top-cue weight makes their score lower than TTB-B trials, giving a negative contrast. Neither current theory can be repaired by reparameterization because their core descriptions force non-negative or one-cue use. Although pi_2 is numerically closer in Experiment 2, pi_1's core one-reason assumption is directly contradicted by Experiment 1's large lower-cue effect, so I target pi_1 for replacement.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a new theory: Negative Validity-Weighted Additive (NWA). People integrate all feature differences linearly but assign negative weights to feature advantages, with weight magnitude increasing with stated validity, e.g. w_j = -v_j^gamma with gamma > 0, or w_j = -(a + b*v_j). Choice probability is a softmax/lapse over the weighted score: p(A) = (1 - epsilon) * sigmoid(beta * sum_j w_j * (a_j - b_j)) + epsilon/2. Suggested ranges: beta in 0.5-1.5, epsilon in 0-0.2, gamma in 0.5-2. This differs from TTB by using all cues rather than a single discriminating cue, and from Tallying by using negative validity-scaled weights rather than equal positive weights. It should be tested on designs that independently vary top-cue direction and lower-cue tally to confirm the predicted negative relations."
}
```

## Usage

```json
{
  "input_tokens": 5774,
  "output_tokens": 23961,
  "reasoning_tokens": 19411,
  "provider": "CoreWeave"
}
```
