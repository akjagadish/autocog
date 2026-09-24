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
**Validities (n_features=6):** [0.92, 0.84, 0.76, 0.68, 0.6, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 10: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 14: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 16: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 17: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 18: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 19: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 20: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 21: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 22: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 23: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  trial 24: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 25: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 26: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 27: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 28: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 29: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 30: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 31: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 32: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design makes the highest-validity cue disagree with the majority of the remaining cues on every trial. All six validities are ordered so feature 1 is the top TTB cue; every critical pair has that top cue discriminate in one direction while at least three of the five lower-validity cues discriminate in the opposite direction. Take The Best stops at the top cue and predicts the top-cue winner; Tallying counts all feature-wise wins and predicts the opposite majority. The 32 unique pairs contain 16 top-cue-favors-A cases and 16 top-cue-favors-B cases to balance any position bias. For a top-A pair where m lower cues favor B (here m = 3, 4, or 5), TTB predicts choice A with probability about 1/(1+exp(-beta)) before lapse, while Tallying predicts choice B with probability about 1/(1+exp(-beta*(m-1))), so the theories are separated in direction at moderate and high beta. The 32 unique pairs are repeated K = 3 times for 96 total trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This design makes the highest-validity cue (feature 0) discriminate in the opposite direction to the majority of the remaining cues on every trial. Take The Best therefore predicts the option favored by the first discriminating cue, while Tallying predicts the opposite option favored by the lower-cue tally. The metric is the proportion of trials on which a subject follows the first discriminating cue. It uses all 96 trials per subject, keeping per-subject binomial variance small (about p(1-p)/96). Under TTB the score is above 0.5, approaching 1 with stronger beta; under Tallying it is below 0.5 because the top cue is the minority tally cue. The design balances top-cue-favors-A and top-cue-favors-B trials, so the raw proportion is not confounded by position bias. This gives a reliable mean separation with tight between-subject variance at N=25.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        # Cues are in descending-validity order, so the first discriminating
        # feature is the Take The Best cue.
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            if b[j] > a[j]:
                pred = 1
                break
        if pred is not None:
            matches.append(1.0 if r == pred else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8723 (var=0.0096)
- Predicted under pi_2 (simulated): 0.1175 (var=0.0068)
- Observed on real data: 0.2487 (var=0.0065)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.78, 0.66, 0.58, 0.52]

**Trial pairs (n=32):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  trial 13: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  trial 14: A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 15: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 18: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  trial 19: A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 20: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 21: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  trial 22: A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 23: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 24: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 25: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 26: A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 27: A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 28: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  trial 29: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0]
  trial 30: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 31: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 32: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** This design isolates a second-cue stopping signature. The highest-validity cue (feature 0, validity 0.90) is tied on every trial, so Take The Best skips it and stops at the second cue (feature 1, validity 0.78). Feature 1 always discriminates, so TTB never consults the three lower cues; it predicts the same choice probability for all feature-1-favors-A trials and the opposite for all feature-1-favors-B trials. Tallying, by contrast, treats the top tie as uninformative and counts wins on all remaining cues. The three lower cues are varied so that Tallying's tally difference d = A_wins - B_wins takes values -2, -1, and 0 for feature-1-A trials, and 0, +1, and +2 for feature-1-B trials. Thus TTB predicts a flat choice curve across lower-cue patterns within each direction, whereas Tallying predicts a logistic curve in d, producing opposite choices on the d=-1/-2 and d=+1/+2 patterns and guessing on d=0. The 32 unique pairs are balanced over top-tie value (0/0 vs 1/1) and second-cue direction, and are repeated K=3 for 96 total trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
The metric contrasts, trial by trial, whether a choice follows the full-tally prediction or the stopping-at-the-first-discriminating-cue prediction. Because feature 0 is tied everywhere, feature 1 is the first discriminating cue for Take The Best. On every non-tied trial in this design, the second cue prediction and the tally prediction point in opposite directions, so tally_score - second_score is positive when choices use lower cues and negative when choices stop at feature 1. Averaging over all 96 trials per subject makes the statistic stable under response noise while preserving the large direction-of-effect gap between Tallying and Take The Best.

Source:
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        response_a = 1.0 if int(row["response"]) == 0 else 0.0

        # Take The Best style second-cue prediction.
        # By design, feature 0 is tied on every trial, so feature 1 is
        # the first discriminating cue.
        if a[1] > b[1]:
            second_wins_a = True
        elif b[1] > a[1]:
            second_wins_a = False
        else:
            second_wins_a = True

        second_score = response_a if second_wins_a else (1.0 - response_a)

        # Tallying total score. Feature-specific ties contribute zero.
        wins_a = float(np.sum(a > b))
        wins_b = float(np.sum(b > a))
        tally_d = wins_a - wins_b

        if tally_d > 0.0:
            tally_score = response_a
        elif tally_d < 0.0:
            tally_score = 1.0 - response_a
        else:
            # Tallying has no preference on a tied tally.
            tally_score = 0.5

        scores.append(tally_score - second_score)

    return float(np.mean(scores)) if scores else 0.0

### RESULTS
- Predicted under pi_1 (simulated): -0.6075 (var=0.0314)
- Predicted under pi_2 (simulated): 0.6304 (var=0.0462)
- Observed on real data: 0.4196 (var=0.0128)

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
  "interpretation": "Across both experiments, pi_2 (Tallying) is clearly the better theory. In Experiment 1 it predicts 0.1175 against an observed 0.2487, a miss of 0.131, while pi_1 predicts 0.8723, a miss of 0.624 and the wrong side of 0.5. In Experiment 2 pi_2 predicts 0.6304 against an observed 0.4196, a miss of 0.211 and the correct sign, while pi_1 predicts -0.6075, a miss of about 1.03 and the opposite direction. So pi_1 is not task-invariant: it is directionally wrong in both experiments. Pi_2 is directionally right in both but is systematically less extreme than its prediction. The observed values are remarkably close to what a within-trial mixture of TTB and Tallying would produce. In Experiment 1 the observed top-cue following rate is 0.2487, implying a TTB mixture weight q of about 0.25. In Experiment 2, using approximate deterministic signatures of -0.922 for TTB and 0.844 for Tallying, the observed 0.4196 implies q = (0.844 - 0.4196) / (0.844 + 0.922) = 0.24. This convergence across independent designs is a strong signature of a stochastic mixture, not an artifact of a single beta or lapse correction. The low between-subject variances, especially 0.0065 and 0.0128 in the observed data, further favor a within-subject probabilistic mixture over a subgroup-level strategy split.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a new trial-level strategy-mixture theory. The theory should state that on each choice a decision maker probabilistically selects between a one-reason stopping process and a full tallying process: with probability q the choice uses the first discriminating cue in validity order, and with probability 1 - q the choice uses the full feature-win tally. The mixture probability q should be centered near 0.25, with a narrow prior such as Beta(5, 15) or Uniform(0.15, 0.35). The two core strategies should then be passed through modest softmax or lapse noise, with beta_tally around 1 or slightly lower and beta_ttb moderately high, while epsilon should remain small, roughly 0.0 to 0.1. Because observed between-subject variance is low, the mixture should be modeled at the trial level within subjects, or with only very narrow subject-level variation in q. This new theory differs from pi_1 because it does not always stop at the first discriminating cue, and it differs from pi_2 because it allows first-cue stopping with a non-zero probability. It should recover pi_2 as a special case when q = 0 and pi_1 as a special case when q = 1, but the data indicate an intermediate q of approximately 0.25."
}
```

## Usage

```json
{
  "input_tokens": 7390,
  "output_tokens": 18631,
  "reasoning_tokens": 17964,
  "provider": "Baidu"
}
```
