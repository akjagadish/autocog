# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

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
    return int(np.argmax(probabilities))


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 5

**Trial pairs (n=8):**
  trial 1: A=[4, 1, 1, 1]  B=[3, 2, 2, 2]
  trial 2: A=[2, 3, 3, 3]  B=[3, 1, 1, 1]
  trial 3: A=[5, 5, 1, 1]  B=[5, 4, 2, 2]
  trial 4: A=[3, 2, 4, 4]  B=[3, 3, 1, 1]
  trial 5: A=[4, 2, 1, 1]  B=[3, 3, 2, 2]
  trial 6: A=[1, 4, 4, 4]  B=[2, 2, 2, 2]
  trial 7: A=[5, 4, 1, 1]  B=[5, 3, 2, 2]
  trial 8: A=[2, 2, 4, 4]  B=[2, 3, 1, 1]

**Rationale:** To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue (or the first discriminating cue) points to one option, while the simple majority of features point to the other. TTB decides solely based on the highest-validity discriminating cue, ignoring all others. Tallying, on the other hand, ignores validities and simply counts the number of features on which each option is superior. We include trials where TTB and Tallying make strictly opposing predictions, as well as trials where the top cue ties and the conflict cascades to the second cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design consists entirely of conflict trials where Take The Best (TTB) and Tallying make strictly opposing predictions. By computing the proportion of trials where the subject's choice aligns with the TTB prediction, we can perfectly discriminate the two heuristics. Under TTB, this metric will be close to 1.0 (subject to noise), whereas under Tallying it will be close to 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = None
        # Features are implicitly ordered by validity (highest to lowest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_match += 1
            total += 1
    return ttb_match / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.3308 (var=0.0607)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**rating_max:** 5

**Trial pairs (n=8):**
  trial 1: A=[5, 2, 2, 2, 2]  B=[2, 5, 5, 5, 2]
  trial 2: A=[2, 5, 5, 5, 2]  B=[5, 2, 2, 2, 2]
  trial 3: A=[5, 5, 2, 2, 2]  B=[5, 2, 5, 5, 5]
  trial 4: A=[4, 4, 4, 2, 2]  B=[4, 4, 2, 5, 5]
  trial 5: A=[3, 2, 5, 5, 5]  B=[3, 5, 2, 2, 2]
  trial 6: A=[4, 3, 3, 3, 3]  B=[3, 4, 4, 4, 4]
  trial 7: A=[1, 5, 1, 1, 1]  B=[1, 1, 5, 5, 5]
  trial 8: A=[5, 1, 5, 5, 5]  B=[1, 5, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tallying from Take The Best (TTB), we present a 5-feature design using cardinal ratings (0-5 scale). TTB is a non-compensatory, one-reason decision rule that stops at the first discriminating feature, ordered by validity. Tallying ignores both the feature validities and the cardinal magnitude of the differences, counting only the number of features on which an option is strictly better. We construct trials where the highest-validity discriminating cue points to one option, but a simple majority of the remaining lower-validity cues point to the other option. By including trials where the top cue discriminates, as well as trials where the top cues tie and the conflict cascades to the 2nd or 3rd cue, we can isolate whether subjects are using a single-feature TTB strategy or a multi-feature Tallying strategy.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the Tallying heuristic. In 7 out of the 8 unique trial pairs in this experimental design, Tallying and Take The Best (TTB) predict opposite choices. Therefore, data generated by the Tallying theory will yield a metric value close to 1.0, while data generated by the TTB theory will yield a value close to 0.125 (since it only agrees on 1 out of 8 trial types). This creates a massive mean difference with low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def tally_predicts(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(x > y for x, y in zip(a, b))
        b_wins = sum(y > x for x, y in zip(a, b))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        return -1
    predictions = data.apply(tally_predicts, axis=1)
    return float(np.mean(data['response'] == predictions))

### RESULTS
- Predicted under pi_1 (simulated): 0.1250 (var=0.0000)
- Predicted under pi_2 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.7270 (var=0.0429)

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
  "interpretation": "Across both experiments, the observed data falls between the predictions of Take The Best (TTB) and Tallying, but consistently leans much closer to Tallying. In Experiment 1, subjects chose the TTB-predicted option 33% of the time (meaning they chose the Tallying-predicted option ~67% of the time). In Experiment 2, subjects chose the Tallying-predicted option 73% of the time. While Tallying is a better fit than TTB, neither theory perfectly captures the data. Both TTB and Tallying discard the cardinal magnitudes of the ratings. A compensatory strategy that integrates both cue validities and the cardinal magnitudes of the feature ratings (such as a Weighted Additive model) would likely predict this intermediate behavior, as large magnitude differences on lower-validity cues might outweigh small differences on higher-validity cues.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a Weighted Additive (WADD) theory. According to WADD, people evaluate options by multiplying each feature's cardinal rating by its cue validity, summing these products to form an overall expected value for each option, and choosing the option with the higher total value. This differs from TTB (which ignores all cues but the best discriminating one and ignores magnitudes) and Tallying (which ignores cue validities and cardinal magnitudes). Response noise should enter through a softmax over the options' total values."
}
```

## Usage

```json
{
  "prompt_token_count": 4306,
  "candidates_token_count": 325,
  "total_token_count": 5278
}
```
