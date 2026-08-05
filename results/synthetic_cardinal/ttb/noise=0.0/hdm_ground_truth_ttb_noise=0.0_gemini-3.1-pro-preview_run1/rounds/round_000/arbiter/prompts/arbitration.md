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

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 5

**Trial pairs (n=5):**
  trial 1: A=[1, 4, 4, 4]  B=[5, 3, 3, 3]
  trial 2: A=[5, 2, 2, 2]  B=[1, 3, 3, 3]
  trial 3: A=[2, 5, 2, 2]  B=[3, 1, 3, 3]
  trial 4: A=[4, 4, 1, 5]  B=[5, 5, 2, 0]
  trial 5: A=[0, 0, 5, 5]  B=[1, 1, 4, 4]

**Rationale:** To dissociate Tallying from WADD, we use cardinal features (rating_max=5) and vary both the number of winning features and the magnitude of those wins. Tallying ignores both feature validities and the magnitude of differences, counting only the number of features where one option strictly exceeds the other. WADD computes a validity-weighted sum of the cardinal ratings. The trials are designed such that in some cases, one option wins on a majority of features by a small margin (favored by Tallying), while the other option wins on fewer, high-validity features by a large margin (favored by WADD).

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Calculates the proportion of choices consistent with the Tallying heuristic on trials 1-4, where Tallying and WADD have strictly opposite preferences. Tallying will yield a metric value near 1.0, whereas WADD will yield a value near 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        if a[0] == 1:
            matches.append(r == 0)
        elif a[0] == 5:
            matches.append(r == 1)
        elif a[0] == 2:
            matches.append(r == 1)
        elif a[0] == 4:
            matches.append(r == 1)
    if not matches:
        return 0.5
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.3600 (var=0.0729)
- Observed on real data: 0.7250 (var=0.0281)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=5):**
  trial 1: A=[5, 5, 5, 5]  B=[0, 6, 6, 6]
  trial 2: A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  trial 3: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 4: A=[10, 0, 0, 0]  B=[0, 5, 5, 4]
  trial 5: A=[8, 4, 4, 4]  B=[5, 5, 5, 5]

**Rationale:** To rigorously dissociate WADD from Tallying, this design uses four cardinal features (max rating 10) and a steep validity gradient. Tallying ignores both magnitude and validities, relying solely on the number of winning features. The trial pairs are constructed to pit these mechanisms against each other: in some trials, one option wins on a single high-validity feature by a large magnitude (dominating the WADD score) while the other option wins on all remaining features by small margins (dominating the Tallying score). We also include trials that produce a perfect tie for one model but a strong preference for the other, isolating the specific computational mechanisms (weighted sum vs. unweighted count) driving choice.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
On Trial 2, option A has values [0, 4, 4, 4] and option B has [10, 0, 0, 0]. WADD heavily weights the first feature (validity 0.9), leading to a strong preference for B (score 9.0 vs 7.6). Tallying completely ignores the magnitudes and validities, simply counting that A wins on 3 out of 4 features, leading to a strong preference for A. The mean response (where 1 = chose B) on this trial will be near 1 for WADD and near 0 for Tallying, providing a maximal, stable contrast between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[0, 4, 4, 4], B=[10, 0, 0, 0]
    # WADD prefers B (response=1) because 10*0.9 = 9.0 > 4*(0.8+0.6+0.5) = 7.6
    # Tallying prefers A (response=0) because A wins on 3 features vs B's 1
    
    # Identify trial 2 by checking the first element of option A
    is_trial_2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 4, 4, 4))
    trial_data = data[is_trial_2]
    
    if len(trial_data) == 0:
        return 0.5
        
    return float(trial_data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.3600 (var=0.2304)
- Observed on real data: 0.2600 (var=0.1924)

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
  "interpretation": "Tallying (THEORY 1) captures the qualitative direction of human behavior much better than WADD (THEORY 2). In Experiment 1, humans chose the Tallying-favored option 72.5% of the time when pitted against WADD. In Experiment 2, WADD strongly preferred option B (due to a single large value on the highest-validity feature), but humans only chose B 26% of the time, aligning with Tallying's preference for option A (which won on 3 out of 4 features). While Tallying correctly predicts the majority choice, its deterministic predictions (1.0 and 0.0) are too extreme compared to the observed human data (0.725 and 0.26), suggesting humans exhibit noise or use a mix of strategies. WADD completely fails to predict the majority direction in both experiments. Therefore, THEORY 2 should be replaced with a stronger competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose an 'Equal-Weight' (EQW) theory. Unlike Tallying, EQW does not discard cardinal feature magnitudes; instead, it sums up the raw feature values for each option and chooses the one with the highest total. Unlike WADD, it ignores the cue validities, treating all features as equally important. This will serve as a strong baseline that isolates whether subjects are ignoring validities (like Tallying and EQW) or ignoring cardinal magnitudes (like Tallying), providing a cleaner contrast to the winning Tallying theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3874,
  "candidates_token_count": 353,
  "total_token_count": 5962
}
```
