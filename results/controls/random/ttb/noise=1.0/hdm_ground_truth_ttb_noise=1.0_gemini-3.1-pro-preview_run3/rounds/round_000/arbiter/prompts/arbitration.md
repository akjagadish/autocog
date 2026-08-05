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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  trial 2: A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  trial 3: A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  trial 4: A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  trial 5: A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  trial 6: A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  trial 7: A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  trial 8: A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Rationale:** To dissociate Tallying from WADD, we exploit two core differences: Tallying ignores both feature validities and cardinal feature magnitudes, relying solely on the count of strict feature-wise wins. By using a rating_max of 10, we can create trials where Option A wins on a majority of features by a small margin, leading Tallying to strongly prefer A. However, Option B can be given a massive magnitude advantage on the remaining high-validity features, causing WADD to strongly prefer B. We also include trials where Tallying sees a tie (2 wins each) but WADD has a strong preference due to magnitudes and validities, isolating the compensatory nature of WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Option A wins on a strict majority of features (3 out of 4) but Option B wins on the single most valid feature by a large margin (trials 1, 4, 6, 8). Tallying ignores feature validities and magnitudes, simply counting the wins (3 vs 1), and therefore strongly prefers Option A (response = 0). WADD, on the other hand, integrates the large magnitude advantage on the highest-validity cue, allowing it to compensate for the three smaller losses, and thus leans towards Option B (response = 1). The metric computes the mean response on these specific trials, predicting a value close to 0 for Tallying and significantly higher for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.5850 (var=0.1940)
- Observed on real data: 0.5042 (var=0.0050)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  trial 3: A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  trial 4: A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  trial 5: A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  trial 6: A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Rationale:** This design systematically dissociates the Weighted Additive (WADD) strategy from the Tallying heuristic. Tallying chooses strictly based on the count of features where one option exceeds another, ignoring both the validity of the features and the magnitude of the difference. WADD, by contrast, considers both feature validities and the cardinal magnitude of differences. The trials are constructed such that in several cases, Option A has a small advantage on a majority of features (causing Tallying to strongly prefer A), but Option B has a massive magnitude advantage on the highest-validity feature (causing WADD to strongly prefer B). We also include tie-trials for Tallying (equal number of feature wins) where WADD has a strong preference due to validities, and vice versa.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the four specific trial types where the Weighted Additive (WADD) model and the Tallying heuristic make strictly opposing predictions. Tallying relies purely on the count of winning features, ignoring the magnitude of those wins, while WADD computes the validity-weighted sums of the cardinal values. On these filtered trials, Tallying strongly prefers one option due to a 3-to-1 feature win advantage, whereas WADD prefers the opposite option because the single feature loss is overwhelmingly large in magnitude and on the highest validity cue. The metric calculates the proportion of choices on these trials that align with the WADD prediction. A value close to 1 indicates behavior consistent with WADD, while a value close to 0 indicates behavior consistent with Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.7800 (var=0.0816)
- Observed on real data: 0.5156 (var=0.0035)

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
  "interpretation": "Across both experiments, Tallying (pi_1) predicts a strict preference (0.0) because it only counts feature-wise wins, ignoring magnitudes and validities. WADD (pi_2) predicts a strong preference for the other option (>0.5) because it integrates the massive magnitude advantage on the highest-validity cue. However, the observed data in both experiments is centered remarkably close to 0.5 with extremely low between-subject variance (var ~ 0.004). This low variance indicates that individual subjects are not strictly using Tallying (which would yield 0) nor TTB/WADD (which would yield 1), nor are they bimodally split (which would produce high variance). Instead, every subject is choosing A and B with roughly equal probability on these specific trial types. This suggests that the compensatory mechanisms in WADD are capturing the trade-off, but the current parameterization overestimates the compensatory weight of the single highest-validity cue. WADD is conceptually closer to capturing this balance than Tallying, which rigidly predicts 0.0.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Preserve the WADD theory description, but regenerate the model's parameter ranges. Specifically, broaden the prior over `validities` mapping and `beta` (inverse temperature). A lower `beta` or a more compressed mapping of subjective weights will allow the model to predict choice probabilities closer to 0.5 on these highly conflicting trials, accurately reflecting the observed indifference in the real data."
}
```

## Usage

```json
{
  "prompt_token_count": 4227,
  "candidates_token_count": 346,
  "total_token_count": 8732
}
```
