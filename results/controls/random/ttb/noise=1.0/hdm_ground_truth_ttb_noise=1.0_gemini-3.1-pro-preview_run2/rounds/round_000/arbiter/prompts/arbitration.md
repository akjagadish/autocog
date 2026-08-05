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
  trial 1: A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[5, 5, 5, 5]  B=[4, 4, 4, 10]
  trial 3: A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 10, 0, 0]  B=[1, 0, 1, 1]
  trial 5: A=[5, 5, 6, 6]  B=[6, 6, 0, 0]
  trial 6: A=[2, 2, 2, 2]  B=[0, 0, 10, 10]
  trial 7: A=[4, 4, 4, 4]  B=[3, 3, 5, 5]
  trial 8: A=[0, 10, 10, 10]  B=[10, 0, 0, 0]

**Rationale:** To dissociate Tallying from WADD, we exploit two key differences: Tallying ignores feature validities and feature magnitudes, whereas WADD integrates both. By setting rating_max to 10, we can create choices where one option wins on a majority of features (thus favored by Tallying) but loses on the weighted sum due to large deficits on high-validity features (thus the other option is favored by WADD). The design also includes ties in Tallying where WADD has a strong preference, to further distinguish the models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with Tallying's strict preferences on three specific trials that perfectly dissociate Tallying from WADD. On these trials, Tallying prefers an option that wins on the majority of features (e.g., 3 out of 4), while WADD prefers the other option due to a massive magnitude advantage on a single high-validity feature. The metric will be near 1.0 for data simulated under Tallying and near 0.0 for WADD, providing strong discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    relevant_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        
        # Trial 1: A=[1, 6, 6, 6], B=[10, 5, 5, 5]
        if a[0] == 1 and a[1] == 6 and a[2] == 6:
            tallying_consistent += (1 if r == 0 else 0)
            relevant_trials += 1
            
        # Trial 3: A=[10, 0, 0, 0], B=[0, 1, 1, 1]
        elif a[0] == 10 and a[1] == 0 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
        # Trial 4: A=[0, 10, 0, 0], B=[1, 0, 1, 1]
        elif a[0] == 0 and a[1] == 10 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
    return float(tallying_consistent / relevant_trials) if relevant_trials > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.1333 (var=0.0578)
- Observed on real data: 0.5056 (var=0.0050)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=3):** [0.9, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[2, 6, 6]  B=[10, 5, 5]
  trial 2: A=[8, 4, 3]  B=[2, 5, 4]
  trial 3: A=[10, 0, 5]  B=[0, 10, 5]
  trial 4: A=[0, 10, 5]  B=[10, 0, 5]
  trial 5: A=[5, 5, 5]  B=[1, 6, 6]
  trial 6: A=[8, 2, 2]  B=[10, 1, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, this design uses cardinal ratings (rating_max=10) and varying expert validities. Tallying only counts strict feature-wise wins, ignoring both the magnitude of the difference and the validity of the feature. WADD, in contrast, integrates both cardinal magnitudes and validity weights. The trial pairs are constructed to pit these mechanisms against each other: in several trials, one option wins on a majority of features by a small margin (favored by Tallying) while the other option wins on fewer, higher-validity features by a large margin (favored by WADD). Additionally, trials where Tallying results in a tie but WADD shows a strong preference due to validity weighting are included to further separate the predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that agree with the deterministic predictions of the WADD model using the provided expert validities. Because the experimental design pits WADD against Tallying directly (Tallying predicts the exact opposite choice on 4 of the 6 trial types, and is indifferent on the other 2), this metric will yield a value close to 1.0 for WADD-simulated data and close to 0.167 for Tallying-simulated data, providing a massive, highly stable contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    validities = np.array([0.9, 0.6, 0.5])
    
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    
    wadd_pred = (score_b > score_a).astype(int)
    
    responses = data['response'].values
    return float(np.mean(responses == wadd_pred))


### RESULTS
- Predicted under pi_1 (simulated): 0.1667 (var=0.0000)
- Predicted under pi_2 (simulated): 0.8433 (var=0.0738)
- Observed on real data: 0.4963 (var=0.0025)

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
  "interpretation": "Across both experiments, the observed metric values are extremely close to 0.5 with very low between-subject variance. This indicates that on these highly conflicting trials\u2014where an option with a large magnitude advantage on a single high-validity cue is pitted against an option with small advantages on multiple lower-validity cues\u2014individual subjects are essentially indifferent or guessing. Tallying (Theory 1) predicts a near-deterministic preference for the option with more winning features (~1.0 or ~0.16), while WADD (Theory 2) predicts a near-deterministic preference for the option with the higher valid-weighted sum (~0.13 or ~0.84). Neither theory correctly captures the robust indifference observed in the real data. Given that both foundational compensatory (WADD) and non-compensatory tallying heuristics fail to capture this, we should introduce the quintessential non-compensatory lexicographic heuristic\u2014Take The Best (TTB)\u2014to see if a one-reason decision mechanism with high noise or strict discrimination thresholds better models this uncertainty.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic. People search through features in descending order of their subjective validity. They stop at the first feature that discriminates between the two options (i.e., where one option has a strictly higher value than the other, perhaps exceeding a small threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents the classic fast-and-frugal one-reason decision making benchmark."
}
```

## Usage

```json
{
  "prompt_token_count": 4150,
  "candidates_token_count": 365,
  "total_token_count": 6013
}
```
