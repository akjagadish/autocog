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
**Validities (n_features=5):** [0.9, 0.8, 0.5, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 12: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design aims to dissociate Tallying (which counts the number of features favoring each option, ignoring validities and magnitudes) from Weighted Additive (WADD) (which computes the weighted sum of feature values). By using five features with a steep validity drop-off, we can create pairs where a few high-validity features outweigh several lower-validity features (WADD preference) versus pairs where the sheer number of lower-validity features dictates the choice (Tallying preference). For instance, in Trial 1, Option A has the two best features (WADD wins) while Option B has the three worst features (Tallying wins).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the Tallying heuristic (i.e., choosing the option that wins on the most features, ignoring validities). Ties are excluded. The advocated theory (Tallying) will score highly on this metric. The competing theory (WADD) will score lower because it will frequently choose an option with fewer winning features if those features have higher validities (e.g., in Trial 1, WADD often prefers Option A which has 2 high-validity features, whereas Tallying prefers Option B which has 3 low-validity features).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
        if row['response'] == tally_pref:
            match_count += 1
        total += 1
    return match_count / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.8550 (var=0.0092)
- Predicted under pi_2 (simulated): 0.6618 (var=0.0121)
- Observed on real data: 0.4832 (var=0.0024)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design quantitatively dissociates Weighted Additive (WADD) from Tallying using 5 features. The validities are chosen such that the sum of the top two validities (0.9 + 0.8 = 1.7) is slightly greater than the sum of the bottom three validities (0.6 + 0.55 + 0.5 = 1.65). This enables critical trials where Option A has the top two features and Option B has the bottom three: WADD prefers Option A (higher weighted sum), whereas Tallying prefers Option B (higher count of winning features, 3 vs 2). Additional trials include cases where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to validity differences, as well as baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the WADD model on the critical trials where WADD and Tallying make opposite predictions. Specifically, it looks at trials where one option has the top two most valid features and the other option has the bottom three. WADD prefers the option with the top two features because their combined validity outweighs the bottom three, whereas Tallying prefers the option with the bottom three features because it simply counts the number of winning features (3 vs 2). Thus, WADD will score highly on this metric, while Tallying will score poorly.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has top 2 features, B has bottom 3
    t1 = (a_str == '11000') & (b_str == '00111')
    # Trial 2: A has bottom 3 features, B has top 2
    t2 = (a_str == '00111') & (b_str == '11000')
    
    critical = t1 | t2
    if not critical.any():
        return 0.5
        
    # WADD prefers the option with the top 2 features (A in t1, B in t2)
    # Tallying prefers the option with the bottom 3 features (since 3 > 2)
    wadd_choices = (t1 & (data['response'] == 0)) | (t2 & (data['response'] == 1))
    return float(wadd_choices.sum() / critical.sum())

### RESULTS
- Predicted under pi_1 (simulated): 0.1275 (var=0.0075)
- Predicted under pi_2 (simulated): 0.4917 (var=0.0953)
- Observed on real data: 0.4750 (var=0.0061)

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
  "interpretation": "Tallying (THEORY 1) completely fails to capture human behavior across both experiments. In Experiment 1, Tallying predicted an 85% match with its heuristic, but the observed data was near chance (48%). In Experiment 2, Tallying predicted a 12% match on critical trials, but observed behavior was again near 47%. WADD (THEORY 2) performed better on Experiment 2 by predicting the near-chance behavior (due to the small difference in weighted sums: 1.7 vs 1.65), but it overestimated the Tallying metric in Experiment 1. The near-chance performance on the Tallying metric in Experiment 1 suggests subjects are frequently choosing the option with fewer total winning features, heavily weighting the most valid feature instead. This strongly points toward a non-compensatory heuristic like Take The Best, which Tallying entirely fails to represent.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic as a brand-new theory. Unlike Tallying (which weights all features equally) and WADD (which computes a compensatory weighted sum), TTB assumes subjects order features by their subjective validity and make a choice based solely on the first feature that discriminates between the two options. This non-compensatory approach would explain the strong deviations from Tallying and the reliance on the highest-validity cues."
}
```

## Usage

```json
{
  "prompt_token_count": 4524,
  "candidates_token_count": 328,
  "total_token_count": 5708
}
```
