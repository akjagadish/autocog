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
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** This design uses six features with linearly spaced validities to quantitatively dissociate Tallying (which simply counts feature-wise wins) from WADD (which weights wins by feature validity). The trial pairs include cases where Tallying prefers one option while WADD prefers the other, as well as cases where one model is indifferent but the other strongly prefers an option. For example, comparing an option that wins on three lower-validity cues against one that wins on the two highest-validity cues creates a sharp divergence: Tallying favors the former (3 vs 2 wins), whereas WADD favors the latter (due to higher weighted sum).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the two theories strongly disagree. It identifies trials where one option wins on more features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). It then calculates the proportion of choices on these specific trials that align with the Tallying heuristic. Tallying agents will score near 1.0 (modulo noise), while WADD agents will score near 0.0, providing maximal discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    tally_matches = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        wadd_a = np.dot(a, v)
        wadd_b = np.dot(b, v)
        
        tally_pref = None
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
            
        wadd_pref = None
        if wadd_a > wadd_b:
            wadd_pref = 0
        elif wadd_b > wadd_a:
            wadd_pref = 1
            
        if tally_pref is not None and wadd_pref is not None and tally_pref != wadd_pref:
            disagreement_trials += 1
            if row['response'] == tally_pref:
                tally_matches += 1
                
    if disagreement_trials == 0:
        return 0.5
        
    return float(tally_matches / disagreement_trials)

### RESULTS
- Predicted under pi_1 (simulated): 0.8569 (var=0.0086)
- Predicted under pi_2 (simulated): 0.4875 (var=0.0782)
- Observed on real data: 0.3500 (var=0.0088)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) theory from Tallying using a five-feature structure with highly skewed validities. In trials 1 and 2, Tallying predicts a preference for the option that wins on three low-validity features, whereas WADD predicts a preference for the option that wins on the two highest-validity features. In trials 3 through 8, the options are tied in terms of the total number of winning features (either 2 vs 2 or 1 vs 1), leading Tallying to predict complete indifference (guessing). However, WADD strongly predicts preferences on these trials because the features won by one option have substantially higher validities than those won by the other.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials 1 and 2, which are the only trials where the total number of features favoring each option differs (one option has 3 positive features, the other has 2). Tallying strongly predicts choosing the option with 3 features (yielding a metric near 1.0), while WADD strongly predicts choosing the option with 2 features because those features have much higher validities (yielding a metric near 0.0). By calculating the proportion of choices that align with the simple feature count on these specific trials, we achieve a massive, robust separation between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    def tally_match(row):
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        if sum_a > sum_b:
            return 1.0 if row['response'] == 0 else 0.0
        elif sum_b > sum_a:
            return 1.0 if row['response'] == 1 else 0.0
        else:
            return np.nan
            
    matches = data.apply(tally_match, axis=1)
    return float(matches.dropna().mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8558 (var=0.0129)
- Predicted under pi_2 (simulated): 0.4275 (var=0.0930)
- Observed on real data: 0.3283 (var=0.0114)

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
  "interpretation": "Across both experiments, the observed metrics (0.3500 and 0.3283) fall well below 0.5, indicating that subjects systematically prefer the option favored by the Weighted Additive (WADD) rule over the one favored by Tallying. Theory 1 (Tallying) incorrectly predicts values around 0.85. Theory 2 (WADD) correctly predicts values below 0.5, but its predictions (~0.43-0.48) are too close to 0.5 (random guessing) compared to the observed data. This under-prediction of the WADD effect is due to a flaw in Theory 2's model implementation: it multiplies the known `validities` by an additional free parameter vector `w`. If `w` is sampled broadly across simulations, it effectively scrambles or dilutes the true validities, pulling the aggregate predictions toward 0.5. Removing this redundant parameter will allow Theory 2 to strongly predict the observed behavior.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Modify the `predict` function to compute scores directly using the provided validities: `scores = stim @ validities`. Remove the redundant `weights` parameter (`w`) that is currently being multiplied element-wise with `validities`, as it scrambles the objective cue validities and dilutes the model's predictions. Keep the `beta` and `epsilon` parameters for noise."
}
```

## Usage

```json
{
  "prompt_token_count": 4322,
  "candidates_token_count": 329,
  "total_token_count": 5589
}
```
