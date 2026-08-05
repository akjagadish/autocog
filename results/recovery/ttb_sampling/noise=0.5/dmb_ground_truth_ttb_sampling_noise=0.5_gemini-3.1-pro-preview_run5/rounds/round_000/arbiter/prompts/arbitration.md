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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design dissociates Tallying from the Weighted Additive (WADD) rule by pitting feature quantity against feature quality. Tallying simply counts the number of features on which an option is superior, ignoring the validities (weights) of those features. WADD, on the other hand, weights each feature's value by its validity. By providing a steep drop-off in validities (e.g., highly predictive first and second features, but moderately predictive third, fourth, and fifth features), we can construct trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by WADD). We also include trials where both models agree, and trials where Tallying sees a tie but WADD strongly prefers one option.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific trials in the design where Tallying and WADD make diametrically opposite predictions. On trials where one option has 3 less-valid features and the other has 2 highly-valid features, Tallying strictly prefers the option with 3 features (ignoring validity), while WADD prefers the option with 2 features (because their combined validity outweighs the 3). By calculating the proportion of choices that align with Tallying on these specific trials, we expect a value significantly above 0.5 for Tallying and significantly below 0.5 for WADD, cleanly discriminating the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_consistent = 0
    total_relevant = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: Tallying prefers A (count 3 vs 2), WADD prefers B (weight 1.8 vs 1.7)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if row['response'] == 0:
                tally_consistent += 1
            total_relevant += 1
        # Trial 2: Tallying prefers B (count 3 vs 2), WADD prefers A (weight 1.8 vs 1.7)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if row['response'] == 1:
                tally_consistent += 1
            total_relevant += 1
            
    if total_relevant == 0:
        return 0.5
    return float(tally_consistent / total_relevant)

### RESULTS
- Predicted under pi_1 (simulated): 0.8408 (var=0.0182)
- Predicted under pi_2 (simulated): 0.5383 (var=0.0797)
- Observed on real data: 0.3683 (var=0.0161)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Rationale:** This design systematically dissociates the Weighted Additive (WADD) model from the Tallying heuristic by pitting feature quantity against feature quality. By using a set of validities with a steep drop-off between the top two features and the remaining four, we can construct pairs where one option is supported by a greater number of low-validity features (favored by Tallying) while the other option is supported by fewer but highly predictive features (favored by WADD). The design includes a mix of trials where the models strongly disagree, trials where Tallying perceives a tie but WADD has a clear preference, and baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on the two trials where WADD and Tallying have completely opposing, deterministic preferences. In Trial 1, WADD strongly prefers B (response 1) because of the two highest-validity features, while Tallying prefers A (response 0) due to a higher quantity of lower-validity features. In Trial 4, the options are swapped, so WADD prefers A (response 0) and Tallying prefers B (response 1). The difference in mean response between Trial 1 and Trial 4 will be strongly positive for WADD (~1.0) and strongly negative for Tallying (~-1.0).

Source:
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    
    m1 = data.loc[is_t1, 'response'].mean()
    m4 = data.loc[is_t4, 'response'].mean()
    
    if pd.isna(m1):
        m1 = 0.5
    if pd.isna(m4):
        m4 = 0.5
        
    return float(m1 - m4)

### RESULTS
- Predicted under pi_1 (simulated): -0.6600 (var=0.0611)
- Predicted under pi_2 (simulated): 0.0733 (var=0.4268)
- Observed on real data: 0.3600 (var=0.0410)

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
  "interpretation": "Across both experiments, Tallying (Theory 1) consistently predicts the wrong direction of the effect. In Experiment 1, subjects preferred the higher-validity option over the higher-quantity option (observed 0.3683, whereas Tallying predicted 0.8408). In Experiment 2, subjects again favored the higher-validity features (observed 0.3600, whereas Tallying predicted -0.6600). WADD (Theory 2) correctly predicts the direction of the effects but underestimates their magnitude, likely because its parameter ranges for inverse temperature (beta) or subjective weights smooth out the differences too much. Since Tallying is fundamentally failing to capture the qualitative behavior, it should be replaced with a stronger competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the 'Take The Best' (TTB) heuristic. TTB posits a lexicographic decision rule: people search through features in descending order of validity and stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. This differs from Tallying (which ignores validities and counts all features) and WADD (which compensates lower validities with higher quantities). TTB will strongly predict the choice of the option with the highest-validity cues, providing a distinct and often more deterministic contrast to WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 4454,
  "candidates_token_count": 321,
  "total_token_count": 6202
}
```
