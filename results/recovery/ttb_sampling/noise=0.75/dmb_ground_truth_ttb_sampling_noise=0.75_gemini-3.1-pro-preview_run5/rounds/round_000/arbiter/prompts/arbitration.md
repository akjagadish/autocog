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
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design where the first two features have very high validities and the remaining three have low validities. Tallying simply counts the number of features on which an option is superior, ignoring validity magnitudes. WADD, however, integrates both the feature values and their validities. In the critical dissociation trials (Trials 1 and 2), one option wins on 3 low-validity features while the other wins on 2 high-validity features. Tallying strongly predicts the option with 3 wins, whereas WADD strongly predicts the option with 2 high-validity wins. We also include trials where Tallying predicts a tie (equal number of wins) but WADD has a strict preference, and agreement trials where both heuristics favor the same option, to serve as a baseline for overall choice consistency.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the Tallying heuristic on the critical dissociation trials (Trials 1 and 2). In these trials, one option has a higher total number of positive features (favored by Tallying) but lacks the highest-validity features (meaning WADD strongly favors the other option). Tallying will produce a metric value close to 1, whereas WADD will produce a metric value close to 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tallying_consistent = 0
    total_dissociation_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Identify dissociation trials: one option has more features, but lacks the first (highest validity) feature
        if sum_a > sum_b and a[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 0:
                tallying_consistent += 1
        elif sum_b > sum_a and b[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 1:
                tallying_consistent += 1
                
    if total_dissociation_trials == 0:
        return 0.5
        
    return float(tallying_consistent / total_dissociation_trials)

### RESULTS
- Predicted under pi_1 (simulated): 0.8669 (var=0.0105)
- Predicted under pi_2 (simulated): 0.4050 (var=0.0960)
- Observed on real data: 0.4113 (var=0.0093)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.55, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To cleanly dissociate the Weighted Additive (WADD) model from the Tallying heuristic, we use a 5-feature design with a mix of very high and very low validities: [0.95, 0.80, 0.55, 0.50, 0.50]. Tallying ignores validity magnitudes and simply counts the number of features on which an option is superior. WADD integrates both the feature values and their validities. The design includes trials where Tallying predicts a strict preference but WADD predicts the opposite (e.g., one option wins on 3 low-validity features, while the other wins on 2 high-validity features), trials where Tallying predicts a tie but WADD has a strict preference (e.g., both options win on 2 features, but one wins on higher validity features), and agreement trials where both models predict the same choice.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits a single high-validity feature (feature 1, validity 0.95) against multiple lower-validity features. In every trial pair, the option that possesses feature 1 is either tied or outnumbered in the total tally of winning features. Consequently, the Tallying heuristic will never deterministically prefer the option with feature 1 (it either prefers the alternative or ties). In contrast, the Weighted Additive (WADD) model heavily weights feature 1 due to its high validity, frequently choosing the option that possesses it. This metric computes the overall proportion of trials where the subject chose the option having feature 1, which will be significantly higher under WADD than Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_f1 = np.array([a[0] for a in data['option_a_ratings']])
    b_f1 = np.array([b[0] for b in data['option_b_ratings']])
    resp = data['response'].values
    chosen_f1 = np.where(resp == 0, a_f1, b_f1)
    return float(np.mean(chosen_f1))

### RESULTS
- Predicted under pi_1 (simulated): 0.2685 (var=0.0055)
- Predicted under pi_2 (simulated): 0.5321 (var=0.0463)
- Observed on real data: 0.5900 (var=0.0027)

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
  "interpretation": "Across both experiments, the Weighted Additive (WADD) model (pi_2) significantly outperforms the Tallying heuristic (pi_1). In Experiment 1, pi_2 almost perfectly predicts the proportion of Tallying-consistent choices on dissociation trials (observed: 0.41, pi_2: 0.40, pi_1: 0.86). In Experiment 2, pi_2 again correctly predicts that subjects will favor the highest-validity feature much more often than Tallying predicts (observed: 0.59, pi_2: 0.53, pi_1: 0.26). Tallying is fundamentally mismatched to the data because it ignores cue validities, whereas the subjects clearly use them. Since pi_1 is degenerate, we should replace it with another classical heuristic that utilizes cue validities, such as Take-The-Best.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'Take-The-Best' (TTB) as the new theory. TTB searches through features in descending order of their validity and stops at the first feature that discriminates between the two options, choosing the option with the higher value on that feature. If no features discriminate, it guesses. This will provide a strong non-compensatory alternative to the compensatory WADD model, as it uses validities (unlike Tallying) but in a lexicographic rather than additive way."
}
```

## Usage

```json
{
  "prompt_token_count": 4183,
  "candidates_token_count": 331,
  "total_token_count": 4878
}
```
