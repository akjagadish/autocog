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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This experiment is designed to strongly dissociate Tallying (which counts the number of features favoring each option, ignoring validities) from the Weighted Additive rule (WADD, which weighs each feature by its validity). We use 5 features with a steep drop-off in validities between the top two and the bottom three. In several critical trials, Option A is superior on the two most valid features (thus strongly favored by WADD), while Option B is superior on the three least valid features (thus favored by Tallying, as it wins 3 to 2). We also include trials where Tallying predicts indifference (2 wins vs 2 wins) but WADD predicts a clear preference, and baseline trials where both heuristics agree, to ensure overall model identifiability.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric zeroes in on two critical trial types where Tallying and WADD have strictly opposite preferences. In Trial 1, Option B has more winning features (3 vs 2) but Option A's features are highly valid (0.95+0.90 > 0.60+0.55+0.50). In Trial 8, Option A has more winning features (2 vs 1) but Option B's single winning feature is the most valid (0.95 > 0.60+0.55). The metric computes the proportion of times the subject chooses the Tallying-preferred option on these critical trials. Tallying will yield a high score, while WADD will yield a low score.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    score = 0
    count = 0
    for _, row in data.iterrows():
        A = tuple(row['option_a_ratings'])
        B = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers B (3 to 2 wins), WADD prefers A (1.85 to 1.65)
        if A == (1, 1, 0, 0, 0) and B == (0, 0, 1, 1, 1):
            score += (resp == 1)
            count += 1
        # Trial 8: Tallying prefers A (2 to 1 wins), WADD prefers B (0.95 to 1.15)
        elif A == (0, 0, 1, 1, 0) and B == (1, 0, 0, 0, 0):
            score += (resp == 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)

### RESULTS
- Predicted under pi_1 (simulated): 0.8525 (var=0.0142)
- Predicted under pi_2 (simulated): 0.4650 (var=0.0585)
- Observed on real data: 0.1625 (var=0.0238)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To maximally distinguish Weighted Additive (WADD) from Tallying, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. This allows us to construct 'dissociation' trials where one option wins on the two most valid features (favored by WADD) while the other option wins on the three least valid features (favored by Tallying). We also include 'tie' trials where Tallying predicts indifference (equal number of wins) but WADD has a clear preference due to validity weighting, and 'agreement' trials where both heuristics favor the same option. This mix ensures the models are identifiable and their distinct mechanisms (magnitude/validity weighting vs. simple counting) are cleanly separated.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Weighted Additive (WADD) model on 'dissociation' trials. On these specific trials (Trial 1 and Trial 2), one option is superior on the two most valid features (favored by WADD), while the other option is superior on the three least valid features (favored by Tallying). WADD agents will consistently score high on this metric (close to 1), whereas Tallying agents will score low (close to 0), producing a large and stable gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    trial_1 = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    trial_2 = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 1, 0, 0, 0))
    
    t1_wadd_aligned = (data['response'] == 0) & trial_1
    t2_wadd_aligned = (data['response'] == 1) & trial_2
    
    wadd_choices = t1_wadd_aligned.sum() + t2_wadd_aligned.sum()
    total_dissociation = trial_1.sum() + trial_2.sum()
    
    return float(wadd_choices / total_dissociation) if total_dissociation > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.1533 (var=0.0142)
- Predicted under pi_2 (simulated): 0.5617 (var=0.0991)
- Observed on real data: 0.8217 (var=0.0115)

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
  "interpretation": "The observed data across both experiments strongly refutes Tallying (THEORY 1) and supports a compensatory, validity-weighted mechanism like Weighted Additive (THEORY 2). In both experiments, subjects consistently chose the option favored by WADD over the one favored by Tallying (yielding an ~82-84% WADD alignment). However, THEORY 2's simulated predictions hovered near 0.5. This discrepancy is due to a flaw in THEORY 2's model implementation: it multiplies the ground-truth `validities` by an unconstrained `weights` parameter vector, effectively scrambling the validities. Furthermore, its temperature/lapse parameters likely produce too much noise, whereas human subjects were highly deterministic.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Revise the `predict` function to compute scores using exactly the provided validities (e.g., `scores = stim @ validities`), removing the redundant and confounding `w` parameter. Additionally, shift the parameter ranges for `beta` to higher values (allowing for sharper, more deterministic softmax outputs) and restrict `epsilon` to lower values to match the high consistency (low variance, high accuracy) observed in the human data."
}
```

## Usage

```json
{
  "prompt_token_count": 4643,
  "candidates_token_count": 277,
  "total_token_count": 5978
}
```
