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
**Validities (n_features=5):** [0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Tallying (which simply counts the number of features favoring each option) from Weighted Additive (WADD) (which integrates the cardinal validity of each feature). The validities are chosen such that the sum of the three least valid features (0.6 + 0.55 + 0.5 = 1.65) is less than the sum of the two most valid features (0.9 + 0.85 = 1.75). This creates a critical dissociation: Tallying will strongly favor an option that wins on the three minor features (3 wins vs 2 wins), whereas WADD will favor the option that wins on the two major features. Additional trials include cases where Tallying predicts a tie but WADD has a strong preference, and cases where both models agree, ensuring a comprehensive test space.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In trials 1 and 2, one option has the two highest validity features (0.9 and 0.85) while the other option has the three lowest validity features (0.6, 0.55, 0.5). Tallying will robustly choose the option with three features (since 3 > 2), completely ignoring the cardinal validities. In contrast, WADD integrates cardinal validities and will generally prefer the option with the two highest validity features, because 0.9 + 0.85 > 0.6 + 0.55 + 0.5. This metric isolated these critical trials and calculates the proportion of times the subject chooses the Tallying-predicted option. Tallying will yield a high value, whereas WADD will yield a low value.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = np.array([x[0] for x in data['option_a_ratings']])
    a_f1 = np.array([x[1] for x in data['option_a_ratings']])
    b_f0 = np.array([x[0] for x in data['option_b_ratings']])
    b_f1 = np.array([x[1] for x in data['option_b_ratings']])
    
    a_sum2 = a_f0 + a_f1
    b_sum2 = b_f0 + b_f1
    
    # Identify critical trials (trials 1 and 2) where one option has the two highest 
    # validity features (sum=2) and the other has none of them (sum=0) but wins on the rest.
    mask = ((a_sum2 == 0) & (b_sum2 == 2)) | ((a_sum2 == 2) & (b_sum2 == 0))
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    a_sum2_rel = a_sum2[mask]
    
    # Tallying prefers the option with more features (which here means the one with 0 on the first two features)
    tally_chose_a = (a_sum2_rel == 0) & (responses == 0)
    tally_chose_b = (a_sum2_rel == 2) & (responses == 1)
    
    return float(np.mean(tally_chose_a | tally_chose_b))


### RESULTS
- Predicted under pi_1 (simulated): 0.8483 (var=0.0104)
- Predicted under pi_2 (simulated): 0.5358 (var=0.1008)
- Observed on real data: 0.4867 (var=0.0082)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.52, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design strictly dissociates WADD from Tallying by manipulating the sum of validities versus the raw count of features. The validities are heavily skewed: the top two features have very high validities (0.95, 0.90), while the remaining three have low validities (0.55, 0.52, 0.50). In critical trials, one option wins on the two highly valid features (thus preferred by WADD, sum = 1.85), while the other option wins on the three less valid features (thus preferred by Tallying, 3 wins vs 2 wins). Additional trials include cases where both heuristics predict a tie or agree, providing a comprehensive test of whether subjects integrate cardinal weights or simply count winning features.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with the first two (highly valid) features over the option with the last three (less valid) features. Tallying strictly prefers the option with three features (score approaches 0), whereas WADD, on average, assigns a higher weighted sum to the two highly valid features (score > 0.5). Thus, this specific trial type perfectly dissociates the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if a[0] == 1 and a[1] == 1 and b[0] == 0 and b[1] == 0:
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a[0] == 0 and a[1] == 0 and b[0] == 1 and b[1] == 1:
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(wadd_choices / total)


### RESULTS
- Predicted under pi_1 (simulated): 0.1611 (var=0.0137)
- Predicted under pi_2 (simulated): 0.5700 (var=0.1165)
- Observed on real data: 0.5089 (var=0.0125)

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
  "interpretation": "Across both experiments, Tallying (Theory 1) strongly predicts that subjects will choose the option with more winning features (predicting ~0.85 and ~0.16 on the respective metrics), completely ignoring cue validities. However, the observed data in both experiments is around 0.50 with very low variance, indicating that subjects do not consistently favor the option with more features. WADD (Theory 2) predicts closer to the mean (~0.54 and ~0.57) but with massive variance, suggesting that its randomly sampled subjective weights inconsistently favor one option or the other across simulated subjects, unlike the real subjects who consistently choose near chance level for these specific critical trials. Since Tallying completely fails to capture the aggregate indifference or heterogeneity in these trials, it should be replaced by another foundational heuristic from the literature.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the Take-The-Best (TTB) heuristic. TTB proposes that decision-makers do not integrate all features (neither by counting nor by weighting). Instead, they search through cues in descending order of validity (or subjective validity) and stop at the first cue that discriminates between the two options, choosing the option with the higher value on that cue. This represents a non-compensatory, lexicographic decision strategy that contrasts strongly with both Tallying and WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 4622,
  "candidates_token_count": 316,
  "total_token_count": 6225
}
```
