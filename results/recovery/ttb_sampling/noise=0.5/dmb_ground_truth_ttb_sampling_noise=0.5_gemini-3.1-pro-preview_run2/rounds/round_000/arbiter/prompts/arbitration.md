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
**Validities (n_features=6):** [0.95, 0.85, 0.65, 0.55, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Tallying (which counts strict feature-wise wins irrespective of validities) from WADD (which computes a validity-weighted sum). The selected validities include a combination where the sum of three lower validities exactly equals the sum of two higher validities (0.55 + 0.55 + 0.50 = 1.60; 0.95 + 0.65 = 1.60), allowing trials where WADD predicts a perfect tie while Tallying strongly favors the option with more winning features. Other trials pit a small number of highly valid features against a larger number of less valid features, causing Tallying and WADD to predict opposite choices.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the participant's choice aligns with the strict prediction of the Tallying heuristic, computed only on trials where Tallying has a strict preference (i.e., one option has more winning features than the other). The experimental design includes multiple trials where Tallying and WADD make diametrically opposed predictions (e.g., Tallying favors the option with many weak features, while WADD favors the option with fewer strong features). Consequently, data generated by the Tallying model will yield a high score on this metric, whereas data generated by the WADD model will yield a score significantly below 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_matches.append(1 if row['response'] == 0 else 0)
        elif b_wins > a_wins:
            tally_matches.append(1 if row['response'] == 1 else 0)
    if len(tally_matches) == 0:
        return 0.5
    return float(np.mean(tally_matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8713 (var=0.0084)
- Predicted under pi_2 (simulated): 0.5197 (var=0.0620)
- Observed on real data: 0.2967 (var=0.0025)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates WADD (Weighted Additive) from Tallying using 5 features with carefully chosen validities. Tallying simply counts the number of winning features, ignoring their validities, whereas WADD computes a validity-weighted sum. By pitting a small number of high-validity features against a larger number of low-validity features, we create clear dissociations. For example, in the first trial type, Option A wins on 2 high-validity features while Option B wins on 3 low-validity features; Tallying favors B (3 > 2) whereas WADD strongly favors A (due to the high weights). Other trials include cases where Tallying predicts a perfect tie but WADD strongly prefers one option, as well as congruent trials to ensure task engagement.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Focuses exclusively on Trial 1, the most discriminative trial type in the design. On this trial, one option has 2 high-validity features (favored by WADD) while the other has 3 low-validity features (favored by Tallying). The metric computes the proportion of choices for the Tallying-favored option. Tallying will yield a high value (close to 1), while WADD will yield a low value (close to 0).

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying and WADD make opposite predictions
    # A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    is_trial_1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                     data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                     
    tally_choices = 0
    total = 0
    
    if is_trial_1.sum() > 0:
        # Tallying favors B (response 1) because it has 3 features vs 2.
        tally_choices += data[is_trial_1]['response'].sum()
        total += is_trial_1.sum()
        
    if is_trial_1_rev.sum() > 0:
        # Tallying favors A (response 0).
        tally_choices += (1 - data[is_trial_1_rev]['response']).sum()
        total += is_trial_1_rev.sum()
        
    if total == 0:
        return 0.5
        
    return float(tally_choices / total)

### RESULTS
- Predicted under pi_1 (simulated): 0.8737 (var=0.0110)
- Predicted under pi_2 (simulated): 0.4853 (var=0.0941)
- Observed on real data: 0.3600 (var=0.0119)

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
  "interpretation": "The observed data across both experiments strongly contradicts the Tallying theory (pi_1). In Experiment 1, Tallying predicted a metric of ~0.87, but the observed value was ~0.30, meaning participants systematically chose the option with fewer, but more valid, features. Similarly, in Experiment 2, participants favored the option with fewer high-validity features over the one with more low-validity features (observed ~0.36 vs pi_1 predicted ~0.87). While pi_2 (WADD) captures the direction of this effect better than Tallying, its simulated predictions hovered around 0.5 because its model implementation includes an extra unconstrained weight vector `w` that dilutes the provided validities. Instead of merely fixing pi_2's model, we should replace the completely degenerate pi_1 (Tallying) with a new theory that strongly prioritizes high-validity features, such as the Take-The-Best (TTB) heuristic.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic as a brand-new theory. Unlike Tallying (which ignores validities) or WADD (which compensates for low validities by summing them up), TTB posits a non-compensatory lexicographic process: people search through features in descending order of their validities and stop at the first feature that discriminates between the two options, choosing the option with the higher value on that cue. The model should implement this by sorting features by validity, finding the first index where the options differ, and choosing deterministically based on that feature (subject to softmax/lapse noise)."
}
```

## Usage

```json
{
  "prompt_token_count": 4431,
  "candidates_token_count": 380,
  "total_token_count": 6154
}
```
