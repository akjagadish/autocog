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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** To strongly dissociate Tallying (which counts unweighted feature-wise wins) from WADD (which computes a validity-weighted sum), we use a 6-feature design with a steep drop-off in validities. In several critical trials, Option A wins on a larger number of low-validity features (e.g., 3 wins), while Option B wins on fewer but much higher-validity features (e.g., 2 wins). Tallying strictly predicts Option A because it ignores cardinal validities and simply counts wins, whereas WADD predicts Option B because the sum of the validities for B's winning features exceeds A's. We also include agreement trials where both models predict the same option, ensuring the experiment covers a range of trade-offs.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'conflict' trials where Tallying predicts Option A (because Option A wins on more features overall) but WADD predicts Option B (because Option B wins on fewer, but higher-validity features). We calculate the mean response on these specific trials. Tallying will produce a metric near 0 (choosing A), whereas WADD will produce a metric near 1 (choosing B), creating a maximal and highly reliable contrast between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    a_wadd = np.sum(a_ratings * validities, axis=1)
    b_wadd = np.sum(b_ratings * validities, axis=1)
    
    conflict_mask = (a_wins > b_wins) & (b_wadd > a_wadd)
    
    if np.sum(conflict_mask) == 0:
        return 0.5
        
    return float(np.mean(data['response'].values[conflict_mask]))

### RESULTS
- Predicted under pi_1 (simulated): 0.1633 (var=0.0132)
- Predicted under pi_2 (simulated): 0.5661 (var=0.0650)
- Observed on real data: 0.8844 (var=0.0081)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To strongly dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities. In critical trials, one option wins on a larger number of low-validity features (e.g., 3 wins out of 5), while the other option wins on fewer but much higher-validity features (e.g., 2 wins out of 5). Tallying strictly predicts the option with more wins because it ignores cardinal validities, whereas WADD predicts the option with fewer wins because the sum of the validities for its winning features is greater. We also include agreement trials where both models predict the same option, to ensure a broad coverage of the decision space.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the critical trials where the two theories make opposing predictions. In these trials, one option ('heavy') has only two positive features but they are the most valid ones (0.95 and 0.85), whereas the other option ('many') has three positive features but they are less valid (0.6, 0.55, 0.5). WADD tends to favor the 'heavy' option because the weighted sum of the two highly valid features is generally greater than the sum of the three less valid features. Tallying, on the other hand, ignores cardinal validities and strictly favors the 'many' option because 3 wins > 2 wins. The metric calculates the proportion of times the 'heavy' option is chosen on these critical trials. We expect a high value for WADD and a low value for Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    is_a_heavy = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_b_heavy = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    is_a_many = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_b_many = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    trial_1 = is_a_heavy & is_b_many
    trial_2 = is_a_many & is_b_heavy
    
    critical_trials = trial_1 | trial_2
    
    if not critical_trials.any():
        return 0.5
        
    heavy_chosen = (trial_1 & (data['response'] == 0)) | (trial_2 & (data['response'] == 1))
    
    return float(heavy_chosen[critical_trials].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1292 (var=0.0084)
- Predicted under pi_2 (simulated): 0.4992 (var=0.0988)
- Observed on real data: 0.8533 (var=0.0160)

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
  "interpretation": "Across both experiments, the observed data strongly contradicts Theory 1 (Tallying) and qualitatively supports Theory 2 (WADD). In conflict trials where one option has more winning features but the other has fewer, higher-validity features, participants overwhelmingly chose the latter (observed means of ~0.88 and ~0.85). However, Theory 2's quantitative predictions are far too close to 0.5 (random guessing). This is likely because its current `predict` implementation multiplies the true validities by an additional random subjective weight vector `w`, which dilutes the validities' impact, and its parameter ranges for noise (beta, epsilon) may be too permissive of guessing.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Revise the WADD model implementation to remove the redundant and noisy per-feature `weights` parameter `w` that is currently multiplied by the objective validities. Instead, compute scores directly using the provided `validities` (optionally allowing a single non-linear scaling parameter like `gamma` where `subjective_weights = validities ** gamma`). Additionally, adjust the parameter ranges for `beta` (allow higher values for more deterministic choices) and `epsilon` (constrain to lower values) so the model can predict the highly consistent >0.85 choice proportions observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 4541,
  "candidates_token_count": 305,
  "total_token_count": 5537
}
```
