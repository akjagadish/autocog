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
**Validities (n_features=4):** [0.95, 0.76, 0.92, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 10: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 11: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 12: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 14: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 15: A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
By restricting the analysis to subsets of trials where Tallying computes the exact same score difference (e.g., A wins exactly 1 more feature than B), Tallying predicts a constant probability of choosing A across those trials. However, WADD uses the validities and cardinal feature magnitudes, so it still predicts variation in choice probabilities within these subsets. The metric computes the difference in the empirical probability of choosing A between the trials WADD strongly favors and the trials WADD weakly favors, holding the Tallying score difference strictly constant. This yields an expected value of 0 for Tallying and a positive value for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.76, 0.92, 0.55])
    
    # Extract matrices of A and B features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying differences
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = a_wins - b_wins
    
    # WADD differences (expected without subject-specific weights)
    wadd_diff = np.sum((a_mat - b_mat) * validities, axis=1)
    
    # 1 if chose A, 0 if chose B
    chose_a = 1.0 - data['response'].values
    
    # We compare choice probabilities within trials where Tallying predicts the exact same difference.
    # For tally_diff == 1, WADD predicts some trials favor A more strongly than others.
    mask1_high = (tally_diff == 1) & (wadd_diff > 0.85)
    mask1_low = (tally_diff == 1) & (wadd_diff < 0.85)
    
    diff1 = 0.0
    if np.any(mask1_high) and np.any(mask1_low):
        diff1 = np.mean(chose_a[mask1_high]) - np.mean(chose_a[mask1_low])
        
    # For tally_diff == -1, WADD predicts some trials favor B more strongly than others.
    mask_m1_high = (tally_diff == -1) & (wadd_diff > -0.85)
    mask_m1_low = (tally_diff == -1) & (wadd_diff < -0.85)
    
    diff_m1 = 0.0
    if np.any(mask_m1_high) and np.any(mask_m1_low):
        diff_m1 = np.mean(chose_a[mask_m1_high]) - np.mean(chose_a[mask_m1_low])
        
    # Under Tallying, both diff1 and diff_m1 should be 0.
    # Under WADD, both diff1 and diff_m1 should be positive.
    return float(diff1 + diff_m1)

### RESULTS
- Predicted under pi_1 (simulated): -0.0017 (var=0.0438)
- Predicted under pi_2 (simulated): 0.0792 (var=0.0977)
- Observed on real data: 0.1467 (var=0.0250)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.89, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 14: A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 16: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 8, Option A is better on the lowest-validity feature while Option B is better on the highest-validity feature. Tallying ignores validities and counts this as a 1-1 tie, predicting a 50% chance of choosing either option. WADD weights the features by their validities, so it strongly prefers Option B. The metric measures the proportion of times Option B is chosen on this specific trial, which cleanly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Tallying predicts a tie but WADD predicts a strong preference.
    # Trial 8: A=[0, 1, 0, 1], B=[1, 1, 0, 0]
    # A wins on feature 4 (validity 0.55). B wins on feature 1 (validity 0.95).
    # Tallying sees 1 win for A and 1 win for B, predicting exactly 50% choice for B.
    # WADD sees B's win on the most important feature as outweighing A's win on the least important, predicting >50% choice for B.
    is_target = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1]) & \
                data['option_b_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0])
    
    if is_target.sum() == 0:
        return 0.5
        
    return float(data.loc[is_target, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.4967 (var=0.0417)
- Predicted under pi_2 (simulated): 0.6533 (var=0.0887)
- Observed on real data: 0.8200 (var=0.0532)

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
  "interpretation": "Across both experiments, Theory 2 (WADD) significantly outperforms Theory 1 (Tallying). Experiment 1 shows that even when Tallying predicts no difference in choice probability, subjects exhibit strong variations that align with feature validities. Experiment 2 isolates a trial where Tallying predicts a tie, but subjects overwhelmingly prefer the option favored by the highest-validity feature. In fact, the observed data in both experiments show an even stronger reliance on validities than WADD's simulated predictions, suggesting that subjects might be using a non-compensatory or lexicographic strategy rather than a simple unweighted tally.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the 'Take The Best' (TTB) heuristic. TTB proposes that decision-makers rank features by their validity and evaluate them lexicographically. The choice is determined entirely by the highest-validity feature that discriminates between the two options, ignoring all lower-validity features. This non-compensatory approach would predict the extreme sensitivity to high-validity features observed in the data (e.g., the 82% preference in Experiment 2), serving as a much stronger competitor to WADD than Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 4911,
  "candidates_token_count": 279,
  "total_token_count": 5900
}
```
