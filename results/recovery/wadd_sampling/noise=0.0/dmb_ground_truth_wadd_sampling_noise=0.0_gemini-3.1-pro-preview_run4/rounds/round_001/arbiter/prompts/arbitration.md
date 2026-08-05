# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Decision-makers evaluate options by computing a weighted sum of their features, where the weights are a subjective power transformation of the cue validities. This Weighted Additive (WADD) model is fully compensatory, allowing the combined evidence of multiple lower-validity cues to potentially outweigh a single higher-validity cue, while still remaining sensitive to the varying diagnosticity of different features. The parameter governing the power transformation is constrained to prevent extreme exponentiation, ensuring the model remains compensatory rather than collapsing into a non-compensatory heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities into weights
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_2
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design strictly dissociates the Weighted Additive (WADD) model from the Tallying heuristic by pitting a small number of highly valid features against a larger number of less valid features. Tallying ignores validities and simply counts the number of features on which each option is superior, leading to predictions that favor the option with a greater quantity of winning features, or predicting a tie when the counts are equal. WADD, being a compensatory model that weights features by their subjective validities, will instead favor the option with the higher weighted sum. The selected trials include cases where Tallying strongly prefers one option while WADD prefers the other, as well as trials where Tallying predicts a tie but WADD makes a deterministic choice based on the cumulative weight of the winning features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the choice probability of the option [1, 0, 1, 0, 0] over [0, 1, 0, 1, 0] (Trial 2). Under Tallying, both options win on exactly 2 features, leading to an indifferent 50% choice probability regardless of the beta and epsilon parameters. Under the WADD model, the option [1, 0, 1, 0, 0] strictly dominates because its winning features (validities 0.95 and 0.6) are strictly more valid than the opponent's winning features (validities 0.9 and 0.55). Thus, WADD will consistently predict a choice probability > 50% for the target option.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 0, 1, 0, 0] and [0, 1, 0, 1, 0]
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    is_B_target = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    
    # Count how many times the target option [1, 0, 1, 0, 0] was chosen
    chose_target_when_A = (data.loc[is_A_target, 'response'] == 0).sum()
    chose_target_when_B = (data.loc[is_B_target, 'response'] == 1).sum()
    
    total_trials = is_A_target.sum() + is_B_target.sum()
    if total_trials == 0:
        return 0.5
        
    return float((chose_target_when_A + chose_target_when_B) / total_trials)

### RESULTS
- Predicted under pi_3 (simulated): 0.6775 (var=0.0220)
- Predicted under pi_2 (simulated): 0.4742 (var=0.0081)
- Observed on real data: 0.6050 (var=0.0828)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from the Weighted Additive (WADD) model, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. Tallying completely ignores validities and simply counts the number of features on which one option is strictly better than the other. In contrast, WADD computes a weighted sum where features are weighted by a power-transformation of their validities. The design includes trials where one option has more winning features (favored by Tallying) but the other option wins on the fewer, higher-validity features (favored by WADD). It also includes trials where both options win on an equal number of features (Tallying predicts a tie, resulting in random guessing), while WADD strongly prefers the option that wins on the more valid features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the two theories make maximally divergent predictions (Trials 1 and 3). In these trials, one option has more positive features overall (favored by Tallying), but the other option possesses the fewer, highest-validity features (favored by WADD). Tallying will consistently choose the option with the higher unweighted sum, whereas WADD will penalize it for lacking the top features and choose the other option. By calculating the proportion of times the subject chooses the option with the higher unweighted sum in these specific trials, we expect a value close to 1.0 for Tallying and close to 0.0 for WADD, cleanly discriminating the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the unweighted sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Extract the first (highest validity) feature for each option
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    b_first = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where one option has more positive features (higher sum)
    # but lacks the most important feature (first feature is 0).
    # This corresponds exactly to trials 1 and 3 in the design.
    mask_a_higher_but_lacks_first = (sum_a > sum_b) & (a_first == 0)
    mask_b_higher_but_lacks_first = (sum_b > sum_a) & (b_first == 0)
    
    mask = mask_a_higher_but_lacks_first | mask_b_higher_but_lacks_first
    
    if not mask.any():
        return 0.5
        
    filtered = data[mask]
    chose_a = filtered['response'] == 0
    
    a_is_higher = sum_a[mask] > sum_b[mask]
    
    # Calculate the proportion of times the subject chose the option with the higher unweighted sum
    chose_higher = (chose_a & a_is_higher) | (~chose_a & ~a_is_higher)
    
    return float(chose_higher.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.3719 (var=0.0884)
- Predicted under pi_2 (simulated): 0.8556 (var=0.0073)
- Observed on real data: 0.3713 (var=0.0726)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # The cues are ordered by validity (descending) from left to right.
    # Find the first cue where the two options differ.
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    
    # Get the difference value at that first discriminating cue.
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Calculate the proportion of responses matching the TTB prediction
    matches = (data['response'].values == ttb_pred)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2988 (var=0.0249)
**Predicted under pi_3:** 0.3183 (var=0.0105)
**Predicted under pi_2:** 0.1846 (var=0.0061)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Extract options matrices
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying predictions: count wins for each option
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    tally_preds = np.zeros(len(data)) - 1
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Take The Best (TTB) predictions: first discriminating cue
    ttb_preds = np.zeros(len(data)) - 1
    for i in range(len(data)):
        for j in range(a_mat.shape[1]):
            if a_mat[i, j] > b_mat[i, j]:
                ttb_preds[i] = 0
                break
            elif b_mat[i, j] > a_mat[i, j]:
                ttb_preds[i] = 1
                break
                
    # Identify trials where Tallying and TTB make strict, opposing predictions
    valid = (tally_preds != -1) & (ttb_preds != -1) & (tally_preds != ttb_preds)
    
    if not np.any(valid):
        return 0.5
        
    # Calculate the proportion of responses that align with Tallying
    responses = data['response'].values[valid]
    tally_preds_valid = tally_preds[valid]
    
    return float(np.mean(responses == tally_preds_valid))
```

**Observed (real) value:** 0.7037 (var=0.0287)
**Predicted under pi_3:** 0.8434 (var=0.0130)
**Predicted under pi_2:** 0.8808 (var=0.0067)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (pi_3, Weighted Additive model) consistently outperforms Theory 2 (pi_2, Tallying). In Experiment 1, pi_3 correctly predicts the preference for the target option (~0.67 vs observed 0.60), while Tallying predicts indifference. In Experiment 2, pi_3 perfectly captures the observed metric (~0.37), whereas Tallying predicts a completely opposite effect (~0.85). In Experiment 3, pi_3 again closely tracks the observed metric. Tallying's assumption that subjects completely ignore cue validities is strongly contradicted by the data. Because pi_3 is highly accurate, it should be retained. Theory 2 is fundamentally flawed for this dataset and should be replaced with a more competitive theory.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory based on Rank-Based Weighting (e.g., a variant of the Minimalist or Rank-Dependent model). Instead of using the exact validities or an unweighted tally, this theory should assign weights to features based purely on their ordinal rank in validity (e.g., weight = 1/rank). This provides a middle ground between the extreme compensatory nature of WADD (Theory 1) and the complete disregard for validity in Tallying (Theory 2), offering a strong new competitor."
}
```

## Usage

```json
{
  "prompt_token_count": 5399,
  "candidates_token_count": 318,
  "total_token_count": 6458
}
```
