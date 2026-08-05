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
**Validities (n_features=4):** [0.95, 0.72, 0.73, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 6: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 14: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 15: A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Identifies trials where Tallying predicts a perfect tie (because both options win on exactly 1 feature) but WADD predicts a strong preference (due to differences in cue validities). The metric calculates the proportion of choices that align with WADD's favored option in these specific 'Tally-tied' trials. Tallying will yield ~0.5, while WADD will yield a value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_favored_choices = 0
    tied_trials = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == (0, 1, 0, 1) and b == (1, 1, 0, 0):
            wadd_favored = 1
        elif a == (1, 0, 1, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        elif a == (1, 1, 1, 0) and b == (0, 1, 1, 1):
            wadd_favored = 0
        elif a == (1, 1, 0, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        else:
            continue
            
        tied_trials += 1
        if row['response'] == wadd_favored:
            wadd_favored_choices += 1
            
    if tied_trials == 0:
        return 0.5
    return float(wadd_favored_choices / tied_trials)

### RESULTS
- Predicted under pi_1 (simulated): 0.5083 (var=0.0129)
- Predicted under pi_2 (simulated): 0.6717 (var=0.0328)
- Observed on real data: 0.8667 (var=0.0063)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.8, 0.85, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 15: A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 16: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying predicts a perfect tie (i.e., both options win on an equal number of features). For these trials, Tallying must guess, producing an expected choice probability of 0.5 regardless of its parameters. In contrast, WADD uses the validities as weights to break the tie, consistently favoring the option with the higher weighted sum. By computing the proportion of choices that align with WADD's prediction on these specific trials, we expect Tallying to score around 0.5 and WADD to score significantly higher.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.8, 0.85, 0.55])
    wadd_consistent = 0
    tally_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying predicts a tie
        if a_wins == b_wins:
            score_a = np.sum(a * validities)
            score_b = np.sum(b * validities)
            
            if score_a > score_b:
                wadd_choice = 0
            elif score_b > score_a:
                wadd_choice = 1
            else:
                continue
                
            if row['response'] == wadd_choice:
                wadd_consistent += 1
            tally_ties += 1
            
    if tally_ties == 0:
        return 0.5
    return float(wadd_consistent / tally_ties)

### RESULTS
- Predicted under pi_1 (simulated): 0.4733 (var=0.0257)
- Predicted under pi_2 (simulated): 0.5183 (var=0.0651)
- Observed on real data: 0.8767 (var=0.0106)

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
  "interpretation": "Across both experiments, the metrics isolate trials where Tallying (Theory 1) predicts a perfect tie (expected choice proportion of ~0.5). The observed data in both experiments shows a very strong preference (0.8667 and 0.8767) for the option favored by the Weighted Additive rule (WADD, Theory 2). This definitively falsifies Tallying's assumption that subjects ignore cue validities and merely count feature-wise wins. While WADD conceptually captures the direction of the effect, its simulated predictions (0.67 and 0.51) are heavily diluted, likely due to suboptimal parameter ranges (e.g., high lapse rates, low beta, or noisy subjective weights). Because Tallying is fundamentally unable to account for validity-based tie-breaking, it should be discarded entirely in favor of a stronger validity-aware competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'Take-The-Best' (TTB) as the new theory. TTB assumes that decision-makers do not integrate all information (unlike WADD) nor do they simply count (like Tallying). Instead, they rank features by validity and compare options lexicographically, stopping at the first feature that discriminates between the two options and choosing the one with the higher value on that feature. This provides a stark, non-compensatory contrast to WADD while still heavily utilizing the provided validities, which the data shows subjects are clearly doing."
}
```

## Usage

```json
{
  "prompt_token_count": 4614,
  "candidates_token_count": 338,
  "total_token_count": 6294
}
```
