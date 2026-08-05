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
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 9: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, we use a 5-feature design with a skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, ignoring the validities entirely. WADD weights each feature by its validity, meaning a few highly valid features can outweigh several less valid ones. The trial pairs are designed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option. This mix of congruent, incongruent, and tie trials provides a robust test to identify which strategy subjects are using.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific incongruent trials (Trials 1 and 2) where Tallying and WADD make opposing predictions. On these trials, one option wins on 3 weak features while the other wins on 2 strong features. Tallying ignores feature validities and strictly prefers the option with 3 winning features. WADD weighs features by their validity, thus favoring the option with the 2 stronger features. By calculating the proportion of choices that align with the Tallying prediction on these specific trials, we expect a high score for data simulated under Tallying and a low score for data simulated under WADD.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: A has 3 wins on weak features, B has 2 wins on strong features
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            # Tallying prefers A
            tally_choices.append(1.0 if row['response'] == 0 else 0.0)
        # Trial 2: A has 2 wins on strong features, B has 3 wins on weak features
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            # Tallying prefers B
            tally_choices.append(1.0 if row['response'] == 1 else 0.0)
            
    if not tally_choices:
        return 0.5
    return float(np.mean(tally_choices))

### RESULTS
- Predicted under pi_1 (simulated): 0.8788 (var=0.0105)
- Predicted under pi_2 (simulated): 0.4338 (var=0.0954)
- Observed on real data: 0.3975 (var=0.0118)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, this design uses a 5-feature task with a highly skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, strictly ignoring the magnitude of the validities. In contrast, WADD weights each feature by its validity, allowing a few highly valid features to outweigh a larger number of less valid ones. The trial pairs are strategically constructed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as tie-breaking trials (where Tallying predicts a tie due to an equal number of winning features, while WADD strongly prefers the option with the higher-validity features).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials 1 and 2, where one option has the two most valid features ([1, 1, 0, 0, 0]) and the other has the three least valid features ([0, 0, 1, 1, 1]). Tallying simply counts the number of winning features, so it will strongly prefer the option with three features (leading to a choice rate of [1, 1, 0, 0, 0] near 0, plus lapse noise). In contrast, WADD weights features by validity; the combined validity of the top two features (0.95 + 0.90 = 1.85) outweighs the bottom three (0.60 + 0.55 + 0.50 = 1.65), meaning WADD will frequently choose the option with only two features. Measuring the selection rate of the [1, 1, 0, 0, 0] option on these specific trials provides a massive, stable dissociation between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 1, 0, 0, 0] and [0, 0, 1, 1, 1]
    is_11000_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_11000_B = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    mask = is_11000_A | is_11000_B
    if mask.sum() == 0:
        return 0.5
    
    subset = data[mask]
    is_11000_A_sub = is_11000_A[mask]
    
    # Check if the subject chose the option with [1, 1, 0, 0, 0]
    chose_11000 = (is_11000_A_sub & (subset['response'] == 0)) | (~is_11000_A_sub & (subset['response'] == 1))
    return float(chose_11000.mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1292 (var=0.0114)
- Predicted under pi_2 (simulated): 0.5050 (var=0.0797)
- Observed on real data: 0.5833 (var=0.0110)

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
  "interpretation": "Across both experiments, Tallying (Theory 1) consistently fails to capture human behavior. In Experiment 1, Tallying predicted a strong preference for the option with a higher count of winning but weaker features (0.88), whereas humans favored the option with fewer but stronger features (observed: 0.40), closely matching the Weighted Additive model (Theory 2). In Experiment 2, Tallying predicted a very low selection rate (0.13) for the option with 2 strong features over 3 weak features, but humans chose the 2 strong features a majority of the time (0.58), again moving in the direction of WADD. Tallying is definitively falsified because subjects are clearly sensitive to feature validities rather than just the raw count of winning features.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with 'Take-The-Best' (TTB). TTB is a lexicographic, non-compensatory heuristic where subjects rank features by validity and decide based solely on the first feature that discriminates between the two options. This provides a strong non-compensatory alternative to the compensatory WADD model, differing from Tallying by relying entirely on the validity hierarchy rather than an unweighted count."
}
```

## Usage

```json
{
  "prompt_token_count": 4783,
  "candidates_token_count": 292,
  "total_token_count": 5930
}
```
