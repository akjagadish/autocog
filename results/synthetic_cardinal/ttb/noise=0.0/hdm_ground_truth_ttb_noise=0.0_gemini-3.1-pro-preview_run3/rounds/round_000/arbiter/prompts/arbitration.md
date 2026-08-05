# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

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
    return int(np.argmax(probabilities))


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

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  trial 3: A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 4: A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  trial 5: A=[2, 8, 8, 8]  B=[10, 5, 5, 5]
  trial 6: A=[10, 1, 1, 1]  B=[0, 2, 2, 2]
  trial 7: A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  trial 8: A=[9, 3, 3, 3]  B=[0, 4, 4, 4]

**Rationale:** This design dissociates Tallying (advocated) from Weighted Additive (WADD; competing) by exploiting Tallying's insensitivity to both feature validities and cardinal rating magnitudes. We use a 4-feature design with a high rating_max (10) and a spread of validities. In the critical trials, one option wins on a majority of features by a very small margin (which drives Tallying to strongly prefer it), while the other option wins on fewer, higher-validity features by a massive margin (which drives WADD to strongly prefer the latter).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Measures the proportion of trials where the subject's choice aligns with the Tallying heuristic (choosing the option that wins on more features). The experimental design perfectly dissociates Tallying from WADD on all trials, so Tallying will produce values near 1, while WADD will produce values near 0.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = (b_wins > a_wins).astype(int)
    return float((data['response'] == tally_choice).mean())


### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.1950 (var=0.0838)
- Observed on real data: 0.6400 (var=0.2304)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=5):**
  trial 1: A=[10, 0, 0, 0]  B=[0, 3, 3, 3]
  trial 2: A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  trial 3: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 4: A=[0, 2, 2, 2]  B=[10, 0, 0, 0]
  trial 5: A=[5, 5, 5, 5]  B=[6, 4, 4, 4]

**Rationale:** This design dissociates the advocated Weighted Additive (WADD) theory from the competing Tallying heuristic. WADD integrates both the cardinal magnitudes of features and their validities, while Tallying relies solely on the sign of the difference (counting strict feature-wise wins) and ignores both magnitude and validity. By using a cardinal rating scale (rating_max = 10) and a spread of validities, we construct trials where an option wins on a majority of features by a small margin (favored by Tallying) but loses on fewer features by a massive margin or on higher-validity features (favored by WADD). We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD strongly prefers one option due to validity and magnitude differences.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the WADD predictions on the specific trials where WADD and Tallying strictly disagree. On Trials 1, 2, and 4, Tallying predicts one option based on the number of feature wins (3 vs 1), while WADD predicts the opposite option due to the large cardinal magnitude differences on high-validity features. The metric will be close to 1.0 for subjects using WADD and close to 0.0 for subjects using Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (10, 0, 0, 0) and b == (0, 3, 3, 3):
            if resp == 0: wadd_match += 1
            count += 1
        elif a == (2, 2, 2, 0) and b == (0, 0, 0, 10):
            if resp == 1: wadd_match += 1
            count += 1
        elif a == (0, 2, 2, 2) and b == (10, 0, 0, 0):
            if resp == 1: wadd_match += 1
            count += 1
            
    return float(wadd_match / count) if count > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.6800 (var=0.0576)
- Observed on real data: 0.3667 (var=0.0856)

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
  "interpretation": "The observed data across both experiments consistently falls squarely between the predictions of Tallying and Weighted Additive (WADD). In Experiment 1, subjects chose the Tallying option 64% of the time, compared to Tallying's prediction of 100% and WADD's prediction of ~19.5%. In Experiment 2, on trials where the two heuristics strictly disagree, subjects chose the WADD option 36.6% of the time (meaning they chose the Tallying option ~63.4% of the time), compared to Tallying's 0% and WADD's 68%. This robust ~64/36 split indicates that neither pure heuristic captures the population behavior. The population likely consists of a mixture of individuals using Tallying and individuals using WADD, or individuals who probabilistically switch between strategies.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Mixture of Heuristics' theory. The new theory should posit that human decision-making in this domain is not governed by a single universal heuristic. Instead, individuals differ in their strategies, or switch between them, such that the population's choices reflect a mixture of Tallying (which ignores magnitudes and validities) and WADD (which integrates both). The model should introduce a mixture weight parameter 'alpha' that dictates the probability of using Tallying versus WADD on any given trial, along with the standard noise parameters."
}
```

## Usage

```json
{
  "prompt_token_count": 3977,
  "candidates_token_count": 337,
  "total_token_count": 5762
}
```
