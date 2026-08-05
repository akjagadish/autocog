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

**Trial pairs (n=10):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]

**Rationale:** The design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with Weighted Additive (WADD) (which scales feature magnitudes by their validities). To create a quantitative dissociation, we use a set of 6 features where a small number of highly valid features can outweigh a larger number of less valid features. In critical trials, Option A is superior on many low-validity features, leading Tallying to predict A, while Option B is superior on fewer but highly valid features, leading WADD to predict B. Other trials present ties for Tallying where WADD has a clear preference, or scenarios where both heuristics agree but with differing confidence levels.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Focuses specifically on the critical trial where Option A has more positive features (favored by Tallying) but Option B has the highest validity features (favored by WADD). The metric calculates the proportion of times the high-validity option (Option B in Trial 1) is chosen. Tallying predicts a value near 0, whereas WADD predicts a value near 1.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask1 = (a_ratings == (0, 0, 0, 1, 1, 1)) & (b_ratings == (1, 1, 0, 0, 0, 0))
    mask2 = (a_ratings == (1, 1, 0, 0, 0, 0)) & (b_ratings == (0, 0, 0, 1, 1, 1))
    
    resp = []
    if mask1.any():
        resp.extend(data.loc[mask1, 'response'].tolist())
    if mask2.any():
        resp.extend((1 - data.loc[mask2, 'response']).tolist())
        
    if len(resp) == 0:
        return 0.5
    return float(np.mean(resp))

### RESULTS
- Predicted under pi_1 (simulated): 0.1533 (var=0.0291)
- Predicted under pi_2 (simulated): 0.5311 (var=0.0885)
- Observed on real data: 0.4844 (var=0.0167)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Weighted Additive (WADD) from Tallying by manipulating the distribution of feature validities. We use 5 features with a steep drop-off in validity: two highly valid features and three weakly valid features. In critical trials, one option wins on the two highly valid features (strongly preferred by WADD) while the other option wins on the three weakly valid features (preferred by Tallying, as it simply counts wins). We also include trials where Tallying predicts indifference (equal number of wins) but WADD strongly prefers one option due to validity differences, and baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the Weighted Additive (WADD) model on the critical trials where WADD and Tallying strongly disagree. In these trials, one option has fewer positive features but those features have much higher validity (e.g., A=[1,1,0,0,0] vs B=[0,0,1,1,1]). Tallying simply counts wins and prefers the option with 3 positive features, whereas WADD weights by validity and generally prefers the option with the 2 highly valid features. A value close to 1 supports WADD, while a value close to 0 supports Tallying.

Source:
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_t1 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    is_t2 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_consistent = (is_t1 & (data['response'] == 0)) | (is_t2 & (data['response'] == 1))
    mask = is_t1 | is_t2
    
    if mask.sum() == 0:
        return 0.5
    return float(wadd_consistent.sum() / mask.sum())

### RESULTS
- Predicted under pi_1 (simulated): 0.1492 (var=0.0174)
- Predicted under pi_2 (simulated): 0.5550 (var=0.0973)
- Observed on real data: 0.4800 (var=0.0103)

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
  "interpretation": "Across both experiments, Tallying (Theory 1) predicts that subjects will overwhelmingly choose the option with more positive features (resulting in metric values ~0.15), completely ignoring the validity of those features. However, the observed data (~0.48 in both experiments) shows that subjects are nearly indifferent or evenly split on these critical trials. Weighted Additive (Theory 2) predicts metrics around 0.53-0.55, which is much closer to the observed data, suggesting that subjects are indeed sensitive to cue validities, but perhaps not in a perfectly compensatory manner, or the population consists of a mixture of heuristic users. Since Tallying completely fails to capture the observed data, it should be replaced by a stronger competing theory that relies on cue validities but in a non-compensatory way.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the 'Take The Best' (TTB) heuristic. TTB proposes that decision-makers search through features in descending order of their validities and stop search at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. This is a classic non-compensatory, one-reason decision-making heuristic that strongly contrasts with the compensatory Weighted Additive model, and will provide a better benchmark for evaluating whether subjects integrate all cues or just rely on the most predictive ones."
}
```

## Usage

```json
{
  "prompt_token_count": 4451,
  "candidates_token_count": 317,
  "total_token_count": 5666
}
```
