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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[2, 8, 8, 8]  B=[9, 7, 7, 7]
  trial 3: A=[10, 4, 4, 4]  B=[5, 5, 5, 5]
  trial 4: A=[9, 3, 3, 3]  B=[2, 4, 4, 4]
  trial 5: A=[6, 6, 6, 2]  B=[5, 5, 5, 9]
  trial 6: A=[3, 3, 3, 10]  B=[4, 4, 4, 1]
  trial 7: A=[8, 8, 2, 2]  B=[7, 7, 4, 4]
  trial 8: A=[2, 2, 8, 8]  B=[4, 4, 7, 7]

**Rationale:** This design dissociates Tallying from WADD by exploiting Tallying's insensitivity to feature magnitude and validities. We use a rating_max of 10 to allow for large magnitude differences. In trials 1, 2, and 5, option A wins on three out of four features by a small margin, while B wins on one feature by a massive margin (often on a high-validity cue). Tallying will robustly choose A because it simply counts wins, whereas WADD will choose B because the large magnitude difference on a heavily weighted feature outweighs the small deficits on the others. Conversely, in trials 3, 4, and 6, the situation is reversed, leading Tallying to choose B and WADD to choose A. Finally, trials 7 and 8 present ties for Tallying (each option wins on exactly two features), resulting in a guessing prediction, whereas WADD retains strong preferences based on the validities and magnitudes.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the Tallying heuristic on trials where Tallying makes a strict prediction (i.e., one option has more feature-wise wins than the other). The experimental design explicitly pits Tallying against WADD on these trials: Tallying chooses the option with more small wins, whereas WADD chooses the option with a single massive win on a heavily weighted feature. Consequently, data simulated under Tallying will yield a metric value close to 1.0, while data simulated under WADD will yield a value close to 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    mask = tally_a != tally_b
    if not np.any(mask):
        return 0.5
        
    preds = np.where(tally_a > tally_b, 0, 1)
    responses = data['response'].values
    
    accuracy = np.mean(preds[mask] == responses[mask])
    return float(accuracy)


### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.2233 (var=0.0673)
- Observed on real data: 0.8000 (var=0.0667)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 5

**Trial pairs (n=8):**
  trial 1: A=[0, 4, 4, 4]  B=[5, 3, 3, 3]
  trial 2: A=[1, 5, 5, 5]  B=[5, 4, 4, 4]
  trial 3: A=[5, 4, 1, 1]  B=[1, 1, 5, 5]
  trial 4: A=[5, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 5: A=[4, 1, 1, 1]  B=[0, 2, 2, 2]
  trial 6: A=[1, 2, 3, 5]  B=[2, 3, 4, 0]
  trial 7: A=[5, 1, 1, 1]  B=[1, 2, 2, 2]
  trial 8: A=[2, 5, 1, 1]  B=[3, 1, 2, 2]

**Rationale:** This design leverages the magnitude and weighting components of WADD to strongly dissociate it from the sign-based, unweighted Tallying heuristic. By setting `rating_max` to 5 and using a steep validity gradient, we construct trials where one option wins on the majority of features by small margins (which Tallying strongly prefers as it simply counts wins), while the other option wins on fewer but more valid features by large margins (which WADD strongly prefers due to compensatory weighted sums). We also include trials where Tallying predicts a complete tie, but WADD makes clear deterministic predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Trial 4 and Trial 1 are perfectly opposed for WADD versus Tallying. In Trial 4 (A=[5, 0, 0, 0], B=[0, 1, 1, 1]), WADD strongly prefers A due to the high validity of the first cue and its large magnitude difference, whereas Tallying prefers B because it wins on 3 out of 4 features. In Trial 1 (A=[0, 4, 4, 4], B=[5, 3, 3, 3]), WADD prefers B due to the first cue, while Tallying prefers A because A wins on 3 features. The metric calculates the difference in the probability of choosing A between Trial 4 and Trial 1. WADD predicts a value near +1, while Tallying predicts a value near -1.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trial 4 (A=[5,0,0,0], sum=5) and trial 1 (A=[0,4,4,4], sum=12)
    a_sum = data['option_a_ratings'].apply(sum)
    chose_a = (data['response'] == 0).astype(float)
    
    mask_t4 = (a_sum == 5)
    mask_t1 = (a_sum == 12)
    
    p_a_t4 = chose_a[mask_t4].mean() if mask_t4.any() else 0.5
    p_a_t1 = chose_a[mask_t1].mean() if mask_t1.any() else 0.5
    
    return float(p_a_t4 - p_a_t1)

### RESULTS
- Predicted under pi_1 (simulated): -1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.5600 (var=0.6864)
- Observed on real data: -0.4000 (var=0.8400)

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
  "interpretation": "Tallying (pi_1) correctly predicts the direction of the observed effects in both experiments, successfully capturing the human tendency to favor options that win on a larger number of features regardless of the validities or the cardinal magnitudes of those wins. In Experiment 1, Tallying predicts 1.0 and the observed is 0.8. In Experiment 2, Tallying predicts -1.0 and the observed is -0.4. While Tallying's predictions are overly deterministic, it is directionally correct. WADD (pi_2), on the other hand, predicts effects in the completely opposite direction to the observed data (e.g., predicting 0.56 in Exp 2 where the observed is -0.4). Thus, Tallying is the clear winner, but WADD is heavily degenerate and not a competitive baseline for these tasks.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace WADD with a new theory that serves as a stronger competitor to Tallying. A good candidate would be the 'Equal Weight' heuristic, which sums the raw cardinal feature values for each option while ignoring the expert validities. Alternatively, propose a 'Take-The-Best' (lexicographic) heuristic, which makes decisions based solely on the single most valid cue that discriminates between the options. This will help isolate whether subjects are completely ignoring magnitudes (Tallying) or just ignoring validities (Equal Weight), or relying on single cues (TTB)."
}
```

## Usage

```json
{
  "prompt_token_count": 4238,
  "candidates_token_count": 337,
  "total_token_count": 5910
}
```
