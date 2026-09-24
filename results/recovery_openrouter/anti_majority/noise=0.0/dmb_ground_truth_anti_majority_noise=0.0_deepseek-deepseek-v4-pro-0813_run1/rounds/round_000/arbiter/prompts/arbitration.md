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
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.91, 0.63, 0.91, 0.63, 0.63]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 2: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 6: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** Five experts are used with the high-validity cue at display position 3 (list index 2, validity 0.91) so visual display order is not confounded with TTB cue order; TTB re-sorts it to first. Every trial has that top cue discriminating. The lower four cues are arranged to produce tallying margins d = wins(A) - wins(B) of -3, -2, -1, 0 in trials 1-4 and +3, +2, +1, 0 in the mirrored trials 5-8. Under TTB, P(A) should be flat within a top-cue direction: high for all top-cue favors A trials and low for all top-cue favors B trials, regardless of d. Under Tallying, P(A) should increase monotonically with d and stay at chance when d = 0. This creates opposite qualitative predictions at d = -3, -2, -1, +1, +2, +3, and a chance-versus-directional difference at d = 0. Eight unique pairs repeated twelve times give 96 trials per subject.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is the empirical effect of the second high-validity cue when the Tallying score is tied. In this fixed design, the two d=0 pairs have feature 0 tied and feature 2 discriminating. Take The Best says feature 2 then breaks the tie, so the B-choice rate should be higher when feature 2 favors B than when feature 2 favors A; in expectation the contrast is (1 - epsilon) * (2 * sigmoid(beta) - 1), where sigmoid(beta) = 1 / (1 + exp(-beta)). Tallying gives feature 2 no special weight; with equal win counts both d=0 cells have choice probability 0.5 under softmax and lapse, so the signed contrast is zero. The per-subject variance is limited because each subject contributes 12 repetitions to each side of the contrast.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=int)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=int)
    r = data['response'].to_numpy(dtype=float)

    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    tie = a_wins == b_wins

    # Among tally ties, the high-validity cue at index 2 discriminates.
    cue_favors_b = tie & (b[:, 2] > a[:, 2])
    cue_favors_a = tie & (a[:, 2] > b[:, 2])

    if not np.any(cue_favors_b) or not np.any(cue_favors_a):
        return 0.0

    p_b_given_b_cue = float(r[cue_favors_b].mean())
    p_b_given_a_cue = float(r[cue_favors_a].mean())

    return p_b_given_b_cue - p_b_given_a_cue


### RESULTS
- Predicted under pi_1 (simulated): 0.6650 (var=0.0546)
- Predicted under pi_2 (simulated): 0.0383 (var=0.0359)
- Observed on real data: -0.4400 (var=0.1503)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.78, 0.55, 0.91, 0.5, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** Five experts are used. The highest-validity expert (91%) is placed at display index 2, so visual order is not confounded with Take The Best's validity order (which re-sorts to: index 2, then 0, then 4, then 1, then 3). Every trial is constructed so that the net feature tally is exactly balanced: A wins on exactly as many features as B wins, so Tallying predicts p(A)=0.5 on every trial. By contrast, Take The Best never tallies; it searches down the validity-ordered cue cascade and uses the first discriminating cue. In trials 1-2 the first discriminator is the top expert and favors A; trials 3-4 favor B; trials 5-6 tie the top expert and use the second expert (favoring A/B); trials 7-8 tie the top two cues and use the third expert (favoring A/B). Thus TTB predicts a high cluster for A-predicting cues and a low cluster for B-predicting cues, while Tallying predicts a flat line at chance. This isolates the core dissociation: balanced evidence produces guessing under tallying but one-reason directional choice under Take The Best, and it does so without repeating the already-explored monotone tally-margin design.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric is a within-subject first-discriminating-cue contrast: P(choose A | the highest-validity cue that differs favors A) minus P(choose A | it favors B). In this design every trial is exactly tally-balanced, so Tallying predicts P(A)=0.5 in both trial classes and the contrast is 0. Take The Best predicts choices follow the first discriminating cue in validity order, so the A-favoring class should be well above 0.5 and the B-favoring class below 0.5, pushing the contrast toward positive values. Each class contains four unique pairs repeated 12 times, giving stable per-subject proportions.

Source:
import numpy as np

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.78, 0.55, 0.91, 0.50, 0.65], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    a = np.vstack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.vstack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b

    ordered_diff = diff[:, cue_order]
    nonzeros = ordered_diff != 0
    has_discriminator = nonzeros.any(axis=1)
    first_pos = nonzeros.argmax(axis=1)
    first_values = ordered_diff[np.arange(ordered_diff.shape[0]), first_pos]

    choose_a = (data['response'].to_numpy() == 0).astype(float)
    first_cue_favors_a = has_discriminator & (first_values > 0)
    first_cue_favors_b = has_discriminator & (first_values < 0)

    if not (first_cue_favors_a.any() and first_cue_favors_b.any()):
        return 0.0

    p_a_given_cue_a = choose_a[first_cue_favors_a].mean()
    p_a_given_cue_b = choose_a[first_cue_favors_b].mean()
    return float(p_a_given_cue_a - p_a_given_cue_b)

### RESULTS
- Predicted under pi_1 (simulated): 0.6875 (var=0.0362)
- Predicted under pi_2 (simulated): -0.0029 (var=0.0086)
- Observed on real data: -0.3750 (var=0.0321)

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
  "interpretation": "Neither current theory explains the observed sign pattern. In Experiment 1, pi_1 predicts a large positive contrast (+0.665) and pi_2 predicts roughly zero (+0.038), but the observed effect is strongly negative (-0.440). In Experiment 2, pi_1 again predicts a large positive contrast (+0.688) and pi_2 predicts roughly zero (-0.003), but the observed effect is again negative (-0.375). Numerically pi_2 is less wrong than pi_1, with absolute errors of 0.478 and 0.372 versus pi_1 errors of 1.105 and 1.063, but pi_2 still fails qualitatively because its tally-equality prediction cannot move off zero on these balanced designs. The consistent negative observed contrasts indicate that subjects are not following the communicated validity order, and they are also not evenly splitting tally-tied pairs. Instead, the pattern strongly matches a lexicographic rule that follows the visual or display order of the features rather than their stated validity. In Experiment 1, the two tally-tied trials are tied on feature 0, so the first displayed discriminating cue is feature 1, which favors B in one trial and A in the other, producing exactly the observed negative contrast. In Experiment 2, rescoring the trials by display order shows that the validity-first-cue A class often has the first displayed feature favoring B, while the validity-first-cue B class often has it favoring A, again producing a negative contrast close to the observed value. Thus pi_1 is contradicted in its core ordering claim, while pi_2 is contradicted by its enforced feature equality. A new theory is warranted rather than a parameter refit of either current model.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace THEORY 1 with a display-order lexicographic heuristic. The new theory should claim that people scan the feature vector in its physical display order, typically left-to-right, and stop at the first feature on which the two options differ. That first discriminating displayed feature determines the choice, later features are ignored, and complete ties lead to guessing. Stated validities may be ignored or may only weakly modulate salience. This differs from pi_1 by replacing validity-based cue order with display-based cue order, and differs from pi_2 by preserving one-cue stopping rather than equal-wins tallying. To capture the data: in Experiment 1, feature 0 is tied in the relevant pairs, so feature 1 becomes the first displayed discriminator, yielding a negative P(B given feature2 favors B) minus P(B given feature2 favors A) contrast. In Experiment 2, learning re-ordered by display position produces a negative contrast in the first-validity-cue metric around -0.5 before noise. The new predict function should order cues by display index, apply the existing TTB-style first-discriminator rule, and retain softmax plus lapse noise. Parameter ranges should include moderate beta and small epsilon so simulated contrasts can reach the observed values of -0.44 and -0.375."
}
```

## Usage

```json
{
  "input_tokens": 5509,
  "output_tokens": 11032,
  "reasoning_tokens": 10369,
  "provider": "Baidu"
}
```
