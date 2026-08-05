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
    return int(np.argmax(probabilities))


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 5

**Trial pairs (n=6):**
  trial 1: A=[4, 2, 2, 2, 2]  B=[3, 4, 4, 4, 4]
  trial 2: A=[2, 4, 4, 4, 4]  B=[4, 2, 2, 2, 2]
  trial 3: A=[3, 4, 2, 2, 2]  B=[3, 3, 4, 4, 4]
  trial 4: A=[5, 2, 4, 4, 4]  B=[5, 4, 2, 2, 2]
  trial 5: A=[1, 1, 4, 2, 2]  B=[1, 1, 3, 4, 4]
  trial 6: A=[2, 2, 2, 4, 4]  B=[2, 2, 4, 2, 2]

**Rationale:** This design strongly dissociates Take The Best (TTB) from Tallying by pitting the single most-valid discriminating cue against the sheer number of winning cues. On every trial, the option that wins on the highest-validity discriminating feature (which TTB chooses) explicitly loses on the majority of the remaining lower-validity features (which Tallying heavily weights). By using 5 features, we ensure that Tallying will firmly select the option with the most feature-wise wins, while TTB ignores those counts and relies solely on the highest-validity cue. A rating_max of 5 is used to allow for cardinal variance, ensuring the design can also be used to evaluate magnitude-sensitive heuristics like WADD in future analyses.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of trials where the subject's choice matches the prediction of the Take The Best (TTB) heuristic. Because the experimental design explicitly pits TTB against Tallying on every single trial (the option that wins on the highest-validity discriminating cue always loses on the sheer number of winning cues), TTB will score near 1.0 on this metric, whereas Tallying will score near 0.0. The large mean difference and low within-theory variance make it a highly discriminative metric.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = None
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                ttb_pred = 0
                break
            elif val_b > val_a:
                ttb_pred = 1
                break
        if ttb_pred is not None:
            matches.append(row['response'] == ttb_pred)
    if not matches:
        return 0.0
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.2933 (var=0.0695)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.0733 (var=0.0124)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 5

**Trial pairs (n=8):**
  trial 1: A=[5, 2, 2, 2]  B=[4, 4, 4, 4]
  trial 2: A=[2, 5, 5, 5]  B=[4, 3, 3, 3]
  trial 3: A=[5, 2, 3, 3]  B=[4, 4, 3, 3]
  trial 4: A=[2, 5, 3, 3]  B=[4, 4, 3, 3]
  trial 5: A=[3, 5, 2, 2]  B=[3, 4, 4, 4]
  trial 6: A=[3, 2, 5, 5]  B=[3, 4, 4, 4]
  trial 7: A=[3, 3, 5, 2]  B=[3, 3, 4, 4]
  trial 8: A=[3, 3, 2, 5]  B=[3, 3, 4, 4]

**Rationale:** This design systematically dissociates Tallying from Take The Best (TTB) by exploiting both preference reversals and predicted indifference. In half of the trials, the highest-validity discriminating cue favors one option while the simple count of winning cues favors the other, causing a strict preference reversal between TTB and Tallying. In the other half of the trials, the options tie in the total number of winning cues (e.g., 1 win vs 1 win, with the rest tied), leading Tallying to predict complete indifference (guessing), whereas TTB still predicts a deterministic choice based on the single most valid discriminating cue. This provides a clear quantitative divergence: Tallying predicts 50/50 choice probabilities on trials where TTB predicts near 100/0.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the Tallying heuristic on trials where Tallying has a strict preference (i.e., one option has more winning features than the other). In the chosen experimental design, these trials are exactly the ones designed to cause a strict preference reversal between Tallying and Take The Best (TTB). Because TTB predicts the exact opposite choice on these trials, data simulated under Tallying will yield a metric value close to 1.0 (subject to noise), whereas data simulated under TTB will yield a value close to 0.0. This creates a massive, reliable gap between the two theories.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pred = 0 if a_wins > b_wins else 1
            tally_match.append(1.0 if row['response'] == tally_pred else 0.0)
            
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))

### RESULTS
- Predicted under pi_1 (simulated): 0.7100 (var=0.0809)
- Predicted under pi_2 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.8350 (var=0.0340)

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
  "interpretation": "Across both experiments, Tallying (Theory 2) significantly outperforms Take The Best (Theory 1). In Experiment 1, the experimental design strictly pits TTB against Tallying. The observed TTB match rate is extremely low (0.0733), meaning subjects overwhelmingly chose the option favored by Tallying (and opposing TTB). In Experiment 2, the observed match rate for Tallying on strict-preference trials was 0.8350, again heavily favoring the Tallying prediction over the TTB prediction. However, Tallying's deterministic predictions (0.0 in Exp 1, 1.0 in Exp 2) miss some of the nuance in the observed data, which likely reflects sensitivity to the cardinal magnitudes of the features (which both TTB and Tallying ignore). Because TTB is fundamentally misaligned with the data, it should be replaced with a theory that can account for magnitude sensitivity while still capturing the multi-cue compensatory behavior observed in the experiments.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Take The Best with a Weighted Additive (WADD) theory. Unlike TTB (which stops at the first discriminating cue) and Tallying (which counts strict wins but ignores magnitudes), WADD computes a weighted sum of the raw cue values for each option, using the cue validities as weights. The option with the higher weighted sum is chosen. This will allow the model to capture compensatory decision-making while also being sensitive to the cardinal magnitudes of the feature differences, which likely explains the variance in the observed data better than pure Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 4269,
  "candidates_token_count": 363,
  "total_token_count": 5597
}
```
