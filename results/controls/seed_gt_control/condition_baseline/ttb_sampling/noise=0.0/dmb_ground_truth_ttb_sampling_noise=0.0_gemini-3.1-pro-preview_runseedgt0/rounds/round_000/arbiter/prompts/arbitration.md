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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 6: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with strictly descending validities. We design trials where the highest-validity discriminating cue strongly favors one option (driving TTB's choice), while the sheer number of lower-validity cues favors the other option (driving Tallying's choice). We also include trials where Tallying sees a tie (equal number of wins) but TTB has a clear preference, and vice versa. This allows us to separate the non-compensatory, validity-ordered stopping rule of TTB from the compensatory, validity-ignoring counting rule of Tallying.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of trials where the subject's choice aligns with the prediction of Take The Best (TTB). Because the experimental design pits TTB and Tallying directly against each other on most trials (where TTB has a strong preference for one option and Tallying has a strong preference for the other), TTB simulated subjects will score very close to 1.0, whereas Tallying simulated subjects will score close to 0.125 (since they choose the opposite of TTB on 6 out of 8 trial types, and guess on the remaining 2). This creates an extremely large and stable gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    agreements = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                agreements += 1
            total += 1
            
    return float(agreements / total) if total > 0 else 0.0

### RESULTS
- Predicted under pi_1 (simulated): 0.8342 (var=0.0101)
- Predicted under pi_2 (simulated): 0.2537 (var=0.0064)
- Observed on real data: 0.8492 (var=0.0099)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** This design uses 5 features with strictly descending validities to strongly dissociate Tallying from Take The Best (TTB). We include 'compensatory' trials where the highest-validity discriminating cue strongly favors one option (driving TTB's choice), while a larger number of lower-validity cues favors the other option (driving Tallying's choice). We also include trials where Tallying results in a tie (equal number of wins for both options) but TTB has a clear preference due to the validity of the first discriminating cue. This directly contrasts TTB's non-compensatory, validity-ordered stopping rule against Tallying's compensatory, validity-ignoring counting rule.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the Tallying heuristic on trials where Tallying predicts a strict preference (i.e., excluding ties). Given the experimental design, the trials where Tallying has a strict preference are precisely those where Take The Best (TTB) makes the opposite prediction. Therefore, data generated by the Tallying model will yield a metric value close to 1.0, whereas data generated by TTB will yield a value close to 0.0. This maximal divergence guarantees strong discrimination between the two theories.

Source:
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            matches += (resp == 0)
            total += 1
        elif b_wins > a_wins:
            matches += (resp == 1)
            total += 1
    return float(matches / total) if total > 0 else 0.5


### RESULTS
- Predicted under pi_1 (simulated): 0.1358 (var=0.0099)
- Predicted under pi_2 (simulated): 0.8858 (var=0.0098)
- Observed on real data: 0.1739 (var=0.0108)

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
  "interpretation": "Across both experiments, the data overwhelmingly supports Theory 1 (Take The Best) over Theory 2 (Tallying). In Experiment 1, the observed metric (0.8492) closely matches the prediction of Take The Best (0.8342), whereas Tallying predicted a low alignment (0.2537). In Experiment 2, the observed proportion of choices consistent with Tallying is very low (0.1739), which again aligns with Take The Best's prediction (0.1358) and thoroughly refutes Tallying's prediction (0.8858). Since Take The Best is extremely successful at explaining the observed behavior, Theory 2 is fundamentally degenerate for this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) model or a rational Bayesian integration model. Unlike TTB (which stops at the first discriminating cue) and Tallying (which counts unweighted wins), a Weighted Additive model computes a sum of feature differences weighted by their respective validities (or log-odds). This provides a strong compensatory alternative that still utilizes validity information, offering a more rigorous test against the non-compensatory Take The Best model in future experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 4325,
  "candidates_token_count": 292,
  "total_token_count": 4950
}
```
