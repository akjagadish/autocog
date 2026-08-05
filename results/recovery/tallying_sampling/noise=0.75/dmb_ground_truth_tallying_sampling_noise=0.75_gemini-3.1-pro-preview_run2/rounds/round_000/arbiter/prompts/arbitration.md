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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To cleanly dissociate Take The Best (TTB) from Weighted Additive (WADD), we must construct 'compensatory' trials where the highest-validity discriminating cue points to one option, but the sum of the remaining lower-validity cues points to the other option. TTB, being a non-compensatory one-reason heuristic, will exclusively follow the highest-validity cue. WADD, being compensatory, will integrate all cues and can choose the option favored by multiple lower-validity cues if their summed weight exceeds that of the single highest-validity cue. We include a mix of these critical dissociation trials and agreement (catch) trials to accurately estimate response noise and model parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the provided experimental design, the first feature (which has the highest validity of 0.95) discriminates between the two options on every single trial. Because Take The Best (TTB) is a non-compensatory one-reason heuristic, it will exclusively rely on this highest-validity feature and always choose the option it favors. Weighted Additive (WADD), on the other hand, is compensatory and integrates all features. On 6 out of the 8 trial types, the sum of the lower-validity features strongly favors the opposite option, leading WADD to frequently choose the option that is NOT favored by the first feature. Therefore, the proportion of choices that align with the first feature will be very close to 1.0 for TTB, and much lower (around 0.25) for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_choice = np.where(b_0 > a_0, 1, 0)
    return float(np.mean(data['response'] == ttb_choice))

### RESULTS
- Predicted under pi_1 (simulated): 0.8585 (var=0.0065)
- Predicted under pi_2 (simulated): 0.4006 (var=0.0278)
- Observed on real data: 0.4550 (var=0.0021)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we employ a 4-feature design where the highest validity cue strongly discriminates, but the sum of lower validity cues outweighs it. TTB strictly follows the highest validity cue, whereas WADD integrates all cues. We include compensatory trials where the models predict opposite choices, and non-compensatory trials where both models agree, ensuring robust estimation of response noise and specific model mechanisms.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the 'compensatory' trials where Take The Best (TTB) and Weighted Additive (WADD) make opposite predictions. TTB stops at the first discriminating cue, while WADD integrates all cues. By computing the proportion of choices that align with TTB specifically on trials where the sum of lower-validity cues outweighs the highest-validity cue, we can sharply discriminate the two theories: data generated by TTB will yield a value close to 1, whereas data generated by WADD will yield a value below 0.5.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    def is_ttb_match(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # WADD expected prediction (assuming uniform weights)
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        # Only consider compensatory trials where the models disagree
        if ttb_winner is not None and ttb_winner != wadd_winner:
            return 1.0 if row['response'] == ttb_winner else 0.0
        return np.nan

    matches = data.apply(is_ttb_match, axis=1)
    return float(matches.mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8517 (var=0.0098)
- Predicted under pi_2 (simulated): 0.3225 (var=0.0383)
- Observed on real data: 0.4225 (var=0.0057)

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
  "interpretation": "Across both experiments, the observed data strongly contradict the predictions of Theory 1 (Take The Best, pi_1). The metrics measure the proportion of choices aligning with the highest-validity cue or the TTB prediction on compensatory trials. TTB predicts values around 0.85, whereas the observed data are around 0.42-0.45, closely matching the predictions of Theory 2 (Weighted Additive, pi_2) which predicts 0.32-0.40. Since TTB fundamentally fails to capture the compensatory nature of the subjects' choices, it should be discarded in favor of a new compensatory or semi-compensatory heuristic that can act as a stronger competitor to WADD.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'Tallying' (Equal Weighting) as the new theory. Unlike TTB, which stops at the first discriminating cue, and unlike WADD, which scales cues by their exact validities, Tallying simply counts the total number of positive features for each option and chooses the one with the higher count. This provides a frugal but compensatory alternative to WADD, testing whether subjects actually use the provided validities or just sum the raw evidence."
}
```

## Usage

```json
{
  "prompt_token_count": 4443,
  "candidates_token_count": 281,
  "total_token_count": 5296
}
```
