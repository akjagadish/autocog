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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, this design exploits compensatory vs. non-compensatory feature weighting. By using four features with validities [0.9, 0.8, 0.7, 0.6], we can create 'compensatory' trials where the highest validity cue favors one option, but the sum of the remaining lower-validity cues favors the other option. TTB relies strictly on the first discriminating cue (the highest validity one) and ignores the rest, whereas WADD integrates all features and will choose the option with more lower-validity cues if their summed weight exceeds the top cue's weight. The trial pairs systematically reverse which option is favored by the top cue vs. the sum of the remaining cues, ensuring that overall choice rates and trial-by-trial responses will clearly separate the two models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of trials where the subject's choice aligns with the Take The Best (TTB) prediction. The experimental design specifically pits the highest-validity cue (favored by TTB) against the sum of the remaining lower-validity cues (favored by WADD). Because TTB is a non-compensatory, one-reason heuristic, it will consistently choose the option with the highest validity discriminating cue. By contrast, WADD integrates all cues, and since the sum of the remaining cues outweighs the top cue in these trials, WADD will systematically choose the opposite option. Thus, this metric will be close to 1 for TTB and close to 0 for WADD, providing maximum discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    diff = A - B
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(A.shape[1]):
            if diff[i, j] == 1:
                ttb_preds[i] = 0
                break
            elif diff[i, j] == -1:
                ttb_preds[i] = 1
                break
                
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8521 (var=0.0072)
- Predicted under pi_2 (simulated): 0.3358 (var=0.0293)
- Observed on real data: 0.5025 (var=0.0026)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we use an environment with 5 features where the validities are spread but relatively high overall (e.g., [0.90, 0.80, 0.75, 0.70, 0.65]). This allows the creation of 'compensatory' trials where the highest-validity discriminating cue strongly favors one option, but the sum of the remaining lower-validity cues favors the other. TTB will invariably choose the option favored by the single best discriminating cue, completely ignoring the sheer number of lower-validity cues pointing the other way. By contrast, WADD integrates all features weighted by their validities, meaning that a coalition of multiple lower-validity cues will systematically override the single highest-validity cue. The trial pairs pit the top cue against various combinations of the remaining cues to clearly separate the predictions of the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In the chosen experimental design, all 6 trial types are structurally compensatory: the single most valid discriminating cue always points to Option A, while the sum of the remaining less-valid discriminating cues always points to Option B. Consequently, Take The Best (TTB) will invariably choose Option A (response = 0) because it stops at the first discriminating cue. In contrast, the Weighted Additive (WADD) model integrates all cues, and since the sum of the validities of the lower cues outweighs the top cue, WADD will systematically choose Option B (response = 1). Therefore, the overall proportion of choices for Option B (the mean of the 'response' column) will be close to 0 for TTB and close to 1 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1435 (var=0.0097)
- Predicted under pi_2 (simulated): 0.7156 (var=0.0163)
- Observed on real data: 0.4996 (var=0.0028)

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
  "interpretation": "Across both experiments, the observed data falls almost exactly halfway between the predictions of the pure Take The Best (TTB) theory and the pure Weighted Additive (WADD) theory. In Experiment 1, the observed TTB match rate is ~0.50 (vs 0.85 for TTB and 0.33 for WADD). In Experiment 2, the rate of choosing the WADD-favored option is ~0.50 (vs 0.14 for TTB and 0.71 for WADD). The extremely low between-subject variance in both experiments suggests this 50/50 split is highly consistent across the population. Neither a pure non-compensatory nor a pure compensatory model can independently explain this intermediate behavior, indicating that a strictly monolithic theory is degenerate here.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture' or 'Cognitive Style' theory. This new theory should posit that the population is heterogeneous, consisting of a mixture of individuals (or trial-by-trial strategy selections) where a proportion of choices are made using a non-compensatory heuristic (like TTB) and the rest are made using a compensatory strategy (like WADD). By introducing a mixture parameter, the model can capture the consistent ~0.50 aggregate behavior that purely monolithic theories fail to predict."
}
```

## Usage

```json
{
  "prompt_token_count": 4367,
  "candidates_token_count": 312,
  "total_token_count": 5674
}
```
