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
**Validities (n_features=4):** [0.95, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** To dissociate Take The Best (TTB) from the Weighted Additive (WADD) rule, this design uses a 'compensatory' structure where the most valid cue often points to one option while the sum of the less valid cues points to the other. TTB is a non-compensatory heuristic: it stops at the first discriminating cue (ordered by validity) and ignores all others. In contrast, WADD computes a weighted sum of all features, allowing multiple weaker cues to outvote a single strong cue. By setting the validities such that the highest validity is less than the sum of the lower validities (e.g., 0.95 vs 0.70+0.65+0.60), we create trials where TTB and WADD make diametrically opposed predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Take The Best (TTB) against the Weighted Additive rule (WADD) by creating compensatory trials where the most valid cue points to one option, while the sum of the less valid cues points to the other. TTB, being a non-compensatory 'one-reason' decision heuristic, will consistently choose the option favored by the single most valid discriminating cue. In contrast, WADD integrates all cues and will often favor the option with multiple weaker cues, as their combined validity outweighs the single best cue. By measuring the proportion of trials where the subject's choice aligns with the TTB prediction, we can strongly discriminate between data generated by TTB (high proportion) and WADD (low proportion).

Source:
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None and row['response'] == ttb_choice:
            ttb_matches += 1
            
    return float(ttb_matches / total) if total > 0 else 0.0

### RESULTS
- Predicted under pi_1 (simulated): 0.8638 (var=0.0118)
- Predicted under pi_2 (simulated): 0.3210 (var=0.0381)
- Observed on real data: 0.4071 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 8: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) rule from Take The Best (TTB), this design uses a set of four cues with validities carefully chosen such that the most valid cue is outweighed by combinations of less valid cues. TTB always decides based on the single highest-validity cue that discriminates between the options. In contrast, WADD computes a compensatory weighted sum. By presenting pairs where the highest-validity cue favors one option but the sum of the remaining cues favors the other, we create strong opposite predictions. Additional trials where both models agree serve as internal controls to measure baseline noise and lapse rates independently of strategy differences.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In this experimental design, trials where the sum of features differs between Option A and Option B are specifically constructed to pit the most valid cue against a combination of less valid cues. Take The Best (TTB) is non-compensatory and always chooses the option favored by the single highest-validity cue, which in these specific trials is always the option with the FEWER total positive features. Thus, TTB will score near 0 on choosing the option with the higher feature sum. Weighted Additive (WADD) integrates all features, and because the sum of the less valid cues' validities outweighs the most valid cue, WADD will typically choose the option with the HIGHER feature sum. This metric calculates the proportion of times a subject chooses the option with the higher unweighted sum of features on trials where the sums are unequal, providing a stark divergence between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_resp = data.loc[mask, 'response']
    a_higher = sum_a[mask] > sum_b[mask]
    
    chose_higher = (a_higher & (filtered_resp == 0)) | (~a_higher & (filtered_resp == 1))
    return float(chose_higher.mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1458 (var=0.0146)
- Predicted under pi_2 (simulated): 0.6663 (var=0.0281)
- Observed on real data: 0.6008 (var=0.0065)

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
  "interpretation": "Across both experiments, the observed data strongly contradict Theory 1 (Take The Best). TTB predicts highly non-compensatory choices (e.g., ~86% TTB matches in Exp 1 and ~15% higher-sum choices in Exp 2), but the actual subjects exhibited compensatory behavior, matching TTB only ~41% of the time and choosing the higher-sum option ~60% of the time. Theory 2 (Weighted Additive) captures the qualitative pattern of this compensatory behavior much better, predicting 32% and 66% on the respective metrics. Since TTB is definitively failing to capture the compensatory nature of the subjects' choices, Theory 1 should be completely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying' (Equal Weights) theory or a 'Rank-Weighted Additive' theory. Tallying posits that subjects simply count the number of positive features for each option and choose the one with the higher count, ignoring the precise cue validities. A Rank-Weighted model posits that subjects assign weights based on the ordinal rank of the cues rather than their cardinal validities. Both serve as strong, classic compensatory alternatives to the WADD model that might better capture the exact mixture of choices observed."
}
```

## Usage

```json
{
  "prompt_token_count": 4472,
  "candidates_token_count": 293,
  "total_token_count": 5985
}
```
