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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the compensatory nature of WADD versus the non-compensatory, lexicographic nature of TTB. The experiment uses 5 features with descending validities. In the critical trials, one option possesses the single best discriminating cue (which dictates TTB's choice), while the competing option possesses multiple lower-validity cues whose combined weight exceeds the single best cue (which dictates WADD's choice). By including trials where the 'best' cue is offset by 2, 3, or 4 lower-validity cues, we can robustly measure the degree of compensation subjects exhibit. Control trials where both models agree are also included to ensure basic task engagement.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the Take The Best (TTB) heuristic. Since the experimental design includes several 'compensatory' trials where a single high-validity cue is pitted against multiple lower-validity cues, the TTB model will consistently choose the option with the highest discriminating cue, yielding a high score on this metric. In contrast, the Weighted Additive (WADD) model will often compensate and choose the option with multiple lower-validity cues, resulting in a significantly lower score.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    return float(np.mean(matches)) if matches else 0.5


### RESULTS
- Predicted under pi_1 (simulated): 0.8596 (var=0.0092)
- Predicted under pi_2 (simulated): 0.3973 (var=0.0146)
- Observed on real data: 0.2787 (var=0.0031)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.65, 0.95, 0.55, 0.8]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** To cleanly dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, this design uses 4 features with non-descending validities. By scrambling the order of validities (e.g., placing the most valid cue second), we ensure subjects must actively process the validity numbers rather than relying on left-to-right reading order. The trials include critical compensatory cases where the option possessing the single best discriminating cue is pitted against an option possessing multiple lower-validity cues whose combined weight exceeds the best cue. In these cases, TTB and WADD make opposite predictions. We also include trials where the best cue is tied, forcing TTB to look at the second-best cue, which is then outweighed by the remaining cues for WADD, as well as control trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Weighted Additive (WADD) model specifically on trials where WADD and Take The Best (TTB) make strictly opposite predictions. By isolating these critical trials, we maximize the divergence between the two models. Under WADD, subjects should systematically choose the WADD-favored option (yielding a metric close to 1), while under TTB, subjects will systematically choose the TTB-favored option (yielding a metric close to 0).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.65, 0.95, 0.55, 0.8])
    cue_order = [1, 3, 0, 2]
    
    wadd_aligned = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is None:
            continue
            
        wadd_winner = 0 if wadd_a > wadd_b else (1 if wadd_b > wadd_a else None)
        
        if wadd_winner is not None and ttb_winner != wadd_winner:
            resp = row['response']
            if resp == wadd_winner:
                wadd_aligned += 1
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_aligned / total

### RESULTS
- Predicted under pi_1 (simulated): 0.1525 (var=0.0116)
- Predicted under pi_2 (simulated): 0.6538 (var=0.0294)
- Observed on real data: 0.8967 (var=0.0081)

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
  "interpretation": "Across both experiments, Theory 1 (Take The Best) is decisively rejected. In Experiment 1, the observed TTB adherence (0.28) is far lower than TTB's prediction (0.86) and even lower than WADD's prediction (0.40). In Experiment 2, on trials where TTB and WADD conflict, subjects overwhelmingly chose the WADD-favored option (0.90 observed vs. 0.15 predicted by TTB and 0.65 predicted by WADD). The data strongly indicate a compensatory decision strategy. However, the fact that observed TTB adherence in Exp 1 is even lower than WADD's prediction, and WADD alignment in Exp 2 is even higher than WADD's prediction, suggests that subjects might be using an even simpler compensatory rule that heavily penalizes single-cue advantages, such as Tallying (Equal Weights). TTB is degenerate here and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Take The Best with 'Tallying' (or Equal Weight heuristic). In this theory, subjects ignore the explicit cue validities and simply count the number of positive features for each option. The option with the highest tally (simple sum of positive cues) is chosen. This provides a strongly compensatory alternative to WADD that does not require the cognitive overhead of multiplying features by exact validities, potentially explaining the extreme deviations from TTB observed in both experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 4723,
  "candidates_token_count": 335,
  "total_token_count": 6193
}
```
