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
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with WADD (which computes a validity-weighted sum). The validities are deliberately highly skewed ([0.95, 0.85, 0.60, 0.55, 0.50]). In critical trials (e.g., A=[0,0,1,1,1] vs B=[1,1,0,0,0]), Tallying strongly favors A because it wins on 3 out of 5 features. In contrast, WADD strongly favors B because the two features it wins on have much higher validities (0.95 + 0.85 = 1.80) than the three features A wins on (0.60 + 0.55 + 0.50 = 1.65). The design also includes pairs where both models agree, and pairs where Tallying predicts a tie but WADD has a clear preference, providing a rich set of quantitative dissociations.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with Tallying on the critical dissociation trials (Trials 1 and 2). In these trials, Tallying favors the option that wins on 3 out of 5 features, while WADD strongly favors the other option because the 2 features it wins on have much higher validities (0.95 and 0.85). Tallying will score close to 1.0, while WADD will score close to 0.0, providing maximum discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers A (0), WADD prefers B (1)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        # Trial 2: Tallying prefers B (1), WADD prefers A (0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total

### RESULTS
- Predicted under pi_1 (simulated): 0.8544 (var=0.0163)
- Predicted under pi_2 (simulated): 0.4400 (var=0.0801)
- Observed on real data: 0.1311 (var=0.0202)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design systematically dissociates the Weighted Additive rule (WADD) and the Tallying heuristic using a 6-feature setup with a wide spread of validities. It includes trials where the models make opposite predictions (e.g., Tallying favors the option with more positive features, while WADD favors the option with fewer but more valid features), as well as trials where one model predicts a tie while the other predicts a strong preference. This enables a precise quantitative evaluation of whether subjects rely on feature validities (WADD) or simply count the number of winning features (Tallying).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 5. In Trial 1, Tallying favors Option A because it has more positive features (3 vs 2), whereas WADD favors Option B because its features have much higher validities. In Trial 5, the situation is exactly reversed: Tallying favors B (3 vs 2 features), while WADD favors A (higher validities). Thus, the difference P(A|Trial 1) - P(A|Trial 5) will be strongly positive under Tallying and strongly negative under WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: Tallying prefers A (3 vs 2 wins), WADD prefers B (higher validity sum)
    t1_a = (0, 0, 0, 1, 1, 1)
    
    # Trial 5: Tallying prefers B (3 vs 2 wins), WADD prefers A (higher validity sum)
    t5_a = (1, 1, 0, 0, 0, 0)
    
    t1_mask = a_tuples == t1_a
    t5_mask = a_tuples == t5_a
    
    t1_resp = data.loc[t1_mask, 'response']
    t5_resp = data.loc[t5_mask, 'response']
    
    if len(t1_resp) == 0 or len(t5_resp) == 0:
        return 0.0
        
    p_a_t1 = (t1_resp == 0).mean()
    p_a_t5 = (t5_resp == 0).mean()
    
    # Tallying: P(A|T1) is high, P(A|T5) is low -> Positive difference
    # WADD: P(A|T1) is low, P(A|T5) is high -> Negative difference
    return float(p_a_t1 - p_a_t5)

### RESULTS
- Predicted under pi_1 (simulated): 0.6958 (var=0.0454)
- Predicted under pi_2 (simulated): -0.1150 (var=0.2399)
- Observed on real data: -0.6650 (var=0.0405)

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
  "interpretation": "The observed data across both experiments strongly contradict the Tallying heuristic (pi_1) and align with the directional predictions of the Weighted Additive rule (WADD, pi_2). In Experiment 1, subjects overwhelmingly chose the option favored by higher validities rather than the one with more winning features (observed metric 0.1311 vs pi_1's 0.8544). In Experiment 2, the observed difference was strongly negative (-0.6650), which matches WADD's predictions and directly opposes Tallying's predicted positive difference (0.6958). However, pi_2's quantitative predictions were heavily muted (e.g., predicting only -0.1150 in Exp 2). This underprediction is likely due to the model multiplying the true validities by an additional, unconstrained 'weights' parameter vector (`validities * w`), which dilutes the validities and introduces noise. WADD is conceptually correct, but its mathematical implementation needs to be fixed.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Revise the WADD model's `predict` function to compute scores using the provided validities directly (e.g., `scores = stim @ validities`), completely removing the redundant and noisy `weights` parameter. Additionally, ensure the `beta` parameter range is wide enough (e.g., up to 10 or 20) to capture the strong, highly deterministic choices observed in the human data."
}
```

## Usage

```json
{
  "prompt_token_count": 4677,
  "candidates_token_count": 342,
  "total_token_count": 6291
}
```
