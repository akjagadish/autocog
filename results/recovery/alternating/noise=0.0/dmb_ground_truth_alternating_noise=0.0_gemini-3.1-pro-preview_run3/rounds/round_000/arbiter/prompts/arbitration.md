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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design specifically pits Take The Best (TTB) against Tallying. TTB relies solely on the highest-validity cue that discriminates between the two options, completely ignoring the sheer number of cues favoring each option. In contrast, Tallying counts the total number of features favoring each option, ignoring cue validities. The trials are constructed so that one option always wins on the highest-validity discriminating cue (favored by TTB), while the other option wins on a greater number of lower-validity cues (favored by Tallying). This provides a strong quantitative dissociation between the two models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the prediction of the Take The Best (TTB) heuristic. Because the experimental design pits TTB against Tallying on almost all trials (where the highest validity cue favors one option and the sheer number of cues favors the other), TTB simulated agents will have a high score on this metric, whereas Tallying agents will have a low score.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
            
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8444 (var=0.0093)
- Predicted under pi_2 (simulated): 0.2267 (var=0.0038)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 8: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Tallying from Take The Best (TTB) by exploiting how each model generates choice probabilities. TTB is a 'one-reason' decision rule: once the highest-validity cue discriminates between options, it stops searching. Consequently, TTB's predicted choice probability depends entirely on a fixed binary score difference (1 vs 0) and remains constant across all trials where the first cue discriminates. In contrast, Tallying counts the total number of features favoring each option, and its choice probability scales with the *magnitude* of the difference in tallies. By holding the highest-validity cue constant (always discriminating in favor of one option) while systematically varying the tally difference on the remaining cues (from strongly opposing the first cue to strongly supporting it), Tallying predicts a graded, monotonic change in choice probabilities across trials, whereas TTB predicts a completely flat probability profile.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing the option favored by the highest-validity cue (cue 0) between trials where the overall tally strongly supports that same option (tally difference >= 3) and trials where the overall tally opposes or ties it (tally difference <= 0). For Take The Best, cue 0 is the only cue ever consulted, so the choice probability is insensitive to the remaining cues, yielding a metric near 0. For Tallying, choice probability is driven entirely by the tally difference, meaning the subject will frequently choose the cue 0 winner when the tally strongly agrees, but will guess or choose the other option when the tally disagrees, yielding a large positive metric.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Determine which option is favored by the highest validity cue (cue 0)
    cue_0_a = a_ratings[:, 0]
    cue_0_b = b_ratings[:, 0]
    cue_0_winner = (cue_0_b > cue_0_a).astype(int)
    
    # Did the subject choose the option favored by cue 0?
    chose_cue_0 = (data['response'] == cue_0_winner).astype(float)
    
    # Calculate the tally difference in favor of the cue 0 winner
    tally_a = (a_ratings > b_ratings).sum(axis=1)
    tally_b = (b_ratings > a_ratings).sum(axis=1)
    tally_diff = np.where(cue_0_winner == 0, tally_a - tally_b, tally_b - tally_a)
    
    # Compare choice probabilities when tally strongly agrees vs. disagrees/ties with cue 0
    high_diff_mask = tally_diff >= 3
    low_diff_mask = tally_diff <= 0
    
    if not np.any(high_diff_mask) or not np.any(low_diff_mask):
        return 0.0
        
    high_diff_rate = chose_cue_0[high_diff_mask].mean()
    low_diff_rate = chose_cue_0[low_diff_mask].mean()
    
    return float(high_diff_rate - low_diff_rate)


### RESULTS
- Predicted under pi_1 (simulated): -0.0025 (var=0.0080)
- Predicted under pi_2 (simulated): 0.6340 (var=0.0248)
- Observed on real data: -0.0237 (var=0.0386)

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
  "interpretation": "In Experiment 1, TTB predicts a high match rate (~0.84) and Tallying predicts a low match rate (~0.22), while the observed data is exactly 0.50, meaning subjects chose the TTB-favored option half the time. In Experiment 2, Tallying predicts a large difference in choice probability when the tally strongly agrees vs. disagrees with the highest-validity cue (~0.63), whereas TTB predicts no difference (~0.0). The observed data in Experiment 2 is roughly 0 (-0.02), which perfectly aligns with TTB's prediction of insensitivity to the tally difference. This suggests that while subjects are not integrating all cues into a tally (rejecting Tallying), they are also not deterministically using TTB as parameterized, given the 50% match in Exp 1. Tallying (pi_2) completely fails Experiment 2, so it should be replaced with a stronger compensatory baseline, such as a Weighted Additive (WADD) model, which weights features by their subjective validities (e.g., log-odds) rather than just counting them.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) theory. Unlike Tallying (which gives equal weight to all cues) or TTB (which only uses the single best discriminating cue), WADD assumes subjects compute a weighted sum of the features for each option, where the weights are proportional to the cues' validities or their log-odds. This provides a formal compensatory model that scales with both cue validity and the number of cues, offering a more realistic alternative to the non-compensatory TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 4782,
  "candidates_token_count": 385,
  "total_token_count": 6474
}
```
