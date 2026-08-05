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
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Rationale:** To distinguish Take The Best (TTB) from Tallying, we design trials where the option favored by the single most valid cue is opposed by the simple majority of features. TTB stops at the first discriminating cue (highest validity), completely ignoring how many subsequent cues favor the other option. Tallying simply counts the number of feature-wise wins for each option, ignoring cue validities. For instance, if Option A has a positive rating only on the most valid cue and Option B has positive ratings on all other cues, TTB will choose A while Tallying will strongly prefer B. We also include trials where the most valid cues are tied to test the fallback mechanisms of both theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
To cleanly discriminate between Take The Best (TTB) and Tallying, we should focus exclusively on the trials where their predictions diverge. This metric dynamically computes the TTB and Tallying predictions for each trial, filters for trials where the two heuristics predict different choices, and calculates the proportion of subject responses that align with TTB. Under TTB, this metric will be close to 1.0 (subject to noise), while under Tallying, it will be close to 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # TTB prediction: first discriminating feature in descending validity order
    ttb_preds = np.full(len(data), -1)
    for i in range(a_ratings.shape[1]):
        mask_a = (a_ratings[:, i] > b_ratings[:, i]) & (ttb_preds == -1)
        mask_b = (b_ratings[:, i] > a_ratings[:, i]) & (ttb_preds == -1)
        ttb_preds[mask_a] = 0
        ttb_preds[mask_b] = 1
        
    # Tally prediction: sum of feature-wise wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_preds = np.full(len(data), -1)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Identify trials where TTB and Tallying make opposite predictions
    disagree_mask = (ttb_preds != -1) & (tally_preds != -1) & (ttb_preds != tally_preds)
    
    if not np.any(disagree_mask):
        return 0.5
        
    # Calculate proportion of choices aligning with TTB on these crucial trials
    responses = data['response'].values
    matches = (responses[disagree_mask] == ttb_preds[disagree_mask])
    
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8393 (var=0.0102)
- Predicted under pi_2 (simulated): 0.1500 (var=0.0114)
- Observed on real data: 0.5360 (var=0.0087)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 10: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design specifically targets trials where Tallying results in a tie (and thus guesses) while TTB has a strong deterministic preference, as well as trials where the two models have completely opposing preferences. By varying which cue is the highest discriminating cue (sometimes the 1st, sometimes the 2nd or 3rd), we can also observe whether subjects follow the TTB cue cascade or simply count the total number of wins. Comparing trials where Tallying guesses vs. strongly prefers an option allows us to precisely measure the influence of the sheer number of winning features against the single highest-validity discriminating cue.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic on trials where Tallying and Take The Best (TTB) make strictly opposing deterministic predictions. The experimental design specifically includes trials where an option is favored by the single highest-validity discriminating cue (TTB's choice) but the other option is favored by a sheer majority of lower-validity cues (Tallying's choice). By isolating these conflicting trials, we expect data generated by Tallying to score close to 1.0, whereas data generated by TTB will score close to 0.0, yielding maximum discriminability.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    diff = a_mat - b_mat
    nz = diff != 0
    has_nz = np.any(nz, axis=1)
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    
    ttb_pref = np.where(has_nz, np.where(first_diff > 0, 0, 1), -1)
    
    mask = (tally_pref != -1) & (ttb_pref != -1) & (tally_pref != ttb_pref)
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    tally_prefs = tally_pref[mask]
    
    return float(np.mean(responses == tally_prefs))

### RESULTS
- Predicted under pi_1 (simulated): 0.1409 (var=0.0112)
- Predicted under pi_2 (simulated): 0.8751 (var=0.0092)
- Observed on real data: 0.5200 (var=0.0096)

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
  "interpretation": "Across both experiments, the observed metric values hover around 0.5 on trials where Take The Best (THEORY 1) and Tallying (THEORY 2) make strictly opposing predictions. THEORY 1 consistently predicts strong alignment with the single best discriminating cue (~0.84 in Exp 1, ~0.14 in Exp 2), while THEORY 2 predicts strong alignment with the sheer majority of cues (~0.15 in Exp 1, ~0.88 in Exp 2). The empirical data falls almost exactly halfway between these two extremes. This suggests that neither pure one-reason decision making (TTB) nor pure unweighted counting (Tallying) captures human behavior. Subjects are likely either using a compensatory strategy that weighs both the validity and the number of cues (such as a Weighted Additive model), or there is a mixture of strategies across the population.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) theory. Unlike TTB (which ignores all but the top discriminating cue) and Tallying (which ignores cue validities), WADD posits that people evaluate options by summing the features weighted by their respective validities (or perceived importance). In conflict trials where a high-validity cue opposes several lower-validity cues, WADD naturally predicts a closer, more compensatory balance of evidence, leading to choice probabilities much closer to 0.5, perfectly aligning with the observed data."
}
```

## Usage

```json
{
  "prompt_token_count": 4819,
  "candidates_token_count": 332,
  "total_token_count": 5676
}
```
