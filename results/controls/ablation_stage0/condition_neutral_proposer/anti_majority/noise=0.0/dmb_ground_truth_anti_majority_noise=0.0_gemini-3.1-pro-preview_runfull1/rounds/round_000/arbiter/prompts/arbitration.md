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
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 7: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 8: A=[1, 0, 1, 0]  B=[0, 1, 0, 0]

**Rationale:** To distinguish between Take The Best (TTB) and Tallying, we need trials where the highest validity cue points to one option, but the simple majority of features (ignoring validity) points to the other option. TTB relies exclusively on the first discriminating cue in descending order of validity, while Tallying simply counts the number of features won by each option. The selected pairs pit a single high-validity advantage against multiple lower-validity advantages, causing TTB and Tallying to make opposite predictions, or causing one to predict a strong preference while the other predicts a tie.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific trials where Take The Best (TTB) and Tallying make strictly opposite predictions (e.g., when one option wins on the single highest-validity cue, but the other option wins on a simple majority of all other cues). By computing the proportion of choices that align with TTB on these critical trials, the metric will yield values close to 1.0 if TTB is the true generating model, and values close to 0.0 if Tallying is the true generating model. The large and consistent gap between these outcomes perfectly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd

    def get_predictions(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    preds = data.apply(get_predictions, axis=1)
    
    # Filter for trials where TTB and Tallying make strictly opposite predictions
    mask = (preds['ttb'] != -1) & (preds['tally'] != -1) & (preds['ttb'] != preds['tally'])
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of choices that align with Take The Best on these critical trials
    ttb_choices = (data.loc[mask, 'response'] == preds.loc[mask, 'ttb']).mean()
    return float(ttb_choices)


### RESULTS
- Predicted under pi_1 (simulated): 0.8546 (var=0.0146)
- Predicted under pi_2 (simulated): 0.1412 (var=0.0101)
- Observed on real data: 0.7858 (var=0.0260)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To maximally distinguish Tallying from Take The Best (TTB), this design pits the highest-validity discriminating cue against the sheer count of winning features. In several trials, the option favored by the single best cue is opposed by an option that wins on multiple lower-validity cues. Tallying will follow the simple majority of features, completely ignoring their validities, while TTB will follow the single highest-validity cue that discriminates, ignoring the number of cues. We also include trials where Tallying predicts a tie (equal number of wins for both options) while TTB predicts a strong preference based on the single highest-validity discriminating cue.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic on trials where Tallying and Take The Best (TTB) make strictly opposite predictions. By isolating the exact trials where the highest-validity discriminating cue opposes the sheer number of winning features, the metric will yield a value close to 1 if the data is generated by Tallying and close to 0 if generated by TTB, providing maximal discrimination between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie
            
        # TTB prediction (cues are ordered by descending validity based on the design)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is None:
            continue  # TTB predicts a tie
            
        # Only consider trials where the two theories make strictly opposite predictions
        if tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            count += 1
            
    if count == 0:
        return 0.5
        
    return float(tally_match / count)


### RESULTS
- Predicted under pi_1 (simulated): 0.1391 (var=0.0083)
- Predicted under pi_2 (simulated): 0.8691 (var=0.0085)
- Observed on real data: 0.2306 (var=0.0195)

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
  "interpretation": "Across both experiments, Theory 1 (Take The Best) significantly outperforms Theory 2 (Tallying). In Experiment 1, the observed alignment with TTB on conflict trials is 0.7858, which is very close to TTB's prediction (0.8546) and far from Tallying's (0.1412). Similarly, in Experiment 2, the alignment with Tallying on conflict trials is only 0.2306, matching TTB's prediction of low Tallying alignment (0.1391) and rejecting Tallying's prediction (0.8691). While Take The Best is clearly the dominant strategy, the observed data shows a slightly lower adherence to pure TTB than the simulated predictions, suggesting either higher response noise or that a minority of choices are compensatory. Tallying is fundamentally mismatched to the data. We should replace Tallying with a stronger competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) theory as a new competitor. Unlike Tallying (which ignores validities) and TTB (which is strictly non-compensatory), WADD computes a weighted sum of feature differences using the cue validities as weights. This allows for compensatory decision-making where multiple lower-validity cues might occasionally override a single high-validity cue, potentially capturing the slight deviations from pure TTB observed in the real data."
}
```

## Usage

```json
{
  "prompt_token_count": 4577,
  "candidates_token_count": 337,
  "total_token_count": 5543
}
```
