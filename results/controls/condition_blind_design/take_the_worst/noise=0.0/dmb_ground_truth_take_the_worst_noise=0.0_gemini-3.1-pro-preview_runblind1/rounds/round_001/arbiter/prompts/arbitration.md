# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Tallying with Recency/Right-Most Tie-Breaker: Decision makers primarily evaluate options by tallying the number of features where one option strictly dominates the other, ignoring feature validities (equal weighting). If one option has more winning features, it is chosen. However, if the feature wins are tied, individuals do not simply guess. Instead, they rely on a 'recency' or 'right-most' bias, breaking the tie based solely on the last feature in the array. Responses are subject to softmax noise over the resulting scores and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Tie-breaking mechanism: Recency/Right-most feature bias
    if a_wins == b_wins:
        if a[-1] > b[-1]:
            a_wins += 1.0
        elif b[-1] > a[-1]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.6, 0.87, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The critical divergence between the two theories occurs on trials where the number of strict feature-wise wins is perfectly tied, but the options differ on the last ('right-most') feature. The advocated theory asserts that decision-makers use this last feature to break the tie, predictably favoring the option with a 1 on the last feature. The competing theory asserts that decision-makers simply guess when the tallies are tied, ignoring the position of the features. This metric isolates exactly those tie-break trials and computes the proportion of choices that align with the last feature. The advocated theory predicts a value substantially above 0.5, whereas the competing theory predicts a value around 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def check_agreement(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins == b_wins and a[-1] != b[-1]:
            expected_resp = 1 if b[-1] > a[-1] else 0
            return 1.0 if row['response'] == expected_resp else 0.0
        return np.nan
        
    agreements = data.apply(check_agreement, axis=1)
    agreements = agreements.dropna()
    if len(agreements) == 0:
        return 0.5
    return float(agreements.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8667 (var=0.0244)
- Predicted under pi_2 (simulated): 0.5333 (var=0.0489)
- Observed on real data: 0.8200 (var=0.0420)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.57, 0.58, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The advocated theory (Tallying) predicts that when the number of winning features is tied, subjects guess randomly (50% chance for either option). The competing theory posits a right-most tie-breaker, meaning subjects will systematically choose the option that wins on the last feature when tallying is tied. By isolating trials where tallying is tied but the last feature is NOT tied, we can measure how often subjects choose the option favored by the last feature. The advocated theory expects this metric to be around 0.5, while the competing theory expects it to be significantly higher than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_target_trial(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        # We only care about trials where tallying results in a tie,
        # AND the last feature is not tied.
        return (a_wins == b_wins) and (a[-1] != b[-1])
    
    def choice_matches_last_feature(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        if a[-1] > b[-1]:
            return row['response'] == 0
        else:
            return row['response'] == 1
            
    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    matches = target_data.apply(choice_matches_last_feature, axis=1)
    return float(matches.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8656 (var=0.0111)
- Predicted under pi_2 (simulated): 0.5056 (var=0.0151)
- Observed on real data: 0.8178 (var=0.0285)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    target_trials_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tally wins
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # TTB winner (validities: 0.95, 0.72, 0.73, 0.55 -> cue order: 0, 2, 1, 3)
        ttb_winner = None
        for j in [0, 2, 1, 3]:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
        
        # Target trials: TTB has a strict preference, but Tallying ties
        if ttb_winner is not None and a_wins == b_wins:
            target_trials_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if target_trials_count == 0:
        return 0.5
        
    return ttb_match_count / target_trials_count
```

**Observed (real) value:** 0.2033 (var=0.0206)
**Predicted under pi_3:** 0.1642 (var=0.0128)
**Predicted under pi_2:** 0.5017 (var=0.0076)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 9 where Tallying and TTB make opposite strict predictions
    # Trial 9: A=[0, 1, 0, 1], B=[0, 0, 1, 0]
    is_trial_9 = (
        data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1)) &
        data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0))
    )
    trial_data = data[is_trial_9]
    if len(trial_data) == 0:
        return 0.5
    
    # Return the proportion of times the subject chose A (response == 0)
    return float(np.mean(trial_data['response'] == 0))
```

**Observed (real) value:** 0.8200 (var=0.0354)
**Predicted under pi_3:** 0.8833 (var=0.0258)
**Predicted under pi_2:** 0.8233 (var=0.0282)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_3) consistently captures the observed data across all four experiments. It correctly predicts that subjects primarily use tallying, but strongly favor the right-most feature when tallying results in a tie (Exp 1 and 2 observed ~0.82, pi_3 predicts ~0.86; pi_2 predicts 0.50). It also correctly predicts that when TTB and Tallying tie, subjects actively choose against the TTB prediction if the right-most feature dictates otherwise (Exp 3 observed 0.20, pi_3 predicts 0.16). Theory 2 (pi_2) is a pure tallying model that completely fails to capture the systematic tie-breaking behavior, consistently predicting 50% guessing on tied trials.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a 'Recency-Weighted Tallying' theory. Instead of a strict unweighted tally with a discontinuous right-most tie-breaker rule (as in Theory 1), this new theory should propose that subjects compute a weighted sum of feature wins, where the weights monotonically increase from left to right due to recency effects in working memory (e.g., later features are more salient). This provides a more elegant, continuous mechanism: it acts like tallying when the number of wins is unequal, but naturally favors the right-most feature when the number of wins is tied, potentially fitting the ~0.82 choice probabilities better than a rigid tie-breaker."
}
```

## Usage

```json
{
  "prompt_token_count": 5962,
  "candidates_token_count": 344,
  "total_token_count": 7515
}
```
