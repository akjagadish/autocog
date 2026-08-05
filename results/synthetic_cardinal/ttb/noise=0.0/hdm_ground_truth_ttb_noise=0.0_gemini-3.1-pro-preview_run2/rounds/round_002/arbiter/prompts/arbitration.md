# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

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
    return int(np.argmax(probabilities))


## THEORY 2 — pi_4
Weighted Tallying Heuristic: People make decisions by integrating across multiple features, but rather than treating all cues equally (as in simple Tallying), they weight each feature-wise win by the explicitly provided expert validity. The score for each option is the sum of the validities for the features where that option has a strictly higher rating than the other. The option with the highest validity-weighted tally is chosen. This captures both the tendency to integrate across features and the sensitivity to explicit validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Weighted Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sum validities for features where an option strictly wins
    score_a = np.sum(validities[a > b])
    score_b = np.sum(validities[b > a])
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.5, 0.5, 0.5]

**rating_max:** 3

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 2, 2, 2]  B=[2, 2, 0, 0, 0]
  trial 2: A=[2, 2, 0, 0, 0]  B=[0, 0, 2, 2, 2]
  trial 3: A=[2, 0, 2, 0, 0]  B=[0, 2, 0, 2, 0]
  trial 4: A=[0, 2, 0, 2, 0]  B=[2, 0, 2, 0, 0]
  trial 5: A=[2, 0, 0, 0, 0]  B=[0, 0, 2, 2, 0]
  trial 6: A=[0, 0, 2, 2, 0]  B=[2, 0, 0, 0, 0]
  trial 7: A=[2, 0, 2, 2, 0]  B=[0, 2, 0, 0, 2]
  trial 8: A=[0, 2, 0, 0, 2]  B=[2, 0, 2, 2, 0]

**Rationale:** This design dissociates the advocated Tallying heuristic (which counts the number of winning features irrespective of their validities) from the competing Weighted Tallying heuristic (which sums the validities of the winning features). By using a set of validities with a steep drop-off ([0.95, 0.85, 0.5, 0.5, 0.5]), we can construct trials where one option wins on a larger number of low-validity features while the other option wins on fewer high-validity features. In these cases, Tallying will prefer the option with more winning features, whereas Weighted Tallying will prefer the option with the higher sum of validities. We also include trials where Tallying predicts a tie (equal number of winning features) but Weighted Tallying makes a deterministic prediction due to unequal validity sums. A rating_max of 3 is used to ensure the features are cardinal and to provide variation, though the core dissociation relies purely on the number of wins vs. validity sums.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the Tallying heuristic on the specific trials designed to strongly dissociate the two models. In these trials, one option wins on three low-validity features (0.5 each) while the other option wins on two high-validity features (0.95 and 0.85). Tallying strictly counts feature wins and prefers the option with three wins, whereas Weighted Tallying sums the validities and prefers the option with two high-validity wins. Tallying will score close to 1, while Weighted Tallying will score close to 0.

Source:
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 2, 2, 2) and b == (2, 2, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        elif a == (2, 2, 0, 0, 0) and b == (0, 0, 2, 2, 2):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_4 (simulated): 0.1858 (var=0.0200)
- Observed on real data: 0.7600 (var=0.1824)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [1.0, 0.8, 0.6, 0.5, 0.5]

**rating_max:** 1

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design dissociates the advocated Weighted Tallying heuristic from the competing Tallying heuristic. Tallying counts the number of winning features while ignoring validities, whereas Weighted Tallying sums the validities of the winning features. By using binary features (rating_max = 1), we remove magnitude information so that both heuristics rely purely on wins and losses. We use a validity set of [1.0, 0.8, 0.6, 0.5, 0.5] to create specific conflicts: trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by Weighted Tallying); trials where the number of winning features is tied but validity sums differ (Tallying guesses, Weighted Tallying has a strong preference); and trials where validity sums are tied but the number of winning features differ (Weighted Tallying guesses, Tallying has a strong preference).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'strict conflict' trials where the Weighted Tallying heuristic and the Tallying heuristic predict opposite choices (i.e., one option wins on the simple count of features, but the other wins when features are weighted by their validities). It calculates the proportion of choices on these specific trials that align with the Weighted Tallying prediction. Data generated by Weighted Tallying will yield values close to 1.0 (subject to noise), whereas data generated by Tallying will yield values close to 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.8, 0.6, 0.5, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    wt_a = np.sum(validities * (A > B), axis=1)
    wt_b = np.sum(validities * (B > A), axis=1)
    wt_sign = np.sign(wt_a - wt_b)
    
    tally_a = np.sum(A > B, axis=1)
    tally_b = np.sum(B > A, axis=1)
    tally_sign = np.sign(tally_a - tally_b)
    
    conflict_mask = (wt_sign != 0) & (tally_sign != 0) & (wt_sign != tally_sign)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    chosen_sign = np.where(responses == 0, 1, -1)
    
    match = (chosen_sign[conflict_mask] == wt_sign[conflict_mask])
    return float(np.mean(match))

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_4 (simulated): 0.7342 (var=0.0269)
- Observed on real data: 0.5000 (var=0.2500)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[2, 8, 8, 8]  B=[9, 7, 7, 7]
  A=[10, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[9, 3, 3, 3]  B=[2, 4, 4, 4]
  A=[6, 6, 6, 2]  B=[5, 5, 5, 9]
  A=[3, 3, 3, 10]  B=[4, 4, 4, 1]
  A=[8, 8, 2, 2]  B=[7, 7, 4, 4]
  A=[2, 2, 8, 8]  B=[4, 4, 7, 7]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    mask = tally_a != tally_b
    if not np.any(mask):
        return 0.5
        
    preds = np.where(tally_a > tally_b, 0, 1)
    responses = data['response'].values
    
    accuracy = np.mean(preds[mask] == responses[mask])
    return float(accuracy)

```

**Observed (real) value:** 0.8000 (var=0.0667)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_4:** 0.8631 (var=0.0088)

### Experiment 4
**Design**
  A=[0, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[1, 5, 5, 5]  B=[5, 4, 4, 4]
  A=[5, 4, 1, 1]  B=[1, 1, 5, 5]
  A=[5, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[4, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[1, 2, 3, 5]  B=[2, 3, 4, 0]
  A=[5, 1, 1, 1]  B=[1, 2, 2, 2]
  A=[2, 5, 1, 1]  B=[3, 1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 4 (A=[5,0,0,0], sum=5) and trial 1 (A=[0,4,4,4], sum=12)
    a_sum = data['option_a_ratings'].apply(sum)
    chose_a = (data['response'] == 0).astype(float)
    
    mask_t4 = (a_sum == 5)
    mask_t1 = (a_sum == 12)
    
    p_a_t4 = chose_a[mask_t4].mean() if mask_t4.any() else 0.5
    p_a_t1 = chose_a[mask_t1].mean() if mask_t1.any() else 0.5
    
    return float(p_a_t4 - p_a_t1)
```

**Observed (real) value:** -0.4000 (var=0.8400)
**Predicted under pi_1:** -1.0000 (var=0.0000)
**Predicted under pi_4:** -0.7633 (var=0.0345)

### Experiment 5
**Design**
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 8]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[9, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[5, 5, 0, 0]  B=[0, 0, 5, 6]
  A=[6, 6, 0, 0]  B=[0, 0, 5, 5]
  A=[0, 0, 0, 10]  B=[2, 2, 2, 2]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(A > B, axis=1)
    b_wins = np.sum(B > A, axis=1)
    
    a_sum = np.sum(A, axis=1)
    b_sum = np.sum(B, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    ew_pref = np.where(a_sum > b_sum, 0, np.where(b_sum > a_sum, 1, -1))
    
    discordant = (tally_pref != -1) & (ew_pref != -1) & (tally_pref != ew_pref)
    
    if not np.any(discordant):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[discordant] == tally_pref[discordant])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7467 (var=0.0780)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_4:** 0.8867 (var=0.0066)

### Experiment 6
**Design**
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[4, 4, 4, 0]  B=[3, 3, 3, 10]
  A=[10, 8, 2, 1]  B=[0, 1, 3, 4]
  A=[1, 1, 0, 0]  B=[0, 0, 10, 10]
  A=[6, 6, 6, 0]  B=[5, 5, 5, 3]
  A=[10, 2, 2, 2]  B=[1, 5, 5, 5]
  A=[0, 0, 0, 10]  B=[1, 1, 1, 0]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filtered = sum_a[mask]
    sum_b_filtered = sum_b[mask]
    
    eqw_pred = (sum_a_filtered < sum_b_filtered).astype(int)
    matches = (filtered_data['response'] == eqw_pred).mean()
    
    return float(matches)
```

**Observed (real) value:** 0.3500 (var=0.0469)
**Predicted under pi_1:** 0.1667 (var=0.0000)
**Predicted under pi_4:** 0.2539 (var=0.0038)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither pure Tallying (Theory 1) nor pure Weighted Tallying (Theory 2) perfectly captures the observed data. In Experiment 1, subjects strongly favored the Tallying prediction (0.76 observed) over the Weighted Tallying prediction, heavily penalizing Theory 2. However, in Experiment 2, choices were split 50/50 between Tallying and Weighted Tallying predictions. Furthermore, in Experiment 5, subjects favored Tallying over Equal Weight 75% of the time, showing that magnitudes are largely ignored in favor of win/loss counts. Theory 1 (pure Tallying) is too rigid, predicting extreme 1.0 or 0.0 values across all experiments and completely failing to capture the 50/50 split in Exp 2. Rather than just tweaking noise parameters, the field needs a theory that systematically bridges these behaviors.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Rank-Weighted Tallying' or 'Mixture of Tallying and Weighted Tallying' theory. Instead of using the exact numerical validities (which Weighted Tallying does) or completely ignoring them (which Tallying does), subjects might convert explicit validities into ordinal ranks and use these rank-based weights, or the population consists of a mixture of pure-Talliers and Weighted-Talliers. This new theory should replace pure Tallying to better capture the intermediate consensus levels observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 6481,
  "candidates_token_count": 343,
  "total_token_count": 8768
}
```
