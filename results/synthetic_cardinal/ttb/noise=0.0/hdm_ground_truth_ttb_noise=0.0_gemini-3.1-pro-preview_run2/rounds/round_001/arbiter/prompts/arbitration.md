# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
Equal Weight Heuristic: People make decisions by summing the raw cardinal feature values for each option, completely ignoring the expert validities. The option with the highest total sum is chosen. This contrasts with Tallying (which ignores magnitudes) and WADD (which weights by validity).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal Weight expects a (2, n_features) stimulus.")
        
    # Sum the raw feature values for each option, ignoring validities
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 8]
  trial 3: A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  trial 4: A=[9, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 5: A=[5, 5, 0, 0]  B=[0, 0, 5, 6]
  trial 6: A=[6, 6, 0, 0]  B=[0, 0, 5, 5]
  trial 7: A=[0, 0, 0, 10]  B=[2, 2, 2, 2]
  trial 8: A=[2, 2, 2, 0]  B=[0, 0, 0, 9]

**Rationale:** This design aims to strongly dissociate Tallying (which only counts the number of winning features, ignoring magnitudes) from Equal Weight (which sums up raw cardinal values across all features, ignoring the number of wins). By setting rating_max to 10, we can create trials where one option wins on the majority of features by very small margins, while the other option wins on a single feature by a massive margin. Tallying will robustly prefer the option with more winning features, whereas Equal Weight will be swayed by the single large magnitude and choose the option with the higher total sum. We also include trials where Tallying predicts a tie (equal number of winning features), leading to guessing, but Equal Weight makes a deterministic prediction due to unequal sums.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject's choice aligns with the Tallying heuristic on 'discordant' trials—those where Tallying and Equal Weight make strictly opposing deterministic predictions. Under the Tallying theory, this proportion will be high (close to 1, minus noise/lapses), whereas under the Equal Weight theory, this proportion will be very low (close to 0), since responses would instead align with the Equal Weight predictions. This yields a massive and highly stable gap between the two theories.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_3 (simulated): 0.1489 (var=0.0087)
- Observed on real data: 0.7467 (var=0.0780)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  trial 2: A=[4, 4, 4, 0]  B=[3, 3, 3, 10]
  trial 3: A=[10, 8, 2, 1]  B=[0, 1, 3, 4]
  trial 4: A=[1, 1, 0, 0]  B=[0, 0, 10, 10]
  trial 5: A=[6, 6, 6, 0]  B=[5, 5, 5, 3]
  trial 6: A=[10, 2, 2, 2]  B=[1, 5, 5, 5]
  trial 7: A=[0, 0, 0, 10]  B=[1, 1, 1, 0]
  trial 8: A=[8, 8, 8, 0]  B=[7, 7, 7, 10]

**Rationale:** To dissociate Equal Weight (EQW) from Tallying, we exploit their core difference: EQW sums feature magnitudes, whereas Tallying counts the number of winning features while ignoring magnitudes. We use a high rating_max (10) to create large magnitude discrepancies. In some trials, one option wins on a single feature by a massive margin (dominating the EQW sum) while the other option wins on the remaining features by small margins (winning the Tallying count). We also include trials where Tallying predicts a tie (equal number of winning features) but EQW strongly prefers one option due to unequal sums, and trials where EQW predicts a tie (equal sums) but Tallying prefers one option (more winning features).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the deterministic prediction of the Equal Weight (EQW) heuristic, specifically on trials where EQW predicts a strict preference (i.e., the sums of magnitudes are not equal). Because the experimental design includes many trials where EQW and Tallying strongly disagree (e.g., one option has a massive single-feature advantage dominating the sum, while the other option wins by small margins on all other features), an EQW agent will score very high (near 1.0) on this metric, whereas a Tallying agent will systematically choose the opposite option and score very low (near 0.0).

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.1667 (var=0.0000)
- Predicted under pi_3 (simulated): 0.8767 (var=0.0074)
- Observed on real data: 0.3500 (var=0.0469)

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
**Predicted under pi_3:** 0.1189 (var=0.0057)

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
**Predicted under pi_3:** 0.7383 (var=0.0522)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Tallying) consistently captures the qualitative direction of the observed data, whereas Theory 2 (Equal Weight) completely fails. In Experiment 1 and 3, subjects predominantly chose the option with more winning features rather than the one with the higher cardinal sum. In Experiment 2, the metric measuring alignment with Equal Weight was very low (0.35), which is close to Tallying's prediction (0.16) and far from Equal Weight's prediction (0.87). In Experiment 4, the negative observed value again aligns with Tallying and strongly opposes Equal Weight. However, Tallying's predictions are slightly too deterministic (predicting 1.0 or -1.0) compared to the noisy observed data. Regardless, Theory 2 (Equal Weight) is clearly degenerate and unable to explain the behavior in this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory to replace Equal Weight. Since subjects are clearly sensitive to the number of winning features but both current theories ignore the explicitly provided expert validities, a logical next step is a theory that incorporates validities. Propose 'Take-The-Best' (TTB) or a 'Weighted Tallying' heuristic. TTB would search through features in descending order of their validity and stop at the first feature that discriminates between the options, ignoring all other features and cardinal magnitudes. This would provide a strong contrast to simple Tallying by predicting that subjects heavily overweight the most valid cue rather than treating all cues equally."
}
```

## Usage

```json
{
  "prompt_token_count": 4906,
  "candidates_token_count": 355,
  "total_token_count": 6273
}
```
