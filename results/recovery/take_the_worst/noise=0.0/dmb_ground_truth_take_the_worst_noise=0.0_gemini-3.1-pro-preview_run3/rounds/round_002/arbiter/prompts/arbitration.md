# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Selection (Mixture of Deterministic Tallying and Probabilistic WADD): Decision-makers probabilistically select between a frugal, unweighted strategy (Tallying) and a fully compensatory, validity-weighted strategy (WADD). Critically, Tallying operates as a deterministic rule (choosing the option with more winning cues, or guessing on ties) rather than a probabilistic score-based process. This breaks the assumption that Tallying consistency scales with the absolute difference in cue counts, allowing the model to capture high consistency in scenarios with small cue count differences (e.g., Exp 4) and lower consistency in scenarios with large cue count differences (e.g., Exp 2). The WADD strategy remains probabilistic and tempers the extremeness of the Tallying predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # 1. Deterministic Tallying strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # 2. WADD strategy scores
    val = np.asarray(parameters["validities"], dtype=float)
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores_wadd = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_tally = float(parameters["w_tally"])
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of the two strategies
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [1.0, 0.5, 0.5, 0.6, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]

**Rationale:** This design tests a stark quantitative dissociation between the Advocated Theory (Mixture of Deterministic Tallying + Probabilistic WADD) and the Competing Theory (Probabilistic Tallying). By setting validities to [1.0, 0.5, 0.5, 0.6, 0.6], we create paired trials where the WADD score difference between options is held exactly constant, but the Tallying cue-count difference varies (e.g., a difference of 1 vs. 3). Under the Competing Theory, choice probability scales continuously with the cue-count difference, predicting a much stronger preference in the 3-difference trial than the 1-difference trial. Under the Advocated Theory, Tallying is deterministic (outputting 1.0 for the winner as long as the difference is > 0), and the WADD component is identical across both trials; thus, it predicts the exact same choice probability for both trials. We also include Tally-tie trials where WADD breaks the tie, which Probabilistic Tallying cannot capture.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option favored by the WADD (Weighted Additive) strategy on trials where the Tallying strategy results in a tie. Under the Competing Theory (Probabilistic Tallying), a tie in cue counts leads to exactly a 50% chance of choosing either option, regardless of the validities. Under the Advocated Theory, the deterministic Tallying component guesses on a tie, but the probabilistic WADD component breaks the tie in favor of the higher-validity option. Thus, the Advocated Theory predicts a choice proportion strictly greater than 0.5 for the WADD winner, whereas the Competing Theory firmly predicts 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.5, 0.5, 0.6, 0.6])
    wadd_winner_chosen = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus only on trials where Tallying results in a tie
        if a_wins == b_wins:
            wadd_a = np.dot(a, validities)
            wadd_b = np.dot(b, validities)
            
            if wadd_a > wadd_b:
                wadd_winner_chosen.append(1 if row['response'] == 0 else 0)
            elif wadd_b > wadd_a:
                wadd_winner_chosen.append(1 if row['response'] == 1 else 0)
                
    if len(wadd_winner_chosen) == 0:
        return 0.5
        
    return float(np.mean(wadd_winner_chosen))

### RESULTS
- Predicted under pi_4 (simulated): 0.5606 (var=0.0098)
- Predicted under pi_2 (simulated): 0.4838 (var=0.0079)
- Observed on real data: 0.1375 (var=0.0095)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Probabilistic Tallying) from the Competing Theory (Mixture of Deterministic Tallying + Probabilistic WADD), this design holds the Tallying score difference constant across multiple trials while varying the WADD score difference. Under the Advocated Theory, choice probabilities are entirely determined by the difference in the number of winning cues (the Tally difference). Thus, any two trials where Option A wins on 2 cues and Option B wins on 1 cue will yield the exact same choice probability. Under the Competing Theory, Tallying acts as a deterministic baseline, but the final choice probability is heavily modulated by the WADD score difference. By constructing pairs that all have a Tally difference of +1 for Option A but widely varying WADD differences (e.g., +1.0 vs +0.2), the Advocated Theory predicts a flat consistency curve across these pairs, whereas the Competing Theory predicts varying choice probabilities. We also include 'Tally tie' trials where the Advocated Theory predicts exactly 50% guessing, but the Competing Theory predicts a preference driven by WADD.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the Tallying heuristic results in a tie (Option A and Option B win on the same number of features). Under the Advocated Theory (Probabilistic Tallying), subjects have no basis for preference and must guess, meaning the probability of choosing Option A is exactly 0.5. However, in the specific experimental design, Option A always holds the most valid cue (0.9) on these tie trials, giving it a higher Weighted Additive (WADD) score than Option B. Therefore, the Competing Theory (Strategy Selection), which incorporates probabilistic WADD, predicts a systematic preference for Option A (a metric value > 0.5). By computing the simple mean of choosing A on these tie trials, we get a highly stable metric with low variance that sharply discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ties_chose_a = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying produces a tie.
        if a_wins == b_wins:
            ties_chose_a.append(1 if row['response'] == 0 else 0)
            
    if not ties_chose_a:
        return 0.5
        
    return float(np.mean(ties_chose_a))

### RESULTS
- Predicted under pi_4 (simulated): 0.5619 (var=0.0082)
- Predicted under pi_2 (simulated): 0.4894 (var=0.0071)
- Observed on real data: 0.1512 (var=0.0107)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Find the highest validity feature where A and B differ.
    # Since features are ordered by descending validity, this is the first non-zero difference.
    diff = a_mat - b_mat
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B on the discriminating cue, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Return the proportion of choices that align with TTB's prediction
    return float(np.mean(ttb_pred == resp))
```

**Observed (real) value:** 0.2512 (var=0.0077)
**Predicted under pi_4:** 0.2515 (var=0.0045)
**Predicted under pi_2:** 0.2373 (var=0.0059)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_trials = a_wins != b_wins
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    match = (tally_pred[valid_trials] == responses[valid_trials])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6089 (var=0.0019)
**Predicted under pi_4:** 0.8489 (var=0.0081)
**Predicted under pi_2:** 0.8428 (var=0.0102)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    mask3 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    mask4 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_choices = 0
    total = 0
    
    if mask3.any():
        wadd_choices += (data.loc[mask3, 'response'] == 0).sum()
        total += mask3.sum()
        
    if mask4.any():
        wadd_choices += (data.loc[mask4, 'response'] == 1).sum()
        total += mask4.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_choices / total)
```

**Observed (real) value:** 0.1500 (var=0.0147)
**Predicted under pi_4:** 0.2658 (var=0.0142)
**Predicted under pi_2:** 0.1400 (var=0.0151)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0]
    # Tallying prefers A (3 features vs 2), WADD prefers B (val 1.65 vs 1.85)
    t1_mask = (a_tuples == (0, 0, 1, 1, 1))
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1]
    # Tallying prefers B, WADD prefers A
    t2_mask = (a_tuples == (1, 1, 0, 0, 0))
    
    t1_responses = data.loc[t1_mask, 'response']
    t2_responses = data.loc[t2_mask, 'response']
    
    val = 0.0
    count = 0
    if len(t1_responses) > 0:
        val += (t1_responses == 0).mean()
        count += 1
    if len(t2_responses) > 0:
        val += (t2_responses == 1).mean()
        count += 1
        
    return float(val / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.8583 (var=0.0155)
**Predicted under pi_4:** 0.7696 (var=0.0136)
**Predicted under pi_2:** 0.8671 (var=0.0080)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Tallying appears to be the dominant choice rule (as seen in Exp 6 where Tallying beats WADD 85% to 15%, and Exp 5 where WADD choices are only 15%). However, both theories fail dramatically on Tallying ties. Theory 1 (pi_4) posits that ties are broken by WADD, predicting a preference for the WADD winner (>0.5). Theory 2 (pi_2) posits random guessing on ties (0.5). The observed data in Exp 1 and Exp 2 shows a strong, systematic preference (~0.14-0.15) for the WADD *loser* on tie trials. Similarly, Exp 3 shows that TTB is only followed ~25% of the time, meaning subjects systematically choose the option with the lower-validity cues when cue counts are matched or when TTB is isolated. Neither theory can explain this 'anti-WADD' or 'anti-TTB' tie-breaking behavior.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory centered around 'Tallying with a Sub-optimal Tie-Breaker' or 'Negative Cue-Weighting in Ties'. The new theory should maintain deterministic or probabilistic Tallying as the primary driver of choice (to capture the ~85% success in Exp 5 and 6), but introduce a novel tie-breaking mechanism that systematically favors options with lower-ranked or lower-validity cues when cue counts are tied. This will replace the flawed Mixture theory (Theory 1) which incorrectly assumes WADD breaks ties in the positive direction."
}
```

## Usage

```json
{
  "prompt_token_count": 6996,
  "candidates_token_count": 376,
  "total_token_count": 9520
}
```
