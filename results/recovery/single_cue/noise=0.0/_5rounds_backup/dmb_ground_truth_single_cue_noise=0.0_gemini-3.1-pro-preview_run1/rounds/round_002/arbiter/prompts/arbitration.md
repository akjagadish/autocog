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
Validity-Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying), ignoring the magnitudes of the differences and the exact cue validities. However, they are not completely blind to cue validities. Instead of integrating all validities in a compensatory manner (WADD) or strictly following a non-compensatory rule (Take The Best), they give a soft bonus weight to the single most predictive cue. This validity bias acts primarily as a tie-breaker when the tallies are close or equal, slightly skewing choice probabilities toward the option endorsed by the best cue, while still preserving the overall dominance of the Tallying heuristic. This explains why human choices overwhelmingly follow Tallying but show slight deviations toward the most valid cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Bonus for winning the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    score_a = a_wins + w * a_best_win
    score_b = b_wins + w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design specifically dissociates standard Tallying from Validity-Biased Tallying. Standard Tallying counts feature wins and predicts exactly 50/50 when both options win on the same number of features, treating all features equally. Validity-Biased Tallying behaves similarly but adds a fractional bonus to the option that wins on the single most valid cue. We include trials where options tie in the tally but differ on the most valid cue, forcing standard Tallying to guess while VBT predicts a preference. We also include trials where one option wins the tally by 1, but the most valid cue is won by either the tally-winner or the tally-loser. Standard Tallying predicts identical choice certainty across these latter trials, whereas VBT predicts higher certainty when the most valid cue aligns with the tally winner and lower certainty when it opposes.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times subjects choose the option that wins on the most valid cue (cue 0) specifically on trials where the simple tally of features is tied. Tallying treats all features equally and predicts random guessing (0.5) on these tie trials. Validity-Biased Tallying, however, gives a soft bonus weight to the most valid cue, predicting a choice probability strictly greater than 0.5 for the option endorsed by cue 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    ties = (a_wins == b_wins)
    
    if not np.any(ties):
        return 0.5
        
    responses = data['response'].values
    
    # Did the chosen option win on the most valid cue (index 0)?
    a_wins_cue0 = (a_ratings[:, 0] > b_ratings[:, 0])
    b_wins_cue0 = (b_ratings[:, 0] > a_ratings[:, 0])
    
    chosen_wins_cue0 = np.where(responses == 0, a_wins_cue0, b_wins_cue0)
    
    return float(np.mean(chosen_wins_cue0[ties]))

### RESULTS
- Predicted under pi_4 (simulated): 0.7719 (var=0.0310)
- Predicted under pi_2 (simulated): 0.5131 (var=0.0062)
- Observed on real data: 0.4750 (var=0.0075)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]

**Rationale:** To conclusively dissociate standard Tallying from Validity-Biased Tallying (VBT), this design systematically varies the tally difference (0, 1, or 2) and the ownership of the single most valid cue (Option A, Option B, or tied). The advocated theory (Tallying) predicts that choice probabilities depend purely on the tally difference, yielding exactly 50/50 guessing on all tally ties and uniform confidence across trials with the same tally difference. The competing theory (VBT) predicts a specific probability shift whenever the single most valid cue is not tied, breaking tally ties and modulating confidence on tally wins. Crucially, we include trials where the most valid cue is tied but other validities differ; VBT predicts these will behave identically to standard Tallying, allowing us to isolate the unique mechanism of VBT's top-cue bonus.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the overall tally is tied but the single most valid cue favors one option over the other (Trials 1 and 2 in the design). The standard Tallying model predicts exactly 50% guessing behavior on these trials, regardless of its noise parameters. Validity-Biased Tallying, however, predicts a systematic shift toward the option endorsed by the best cue, breaking the tie. By computing the proportion of choices that align with the best cue on these specific trials, we can cleanly discriminate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    alignments = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where the tally is tied, but the most valid cue (index 0) is not tied.
        if a_wins == b_wins and a[0] != b[0]:
            best_cue_choice = 0 if a[0] > b[0] else 1
            alignments.append(1 if resp == best_cue_choice else 0)
            
    if not alignments:
        return 0.5
    return float(np.mean(alignments))

### RESULTS
- Predicted under pi_4 (simulated): 0.8008 (var=0.0186)
- Predicted under pi_2 (simulated): 0.4900 (var=0.0052)
- Observed on real data: 0.1450 (var=0.0070)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        # Only consider trials where Tallying does not predict a tie
        if a_wins != b_wins:
            ttb_pred = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_pred = 0
                    break
                elif b[i] > a[i]:
                    ttb_pred = 1
                    break
            
            if ttb_pred is not None:
                if resp == ttb_pred:
                    ttb_matches += 1
                total += 1
                
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1194 (var=0.0110)
**Predicted under pi_4:** 0.1534 (var=0.0108)
**Predicted under pi_2:** 0.1291 (var=0.0066)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        if row['response'] == tally_pred:
            matches += 1
        total += 1
        
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.7608 (var=0.0062)
**Predicted under pi_4:** 0.8625 (var=0.0093)
**Predicted under pi_2:** 0.8690 (var=0.0080)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        # Trial 1: A has top 2 validities (0.9+0.8=1.7), B has bottom 3 (0.6+0.5+0.5=1.6)
        # WADD prefers A (0), Tallying prefers B (1) since B wins 3-2
        if a == (1, 1, 0, 0, 0):
            if resp == 0:
                wadd_match += 1
            count += 1
        # Trial 2: A has bottom 3, B has top 2
        # WADD prefers B (1), Tallying prefers A (0)
        elif a == (0, 0, 1, 1, 1):
            if resp == 1:
                wadd_match += 1
            count += 1
            
    if count == 0:
        return 0.5
    return float(wadd_match / count)
```

**Observed (real) value:** 0.1562 (var=0.0200)
**Predicted under pi_4:** 0.1706 (var=0.0123)
**Predicted under pi_2:** 0.1475 (var=0.0146)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying predicts B (3 wins vs 2) but WADD predicts A (1.8 vs 1.65)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[is_trial_1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1533 (var=0.0127)
**Predicted under pi_4:** 0.1933 (var=0.0203)
**Predicted under pi_2:** 0.1467 (var=0.0111)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 2 (Pure Tallying) consistently outperforms Theory 1 (Validity-Biased Tallying). Theory 1 posits that subjects use the single most valid cue as a tie-breaker, which leads to predictions of ~0.77-0.80 for the metric in Experiments 1 and 2. However, the observed data show no such preference: in Experiment 1, subjects guess (~0.475) on tally ties, and in Experiment 2, they actively choose against the top cue (~0.145). Furthermore, in Experiments 3-6, Pure Tallying accurately predicts the strong tendency of subjects to simply count feature wins, largely ignoring cue validities. Theory 1 is thus fundamentally degenerate because its core secondary mechanism (top-cue bias) contradicts the observed data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that retains the core feature-counting mechanism of Tallying (which strongly drives behavior in Exps 3-6) but introduces a different tie-breaking or weighting mechanism to explain the severe drop to 0.145 in Experiment 2. Instead of a top-cue bonus, consider a mechanism where subjects might penalize options that rely too heavily on a single extreme cue, or a 'moderate-cue' bias where a cluster of medium-validity features is preferred over one high-validity feature when tallies are tied. The new theory must preserve the baseline 50/50 guessing on symmetric ties (Exp 1) while capturing the systematic anti-top-cue preference in Exp 2."
}
```

## Usage

```json
{
  "prompt_token_count": 6338,
  "candidates_token_count": 366,
  "total_token_count": 8131
}
```
