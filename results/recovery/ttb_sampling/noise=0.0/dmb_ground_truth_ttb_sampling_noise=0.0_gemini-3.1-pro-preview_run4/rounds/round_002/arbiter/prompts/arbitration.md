# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
People's decision making in multi-attribute choice is best described by a Strategy Mixture model. While decision-makers predominantly rely on the fast and frugal 'Take-The-Best' (TTB) heuristic—searching through cues in order of validity and stopping at the first discriminating cue—they occasionally switch to a compensatory 'Tallying' strategy (equal-weight counting of winning features). This mixture allows the model to capture both the high overall success of non-compensatory heuristics and the systematic, compensatory-like deviations observed in specific trial types where multiple lower-validity cues contradict a single high-validity cue. Response noise is incorporated as a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Strategy Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take-The-Best (TTB) Prediction
    order = np.argsort(-validities)
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # 2. Tallying Prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Blend strategies and add uniform lapse
    p_mix = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    p_core = (1.0 - p_mix) * p_ttb + p_mix * p_tally
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** This design tests the quantitative dissociation between a pure Take-The-Best (TTB) model and a Strategy Mixture model (which occasionally uses Tallying). By employing five features with descending validities, we construct critical trials where the highest-validity cue points to one option, but the majority of the remaining cues point to the other option. In these trials, pure TTB will consistently choose the option favored by the single best cue, whereas the Mixture model predicts systematic deviations (a lower choice probability for the TTB-favored option) due to the Tallying component favoring the option with more positive cues. Additional trials where both strategies agree, or where Tallying predicts a tie while TTB has a strict preference, allow precise estimation of the lapse rate versus the specific tallying mixing proportion.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By isolating trials where the Take-The-Best (TTB) and Tallying strategies disagree and comparing them to trials where they agree, we can effectively cancel out the baseline lapse rate (epsilon). For a pure TTB model, the deviation rate from the TTB prediction should be identical (driven purely by random guessing) across both trial types, yielding a difference close to 0. For the Strategy Mixture model, deviations from TTB will be significantly higher on disagree trials due to the Tallying component, yielding a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_choices(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    choices = data.apply(get_choices, axis=1)
    ttb_choice = choices['ttb']
    tally_choice = choices['tally']
    resp = data['response']
    
    disagree_mask = (ttb_choice != -1) & (tally_choice != -1) & (ttb_choice != tally_choice)
    agree_mask = (ttb_choice != -1) & (tally_choice != -1) & (ttb_choice == tally_choice)
    
    if disagree_mask.sum() == 0 or agree_mask.sum() == 0:
        return 0.0
        
    disagree_error_rate = (resp[disagree_mask] != ttb_choice[disagree_mask]).mean()
    agree_error_rate = (resp[agree_mask] != ttb_choice[agree_mask]).mean()
    
    return float(disagree_error_rate - agree_error_rate)

### RESULTS
- Predicted under pi_3 (simulated): 0.0044 (var=0.0067)
- Predicted under pi_4 (simulated): 0.0986 (var=0.0068)
- Observed on real data: -0.0300 (var=0.0162)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To isolate the Strategy Mixture model from pure TTB, we use five features to create trials where the most valid cue favors Option A, but a simple count of positive features (Tallying) favors Option B. Pure TTB views any deviations from the highest-validity cue solely as random noise (constant lapse rate epsilon). In contrast, the Strategy Mixture model predicts a specific, systematic increase in 'errors' on these conflict trials due to the p_tally parameter, while maintaining high accuracy on trials where TTB and Tallying agree. By including a spectrum of trials—full agreement, direct conflict, and partial ties—we can robustly distinguish whether deviations from TTB are purely random or driven by a compensatory Tallying process.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By calculating the difference in the rate at which subjects choose Option A between 'Agreement' trials (where both TTB and Tallying favor A) and 'Conflict' trials (where TTB favors A but Tallying favors B), we can discriminate the models. The pure TTB model predicts this difference to be near zero, because deviations are driven only by a uniform lapse rate. The Strategy Mixture model predicts a significantly positive difference, as the compensatory Tallying strategy pulls choices away from the TTB prediction specifically on Conflict trials.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Agreement trial: Both TTB and Tallying favor Option A
    t2 = data[(data['A_str'] == '11100') & (data['B_str'] == '00011')]
    
    # Conflict trials: TTB favors Option A, but Tallying favors Option B
    conflict_trials = data[
        ((data['A_str'] == '10000') & (data['B_str'] == '01111')) |
        ((data['A_str'] == '11000') & (data['B_str'] == '10111')) |
        ((data['A_str'] == '00100') & (data['B_str'] == '00011'))
    ]
    
    if len(t2) == 0 or len(conflict_trials) == 0:
        return 0.0
        
    p_A_agree = (t2['response'] == 0).mean()
    p_A_conflict = (conflict_trials['response'] == 0).mean()
    
    return float(p_A_agree - p_A_conflict)

### RESULTS
- Predicted under pi_3 (simulated): 0.0008 (var=0.0100)
- Predicted under pi_4 (simulated): 0.1038 (var=0.0044)
- Observed on real data: 0.0083 (var=0.0117)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    t1_match = (t1_data['response'] == 0).sum()
    t2_match = (t2_data['response'] == 1).sum()
    
    total = len(t1_data) + len(t2_data)
    
    if total == 0:
        return 0.5
    return float(t1_match + t2_match) / total

```

**Observed (real) value:** 0.1062 (var=0.0063)
**Predicted under pi_3:** 0.1212 (var=0.0080)
**Predicted under pi_4:** 0.1119 (var=0.0041)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_critical = 0
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        
        sum_a = sum(a)
        sum_b = sum(b)
        sum_a_top = sum(a[:3])
        sum_b_top = sum(b[:3])
        
        # Check if this is a critical trial: one option has exactly 2 high-validity features, 
        # while the other has exactly 3 low-validity features.
        if sum_a == 2 and sum_a_top == 2 and sum_b == 3 and sum_b_top == 0:
            total_critical += 1
            if row['response'] == 1:
                tally_choices += 1
        elif sum_b == 2 and sum_b_top == 2 and sum_a == 3 and sum_a_top == 0:
            total_critical += 1
            if row['response'] == 0:
                tally_choices += 1
                
    return tally_choices / total_critical if total_critical > 0 else 0.5
```

**Observed (real) value:** 0.2022 (var=0.0207)
**Predicted under pi_3:** 0.1183 (var=0.0055)
**Predicted under pi_4:** 0.1156 (var=0.0068)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
                
        if pred is not None:
            if resp == pred:
                matches += 1
            total += 1
            
    return matches / total if total > 0 else 0.5
```

**Observed (real) value:** 0.8579 (var=0.0072)
**Predicted under pi_3:** 0.8931 (var=0.0064)
**Predicted under pi_4:** 0.9137 (var=0.0031)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_more_than_a = data.apply(lambda row: sum(row['option_b_ratings']) > sum(row['option_a_ratings']), axis=1)
    if not b_more_than_a.any():
        return 0.0
    return float(data.loc[b_more_than_a, 'response'].mean())
```

**Observed (real) value:** 0.1383 (var=0.0137)
**Predicted under pi_3:** 0.1246 (var=0.0075)
**Predicted under pi_4:** 0.1196 (var=0.0043)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, we observe that the difference in error rates between trials where lower-validity cues conflict with the highest-validity cue versus when they agree is near zero (-0.0300 and 0.0083). This strongly contradicts the Strategy Mixture model (pi_4), which predicts a large positive difference (~10%) due to the compensatory Tallying component. Pure Take-The-Best (pi_3) successfully predicts this near-zero difference, implying subjects strictly ignore lower-validity cues once a discriminating cue is found. However, in Experiment 4, the observed deviation from TTB (20.2%) is higher than the uniform lapse rate predicted by pi_3 (~11.8%). This suggests that while subjects are using a non-compensatory TTB stopping rule, the noise process is not purely a uniform lapse at the decision stage, but rather might occur at the cue-evaluation stage.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Strategy Mixture theory with a 'Noisy-Perception Take-The-Best' theory. In this theory, decision-makers strictly follow the TTB heuristic (searching cues in descending order of validity and stopping at the first discriminating cue), but their evaluation of each feature is subject to independent perception noise. Specifically, there is a probability `p_error` that a feature's value is misread during the sequential search. This maintains the non-compensatory nature of the strategy (explaining the near-zero metrics in Exps 1 and 2) while allowing for elevated error rates on specific trials where multiple conflicting cues increase the cumulative likelihood of a perception error altering the decision (better capturing Exp 4)."
}
```

## Usage

```json
{
  "prompt_token_count": 6746,
  "candidates_token_count": 387,
  "total_token_count": 9115
}
```
