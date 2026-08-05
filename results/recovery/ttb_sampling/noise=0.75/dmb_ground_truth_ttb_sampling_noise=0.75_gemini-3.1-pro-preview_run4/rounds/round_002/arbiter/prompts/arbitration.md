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
Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Strategy Selection (Mixed Heuristic) with Linear Tallying: Decision makers probabilistically select between a non-compensatory lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). To prevent extreme over-sensitivity when feature differences are large, the compensatory Tallying component uses a simple proportional (linear) rule rather than an exponential softmax.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(validities)[::-1]
    chosen_ttb = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen_ttb = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen_ttb = 1
            break
            
    if chosen_ttb == 0:
        p_ttb_core = np.array([1.0, 0.0])
    elif chosen_ttb == 1:
        p_ttb_core = np.array([0.0, 1.0])
    else:
        p_ttb_core = np.array([0.5, 0.5])
        
    # Tallying prediction (linear proportional rule)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    scores = np.array([a_wins, b_wins])
    
    if np.sum(scores) > 0:
        p_tally_core = scores / np.sum(scores)
    else:
        p_tally_core = np.array([0.5, 0.5])
    
    # Strategy mixture
    p_ttb = float(parameters["p_ttb"])
    p_mixed = p_ttb * p_ttb_core + (1.0 - p_ttb) * p_tally_core
    
    # Response noise (lapse rate)
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To dissociate pure Take The Best (TTB) from a Strategy Selection model that mixes TTB with Tallying, we parametrically vary the number of lower-validity cues that oppose the single highest-validity discriminating cue. For a pure TTB user, the choice probability for the option favored by the most valid cue remains constant across these trials, as TTB completely ignores all subsequent cues. In contrast, the Strategy Selection model predicts that the probability of choosing the TTB-favored option will monotonically decrease as the number of opposing cues increases, because the linear Tallying component becomes increasingly biased toward the option with more winning features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Under pure Take The Best (TTB), the decision maker relies entirely on the most valid cue (which is always index 0 in this design). Thus, the probability of choosing the TTB-favored option should be constant regardless of how many lower-validity cues contradict it. Under the Strategy Selection (Mixed) model, the Tallying component comes into play. Tallying simply counts the winning cues for each option, meaning that as the number of opposing cues increases (from 1 to 4), the probability of choosing the TTB-favored option monotonically decreases. This metric computes the difference in the rate of choosing the TTB-favored option between trials with 1 opposing cue and trials with 4 opposing cues. TTB predicts this difference to be near zero, while the Mixed model predicts a robustly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_favored = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    ttb_chosen = ((a_favored & (data['response'] == 0)) | (~a_favored & (data['response'] == 1))).astype(float)
    
    opposing_cues = data.apply(lambda row: sum(row['option_b_ratings']) if row['option_a_ratings'][0] == 1 else sum(row['option_a_ratings']), axis=1)
    
    mask_1 = opposing_cues == 1
    mask_4 = opposing_cues == 4
    
    if mask_1.sum() == 0 or mask_4.sum() == 0:
        return 0.0
        
    return float(ttb_chosen[mask_1].mean() - ttb_chosen[mask_4].mean())

### RESULTS
- Predicted under pi_3 (simulated): -0.0033 (var=0.0108)
- Predicted under pi_4 (simulated): 0.1142 (var=0.0176)
- Observed on real data: -0.0017 (var=0.0224)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate pure Take The Best (TTB) from the Strategy Selection (Mixed) model, we parametrically vary the number of lower-validity cues that *support* the single highest-validity discriminating cue. Across a series of trials, one option always wins on the most valid cue (ensuring pure TTB deterministically chooses it with a constant probability, as TTB ignores all subsequent cues). However, we parametrically increase the number of additional, lower-validity cues that also favor this chosen option, while keeping the number of opposing cues constant at one. Pure TTB predicts a flat response curve across these trials because the first cue is decisive. In contrast, the Mixed model predicts a monotonically increasing probability of choosing the TTB-favored option, because its linear Tallying component becomes increasingly confident as the proportion of winning features grows.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Mixed model (Strategy Selection with Tallying) predicts that adding supporting cues will incrementally increase the probability of choosing the TTB-favored option due to its proportional Tallying component. Pure TTB predicts a flat response curve across these trials because only the first, most valid cue matters. This metric measures the difference in the probability of choosing the TTB-favored option between trials with 4 supporting cues and trials with 1 supporting cue. The difference will be substantially positive under the Mixed model and near zero under pure TTB.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    supports = []
    chose_ttbs = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # The most valid cue is always the first one (index 0)
        if a[0] > b[0]:
            ttb_fav = 0
            sup = sum(a)
        else:
            ttb_fav = 1
            sup = sum(b)
            
        supports.append(sup)
        chose_ttbs.append(1 if row['response'] == ttb_fav else 0)
        
    df = pd.DataFrame({'support': supports, 'chose_ttb': chose_ttbs})
    
    m4 = df[df['support'] == 4]['chose_ttb'].mean()
    m1 = df[df['support'] == 1]['chose_ttb'].mean()
    
    if pd.isna(m4) or pd.isna(m1):
        return 0.0
        
    return float(m4 - m1)


### RESULTS
- Predicted under pi_3 (simulated): -0.0150 (var=0.0147)
- Predicted under pi_4 (simulated): 0.1367 (var=0.0132)
- Observed on real data: -0.0217 (var=0.0199)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Calculate the sum of features for options A and B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Isolate trials 1 and 2 where Tallying and WADD make opposite predictions.
    # These are uniquely identified by the total number of features across both options being 5.
    mask = (a_sums + b_sums) == 5
    if not mask.any():
        return 0.5
    
    # Tallying predicts the option with more features (lower sum means response 1, i.e., B)
    tally_predictions = (a_sums < b_sums).astype(int)
    
    # Calculate the proportion of choices on these trials that align with Tallying
    matches = (data.loc[mask, 'response'] == tally_predictions.loc[mask]).mean()
    return float(matches)

```

**Observed (real) value:** 0.3883 (var=0.0126)
**Predicted under pi_3:** 0.3117 (var=0.0269)
**Predicted under pi_4:** 0.3942 (var=0.0252)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    mask = (a_str == '11000') & (b_str == '00111')
    if not mask.any():
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5633 (var=0.0246)
**Predicted under pi_3:** 0.7400 (var=0.0299)
**Predicted under pi_4:** 0.6233 (var=0.0392)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5
        
    preds = [ttb_predict(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    resps = data['response'].values
    
    agreements = []
    for p, r in zip(preds, resps):
        if p == 0.5:
            agreements.append(0.5)
        else:
            agreements.append(1.0 if p == r else 0.0)
            
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5971 (var=0.0020)
**Predicted under pi_3:** 0.7942 (var=0.0242)
**Predicted under pi_4:** 0.6246 (var=0.0237)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    diff = b_sum - a_sum
    
    high_diff = data[diff >= 2]['response']
    low_diff = data[diff <= 0]['response']
    
    if len(high_diff) == 0 or len(low_diff) == 0:
        return 0.0
        
    return float(high_diff.mean() - low_diff.mean())
```

**Observed (real) value:** -0.0133 (var=0.0090)
**Predicted under pi_3:** -0.0025 (var=0.0123)
**Predicted under pi_4:** 0.0988 (var=0.0186)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, we see a striking pattern: Experiments 1, 2, and 6 show that varying the number of lower-validity opposing or supporting cues has virtually no effect on choice probabilities. The data yields near-zero differences, perfectly matching the non-compensatory predictions of pure Take The Best (THEORY 1) and strongly rejecting the Strategy Selection/Tallying model (THEORY 2), which predicts a monotonic slope. However, Experiments 3, 4, and 5 show that the overall agreement with the objective TTB prediction is quite low (around 60%), much lower than THEORY 1 predicts. THEORY 2 only fits these latter experiments better because its Tallying mixture artificially drags down the TTB agreement. To capture both phenomena\u2014strictly non-compensatory integration but low objective accuracy\u2014we need a theory where subjects use a non-compensatory heuristic but often rely on the 'wrong' cue due to uncertainty about the validities.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace THEORY 2 with 'Subjective Take The Best' (or 'Noisy-Validity TTB'). In this theory, decision makers strictly follow the non-compensatory Take The Best heuristic, but their internal ranking of cue validities is noisy (e.g., Gaussian noise added to the objective validities before sorting, or sampling the primary cue via a softmax over validities). This ensures that once a cue is selected as the 'best', all other cues are ignored (capturing the flat curves in Exps 1, 2, and 6), but because subjects frequently select a sub-optimal cue as their highest-ranked feature, the overall agreement with the objective TTB predictions is suppressed (capturing the lower performance in Exps 3, 4, and 5)."
}
```

## Usage

```json
{
  "prompt_token_count": 6455,
  "candidates_token_count": 402,
  "total_token_count": 8309
}
```
