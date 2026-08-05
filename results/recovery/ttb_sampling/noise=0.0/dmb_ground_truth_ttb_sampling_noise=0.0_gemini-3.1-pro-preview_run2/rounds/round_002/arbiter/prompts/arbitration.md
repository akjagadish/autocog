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
Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Strategy Selection: Decision makers probabilistically switch between a lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). On any given trial, an individual predominantly uses TTB (with high probability p_ttb) but occasionally lapses into Tallying, reflecting a mixture of non-compensatory and compensatory decision strategies driven by individual differences or cognitive load.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(-validities, kind='stable')
    p_ttb_choice = np.array([0.5, 0.5])
    for i in order:
        if a[i] > b[i]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of strategies
    p_core = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. While pure TTB relies exclusively on the highest-validity discriminating feature, the Strategy Selection model predicts occasional lapses into Tallying. By keeping the TTB prediction constant (e.g., Option A always favored by the highest validity feature) but varying the number of lower-validity features favoring Option B, we manipulate the Tallying prediction. Pure TTB predicts equal choice probability for Option A across these trials. In contrast, the Strategy Selection model predicts a drop in Option A choice probability specifically on trials where Tallying strongly favors Option B, creating a clear quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic in 'conflict trials' (where TTB and Tallying disagree) and subtracts the baseline error rate derived from 'agreement trials' (where TTB and Tallying agree). By subtracting the baseline error rate, we control for the uniform lapse parameter (epsilon) that could otherwise masquerade as Tallying choices. Pure TTB will yield an expected value of 0, whereas the Strategy Selection model, which includes a genuine probability of lapsing into Tallying, will yield a strictly positive expected value.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Agreement trials (where TTB and Tallying agree)
    # Trial 2: A='11100', B='00010' -> Both choose A (response 0)
    # Trial 5: A='00010', B='11100' -> Both choose B (response 1)
    t2_mask = (a_str == '11100') & (b_str == '00010')
    t5_mask = (a_str == '00010') & (b_str == '11100')
    
    errors = 0
    total_agree = 0
    if t2_mask.sum() > 0:
        errors += (data.loc[t2_mask, 'response'] == 1).sum()
        total_agree += t2_mask.sum()
    if t5_mask.sum() > 0:
        errors += (data.loc[t5_mask, 'response'] == 0).sum()
        total_agree += t5_mask.sum()
        
    e_agree = errors / total_agree if total_agree > 0 else 0
    
    # Conflict trials (where TTB and Tallying completely disagree)
    # Trial 1: A='10000', B='01110' -> TTB chooses A (0), Tallying chooses B (1)
    # Trial 4: A='01110', B='10000' -> TTB chooses B (1), Tallying chooses A (0)
    t1_mask = (a_str == '10000') & (b_str == '01110')
    t4_mask = (a_str == '01110') & (b_str == '10000')
    
    tally_choices = 0
    total_conflict = 0
    if t1_mask.sum() > 0:
        tally_choices += (data.loc[t1_mask, 'response'] == 1).sum()
        total_conflict += t1_mask.sum()
    if t4_mask.sum() > 0:
        tally_choices += (data.loc[t4_mask, 'response'] == 0).sum()
        total_conflict += t4_mask.sum()
        
    p_tally = tally_choices / total_conflict if total_conflict > 0 else 0
    
    # Subtracting agreement error rate controls for the lapse rate (epsilon)
    return float(p_tally - e_agree)

### RESULTS
- Predicted under pi_3 (simulated): 0.0256 (var=0.0083)
- Predicted under pi_4 (simulated): 0.1125 (var=0.0126)
- Observed on real data: -0.0062 (var=0.0059)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. Pure TTB relies exclusively on the highest-validity discriminating feature and predicts a constant choice probability for the favored option regardless of how many lower-validity features favor the alternative. In contrast, the Strategy Selection model assumes occasional lapses into Tallying. By keeping the TTB prediction constant but varying the Tallying prediction (Agreement, Conflict, or Tie), we can test whether the choice probability for the TTB-favored option drops systematically when Tallying favors the alternative. The trials vary the highest discriminating cue to ensure the effect is robust across different cue validities.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in adherence to the Take The Best (TTB) prediction between 'Agreement' trials (where TTB and Tallying predict the same option) and 'Conflict' trials (where TTB and Tallying predict different options). Under the pure TTB model, the choice probability for the TTB-favored option is independent of the Tallying prediction, so the difference should be near zero. Under the Strategy Selection model, decision makers probabilistically lapse into Tallying, meaning their adherence to the TTB prediction will systematically drop on Conflict trials compared to Agreement trials, resulting in a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements_match = []
    conflicts_match = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues are already ordered by validity 0 to 4)
        ttb_choice = -1
        for i in range(5):
            if A[i] > B[i]:
                ttb_choice = 0
                break
            elif B[i] > A[i]:
                ttb_choice = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for a, b in zip(A, B) if a > b)
        b_wins = sum(1 for a, b in zip(A, B) if b > a)
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            tally_choice = -1
            
        if ttb_choice == -1:
            continue
            
        is_match = 1 if resp == ttb_choice else 0
        
        if ttb_choice == tally_choice:
            agreements_match.append(is_match)
        elif tally_choice != -1 and ttb_choice != tally_choice:
            conflicts_match.append(is_match)
            
    if not agreements_match or not conflicts_match:
        return 0.0
        
    return float(np.mean(agreements_match) - np.mean(conflicts_match))

### RESULTS
- Predicted under pi_3 (simulated): -0.0076 (var=0.0133)
- Predicted under pi_4 (simulated): 0.1660 (var=0.0107)
- Observed on real data: 0.0070 (var=0.0102)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying strongly prefers A, WADD generally prefers B
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 2: Tallying strongly prefers B, WADD generally prefers A
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            score += resp
            count += 1
        # Trial 3: Tallying is indifferent (50/50), WADD generally prefers B
        elif a == (0, 1, 0, 0, 1) and b == (1, 0, 1, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 4: Tallying is indifferent (50/50), WADD generally prefers A
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 0, 1):
            score += resp
            count += 1
            
    return float(score / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.1575 (var=0.0090)
**Predicted under pi_3:** 0.1037 (var=0.0066)
**Predicted under pi_4:** 0.1358 (var=0.0050)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option A ratings to a string for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1], B = [1, 1, 0, 0, 0]
    t1_mask = a_str == '00111'
    
    if t1_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1400 (var=0.0213)
**Predicted under pi_3:** 0.1263 (var=0.0110)
**Predicted under pi_4:** 0.1800 (var=0.0163)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
        
    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    matches = (np.array(data['response']) == np.array(ttb_choices))
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8538 (var=0.0067)
**Predicted under pi_3:** 0.8667 (var=0.0052)
**Predicted under pi_4:** 0.8404 (var=0.0062)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
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
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.8292 (var=0.0119)
**Predicted under pi_3:** 0.8700 (var=0.0069)
**Predicted under pi_4:** 0.8740 (var=0.0051)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both targeted experiments (Experiment 1 and 2), the observed metrics strongly support the pure Take The Best (TTB) model (pi_3) over the Strategy Selection model (pi_4). Experiment 1 showed that subjects do not systematically lapse into Tallying when it conflicts with TTB (observed value -0.0062, near the 0 predicted by pure TTB). Experiment 2 similarly showed no significant difference in TTB adherence between agreement and conflict trials (observed 0.0070, near pure TTB's prediction, whereas Strategy Selection predicted a large positive difference). Therefore, the assumption that decision makers probabilistically mix TTB with Tallying is incorrect for this domain. However, pure TTB with a uniform lapse rate slightly mispredicts overall accuracy in some other experiments (e.g., Exp 6), suggesting that the error mechanism might be more nuanced than a simple uniform lapse.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory such as 'Take The Best with Validity-Dependent Confidence' or 'Probabilistic Cue Search'. In this theory, the decision maker still searches lexicographically and does not use compensatory tallying, but their probability of lapsing (or making an execution error) depends on the validity of the discriminating cue. If the options are discriminated by a highly valid cue, the lapse rate is low; if they are discriminated only by a low-validity cue, the lapse rate is higher. This maintains the core non-compensatory nature of TTB (explaining the results of Exp 1 and 2) while providing a more realistic noise mechanism that can better capture the variance in overall accuracy across different trial structures."
}
```

## Usage

```json
{
  "prompt_token_count": 7244,
  "candidates_token_count": 379,
  "total_token_count": 8765
}
```
