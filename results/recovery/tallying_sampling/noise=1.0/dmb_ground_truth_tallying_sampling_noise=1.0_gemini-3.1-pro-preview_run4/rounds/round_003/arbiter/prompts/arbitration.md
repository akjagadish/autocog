# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Equal Weights with High Lapse: Subjects often find the integration of complex cue validities too cognitively demanding or disengaging in these conflict paradigms. As a result, they ignore the provided validities entirely and fall back on a simple 'Tallying' (Equal Weights) heuristic, where they just count the number of positive features for each option. Furthermore, due to the high cognitive friction or confusion, subjects exhibit a very high baseline guessing rate (lapse), meaning that on the vast majority of trials they simply guess randomly. This explains why behavior across various conflict and agreement metrics hovers so closely to 0.5 or 0 difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: equal weights for all features (counting positive cues)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallied scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # High uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** To dissociate the Single-Cue Focus model (Advocated) from the Equal Weights model (Competing), both of which assume a high lapse rate, we create trials that pit the single highest-validity cue against the sheer number of positive cues. In some trials, Option A has only the highest-validity cue while Option B has multiple lower-validity cues. The Advocated theory predicts a preference for Option A (driven by the single best cue), whereas the Competing theory predicts a preference for Option B (driven by tallying). We also include trials where the total number of cues is equal, forcing the Competing theory to predict exactly a 50/50 split, while the Advocated theory still predicts a clear preference based on the single best cue.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option containing the single highest-validity cue (cue 0). The Advocated theory (Single-Cue Focus) predicts that subjects will systematically favor the option with the highest-validity cue whenever they do not lapse, yielding a concordance greater than 0.5. Conversely, the Competing theory (Equal Weights) tallies the total number of cues. The experimental design intentionally pits the highest-validity cue against the sheer quantity of cues, such that the Equal Weights model consistently predicts choices contrary to the single best cue, yielding a concordance below 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Cue 0 is the highest validity cue (0.95)
    a_cue0 = a_ratings[:, 0]
    b_cue0 = b_ratings[:, 0]
    
    # Determine which option has the highest validity cue
    expected = np.where(a_cue0 > b_cue0, 0, np.where(b_cue0 > a_cue0, 1, -1))
    
    valid = expected != -1
    if not np.any(valid):
        return 0.5
        
    # Calculate the proportion of choices aligning with the highest validity cue
    match = (data['response'].values[valid] == expected[valid])
    return float(np.mean(match))

### RESULTS
- Predicted under pi_5 (simulated): 0.5175 (var=0.0019)
- Predicted under pi_4 (simulated): 0.4752 (var=0.0030)
- Observed on real data: 0.5004 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated theory (Equal Weights with High Lapse) from the Competing theory (Extreme Cognitive Noise / Single-Cue Focus), we create trials that independently manipulate the option favored by the single highest-validity cue and the option favored by the raw tally of positive cues. In some trials, the single best cue strongly opposes the sheer number of cues (e.g., Option A has only the best cue, while Option B has 3 or 4 lower-validity cues). The Advocated theory predicts a preference for the option with more positive cues, completely ignoring validities, whereas the Competing theory predicts a preference for the option with the single best cue, completely ignoring the count. We also include tie-breaker trials: trials where the total number of cues is equal but the best cue favors one option (Advocated predicts a tie, Competing predicts a clear winner), and trials where both options share the best cue but one has more total cues (Advocated predicts a clear winner, Competing predicts a tie).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'conflict' trials where the Equal Weights (Tallying) heuristic and the Single-Cue (Best Cue) heuristic point in opposite directions. It calculates the proportion of choices on these trials that align with the Tallying prediction. The Advocated theory (Tallying) will score above 0.5, whereas the Competing theory (Single-Cue Focus) will score below 0.5, maximizing the discriminability between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    best_a = data['option_a_ratings'].apply(lambda x: x[0])
    best_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    tally_favors_a = tally_a > tally_b
    tally_favors_b = tally_a < tally_b
    best_favors_a = best_a > best_b
    best_favors_b = best_a < best_b
    
    conflict = (tally_favors_a & best_favors_b) | (tally_favors_b & best_favors_a)
    
    if conflict.sum() == 0:
        return 0.5
        
    aligned_tally = (tally_favors_a & (data['response'] == 0)) | (tally_favors_b & (data['response'] == 1))
    
    return float(aligned_tally[conflict].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.4637 (var=0.0069)
- Predicted under pi_4 (simulated): 0.5306 (var=0.0065)
- Observed on real data: 0.4925 (var=0.0050)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    diff = A - B
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(A.shape[1]):
            if diff[i, j] == 1:
                ttb_preds[i] = 0
                break
            elif diff[i, j] == -1:
                ttb_preds[i] = 1
                break
                
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5025 (var=0.0026)
**Predicted under pi_5:** 0.5258 (var=0.0030)
**Predicted under pi_4:** 0.4506 (var=0.0030)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Predicted under pi_5:** 0.4927 (var=0.0023)
**Predicted under pi_4:** 0.5415 (var=0.0036)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_chosen = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_f0 = a[0]
        b_f0 = b[0]
        
        # Focus on conflict trials where the option with the best cue (f0) 
        # has very few other positive cues, while the other option has many.
        if a_f0 == 1 and b_f0 == 0:
            if sum(a) <= 2 and sum(b) >= 4:
                ttb_chosen.append(1 if resp == 0 else 0)
        elif b_f0 == 1 and a_f0 == 0:
            if sum(b) <= 2 and sum(a) >= 4:
                ttb_chosen.append(1 if resp == 1 else 0)
                
    if not ttb_chosen:
        return 0.5
    return float(np.mean(ttb_chosen))
```

**Observed (real) value:** 0.4947 (var=0.0048)
**Predicted under pi_5:** 0.5133 (var=0.0036)
**Predicted under pi_4:** 0.4503 (var=0.0051)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Safely convert list of ratings to string for easy matching
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Agreement trials: WADD and TTB both strongly favor the '11111' option
    t5_ab = data[(data['a_str'] == '11111') & (data['b_str'] == '00000')]
    t5_ba = data[(data['a_str'] == '00000') & (data['b_str'] == '11111')]
    
    agree_wadd = 0
    agree_total = 0
    if len(t5_ab) > 0:
        agree_wadd += (t5_ab['response'] == 0).sum()
        agree_total += len(t5_ab)
    if len(t5_ba) > 0:
        agree_wadd += (t5_ba['response'] == 1).sum()
        agree_total += len(t5_ba)
    p_agree = agree_wadd / agree_total if agree_total > 0 else 0.5
    
    # Conflict trials: WADD strongly favors '01111' but TTB favors '10000'
    t1_ab = data[(data['a_str'] == '01111') & (data['b_str'] == '10000')]
    t1_ba = data[(data['a_str'] == '10000') & (data['b_str'] == '01111')]
    
    conflict_wadd = 0
    conflict_total = 0
    if len(t1_ab) > 0:
        conflict_wadd += (t1_ab['response'] == 0).sum()
        conflict_total += len(t1_ab)
    if len(t1_ba) > 0:
        conflict_wadd += (t1_ba['response'] == 1).sum()
        conflict_total += len(t1_ba)
        
    p_conflict = conflict_wadd / conflict_total if conflict_total > 0 else 0.5
    
    return float(p_agree - p_conflict)
```

**Observed (real) value:** -0.0111 (var=0.0415)
**Predicted under pi_5:** 0.0589 (var=0.0404)
**Predicted under pi_4:** -0.0378 (var=0.0365)

### Experiment 7
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract the first cue (highest validity) for options A and B
    a_cue1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Determine if the chosen option had a 1 on the most valid cue
    # response == 0 means A was chosen, response == 1 means B was chosen
    chosen_cue1 = np.where(data['response'] == 0, a_cue1, b_cue1)
    
    # Return the proportion of trials where the choice aligned with the most valid cue
    return float(np.mean(chosen_cue1))
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Predicted under pi_5:** 0.5306 (var=0.0022)
**Predicted under pi_4:** 0.4892 (var=0.0028)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_trial_1(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1)
        
    mask = data.apply(is_trial_1, axis=1)
    if not mask.any():
        return 0.5
        
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.5350 (var=0.0082)
**Predicted under pi_5:** 0.5238 (var=0.0164)
**Predicted under pi_4:** 0.4662 (var=0.0191)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics consistently hover around 0.5, indicating that subjects' choices are overwhelmingly random. Theory 1 (Single-Cue Focus) predicts slight deviations above 0.5 on metrics tracking the highest-validity cue, which are generally absent in the real data (e.g., Exp 1: 0.5004, Exp 7: 0.4996). Theory 2 (Equal Weights) predicts deviations favoring the sheer count of cues, which also fail to materialize (e.g., Exp 2: 0.4925, Exp 4: 0.4996). However, in Experiment 8, there is a slight preference (0.5350) for the option with fewer but higher-validity cues, which Theory 1 captures better than Theory 2. Neither theory perfectly captures the nearly absolute lack of signal in most experiments, but Theory 2's reliance on tallying actively pushes predictions in the wrong direction on conflict trials. A better approach would be to assume subjects attempt full cue integration (Weighted Additive) but are overwhelmed by noise, or that they are simply guessing entirely.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a 'Weak Weighted Additive with Extreme Noise' theory. This theory should posit that subjects attempt to weight all cues by their provided validities (unlike Single-Cue or Tallying), but due to cognitive overload or disengagement, their behavior is dominated by an extremely high lapse rate (guessing). This would naturally predict outcomes very close to 0.5 across all conflict metrics, while correctly capturing the very slight systematic preference for higher-validity combinations seen in Experiment 8."
}
```

## Usage

```json
{
  "prompt_token_count": 7216,
  "candidates_token_count": 389,
  "total_token_count": 8888
}
```
