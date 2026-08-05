# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Selection (Mixed Heuristics): Decision-makers probabilistically sample between a non-compensatory heuristic (Take The Best) and a simple compensatory heuristic (Tallying) on a trial-by-trial basis. The probability of using Tallying increases with the ease of the choice, defined by the absolute difference in the number of features favoring each option. By restricting the sensitivity parameter of this mixture, decision-makers preserve a baseline probability of using TTB even when Tallying discriminates, matching empirical reliance on dominant cues while pulling highly conflicting trials toward chance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) score
    ttb_score = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying (Equal-Weights) score
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    if a_wins > b_wins:
        tally_score = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_score = np.array([0.0, 1.0])
    else:
        tally_score = np.array([0.5, 0.5])
        
    # Difficulty defined by tally difference
    diff = abs(a_wins - b_wins)
    
    # Probability of using Tallying over TTB
    gamma = float(parameters["gamma"])
    w_tally = 1.0 - np.exp(-gamma * diff)
    w_ttb = 1.0 - w_tally
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * ttb_score
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * tally_score
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of probabilities
    p_core = w_ttb * p_ttb + w_tally * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.85, 0.8, 0.75, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Theory 1 (Strategy Selection: mixture of TTB and Tallying) from Theory 2 (Sequential Evidence Accumulation). By manipulating the absolute difference in tally scores (diff) alongside the evidence for the Weighted Additive rule (WADD), we create unique divergence points. For example, when diff=0, Theory 1 reduces strictly to Take The Best (TTB). In Trial 1, diff=0 so Theory 1 unequivocally chooses Option B (which has the highest valid cue). However, the remaining cues are structured such that Option A wins on WADD. Thus, Theory 2 will choose Option B if its threshold is low, but will switch to Option A if its threshold is high. Other trials systematically vary 'diff' to map the mixture probability in Theory 1 against the threshold-dependent stopping in Theory 2.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1, the absolute difference in tally scores (diff) is 0. Under Theory 1 (Strategy Selection), when diff=0, the probability of using Tallying is 0, so the decision-maker strictly uses Take The Best (TTB). TTB checks the most valid cue (Cue 1), which favors Option B, leading to a very high probability of choosing B. Under Theory 2 (Sequential Evidence Accumulation), the decision-maker accumulates evidence. Cue 1 favors B, but Cues 2 and 3 strongly favor A, so the total accumulated evidence actually favors Option A. Unless the decision threshold is extremely low (causing premature stopping at Cue 1), Theory 2 will often choose A. Thus, the proportion of choosing Option B on Trial 1 will be significantly higher under Theory 1 than under Theory 2.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Option A is [0, 1, 1, 0, 0] and Option B is [1, 0, 0, 1, 0]
    # Convert lists to tuples to make them hashable/comparable
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    trial_1_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 0))
    trial_1_data = data[trial_1_mask]
    
    if len(trial_1_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B (response == 1) was chosen
    return float(trial_1_data['response'].mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.6562 (var=0.0385)
- Predicted under pi_5 (simulated): 0.4825 (var=0.0430)
- Observed on real data: 0.4000 (var=0.2400)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.85, 0.82, 0.79, 0.76, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Rationale:** To maximally dissociate Theory 1 (Sequential Evidence Accumulation with Normalized Weights) from Theory 2 (Strategy Selection: Mixed Heuristics), we manipulate the absolute difference in tally scores (diff) alongside the accumulated log-odds evidence. When diff=0, Theory 2 strictly relies on Take The Best (TTB) because Tallying cannot discriminate. In Trials 1 and 2, diff=0 so Theory 2 unequivocally chooses the option with the highest valid cue. However, the lower-validity cues are structured such that their combined log-odds weight outweighs the top cue, causing Theory 1 (if its threshold is sufficiently high) to choose the opposite option. The remaining trials systematically vary 'diff' (from 1 to 3) to map out Theory 2's mixture probability against Theory 1's threshold-dependent stopping and compensatory integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on trials where the tally difference is zero (Trials 1 and 2). Under the Strategy Selection theory (T2), a tally difference of zero means the Tallying heuristic is completely abandoned in favor of Take The Best (TTB), leading to a near-zero probability of choosing the anti-TTB option. Under the Sequential Evidence Accumulation theory (T1), if the decision threshold is sufficiently high, the decision-maker will continue to accumulate evidence past the first cue. Because the lower-validity cues collectively outweigh the first cue in these specific trials, T1 will frequently choose the anti-TTB option.

Source:
def metric(data: pd.DataFrame) -> float:
    anti_ttb_count = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if sum(a) == sum(b):
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
            if row['response'] != ttb_choice:
                anti_ttb_count += 1
            total += 1
    if total == 0:
        return 0.0
    return anti_ttb_count / total

### RESULTS
- Predicted under pi_4 (simulated): 0.2950 (var=0.0252)
- Predicted under pi_5 (simulated): 0.5392 (var=0.0486)
- Observed on real data: 0.5000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    is_1 = a_tuples == (1, 0, 0, 0)
    is_3 = a_tuples == (0, 1, 1, 1)
    is_5 = a_tuples == (1, 1, 0, 0)
    is_6 = a_tuples == (0, 0, 1, 1)
    
    correct = 0
    total = 0
    
    if is_1.any():
        correct += (data.loc[is_1, 'response'] == 0).sum()
        total += is_1.sum()
    if is_3.any():
        correct += (data.loc[is_3, 'response'] == 1).sum()
        total += is_3.sum()
    if is_5.any():
        correct += (data.loc[is_5, 'response'] == 0).sum()
        total += is_5.sum()
    if is_6.any():
        correct += (data.loc[is_6, 'response'] == 1).sum()
        total += is_6.sum()
        
    if total == 0:
        return 0.5
    return float(correct / total)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5112 (var=0.0128)
**Predicted under pi_5:** 0.5437 (var=0.0229)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        tally_match.append(row['response'] == tally_choice)
        
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4742 (var=0.0221)
**Predicted under pi_5:** 0.4383 (var=0.0189)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None and row['response'] == ttb_pred:
            ttb_matches += 1
            
    return float(ttb_matches / total)
```

**Observed (real) value:** 0.6950 (var=0.1026)
**Predicted under pi_4:** 0.5215 (var=0.0128)
**Predicted under pi_5:** 0.4765 (var=0.0151)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_4:** 0.4517 (var=0.0139)
**Predicted under pi_5:** 0.5262 (var=0.0235)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    relevant_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Check if one option has [1, 0, 0] on the first 3 features and the other has [0, 1, 1]
        a_top3 = [a[0], a[1], a[2]]
        b_top3 = [b[0], b[1], b[2]]
        if a_top3 == [1, 0, 0] and b_top3 == [0, 1, 1]:
            relevant_trials += 1
            if row['response'] == 1:
                wadd_consistent += 1
        elif a_top3 == [0, 1, 1] and b_top3 == [1, 0, 0]:
            relevant_trials += 1
            if row['response'] == 0:
                wadd_consistent += 1
    if relevant_trials == 0:
        return 0.5
    return wadd_consistent / relevant_trials
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.3494 (var=0.0182)
**Predicted under pi_5:** 0.5022 (var=0.0539)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A=[1, 0, 0, 1, 1] and B=[0, 1, 1, 0, 0]
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5600 (var=0.2464)
**Predicted under pi_4:** 0.2900 (var=0.0286)
**Predicted under pi_5:** 0.4587 (var=0.0270)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments, Theory 2 (Sequential Evidence Accumulation, pi_5) provides predictions that are consistently closer to the observed metrics than Theory 1 (Strategy Selection, pi_4). For instance, in Experiments 1, 2, 6, 7, and 8, Theory 1 predicts extreme deviations from chance (e.g., 0.29 in Exp 2 and 8, 0.35 in Exp 7) while the observed data is often centered around 0.5, which Theory 2 handles much better. However, both theories fail to capture the high Take-The-Best (TTB) match rate in Experiment 5 (observed 0.695 vs predictions around ~0.50). This suggests that while subjects do not strictly follow the deterministic mixture proposed by Theory 1, they still exhibit a strong reliance on the most valid cue that is not fully captured by Theory 2's log-odds weighting. Theory 1 is fundamentally misaligned with the empirical data, necessitating a new theoretical approach.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Rank-Weighted Exponential Integration' theory. Instead of a probabilistic mixture of heuristics (like Theory 1) or log-odds evidence accumulation (like Theory 2), this theory posits that decision-makers evaluate all cues simultaneously but weight them exponentially according to their validity rank (e.g., weight = alpha^(-rank)). This creates a highly non-compensatory profile that mimics TTB (explaining the high TTB match rate in Experiment 5) but allows for compensatory overrides when multiple lower-ranked cues strongly align against the top cue, naturally capturing the regressions to chance (~0.5) observed in the highly conflicting trial designs of Experiments 2, 7, and 8 without needing an arbitrary stopping threshold."
}
```

## Usage

```json
{
  "prompt_token_count": 7448,
  "candidates_token_count": 411,
  "total_token_count": 8980
}
```
