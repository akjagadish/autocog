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
Weighted Additive (WADD) Theory with Non-linear Subjective Weights: People evaluate options by considering all available features, weighting each by a subjective transformation of its validity. Rather than using raw validities, people subjectively distort them via an exponent (gamma). A gamma > 1 disproportionately emphasizes higher-validity cues (shifting behavior towards Take The Best), whereas a gamma < 1 flattens the weights (shifting behavior towards Tallying). The subjective value of an option is the weighted sum of its features, and choices are made probabilistically via a softmax over these values, with a lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Weighted additive sum for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Strategy Mixture Theory with Decoupled Determinism: Decision-makers probabilistically choose between Tallying and Take-The-Best on any given trial. Because Tallying operates on raw integer count differences (which can be large) while TTB operates on binary 1/0 differences, the two heuristics have different intrinsic scales. To account for this, the model uses independent inverse temperatures (beta_tally and beta_ttb) for each strategy. This allows the decision process to independently calibrate the determinism of each heuristic, avoiding artificial scale distortions when balancing the two strategies via the mixture weight.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_tally_weight = float(parameters["p_tally_weight"])
    
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # --- Take-The-Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_tally_weight * p_tally + (1.0 - p_tally_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.98, 0.94, 0.9, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To decisively dissociate WADD from the Tallying/TTB Strategy Mixture, we introduce 'Dominance Violation' trials. In these critical trials, Option A possesses the single highest-validity cue (winning TTB) AND a greater total number of winning cues (winning Tallying). Because both strategies in the Mixture model agree, the Mixture model strictly predicts Option A. However, WADD with a non-linear validity transformation (gamma > 1) will predict Option B if Option B's fewer cues are collectively strong enough (e.g., the second and third highest validities) to outweigh Option A's combination of one top cue and several weak cues. This creates a qualitative divergence that no mixture of Tallying and TTB can explain. We also include standard conflict trials to calibrate the mixture weights and gamma.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that violate both TTB and Tallying on the critical 'Dominance Violation' trials. On these trials, one option wins on the single highest-validity cue AND has a greater total number of winning cues. The Strategy Mixture model, which probabilistically chooses between TTB and Tallying, strongly predicts this dominant option. However, the WADD model with a gamma > 1 can predict the opposite option, because the second and third highest validities combined can outweigh the single highest validity plus several weak validities. The metric returns the proportion of these WADD-specific choices, which will be near zero for the Mixture model but substantially higher for the WADD model.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = tuple(a)
        b = tuple(b)
        if a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 0, 0):
            wadd_choices += int(r == 1)
            total += 1
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 1, 1):
            wadd_choices += int(r == 0)
            total += 1
    return float(wadd_choices / total) if total > 0 else 0.0

### RESULTS
- Predicted under pi_3 (simulated): 0.7258 (var=0.0583)
- Predicted under pi_4 (simulated): 0.1492 (var=0.0135)
- Observed on real data: 0.4700 (var=0.0206)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design tests for a qualitative signature of the Strategy Mixture Theory that WADD cannot capture: the decoupling of determinism between TTB and Tallying. By varying the feature count difference (e.g., Option A having 1 vs 4 features, or 2 vs 3 features) while holding the TTB winner constant, WADD strictly predicts a monotonic change in choice probabilities because the weighted sum difference must change. In contrast, the Strategy Mixture Theory can predict perfectly flat (plateaued) choice probabilities across these trials if the Tallying inverse temperature (beta_tally) is high, because Tallying becomes deterministic as long as the count difference is non-zero, regardless of its magnitude. This allows the mixture model to predict identical choice proportions for pairs with drastically different feature-sum differences.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric evaluates the change in the probability of choosing Option A at the extremes of the feature count differences. Under WADD, the probability of choosing A strictly and monotonically increases as the weighted sum difference shifts in favor of A, predicting a positive difference for both (T2 - T1) and (T10 - T9). Under the Strategy Mixture Theory, because Tallying operates on raw count differences, Tallying strongly favors B in both T1 and T2, and strongly favors A in both T9 and T10. This determinism causes the choice probabilities to plateau at the tails, resulting in a metric value close to zero.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Response = 0 means option A was chosen
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t1_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))
    t2_mask = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 1))
    t9_mask = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))
    t10_mask = (data['A_tuple'] == (1, 1, 1, 1, 0)) & (data['B_tuple'] == (0, 0, 0, 0, 1))
    
    p_t1 = data.loc[t1_mask, 'chose_A'].mean()
    p_t2 = data.loc[t2_mask, 'chose_A'].mean()
    p_t9 = data.loc[t9_mask, 'chose_A'].mean()
    p_t10 = data.loc[t10_mask, 'chose_A'].mean()
    
    if pd.isna(p_t1) or pd.isna(p_t2) or pd.isna(p_t9) or pd.isna(p_t10):
        return 0.0
        
    return (p_t2 - p_t1) + (p_t10 - p_t9)


### RESULTS
- Predicted under pi_3 (simulated): 0.4778 (var=0.0920)
- Predicted under pi_4 (simulated): 0.0289 (var=0.0774)
- Observed on real data: 0.1378 (var=0.0902)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            if b[i] > a[i]:
                return 1
        return -1
    
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if ttb_predict(a, b) == r:
            matches += 1
            
    return float(matches) / len(data) if len(data) > 0 else 0.0
```

**Observed (real) value:** 0.3747 (var=0.0051)
**Predicted under pi_3:** 0.3573 (var=0.0363)
**Predicted under pi_4:** 0.5258 (var=0.0613)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate first feature for each option (most valid cue)
    first_a = data['option_a_ratings'].apply(lambda x: x[0])
    first_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where Tallying and TTB make strictly opposite predictions
    # Tallying decides by sum(A) vs sum(B)
    # TTB decides by A[0] vs B[0]
    opposing_mask = ((sum_a > sum_b) & (first_a < first_b)) | ((sum_a < sum_b) & (first_a > first_b))
    opposing_data = data[opposing_mask]
    
    if len(opposing_data) == 0:
        return 0.5
        
    # Determine what Tallying would choose: 0 (A) if sum_a > sum_b else 1 (B)
    tallying_choice = (opposing_data['option_a_ratings'].apply(sum) < opposing_data['option_b_ratings'].apply(sum)).astype(int)
    
    # Calculate the proportion of responses that match the Tallying prediction
    match_rate = (opposing_data['response'] == tallying_choice).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.6400 (var=0.0095)
**Predicted under pi_3:** 0.7929 (var=0.0253)
**Predicted under pi_4:** 0.5837 (var=0.0394)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    wadd_match = 0
    n_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b:
                expected = 0
            elif wadd_b > wadd_a:
                expected = 1
            else:
                continue
                
            if row['response'] == expected:
                wadd_match += 1
            n_ties += 1
            
    if n_ties == 0:
        return 0.5
    return wadd_match / n_ties
```

**Observed (real) value:** 0.5863 (var=0.0190)
**Predicted under pi_3:** 0.7831 (var=0.0163)
**Predicted under pi_4:** 0.6769 (var=0.0151)

### Experiment 6
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_pairs = [
        ( (1,0,0,0), (0,1,0,0) ),
        ( (1,0,1,0), (0,1,0,1) )
    ]
    
    chosen_has_f1 = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        is_target = False
        for p in target_pairs:
            if (a == p[0] and b == p[1]) or (a == p[1] and b == p[0]):
                is_target = True
                break
                
        if is_target:
            chose_a = (row['response'] == 0)
            chosen_opt = a if chose_a else b
            
            if chosen_opt[0] == 1:
                chosen_has_f1.append(1)
            else:
                chosen_has_f1.append(0)
                
    if len(chosen_has_f1) == 0:
        return 0.5
        
    return float(np.mean(chosen_has_f1))
```

**Observed (real) value:** 0.5317 (var=0.0228)
**Predicted under pi_3:** 0.7533 (var=0.0292)
**Predicted under pi_4:** 0.6825 (var=0.0197)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither the non-linear WADD theory (pi_3) nor the Strategy Mixture of Tallying and TTB (pi_4) fully captures the observed data. In Experiment 1, observed dominance violations (0.47) fall midway between WADD's overprediction (0.72) and the Mixture's underprediction (0.15). In Experiment 2, the plateauing effect metric (0.13) is closer to the Mixture model (0.03) than WADD (0.48). Similar intermediate results are observed in Experiments 3-6. WADD fails to capture the insensitivity to feature count magnitude changes (plateauing), while the Tallying/TTB mixture fails to account for trials where a combination of lower-validity cues overcomes the highest-validity cue. This suggests that human decision-making in this domain blends both fully compensatory (WADD-like) and non-compensatory (TTB-like) processes, rather than just two simple heuristics.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'WADD / Take-The-Best Strategy Mixture' theory. Instead of mixing Tallying and TTB (which forces the compensatory process to ignore validities), this theory posits that decision-makers probabilistically mix a fully compensatory Weighted Additive (WADD) strategy with a non-compensatory Take-The-Best (TTB) strategy. This would allow the model to predict dominance violations (via the WADD component) while also predicting choice probability plateaus when the TTB component dominates the mixture. The model should include separate inverse temperatures for WADD and TTB to handle their different intrinsic scales."
}
```

## Usage

```json
{
  "prompt_token_count": 7447,
  "candidates_token_count": 385,
  "total_token_count": 8846
}
```
