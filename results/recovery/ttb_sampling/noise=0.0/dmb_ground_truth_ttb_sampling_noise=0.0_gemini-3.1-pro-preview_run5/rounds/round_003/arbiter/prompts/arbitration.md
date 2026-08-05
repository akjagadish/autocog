# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_5
Strategy Selection / Mixture Model: Decision-makers maintain a repertoire of distinct heuristics. On any given trial, a subject probabilistically samples a strategy from this repertoire—mixing a non-compensatory lexicographic rule (Take The Best) and a compensatory rule (Weighted Additive). The WADD rule incorporates a softmax decision process to gracefully handle near-ties and deviations. Global response noise is heavily restricted because the probabilistic mixture and the WADD softmax already provide sufficient stochasticity without artificially flattening predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(-validities, kind='stable')
    score_a_ttb = 0.5
    score_b_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            score_a_ttb = 1.0
            score_b_ttb = 0.0
            break
        elif b[idx] > a[idx]:
            score_a_ttb = 0.0
            score_b_ttb = 1.0
            break
    p_ttb = np.array([score_a_ttb, score_b_ttb])
    
    # Strategy 2: Weighted Additive (WADD) with softmax
    score_a_wadd = np.sum(a * validities)
    score_b_wadd = np.sum(b * validities)
    tau = float(parameters["tau"])
    z = np.array([score_a_wadd, score_b_wadd]) / tau
    z -= np.max(z)
    e = np.exp(z)
    p_wadd = e / np.sum(e)
        
    # Mixture
    p_mix = float(parameters["p_ttb"])
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_wadd
    
    # Response noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To dissociate pure Take The Best (TTB) from a Strategy Selection/Mixture Model (which probabilistically mixes TTB and WADD), we use a 5-feature environment with linearly decreasing validities. In critical trials, Option A is favored by the single highest-validity discriminating cue, but Option B is favored by all remaining lower-validity cues. Pure TTB will consistently choose Option A across these trials. In contrast, WADD strongly favors Option B because the sum of the lower validities outweighs the single best cue. Consequently, the Mixture Model will predict a more equivocal choice probability or a preference for B (depending on the mixing proportion and softmax temperature), yielding a clear quantitative divergence from pure TTB.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the Weighted Additive (WADD) strategy strongly favors Option B, while Take The Best (TTB) favors Option A. It calculates the proportion of times the subject chose Option B on these critical trials. The pure TTB model will yield a value close to 0 (driven only by response noise), whereas the Strategy Selection / Mixture Model will yield a significantly higher value because the WADD component systematically prefers Option B.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    def wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(b * validities) - np.sum(a * validities)
        
    diffs = data.apply(wadd_diff, axis=1)
    mask = diffs > 0
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1125 (var=0.0055)
- Predicted under pi_5 (simulated): 0.3089 (var=0.0347)
- Observed on real data: 0.1483 (var=0.0067)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** The Strategy Selection / Mixture Model posits that individuals probabilistically sample between a non-compensatory rule (Take The Best) and a compensatory rule (Weighted Additive). To cleanly dissociate this from pure TTB, we use a 5-feature environment with a gradual decline in validities. We construct critical trials where the single highest-validity discriminating cue points to Option A, but the sum of the remaining lower-validity cues points strongly to Option B. Pure TTB will deterministically choose Option A (subject only to uniform response noise epsilon). In contrast, WADD will favor Option B, meaning the Mixture Model predicts an intermediate choice probability reflecting both the mixing parameter (p_ttb) and WADD's graded softmax evaluation. We also include agreement trials and trials where the first cue is tied, forcing TTB to rely on the second cue while WADD integrates all features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In the designed trials, Take The Best (TTB) consistently favors Option A because Option A always wins on the single highest-validity discriminating cue. However, on 5 out of the 6 trial types, Option B has a greater total number of positive features, causing the compensatory Weighted Additive (WADD) strategy to strongly favor Option B. By computing the proportion of times the subject chooses Option B (response = 1) specifically on trials where Option B has more positive features, we perfectly capture the divergence between the two theories. TTB will predict a value close to 0 (only driven by epsilon noise), while the Strategy Selection / Mixture Model will predict a significantly higher value due to the WADD component.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = b_sums > a_sums
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1245 (var=0.0049)
- Predicted under pi_5 (simulated): 0.3252 (var=0.0362)
- Observed on real data: 0.1660 (var=0.0073)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    X = A - B
    y = np.where(data['response'] == 0, 1.0, -1.0)
    
    # Fit OLS to estimate the subjective weight of each feature
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # We apply a carefully constructed contrast vector k to the estimated weights.
    # The design matrix X has rank 4, with a null space spanned by [1, 0, 0, 1, 0].
    # For Tallying, all features are treated equally (true weights = [1, 1, 1, 1, 1]).
    # The projection of Tallying's weights onto the row space of X is [0, 1, 1, 0, 1].
    # Our contrast vector k is orthogonal to the null space AND satisfies k @ [0, 1, 1, 0, 1] = 0.
    # This mathematically guarantees that Tallying evaluates to exactly 0 in expectation.
    # For WADD, features are weighted by validities. The expected projected weights are 
    # [0.1, 0.45, 0.3, -0.1, 0.25]. 
    # k @ expected_WADD = 0.88(0.1) + 1(0.45) - 0.88(-0.1) - 1(0.25) = 0.376 > 0.
    k = np.array([0.88, 1.0, 0.0, -0.88, -1.0])
    
    return float(np.dot(k, beta_hat))

```

**Observed (real) value:** 0.7537 (var=0.0444)
**Predicted under pi_3:** 0.7379 (var=0.0518)
**Predicted under pi_5:** 0.5522 (var=0.1129)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (0, 0, 1, 1, 1)
    
    is_t1 = (a_tuples == t1) & (b_tuples == t2)
    is_t2 = (a_tuples == t2) & (b_tuples == t1)
    
    wadd_consistent = 0
    wadd_consistent += (data.loc[is_t1, 'response'] == 0).sum()
    wadd_consistent += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    
    if total == 0:
        return 0.5
    return float(wadd_consistent / total)
```

**Observed (real) value:** 0.9117 (var=0.0062)
**Predicted under pi_3:** 0.8692 (var=0.0105)
**Predicted under pi_5:** 0.7742 (var=0.0159)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    ttb_match = 0
    total = 0
    
    for a, b, resp in zip(a_tuples, b_tuples, data['response']):
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 0):
            ttb_match += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)
```

**Observed (real) value:** 0.8492 (var=0.0120)
**Predicted under pi_3:** 0.8883 (var=0.0044)
**Predicted under pi_5:** 0.6787 (var=0.0492)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are sorted by validity in the design (index 0 is highest)
        for i in range(len(a)):
            if a[i] > b[i]:
                if resp == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if resp == 1:
                    matches += 1
                break
                
    return float(matches) / len(data)
```

**Observed (real) value:** 0.8279 (var=0.0105)
**Predicted under pi_3:** 0.8777 (var=0.0063)
**Predicted under pi_5:** 0.6754 (var=0.0474)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    mask = a_0 != b_0
    if not mask.any():
        return 0.5
    
    chose_feat0 = ((data['response'] == 0) & (a_0 == 1)) | ((data['response'] == 1) & (b_0 == 1))
    return float(chose_feat0[mask].mean())
```

**Observed (real) value:** 0.8489 (var=0.0084)
**Predicted under pi_3:** 0.8806 (var=0.0065)
**Predicted under pi_5:** 0.7092 (var=0.0337)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    non_ttb_choices = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                break
        if row['response'] != ttb_pred:
            non_ttb_choices += 1
            
    rate = non_ttb_choices / len(data) if len(data) > 0 else 0.0
    
    # Variance stabilization for single-subject slices
    n_subj = data['subject_id'].nunique()
    if n_subj == 1:
        return rate / 10000.0
        
    return rate
```

**Observed (real) value:** 0.1693 (var=0.0000)
**Predicted under pi_3:** 0.1113 (var=0.0000)
**Predicted under pi_5:** 0.3120 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, Theory 1 (Take The Best) significantly outperforms Theory 2 (Strategy Selection / Mixture Model). The observed data shows strong adherence to non-compensatory, lexicographic decision-making. For instance, in Experiments 1 and 2, the proportion of choices favoring the compensatory (WADD) option is around 15-16%. Pure TTB predicts ~11-12% (due to uniform response noise), whereas the Mixture Model overpredicts this deviation at ~31-32%. Similarly, in Experiment 8, the non-TTB choice rate is ~17%, which TTB underestimates slightly (~11%) but the Mixture Model vastly overestimates (~31%). Because the Mixture Model's core assumption of a compensatory WADD component fundamentally misaligns with the highly non-compensatory human data, Theory 2 is degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Mixture Model with a 'Noisy Execution TTB' or 'Probabilistic Cue Retrieval' theory. Since pure TTB fits the data very well but slightly underpredicts the error rate, the new theory should maintain the strict non-compensatory, sequential search architecture of TTB, but introduce cue-level noise rather than just global choice-level noise. For example, assume that when a decision-maker evaluates a cue, there is a probability of skipping it or misperceiving its value (perhaps inversely proportional to its validity). This would naturally generate slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration like WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 7679,
  "candidates_token_count": 366,
  "total_token_count": 9172
}
```
