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
Probabilistic Cue Search (Sequential Evidence Accumulation): Decision-makers evaluate features sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue (as in pure Take-The-Best) or exhaustively accumulating all cues (as in Weighted Additive), they exhibit a probabilistic stopping rule. After evaluating each discriminating cue and updating their internal evidence, they stop searching and make a choice with probability `theta`. If they do not stop, they continue to the next discriminating cue, accumulating its evidence. This naturally produces a graded interpolation between non-compensatory and compensatory decision-making without relying on a discrete mixture of distinct strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Subjective weights normalized so the sum of weights is 1.0
    w = validities ** gamma
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Sort all features by validity descending
    order = np.argsort(validities)[::-1]
    
    # Find discriminating cues in order of validity
    discrim_indices = [i for i in order if a[i] != b[i]]
    
    if len(discrim_indices) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_A_total = 0.0
        p_B_total = 0.0
        p_continue = 1.0
        
        score_A = 0.0
        score_B = 0.0
        
        for j, i in enumerate(discrim_indices):
            # Accumulate evidence from the current discriminating cue
            score_A += w[i] * a[i]
            score_B += w[i] * b[i]
            
            # Determine stopping probability
            if j < len(discrim_indices) - 1:
                p_stop = theta
            else:
                p_stop = 1.0  # Must stop at the last discriminating cue
                
            # Softmax over accumulated scores so far
            z_A = beta * score_A
            z_B = beta * score_B
            max_z = max(z_A, z_B)
            e_A = np.exp(z_A - max_z)
            e_B = np.exp(z_B - max_z)
            p_A_given_stop = e_A / (e_A + e_B)
            p_B_given_stop = e_B / (e_A + e_B)
            
            # Marginalize over the stopping probability
            p_A_total += p_continue * p_stop * p_A_given_stop
            p_B_total += p_continue * p_stop * p_B_given_stop
            
            # Update the probability of continuing to the next cue
            p_continue *= (1.0 - p_stop)
            
        p_core = np.array([p_A_total, p_B_total])
        
    # Apply uniform lapse
    n_opts = 2
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Probabilistic Cue Search (PCS) from Strategy Mixture Theory (SMT) by exploiting how the two models handle intermediate evidence. SMT posits a discrete mixture of a single-cue heuristic (TTB) and an exhaustive compensatory rule (WADD). It only cares about the first discriminating cue and the final weighted sum of all cues. In contrast, PCS accumulates evidence sequentially and can stop after any cue. We introduce 'evidence trap' trials where the first cue favors Option A (aligning TTB with A) and the final weighted sum also favors Option A (aligning WADD with A), meaning SMT strongly predicts a choice for A. However, the intermediate cues sequentially build a temporary but strong lead for Option B. Because PCS has a probability of stopping search during this intermediate phase, it predicts a substantially higher probability of choosing Option B than SMT can account for. We also include standard compensatory trials and ties to provide a robust parameter estimation landscape.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Contrasts the preference for the TTB-favored option in Pair 4 ([1, 1, 0, 0, 0] vs [0, 0, 1, 1, 1]) versus Pair 1 ([1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]). Under Strategy Mixture Theory (SMT), the TTB-favored option in Pair 1 is also supported by the compensatory WADD rule, whereas in Pair 4, WADD opposes it. Thus, SMT predicts a higher preference for the TTB option in Pair 1 than in Pair 4 (a negative difference). In contrast, Probabilistic Cue Search (PCS) accumulates evidence sequentially. In Pair 4, the TTB option wins the first two cues, building a strong early lead that captures early stopping. In Pair 1, the TTB option wins the first cue but loses the next two, creating an 'evidence trap' that heavily penalizes it if search stops early. Thus, PCS predicts a higher preference for the TTB option in Pair 4 than in Pair 1 (a positive difference). By pooling across counterbalanced left/right presentations, we halve the per-subject variance, providing a highly reliable discriminative signal.

Source:
def metric(data: pd.DataFrame) -> float:
    # Target option 1: [1, 1, 0, 0, 0] vs [0, 0, 1, 1, 1]
    mask_4_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
               data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    mask_4_B = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
               data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
               
    # Target option 2: [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]
    mask_1_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1)) & \
               data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    mask_1_B = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0)) & \
               data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))
               
    choices_4 = []
    if mask_4_A.any():
        choices_4.extend((data.loc[mask_4_A, 'response'] == 0).astype(float).tolist())
    if mask_4_B.any():
        choices_4.extend((data.loc[mask_4_B, 'response'] == 1).astype(float).tolist())
        
    choices_1 = []
    if mask_1_A.any():
        choices_1.extend((data.loc[mask_1_A, 'response'] == 0).astype(float).tolist())
    if mask_1_B.any():
        choices_1.extend((data.loc[mask_1_B, 'response'] == 1).astype(float).tolist())
        
    p_4 = sum(choices_4) / len(choices_4) if choices_4 else 0.5
    p_1 = sum(choices_1) / len(choices_1) if choices_1 else 0.5
    
    return float(p_4 - p_1)

### RESULTS
- Predicted under pi_5 (simulated): 0.0958 (var=0.0307)
- Predicted under pi_4 (simulated): -0.0333 (var=0.0178)
- Observed on real data: -0.0417 (var=0.0097)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Strategy Mixture Theory (SMT) from Probabilistic Cue Search (PCS) by exploiting how each model handles intermediate evidence. SMT posits a discrete mixture of a single-cue heuristic (TTB) and an exhaustive compensatory rule (WADD). It is influenced only by the first discriminating cue and the final weighted sum of all cues. PCS, however, accumulates evidence sequentially and can stop at any point. We introduce 'anti-trap' trials where both the first discriminating cue and the final weighted sum strongly favor Option A, meaning SMT predicts a near-deterministic choice for A. However, the intermediate cues are arranged to temporarily build a strong lead for Option B. Because PCS has a probability of stopping search during this intermediate phase, it is forced to predict a substantial proportion of B choices. To ensure PCS cannot simply set its stopping probability to 0 (always exhaustive) or 1 (always TTB), we include standard compensatory trials that require an intermediate stopping probability in PCS, or an intermediate mixture weight in SMT, to fit the data. SMT can easily accommodate mixed choices on compensatory trials while maintaining deterministic predictions on the anti-trap trials, whereas PCS cannot.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits a structural difference in how the two theories evaluate evidence. In the experimental design, Trial 1 and Trial 2 share the exact same sequence of score differences for the first 3 cues, only diverging at cues 4 and 5. Because Probabilistic Cue Search (PCS) frequently stops search early, it cannot distinguish between Trial 1 and Trial 2 in those early-stopping cases, leading to highly correlated choice probabilities. Strategy Mixture Theory (SMT), however, computes a holistic Weighted Additive (WADD) score that evaluates all cues simultaneously, perfectly separating these trials from the start. By computing the difference in choice probabilities between pairs of trials that share their early cue trajectories (T1 vs T2, T3 vs T4, T6 vs T5, T8 vs T7), we isolate the impact of late-stage cues. SMT predicts a large divergence (high metric value), while PCS predicts a much smaller divergence due to its probabilistic early stopping.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    trial_map = {
        (1, 0, 0, 1, 1): 1,
        (1, 0, 0, 0, 0): 2,
        (1, 1, 0, 0, 1): 3,
        (1, 1, 0, 0, 0): 4,
        (0, 1, 1, 0, 0): 5,
        (0, 1, 1, 1, 1): 6,
        (1, 0, 1, 1, 0): 7,
        (1, 0, 1, 1, 1): 8
    }
    
    trial_types = []
    for a in data['option_a_ratings']:
        trial_types.append(trial_map.get(tuple(a), 0))
        
    data_eval = data.copy()
    data_eval['trial_type'] = trial_types
    data_eval['chose_A'] = (data_eval['response'] == 0).astype(float)
    
    means = data_eval.groupby('trial_type')['chose_A'].mean()
    p = {i: means.get(i, 0.5) for i in range(1, 9)}
    
    # Calculate the total divergence between trials that share early cues but differ at the end
    M = (p[1] - p[2]) + (p[3] - p[4]) + (p[6] - p[5]) + (p[8] - p[7])
    
    return float(M)


### RESULTS
- Predicted under pi_5 (simulated): 0.2717 (var=0.3038)
- Predicted under pi_4 (simulated): 0.8650 (var=0.6447)
- Observed on real data: 0.0033 (var=0.1397)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_choices = 0
    total_relevant = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_choices += 1
            total_relevant += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_choices += 1
            total_relevant += 1
    if total_relevant == 0:
        return 0.5
    return tallying_choices / total_relevant
```

**Observed (real) value:** 0.3400 (var=0.0108)
**Predicted under pi_5:** 0.1958 (var=0.0179)
**Predicted under pi_4:** 0.1733 (var=0.0166)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the specific trial where WADD and Tallying strongly disagree.
    # Trial: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    # WADD prefers A (validity sum 1.8 > 1.7) while Tallying prefers B (tally 3 > 2).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.3567 (var=0.0242)
**Predicted under pi_5:** 0.1983 (var=0.0197)
**Predicted under pi_4:** 0.2117 (var=0.0267)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_mismatch = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 1, 1, 1, 0) and b == (1, 1, 0, 0, 0, 0):
            tally_choices += (resp == 0)
            total_mismatch += 1
        elif a == (0, 0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0, 0):
            tally_choices += (resp == 0)
            total_mismatch += 1
        elif a == (1, 1, 0, 0, 0, 0) and b == (0, 0, 1, 1, 1, 0):
            tally_choices += (resp == 1)
            total_mismatch += 1
            
    return float(tally_choices / total_mismatch) if total_mismatch > 0 else 0.5

```

**Observed (real) value:** 0.3256 (var=0.0090)
**Predicted under pi_5:** 0.2006 (var=0.0262)
**Predicted under pi_4:** 0.1861 (var=0.0187)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_is_11000 = np.all(a_ratings == [1, 1, 0, 0, 0], axis=1)
    b_is_00111 = np.all(b_ratings == [0, 0, 1, 1, 1], axis=1)
    
    a_is_00111 = np.all(a_ratings == [0, 0, 1, 1, 1], axis=1)
    b_is_11000 = np.all(b_ratings == [1, 1, 0, 0, 0], axis=1)
    
    trial_type_1 = a_is_11000 & b_is_00111
    trial_type_2 = a_is_00111 & b_is_11000
    
    target_trials = trial_type_1 | trial_type_2
    
    if not np.any(target_trials):
        return 0.5
        
    responses = data['response'].values
    
    wadd_chosen = np.zeros_like(responses, dtype=bool)
    wadd_chosen[trial_type_1 & (responses == 0)] = True
    wadd_chosen[trial_type_2 & (responses == 1)] = True
    
    return float(np.mean(wadd_chosen[target_trials]))
```

**Observed (real) value:** 0.6717 (var=0.0180)
**Predicted under pi_5:** 0.7825 (var=0.0161)
**Predicted under pi_4:** 0.7950 (var=0.0171)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Validities are strictly decreasing from feature 0 to 4.
    # TTB evaluates features in order 0, 1, 2, 3, 4.
    diff = a_mat - b_mat
    
    # Weight features exponentially to find the first differing feature's sign
    weights = np.array([10000, 1000, 100, 10, 1])
    scores = diff.dot(weights)
    
    # Positive score means A dominates on the most valid discriminating feature (TTB chooses A -> 0)
    # Negative score means B dominates (TTB chooses B -> 1)
    ttb_choices = np.where(scores > 0, 0, 1)
    
    valid = scores != 0
    if not np.any(valid):
        return 0.5
        
    match = (ttb_choices[valid] == resp[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6817 (var=0.0051)
**Predicted under pi_5:** 0.6894 (var=0.0260)
**Predicted under pi_4:** 0.7200 (var=0.0240)

### Experiment 8
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where TTB prefers B (first differing feature favors B)
    # and WADD tends to prefer A (sum of features favors A)
    def is_compensatory_B(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # TTB prefers B?
        ttb_b = False
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_b = (b[i] > a[i])
                break
        # Sum prefers A?
        sum_a = sum(a) > sum(b)
        return ttb_b and sum_a
        
    mask = data.apply(is_compensatory_B, axis=1)
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6725 (var=0.0056)
**Predicted under pi_5:** 0.5980 (var=0.0328)
**Predicted under pi_4:** 0.6090 (var=0.0209)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    ttb_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Identify compensatory trials where TTB and WADD strongly disagree
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):
            ttb_consistent += (resp == 0)
            total += 1
        elif a == (0, 1, 1, 1, 1) and b == (1, 0, 0, 0, 0):
            ttb_consistent += (resp == 1)
            total += 1
        elif a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            ttb_consistent += (resp == 0)
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (0, 1, 0, 0, 0):
            ttb_consistent += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent / total)
```

**Observed (real) value:** 0.6778 (var=0.0087)
**Predicted under pi_5:** 0.5656 (var=0.0390)
**Predicted under pi_4:** 0.5678 (var=0.0307)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # The features are ordered by validity in the experimental design.
    # Find the first feature where A and B differ.
    diff = a_ratings - b_ratings
    mask = diff != 0
    first_diff_idx = np.argmax(mask, axis=1)
    
    row_indices = np.arange(len(data))
    first_diffs = diff[row_indices, first_diff_idx]
    
    # If A > B on the first discriminating feature, TTB favors A (response 0).
    # If B > A, TTB favors B (response 1).
    ttb_pred = np.where(first_diffs < 0, 1, 0)
    
    # Return the proportion of choices that are consistent with TTB.
    return float(np.mean(responses == ttb_pred))
```

**Observed (real) value:** 0.6696 (var=0.0069)
**Predicted under pi_5:** 0.6123 (var=0.0317)
**Predicted under pi_4:** 0.5910 (var=0.0248)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither theory consistently captures the observed data. In Experiment 1, Strategy Mixture Theory (pi_4) accurately predicts the negative metric (-0.03 vs observed -0.04), while Probabilistic Cue Search (pi_5) predicts a positive metric. However, in Experiment 2, pi_4 catastrophically fails by predicting a massive divergence (0.865) compared to the observed near-zero divergence (0.003), whereas pi_5 is closer (0.27) but still misses. In the older experiments (3-10), both models consistently underpredict the rate of compensatory choices (e.g., predicting ~0.20 when observed is ~0.35) and miss the mark on TTB-consistency (predicting ~0.58 when observed is ~0.67). Because both theories are fundamentally limited in capturing the nuance of how intermediate evidence and compensatory cues are integrated, a completely new theoretical framework is warranted.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Parallel Leaky Evidence Accumulation' theory. Unlike SMT (which mixes a discrete heuristic and a compensatory rule) or PCS (which sequentially searches and stops probabilistically), this new theory assumes all features are evaluated in parallel, but their evidence is accumulated into a relative decision variable with a leak or decay over time. More valid cues drive the accumulator faster, but the leak prevents perfect compensatory integration, inherently bounding the influence of many weak cues. This provides a natural, single-process mechanism that mimics both heuristic-like and compensatory-like behavior depending on the time course and validities of the cues, potentially capturing the near-zero divergence in Exp 2 and the balanced metric values in Exps 3-10."
}
```

## Usage

```json
{
  "prompt_token_count": 11014,
  "candidates_token_count": 397,
  "total_token_count": 12424
}
```
