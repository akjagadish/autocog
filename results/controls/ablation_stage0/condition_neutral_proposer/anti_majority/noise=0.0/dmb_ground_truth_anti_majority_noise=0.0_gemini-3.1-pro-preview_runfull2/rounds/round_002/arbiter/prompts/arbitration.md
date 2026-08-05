# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
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
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_3
The Weighted Additive (WADD) model with non-linear weight scaling posits that decision makers integrate all available information by computing a weighted sum of features. However, the weighting of cues is not strictly proportional to their log-odds validity. Instead, decision makers apply a non-linear transformation (parameterized by gamma) to the log-odds, allowing them to stretch the weight differential. This permits WADD to approximate lexicographic (TTB-like) choice when gamma > 1, or more uniform (Tallying-like) weighting when gamma < 1, while remaining a fully compensatory integration process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid infinity
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # Compute weighted sum for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the weighted scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.92, 0.82, 0.72, 0.62, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To maximally distinguish Theory 1 (Mixture of TTB and linear WADD) from Theory 2 (Non-linear WADD), we exploit a fundamental difference in how they weight cues. Theory 1's WADD component uses raw validities as linear weights, while Theory 2 transforms validities into log-odds before applying a non-linear scaling parameter (gamma). Because raw validities are bounded between 0.5 and 1.0, their sum grows quickly, making linear WADD strongly favor options with more positive cues (Tallying-like behavior). In contrast, log-odds can be heavily skewed (e.g., a 0.92 validity has a much higher log-odds than a 0.72 validity), meaning Theory 2 can easily favor the single best cue even without a strict TTB mechanism, especially when gamma > 1. By pitting the highest-validity cue against various combinations of lower-validity cues, we create trials where the raw validity sum strongly opposes the highest cue, but the log-odds sum (depending on gamma) may still favor it. The pattern of choices across these specific trade-offs will differentiate a probabilistic mixture of step-function TTB and linear WADD from a deterministic, non-linearly scaled log-odds WADD.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits a perfect reversal in cue weighting between Linear WADD (used in Strategy Mixture) and Non-Linear WADD.

In Trial 8 (10011 vs 01100), Linear WADD sums raw validities and strictly favors Option A (2.06 vs 1.54). Since Take-The-Best (TTB) also favors A, the Strategy Mixture model predicts a very high P(A). In Trial 7 (11000 vs 00111), Linear WADD favors Option B (1.74 vs 1.86), opposing TTB. Consequently, the Strategy Mixture model's P(A) drops to roughly the mixture parameter 'alpha'. Thus, Strategy Mixture predicts P(A|T8) - P(A|T7) > 0.

In contrast, Non-Linear WADD transforms validities to log-odds. In Trial 7, Option A possesses the top two cues, which under log-odds scaling sum to a massive, insurmountable advantage over the bottom three cues for any gamma > 0. In Trial 8, Option B holds the strong second cue, making the contest much closer. Therefore, Non-Linear WADD predicts a stronger preference for A in Trial 7 than in Trial 8, yielding P(A|T8) - P(A|T7) < 0.

This robust sign flip clearly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_p_a(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        
        chose_a = 0
        total = 0
        
        if mask1.any():
            chose_a += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.any():
            chose_a += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total == 0:
            return 0.5
        return chose_a / total

    # Trial 8: Concordant for Strategy Mixture, weak advantage for NL-WADD
    p_a_t8 = get_p_a('10011', '01100')
    
    # Trial 7: Discordant for Strategy Mixture, massive advantage for NL-WADD
    p_a_t7 = get_p_a('11000', '00111')
    
    return float(p_a_t8 - p_a_t7)

### RESULTS
- Predicted under pi_4 (simulated): 0.1625 (var=0.0470)
- Predicted under pi_3 (simulated): -0.0175 (var=0.0350)
- Observed on real data: -0.3900 (var=0.1891)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.93, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To maximally distinguish Theory 1 (Non-linear log-odds WADD) from Theory 2 (Mixture of TTB and Linear WADD), we exploit the fundamental difference in how they weight cues. Theory 2's WADD component uses raw validities as linear weights, which sums up quickly and strongly favors options with more positive cues (similar to Tallying). Theory 1 transforms validities into log-odds before applying a non-linear scaling parameter (gamma). Because log-odds grow asymptotically as validity approaches 1.0, Theory 1 can strongly favor a single highly valid cue over multiple moderately valid cues, even without a strict TTB mechanism. By creating trials where the linear sum of raw validities strongly opposes the highest cue, but the log-odds sum supports it (or vice-versa depending on the exact combination), we can cleanly separate the probabilistic mixture of step-function TTB and linear WADD from the deterministic, non-linearly scaled log-odds WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the rate of choosing the TTB-favored option between Trial 7 and Trials 3/6. Under the Competing theory (Strategy Mixture), the agent mixes between TTB and Linear WADD. In both Trial 7 and Trials 3/6, Linear WADD strongly favors the non-TTB option. Thus, the agent only chooses the TTB option when using the TTB heuristic, which happens with probability 'alpha'. Because 'alpha' is consistent, the difference in TTB choice rate between these trials is near 0. Under the Advocated theory (Non-linear log-odds WADD), the decision depends on the non-linear scaling of log-odds validities. In Trial 7, the log-odds of the TTB option's cues are strong enough to outweigh the opposing cues for ANY value of the scaling parameter 'gamma', leading to a consistently high TTB choice rate (~1.0). In Trials 3/6, the TTB option has only the top cue against all four other cues; for smaller values of 'gamma', the sum of the four non-TTB cues outweighs the top cue, pulling the average TTB choice rate down significantly. Therefore, the Advocated theory predicts a large positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t7_a = (a_str == '10100') & (b_str == '01011')
    t7_b = (a_str == '01011') & (b_str == '10100')
    
    t36_a = (a_str == '10000') & (b_str == '01111')
    t36_b = (a_str == '01111') & (b_str == '10000')
    
    t7_ttb = pd.concat([
        data.loc[t7_a, 'response'] == 0,
        data.loc[t7_b, 'response'] == 1
    ])
    
    t36_ttb = pd.concat([
        data.loc[t36_a, 'response'] == 0,
        data.loc[t36_b, 'response'] == 1
    ])
    
    t7_rate = t7_ttb.mean() if len(t7_ttb) > 0 else 0.5
    t36_rate = t36_ttb.mean() if len(t36_ttb) > 0 else 0.5
    
    return float(t7_rate - t36_rate)

### RESULTS
- Predicted under pi_4 (simulated): 0.0000 (var=0.0209)
- Predicted under pi_3 (simulated): 0.2242 (var=0.1096)
- Observed on real data: -0.1983 (var=0.1423)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        
        if winner is not None:
            if row['response'] == winner:
                ttb_consistent += 1
            total += 1
            
    return float(ttb_consistent / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.6583 (var=0.0166)
**Predicted under pi_4:** 0.7150 (var=0.0215)
**Predicted under pi_3:** 0.7360 (var=0.0309)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, 1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    matches = (data['response'].values[mask] == tally_pred[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2750 (var=0.0193)
**Predicted under pi_4:** 0.4446 (var=0.0098)
**Predicted under pi_3:** 0.3937 (var=0.0151)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    comp_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: A has best cue, B has all others (compensatory is B)
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            comp_choices += (resp == 1)
            total += 1
        # Trial 2: B has best cue, A has next two (compensatory is A)
        elif a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            comp_choices += (resp == 0)
            total += 1
        # Trial 4: A has best available cue, B has next two (compensatory is B)
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            comp_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.0
    return float(comp_choices / total)
```

**Observed (real) value:** 0.2278 (var=0.0190)
**Predicted under pi_4:** 0.2911 (var=0.0192)
**Predicted under pi_3:** 0.3717 (var=0.0906)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has only 1 positive feature (the most valid one) 
    # and Option B has 3 or 4 positive features (all lower validity).
    # This corresponds exactly to Trial 1 and Trial 3 in the design.
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = (a_sum == 1) & (b_sum >= 3)
    if not mask.any():
        return 0.0
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2333 (var=0.0272)
**Predicted under pi_4:** 0.2808 (var=0.0246)
**Predicted under pi_3:** 0.3633 (var=0.0726)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, both Strategy Mixture (pi_4) and Non-Linear WADD (pi_3) struggle to capture the observed negative metric values in Experiments 1 and 2, though pi_4 performs slightly better on the older experiments (Exps 3-6). In Experiment 1, the observed metric is strongly negative (-0.3900), indicating a much stronger preference for the TTB-favored option when it holds the top two cues compared to when it holds the first, fourth, and fifth. In Experiment 2, the metric is again negative (-0.1983), which surprisingly implies that subjects chose the TTB option more often when it was opposed by four lower-validity cues than when it was opposed by three. Neither a simple probabilistic mixture of TTB/Linear-WADD nor a Non-Linear WADD can predict this pattern. This suggests that decision-makers are not just weighting cues or mixing strategies, but might be using a threshold-based lexicographic rule (like Lexicographic Semiorder) or a tallying-based difference evaluation that heavily discounts cues when too many are presented, or incorporates cue interactions.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory based on a 'Lexicographic Semiorder' or 'Cue Difference Threshold' model. In this theory, decision-makers compare options cue-by-cue in order of validity. However, instead of a strict TTB rule, an option is only chosen if the validity of its cue exceeds the alternative by a certain threshold, or if the sheer number of opposing cues doesn't cross a 'tallying deficit' limit. If the top cue's advantage is not seen as decisive enough (e.g., when facing a concentrated block of moderately high cues), they fall back to a compensatory or tallying process. This can explain the non-monotonic preference reversals seen in Experiments 1 and 2 where extreme cue configurations lead to counterintuitive choice rates."
}
```

## Usage

```json
{
  "prompt_token_count": 7632,
  "candidates_token_count": 439,
  "total_token_count": 9379
}
```
