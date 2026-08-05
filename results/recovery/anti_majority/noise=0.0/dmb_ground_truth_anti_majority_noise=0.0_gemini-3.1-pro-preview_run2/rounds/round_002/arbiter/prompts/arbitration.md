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
Strategy Mixture Theory: Decision-makers do not uniformly rely on a single decision strategy. Instead, they exhibit variability in strategy selection, probabilistically mixing between a non-compensatory heuristic (Take The Best) and a compensatory strategy (Weighted Additive, WADD). The mixture parameter 'alpha' dictates the probability of using TTB on any given trial, while '1 - alpha' is the probability of using WADD. This allows the model to capture intermediate choice proportions that neither pure strategy can explain on its own, with a bias toward the empirically dominant non-compensatory strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Take The Best (TTB) prediction
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
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # WADD prediction
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Strategy Mixture
    alpha = float(parameters["alpha"])
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To dissociate Strategy Mixture Theory from the Generalized WADD theory, we exploit how each model produces intermediate choice probabilities. Strategy Mixture Theory assumes a probabilistic mixture of pure TTB and pure WADD, predicting relatively constant intermediate choice proportions (e.g., matching the mixing parameter alpha) across any trial where TTB and WADD strongly conflict, regardless of the exact margin of the WADD sum. In contrast, Generalized WADD achieves intermediate probabilities either via high noise (low beta) or by precisely tuning the gamma parameter to make the exponentiated sums of the two options nearly equal. By presenting multiple distinct conflict trials with varying differences in raw feature sums, Generalized WADD cannot simultaneously equate the sums for all trials with a single gamma parameter. Consequently, Generalized WADD will predict extreme probabilities on some conflict trials and intermediate on others, whereas Strategy Mixture Theory predicts consistent intermediate probabilities across all such conflict trials.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
For Strategy Mixture Theory, the probability of choosing option A on any trial where TTB prefers A and WADD prefers B is relatively constant (approximately equal to the mixture parameter alpha). Thus, the variance of the mean choice proportions of A across these three conflict trials should be low (driven mainly by binomial noise). For Generalized WADD, equating the exponentiated sums across these three distinct trial geometries with a single gamma parameter is impossible, leading to highly variable predicted choice proportions across the three trials. By computing the variance of P(A) across the three conflict trials per subject and taking the average, we strongly discriminate the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trials 1, 2, and 3 are conflict trials where TTB prefers A and WADD prefers B.
    target_trials = ['10000', '01000', '10100']
    df_conflict = data[data['a_str'].isin(target_trials)].copy()
    df_conflict['chose_A'] = (df_conflict['response'] == 0).astype(float)
    
    subj_vars = []
    for subj, subj_df in df_conflict.groupby('subject_id'):
        means = subj_df.groupby('a_str')['chose_A'].mean()
        if len(means) == 3:
            subj_vars.append(means.var(ddof=1))
            
    if not subj_vars:
        return 0.0
        
    return float(np.mean(subj_vars))

### RESULTS
- Predicted under pi_4 (simulated): 0.0096 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0331 (var=0.0015)
- Observed on real data: 0.0574 (var=0.0093)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate Generalized WADD from Strategy Mixture Theory, we manipulate the margin by which the compensatory WADD strategy opposes the non-compensatory TTB strategy. Strategy Mixture Theory assumes choices are a stable probabilistic coin-flip between pure TTB and pure WADD. Therefore, on any trial where TTB strongly favors Option A and WADD strongly favors Option B, Strategy Mixture predicts a relatively constant intermediate choice probability driven by the mixing parameter 'alpha'. In contrast, Generalized WADD integrates all features non-linearly. A single 'gamma' parameter cannot flatten the varying evidence margins across different trials. By presenting a spectrum of conflicts—ranging from the top cue being opposed by all four remaining cues, to being opposed by only two—Generalized WADD is forced to predict a graded shift in probabilities, whereas Strategy Mixture predicts a step-function or constant mixture across these conflicts.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Strategy Mixture Theory (SMT) and Generalized WADD predict opposite orderings for the probability of choosing Option A in Trial 2 versus Trial 5. In Trial 2, TTB strongly favors A while linear WADD favors B, leading SMT to predict P(A) = alpha. In Trial 5, TTB favors A but linear WADD yields a perfect tie, leading SMT to predict P(A) = 0.5 + 0.5 * alpha. Since alpha <= 1.0, SMT strictly predicts that P(A) in Trial 2 is LESS than or equal to P(A) in Trial 5 (a negative difference). Conversely, Generalized WADD integrates features non-linearly. Because the best cue in Trial 2 (validity 0.95) is much stronger than the best cue in Trial 5 (validity 0.85), exponentiation (gamma > 1) causes the evidence margin favoring A to be substantially larger in Trial 2 than in Trial 5. Therefore, Generalized WADD predicts P(A) in Trial 2 will be GREATER than in Trial 5 (a positive difference). This qualitative reversal provides a highly robust, low-variance discriminant.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    data['chose_A'] = 1 - data['response']
    
    t2_mask = (data['A_str'] == '10001') & (data['B_str'] == '01100')
    t5_mask = (data['A_str'] == '01001') & (data['B_str'] == '00110')
    
    p_a_t2 = data[t2_mask]['chose_A'].mean() if t2_mask.sum() > 0 else 0.5
    p_a_t5 = data[t5_mask]['chose_A'].mean() if t5_mask.sum() > 0 else 0.5
    
    return float(p_a_t2 - p_a_t5)

### RESULTS
- Predicted under pi_4 (simulated): -0.0338 (var=0.0268)
- Predicted under pi_3 (simulated): 0.1713 (var=0.0375)
- Observed on real data: 0.0175 (var=0.0095)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1
            
        # Only consider trials where TTB and Tallying make opposite predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            disagree_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if disagree_count == 0:
        return 0.5
    return ttb_match_count / disagree_count

```

**Observed (real) value:** 0.7581 (var=0.0332)
**Predicted under pi_4:** 0.6731 (var=0.0154)
**Predicted under pi_3:** 0.6753 (var=0.0250)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pref)
        
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2506 (var=0.0294)
**Predicted under pi_4:** 0.2844 (var=0.0129)
**Predicted under pi_3:** 0.2033 (var=0.0222)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    # TTB prediction: heavily weights the first discriminating cue
    # Using powers of 10 ensures strict lexicographical ordering (TTB logic)
    weights = np.array([1000, 100, 10, 1])
    ttb_score = np.dot(diff, weights)
    ttb_pred = np.where(ttb_score > 0, 0, 1)
    
    # WADD prediction (with gamma=1, i.e., linear integration)
    wadd_score = np.dot(diff, val)
    wadd_pred = np.where(wadd_score > 0, 0, 1)
    
    # Identify trials where TTB and baseline WADD disagree
    divergent = ttb_pred != wadd_pred
    
    if not np.any(divergent):
        return 0.5
        
    responses = data['response'].values
    # Calculate proportion of choices matching TTB on these critical trials
    ttb_match = (responses[divergent] == ttb_pred[divergent]).mean()
    
    return float(ttb_match)
```

**Observed (real) value:** 0.7236 (var=0.0302)
**Predicted under pi_4:** 0.6902 (var=0.0128)
**Predicted under pi_3:** 0.7036 (var=0.0145)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.3975 (var=0.0240)
**Predicted under pi_4:** 0.2710 (var=0.0132)
**Predicted under pi_3:** 0.3127 (var=0.0167)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither Strategy Mixture Theory (pi_4) nor Generalized WADD (pi_3) perfectly captures the observed data, though they each capture different aspects. In Experiment 1, the observed variance across conflict trials is high (0.0574), which pi_3 (0.0331) approximates better than pi_4 (0.0096), contradicting pi_4's assumption of constant mixture probabilities. In Experiment 2, the observed difference in probabilities between two specific conflict trials is near zero (0.0175). pi_4 predicts a negative difference (-0.0338), while pi_3 predicts a massive positive difference (0.1713). The reality is that people exhibit variance across conflict configurations (unlike pure Strategy Mixture) but do not wildly scale their choice probabilities based on the exact non-linear evidence margins (unlike Generalized WADD). This suggests a fundamental limitation in both static probabilistic mixture and static non-linear integration.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Strategy Mixture Theory with a Sequential Evidence Accumulation theory. In this theory, decision-makers sample cues sequentially (perhaps ordered by validity or randomly) and accumulate evidence for each option until a threshold is reached. If the threshold is reached early, it mimics Take-The-Best; if evidence accumulates slowly due to conflicting cues, it mimics WADD or Tallying. This natural stopping rule generates variable choice probabilities across different conflict geometries without requiring non-linear scaling of validities, naturally capturing the moderate variance seen in Exp 1 and the muted differences seen in Exp 2."
}
```

## Usage

```json
{
  "prompt_token_count": 6676,
  "candidates_token_count": 364,
  "total_token_count": 7736
}
```
