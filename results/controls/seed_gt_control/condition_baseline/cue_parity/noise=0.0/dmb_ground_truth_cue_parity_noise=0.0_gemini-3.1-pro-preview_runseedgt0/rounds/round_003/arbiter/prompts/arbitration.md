# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Tallying-Biased Parity (Strict Parity Dominance): Decision makers primarily rely on a strict Cue-Parity rule (preferring the option with an odd number of winning cues). However, instead of lapsing into purely uniform random noise, their secondary considerations consist of occasionally falling back to a standard validity-weighted tallying strategy. To ensure that tallying acts only as a weak bias and does not override parity in cases of option dominance, the fallback probability is strictly bounded to a low range.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying-Biased Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Core Strategy: Strict Cue-Parity
    a_wins = int(np.sum(a > b))
    parity_winner = 0 if (a_wins % 2 == 1) else 1
    parity_scores = np.array([1.0, 0.0]) if parity_winner == 0 else np.array([0.0, 1.0])
    
    # Fallback Strategy: Validity-weighted Tallying
    val_a = np.sum(validities * (a > b))
    val_b = np.sum(validities * (b > a))
    tally_scores = np.array([val_a, val_b])
    
    beta_parity = float(parameters["beta_parity"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Parity probabilities via softmax
    z_p = beta_parity * (parity_scores - parity_scores.max())
    e_p = np.exp(z_p)
    p_parity = e_p / e_p.sum()
    
    # Tallying probabilities via softmax
    z_t = beta_tally * (tally_scores - tally_scores.max())
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Mixture of the two strategies
    return (1.0 - epsilon) * p_parity + epsilon * p_tally


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Cue-Parity from Tallying-Biased Parity, we exploit the latter's fallback mechanism. Cue-Parity predicts that the probability of choosing Option A depends entirely and exclusively on whether the number of A-wins is odd or even, remaining completely invariant to the validities of those features or the number of B-wins. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By holding the number of A-wins constant (e.g., exactly 1 win, which is odd) while drastically changing the validity-weighted tally (e.g., A wins on 1 high-validity cue vs. A wins on 1 low-validity cue while B wins on 3 high-validity cues), Tallying-Biased Parity predicts a systematic shift in choice probabilities between the trials. Cue-Parity predicts identical choice probabilities across these paired trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By comparing trials where Cue-Parity makes identical predictions but the validity-weighted tally strongly shifts from favoring Option A to Option B, we can isolate the effect of the tallying fallback mechanism. Cue-Parity predicts identical probabilities (difference around 0), while Tallying-Biased Parity predicts a systematic positive shift in the choice probabilities across these paired trials.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['a_str'] == '10000') & (data['b_str'] == '00000')
    t2_mask = (data['a_str'] == '00010') & (data['b_str'] == '11100')
    t3_mask = (data['a_str'] == '11000') & (data['b_str'] == '00000')
    t4_mask = (data['a_str'] == '00011') & (data['b_str'] == '11100')
    
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    p_a_t4 = 1.0 - data.loc[t4_mask, 'response'].mean()
    
    diff1 = (p_a_t1 - p_a_t2) if pd.notna(p_a_t1) and pd.notna(p_a_t2) else 0.0
    diff2 = (p_a_t3 - p_a_t4) if pd.notna(p_a_t3) and pd.notna(p_a_t4) else 0.0
    
    return float(diff1 + diff2)

### RESULTS
- Predicted under pi_1 (simulated): -0.0625 (var=0.0234)
- Predicted under pi_5 (simulated): 0.2612 (var=0.0429)
- Observed on real data: 0.0250 (var=0.0422)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.95, 0.55, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying-Biased Parity from pure Cue-Parity, we exploit the alignment between the primary Cue-Parity rule and the secondary validity-weighted tallying fallback. Pure Cue-Parity predicts that the probability of choosing A depends exclusively on whether the number of A-wins is odd or even, plus a constant uniform lapse rate. It is completely blind to the validities of the winning features. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By designing pairs of trials where the unweighted parity of A-wins is identical (e.g., A wins 2 features, which is Even -> Parity favors B) but the validity-weighted tally is reversed (Trial A: A wins on 2 high-validity cues vs B's 3 low-validity cues; Trial B: A wins on 2 low-validity cues vs B's 2 high-validity cues), we can observe a systematic shift. Tallying-Biased Parity predicts a larger choice share for the parity-disfavored option when the tallying fallback supports it. Pure Cue-Parity predicts perfectly flat choice probabilities across these paired trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing option A between pairs of trials that have the exact same Cue-Parity status but opposite Tallying (validity-weighted) status. Pure Cue-Parity predicts these differences will be exactly zero on average, as it ignores validities. Tallying-Biased Parity predicts a strictly positive value because the fallback tallying strategy will systematically increase the probability of choosing A when the high-validity cues favor A compared to when they favor B.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Identify the trial types based on option A's features.
    # T1: sum=1, A[0]=1 (Parity favors A, Tally favors A)
    # T2: sum=1, A[0]=0 (Parity favors A, Tally favors B)
    # T3: sum=2, A[0]=1 (Parity favors B, Tally favors A)
    # T4: sum=2, A[0]=0 (Parity favors B, Tally favors B)
    # T5: sum=3, A[0]=1 (Parity favors A, Tally favors A)
    # T6: sum=3, A[0]=0 (Parity favors A, Tally favors B)
    
    def get_trial_type(row):
        a = row['option_a_ratings']
        s = sum(a)
        if s == 1:
            return 'T1' if a[0] == 1 else 'T2'
        elif s == 2:
            return 'T3' if a[0] == 1 else 'T4'
        elif s == 3:
            return 'T5' if a[0] == 1 else 'T6'
        return 'Other'
        
    data = data.copy()
    data['trial_type'] = data.apply(get_trial_type, axis=1)
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # Calculate mean P(choose A) for each trial type
    means = data.groupby('trial_type')['chose_A'].mean()
    
    def get_mean(t):
        return means.get(t, 0.0)
        
    # Difference in P(choose A) when Tally favors A vs when Tally favors B,
    # holding Parity constant.
    diff_1 = get_mean('T1') - get_mean('T2')
    diff_2 = get_mean('T3') - get_mean('T4')
    diff_3 = get_mean('T5') - get_mean('T6')
    
    return float(diff_1 + diff_2 + diff_3)


### RESULTS
- Predicted under pi_1 (simulated): -0.0017 (var=0.0649)
- Predicted under pi_5 (simulated): 0.2767 (var=0.0857)
- Observed on real data: 0.1067 (var=0.0581)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_wins = data.apply(lambda row: sum(a > b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    b_wins = data.apply(lambda row: sum(b > a for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    
    mask_123 = (a_wins.isin([2, 4])) & (b_wins.isin([0, 1]))
    mask_45 = (a_wins == 1) & (b_wins.isin([2, 3]))
    
    val_123 = data.loc[mask_123, 'response'].mean() if mask_123.any() else 0.5
    val_45 = data.loc[mask_45, 'response'].mean() if mask_45.any() else 0.5
    
    return float(val_123 - val_45)
```

**Observed (real) value:** 0.7539 (var=0.0403)
**Predicted under pi_1:** 0.7083 (var=0.0555)
**Predicted under pi_5:** 0.6650 (var=0.0488)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if x > y), axis=1)
    b_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if y > x), axis=1)
    
    mask = a_wins == b_wins
    if not mask.any():
        return 0.5
        
    tied_data = data[mask]
    tied_a_wins = a_wins[mask]
    
    # Cue-Parity predicts A (0) if a_wins is odd, B (1) if a_wins is even
    cp_preds = (tied_a_wins % 2 == 0).astype(int)
    
    matches = (tied_data['response'] == cp_preds).astype(float)
    return float(matches.mean())
```

**Observed (real) value:** 0.8178 (var=0.0117)
**Predicted under pi_1:** 0.8596 (var=0.0092)
**Predicted under pi_5:** 0.9009 (var=0.0082)

### Experiment 5
**Design**
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract options as 2D arrays
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Compute strict A-wins across all 5 cues
    a_wins = np.sum(a_mat > b_mat, axis=1)
    
    # Cue-Parity rule: Odd A-wins -> A (0), Even A-wins -> B (1)
    cp_preds = (a_wins % 2 == 0).astype(int)
    
    # Compute agreement between subjects' responses and Cue-Parity predictions
    responses = data['response'].values
    return float(np.mean(responses == cp_preds))
```

**Observed (real) value:** 0.8358 (var=0.0105)
**Predicted under pi_1:** 0.8479 (var=0.0101)
**Predicted under pi_5:** 0.9079 (var=0.0059)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        top3_awins = np.sum(a[:3] > b[:3])
        all_awins = np.sum(a > b)
        
        if (top3_awins % 2) != (all_awins % 2):
            pred = 0 if top3_awins % 2 == 1 else 1
            matches.append(1.0 if row['response'] == pred else 0.0)
            
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1367 (var=0.0064)
**Predicted under pi_1:** 0.1700 (var=0.0169)
**Predicted under pi_5:** 0.0760 (var=0.0021)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the unweighted sum of cues favoring Option A (since B is all 0s)
    a_sums = data['option_a_ratings'].apply(sum)
    
    # Responses: 0 means A was chosen, 1 means B was chosen
    is_a = (data['response'] == 0).astype(float)
    is_b = (data['response'] == 1).astype(float)
    
    # For odd sums (5 and 7), Cue-Parity predicts A with equal strength.
    # For CW-Parity, sum=5 is pushed towards the half-integer dead zone by validities,
    # reducing P(A), while sum=7 is barely affected.
    p_a_7 = is_a[a_sums == 7].mean()
    p_a_5 = is_a[a_sums == 5].mean()
    
    # For even sums (4 and 6), Cue-Parity predicts B with equal strength.
    # For CW-Parity, sum=6 is pushed towards the half-integer dead zone,
    # reducing P(B), while sum=4 is unaffected.
    p_b_4 = is_b[a_sums == 4].mean()
    p_b_6 = is_b[a_sums == 6].mean()
    
    diff1 = (p_a_7 - p_a_5) if pd.notna(p_a_7) and pd.notna(p_a_5) else 0.0
    diff2 = (p_b_4 - p_b_6) if pd.notna(p_b_4) and pd.notna(p_b_6) else 0.0
    
    return float(diff1 + diff2)
```

**Observed (real) value:** 0.0525 (var=0.0125)
**Predicted under pi_1:** 0.0006 (var=0.0252)
**Predicted under pi_5:** 0.0037 (var=0.0131)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    high_wins = a_wins_mask[:, :5].sum(axis=1)
    
    # T1, T2: 5 wins, strictly uniform validities
    mask_12 = (a_wins == 5) & ((high_wins == 0) | (high_wins == 5))
    
    # T3, T4: 4 wins, strictly uniform validities
    mask_34 = (a_wins == 4) & ((high_wins == 0) | (high_wins == 4))
    
    if mask_12.sum() == 0 or mask_34.sum() == 0:
        return 0.0
        
    # h is the number of times Option A was chosen
    h_12 = (data['response'].values[mask_12] == 0).sum()
    m_12 = mask_12.sum() - h_12
    
    h_34 = (data['response'].values[mask_34] == 0).sum()
    m_34 = mask_34.sum() - h_34
    
    # Compute log-odds of choosing A with Laplace smoothing
    lo_12 = np.log((h_12 + 0.5) / (m_12 + 0.5))
    lo_34 = np.log((h_34 + 0.5) / (m_34 + 0.5))
    
    # We return the negative sum of these log-odds
    return float(-(lo_12 + lo_34))
```

**Observed (real) value:** -0.1572 (var=0.8155)
**Predicted under pi_1:** 0.1336 (var=1.0828)
**Predicted under pi_5:** -2.1229 (var=1.2146)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of the experiments (especially 1, 3, 4, 5, 6, and 8), Theory 1 (Pure Cue-Parity) provides predictions that are significantly closer to the observed data than Theory 2 (Tallying-Biased Parity). Theory 2 consistently over-predicts the influence of the validity-weighted tallying fallback (e.g., in Experiment 1 and 8, where its predictions are wildly off). However, Experiment 2 shows a small but non-zero positive shift (0.1067) where pure Cue-Parity predicts zero, indicating that validities do have some minor influence that pure Cue-Parity fails to capture. Rather than a tallying fallback, this influence might stem from how cues are selected for the parity count.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a 'Thresholded Parity' or 'Top-K Parity' theory. Instead of falling back to a completely different strategy like tallying, subjects might apply the strict Cue-Parity rule but only to a subset of cues that they consider 'important' (e.g., the top K most valid cues, or cues with validities above a certain threshold). Low-validity cues are ignored and do not contribute to the parity count. This would introduce a validity-dependent effect (as low-validity cues failing to flip the parity would explain the small deviations from pure parity) without requiring an additive tallying mixture that grossly overestimates the validity effect."
}
```

## Usage

```json
{
  "prompt_token_count": 9475,
  "candidates_token_count": 349,
  "total_token_count": 11031
}
```
