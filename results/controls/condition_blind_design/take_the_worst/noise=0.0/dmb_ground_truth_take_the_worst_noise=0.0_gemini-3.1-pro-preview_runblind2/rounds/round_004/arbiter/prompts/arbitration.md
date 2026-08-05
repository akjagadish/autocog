# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Majority Rule with Reverse Lexicographic Tie-Breaking. People primarily use Tallying (equal weighting) to compare options, counting the number of features where one option strictly dominates the other. If one option wins on more features, it is chosen. However, if there is a tie in the tally, subjects break the tie by consulting cues in ASCENDING order of validity (least valid first). This perfectly captures why subjects follow Tallying in general, but systematically oppose both Weighted Additive (WADD) and Take The Best (TTB) predictions specifically on trials where Tallying results in a tie.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        scores = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        scores = np.array([0.0, 1.0])
    else:
        # Tie-breaker: Reverse Lexicographic (least valid cue first)
        val = np.asarray(parameters["validities"], dtype=float)
        # ASCENDING order of validity
        cue_order = np.argsort(val, kind="stable").tolist()
        
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Salience-Biased Exponential Weighted Additive Model

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Exponential salience boost over equal weights.
    # When alpha=0, all weights are 1.0 (pure Tallying).
    # As alpha increases, lower validity cues get exponentially larger weights,
    # allowing them to dominate on tie trials without breaking non-tie behavior.
    w = np.exp(alpha * (1.0 - val))
    
    # Score is the sum of weights for features where one option strictly dominates the other
    score_a = np.sum(w * (a > b))
    score_b = np.sum(w * (b > a))
    
    scores = np.array([score_a, score_b])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
**Validities (n_features=4):** [0.95, 0.67, 0.61, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 6: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 8: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  trial 10: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1]  B=[1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 13: A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 15: A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Advocated Theory (Majority Rule with Reverse Lexicographic Tie-Breaking), the predicted probability of choosing the favored option is exactly the same across all trials. This is because the core logit difference is always exactly 1.0, regardless of whether the choice is won by a large tallying margin (H trials) or won by a single tie-breaking cue (L trials). Thus, Acc(H) - Acc(L) will be exactly 0 in expectation.

Under the Competing Theory (Salience-Biased Exponential Weighted Additive), the logit difference scales with the sum of exponential weights. Trials with a large tallying margin (H trials) have a much larger weight difference than trials where options tie on most cues and are decided by the lowest validity cues (L trials). Thus, the Competing theory predicts a significantly higher confidence (and thus higher accuracy) on H trials compared to L trials, making Acc(H) - Acc(L) > 0.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # H_pairs: Trials where the Competing theory predicts a HUGE score difference.
    # T9: '1000' vs '1101' (Favored: '1101')
    # T13: '0110' vs '0000' (Favored: '0110')
    H_pairs = {
        frozenset(['1000', '1101']): '1101',
        frozenset(['0110', '0000']): '0110'
    }
    
    # L_pairs: Trials where the Competing theory predicts a very SMALL score difference.
    # T14 & T16: '0001' vs '0010' (Favored: '0001', as it wins on the lowest validity cue)
    L_pairs = {
        frozenset(['0001', '0010']): '0001'
    }
    
    h_correct, h_total = 0, 0
    l_correct, l_total = 0, 0
    
    for a, b, resp in zip(a_str, b_str, data['response']):
        pair = frozenset([a, b])
        chosen = a if resp == 0 else b
        
        if pair in H_pairs:
            if chosen == H_pairs[pair]:
                h_correct += 1
            h_total += 1
        elif pair in L_pairs:
            if chosen == L_pairs[pair]:
                l_correct += 1
            l_total += 1
            
    h_acc = h_correct / h_total if h_total > 0 else 0.5
    l_acc = l_correct / l_total if l_total > 0 else 0.5
    
    return float(h_acc - l_acc)

### RESULTS
- Predicted under pi_4 (simulated): 0.0183 (var=0.0201)
- Predicted under pi_6 (simulated): 0.0517 (var=0.0302)
- Observed on real data: -0.0267 (var=0.0132)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.73, 0.56, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric targets specific trials (Trials 2 and 9) where the two options are [0, 0, 1, 1] and [1, 1, 0, 1]. In these trials, Tallying favors the option with cues 0 and 1 (2 strict wins vs 1 strict win), so the Competing theory (Majority Rule) will almost always select it. However, the Advocated theory (Salience-Biased Exponential) assigns exponentially larger weights to less valid cues. Cue 2 has a high 'salience' weight that outstrips the combined weights of cues 0 and 1 for large alpha values. Therefore, the Advocated theory will frequently choose the option with cue 2, whereas the Competing theory will not. The metric calculates the proportion of times the option with cue 2 is chosen on these specific trials.

Source:
def metric(data: pd.DataFrame) -> float:
    is_target = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        if list(a) == [0, 0, 1, 1] and list(b) == [1, 1, 0, 1]:
            is_target.append(True)
        elif list(a) == [1, 1, 0, 1] and list(b) == [0, 0, 1, 1]:
            is_target.append(True)
        else:
            is_target.append(False)
            
    target_data = data[is_target]
    if len(target_data) == 0:
        return 0.0
        
    chose_cue_2 = 0
    for a, resp in zip(target_data['option_a_ratings'], target_data['response']):
        if list(a) == [0, 0, 1, 1]:
            if resp == 0:
                chose_cue_2 += 1
        else:
            if resp == 1:
                chose_cue_2 += 1
                
    return chose_cue_2 / len(target_data)

### RESULTS
- Predicted under pi_4 (simulated): 0.1183 (var=0.0114)
- Predicted under pi_6 (simulated): 0.7017 (var=0.0914)
- Observed on real data: 0.8267 (var=0.0233)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 6: TTB picks B (cue 0), Tallying picks A (A wins 2 cues to 1)
        if a == (0, 1, 1, 1) and b == (1, 1, 0, 0):
            ttb_match.append(1 if resp == 1 else 0)
        # Trials 8 & 16: TTB picks A (cue 0), Tallying picks B (B wins 2 cues to 1)
        elif a == (1, 1, 0, 0) and b == (0, 1, 1, 1):
            ttb_match.append(1 if resp == 0 else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.1733 (var=0.0250)
**Predicted under pi_4:** 0.1422 (var=0.0122)
**Predicted under pi_6:** 0.1356 (var=0.0157)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying tallies strict wins across all features
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_prefers_a = a_wins > b_wins
    
    # TTB checks the most valid cue first (cue 0, validity 0.95)
    ttb_prefers_b = b_mat[:, 0] > a_mat[:, 0]
    
    # Identify conflict trials where Tallying prefers A but TTB prefers B
    conflict_mask = tally_prefers_a & ttb_prefers_b
    
    # Return the proportion of times B was chosen on these conflict trials
    # Tallying will yield ~0.0, TTB will yield ~1.0
    if np.any(conflict_mask):
        return float(data.loc[conflict_mask, 'response'].mean())
    return 0.5

```

**Observed (real) value:** 0.1267 (var=0.0206)
**Predicted under pi_4:** 0.1133 (var=0.0160)
**Predicted under pi_6:** 0.1767 (var=0.0293)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.77, 0.8, 0.55])
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Check if it's a tie under Tallying
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tie_count += 1
            
            # WADD predictions
            wadd_a = np.sum(validities * a)
            wadd_b = np.sum(validities * b)
            
            if wadd_a > wadd_b:
                wadd_pred = 0
            elif wadd_b > wadd_a:
                wadd_pred = 1
            else:
                continue
                
            if row['response'] == wadd_pred:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return match_count / tie_count

```

**Observed (real) value:** 0.1240 (var=0.0095)
**Predicted under pi_4:** 0.1247 (var=0.0072)
**Predicted under pi_6:** 0.1287 (var=0.0099)

### Experiment 6
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.84, 0.64, 0.55])
    
    match_wadd = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying sees a tie
        if a_wins == b_wins:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            
            if wadd_a > wadd_b:
                pref = 0
            elif wadd_b > wadd_a:
                pref = 1
            else:
                continue
                
            match_wadd.append(1 if row['response'] == pref else 0)
            
    if len(match_wadd) == 0:
        return 0.5
        
    return float(np.mean(match_wadd))
```

**Observed (real) value:** 0.1589 (var=0.0095)
**Predicted under pi_4:** 0.1306 (var=0.0078)
**Predicted under pi_6:** 0.1628 (var=0.0129)

### Experiment 7
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tie_count += 1
            # Reverse Lexicographic tie-breaker
            # Validities: [0.95, 0.68, 0.88, 0.55]
            # Order of ascending validity: 3, 1, 2, 0
            winner = None
            for j in [3, 1, 2, 0]:
                if a[j] > b[j]:
                    winner = 0
                    break
                elif b[j] > a[j]:
                    winner = 1
                    break
            
            if winner is not None and row['response'] == winner:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
    return float(match_count / tie_count)
```

**Observed (real) value:** 0.8440 (var=0.0062)
**Predicted under pi_4:** 0.8393 (var=0.0087)
**Predicted under pi_6:** 0.8513 (var=0.0146)

### Experiment 8
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    tie_count = 0
    
    # Validities are [0.95, 0.83, 0.65, 0.55]
    # Reverse lexicographic order (least valid first) corresponds to indices [3, 2, 1, 0]
    cue_order = [3, 2, 1, 0]
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying results in a tie
        if a_wins == b_wins:
            tie_count += 1
            
            pred = None
            for j in cue_order:
                if a[j] > b[j]:
                    pred = 0
                    break
                elif b[j] > a[j]:
                    pred = 1
                    break
            
            if pred is not None and row['response'] == pred:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return float(match_count / tie_count)
```

**Observed (real) value:** 0.8422 (var=0.0202)
**Predicted under pi_4:** 0.8533 (var=0.0155)
**Predicted under pi_6:** 0.8444 (var=0.0233)

### Experiment 9
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(A > B, axis=1)
    b_wins = np.sum(B > A, axis=1)
    
    is_tie = a_wins == b_wins
    
    winner = np.zeros(len(data), dtype=int)
    winner[a_wins < b_wins] = 1
    
    tie_indices = np.where(is_tie)[0]
    for idx in tie_indices:
        a = A[idx]
        b = B[idx]
        w = 0
        for j in [3, 1, 2, 0]:
            if a[j] > b[j]:
                w = 0
                break
            if b[j] > a[j]:
                w = 1
                break
        winner[idx] = w
        
    correct = (data['response'].values == winner).astype(float)
    
    if np.sum(is_tie) == 0 or np.sum(~is_tie) == 0:
        return 0.0
        
    acc_nontie = np.mean(correct[~is_tie])
    acc_tie = np.mean(correct[is_tie])
    
    return float(acc_nontie - acc_tie)
```

**Observed (real) value:** -0.0524 (var=0.0076)
**Predicted under pi_4:** 0.0055 (var=0.0048)
**Predicted under pi_6:** -0.0146 (var=0.0170)

### Experiment 10
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
    a_t13 = (0, 1, 1, 0)
    b_t13 = (1, 0, 0, 1)
    
    # Trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
    a_t16 = (0, 0, 1, 0)
    b_t16 = (0, 0, 0, 1)
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t13_mask = (a_tuples == a_t13) & (b_tuples == b_t13)
    t16_mask = (a_tuples == a_t16) & (b_tuples == b_t16)
    
    if t13_mask.sum() == 0 or t16_mask.sum() == 0:
        return 0.0
        
    p_b_16 = data[t16_mask]['response'].mean()
    p_b_13 = data[t13_mask]['response'].mean()
    
    return float(p_b_16 - p_b_13)
```

**Observed (real) value:** 0.0200 (var=0.0385)
**Predicted under pi_4:** 0.0133 (var=0.0520)
**Predicted under pi_6:** 0.0233 (var=0.0233)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 2 (pi_6, Salience-Biased Exponential Weighted Additive Model) is significantly better at capturing the observed data across the experiments. In Experiment 2, Theory 1 completely fails to predict the high choice proportion for the target option (predicting 0.118 vs observed 0.826), whereas Theory 2 accurately predicts a strong preference (0.701). Across the other experiments, Theory 2 maintains parity or slightly outperforms Theory 1. Theory 1's rigid reliance on Majority Rule with a strict Reverse Lexicographic tie-breaker is too constrained and fails to capture the continuous, compensatory nature of the subjects' choices when tallying and cue validities strongly conflict.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Theory 1 with a brand-new theory that can serve as a stronger competitor to the Salience-Biased Exponential model. A promising direction is a 'Rank-Dependent Subjective Weighting Model'. Instead of using absolute validities or an exponential transformation of them, this theory would posit that subjects convert the provided validities into ordinal ranks and assign subjective weights based purely on these ranks (e.g., using a rank-decay function). This would capture the intuition that people often discard absolute probabilities in favor of ordinal heuristics, but do so in a fully compensatory (additive) manner rather than a strict lexicographic or tallying rule."
}
```

## Usage

```json
{
  "prompt_token_count": 11813,
  "candidates_token_count": 317,
  "total_token_count": 13304
}
```
