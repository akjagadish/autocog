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
People make decisions between options by computing a fully compensatory, weighted sum of each option's features, using the provided cue validities as weights (Weighted Additive Model, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), decision-makers integrate all available evidence proportionally to its reliability. Response selection is probabilistic, modeled as a softmax over the weighted sums with an inverse temperature parameter and an independent lapse rate. The inverse temperature parameter is constrained to very low values to reflect the high degree of stochasticity observed in human choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {stim.shape[1]}."
        )

    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Noisy Take-The-Best: Decision-makers evaluate options lexicographically, starting with the most valid cue and stopping at the first cue that discriminates between the options. However, the execution of this heuristic is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic TTB predictions with heavy noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.ones(2) / 2.0
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.94, 0.91, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 2: A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 6: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 13: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 14: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 16: A=[1, 0, 0, 1]  B=[1, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the average difference in WADD score between the chosen option and the unchosen option. The Advocated Theory (WADD) has a relatively low lapse rate (epsilon <= 0.5), meaning subjects will reliably choose the option with the higher weighted sum, resulting in a significantly positive score difference. The Competing Theory (Noisy TTB) posits a massive lapse rate (epsilon >= 0.7), meaning choices are predominantly random, which will drag the average score difference much closer to zero.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.94, 0.91, 0.55])
    
    score_a = data['option_a_ratings'].apply(lambda x: np.sum(val * np.array(x)))
    score_b = data['option_b_ratings'].apply(lambda x: np.sum(val * np.array(x)))
    
    chosen_score = np.where(data['response'] == 0, score_a, score_b)
    unchosen_score = np.where(data['response'] == 0, score_b, score_a)
    
    return float(np.mean(chosen_score - unchosen_score))

### RESULTS
- Predicted under pi_3 (simulated): 0.3962 (var=0.0554)
- Predicted under pi_4 (simulated): 0.0442 (var=0.0232)
- Observed on real data: -0.0231 (var=0.3337)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.92, 0.87, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 10: A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 11: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  trial 14: A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  trial 15: A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
In the experimental design, Trial 1 (A=[0, 1, 1, 0] vs B=[1, 0, 0, 0]) is the unique trial where the deterministic predictions of WADD and TTB diverge. WADD favors Option A because the combined weight of cues 1 and 2 (0.92 + 0.87 = 1.79) outweighs cue 0 (0.95). In contrast, TTB evaluates lexicographically, checking cue 0 first, which strictly favors Option B. By measuring the choice proportion of Option A on this specific trial, we obtain a precise estimate of which decision rule is driving behavior, effectively separating the two theories despite their high noise/lapse rates.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 1, which is the only trial where WADD and TTB have opposite deterministic predictions.
    # Trial 1: A = [0, 1, 1, 0], B = [1, 0, 0, 0]
    # WADD favors A (Score A = 0.92 + 0.87 = 1.79 > Score B = 0.95)
    # TTB favors B (Cue 0 discriminates for B)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 1, 0]) & \
                 data['option_b_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0])
                 
    if not is_trial_1.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_trial_1, 'response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.6500 (var=0.0447)
- Predicted under pi_4 (simulated): 0.4533 (var=0.0445)
- Observed on real data: 0.5200 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where TTB and Tallying make opposite deterministic predictions.
    # Specifically, Trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
    # TTB: Feature 0 (highest validity) favors A, so TTB chooses A.
    # Tallying: A wins on Feature 0, but B wins on Features 1 and 2. Tallying chooses B.
    is_reversal_trial = data.apply(
        lambda row: tuple(row['option_a_ratings']) == (1, 0, 0, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 1),
        axis=1
    )
    if is_reversal_trial.sum() == 0:
        return 0.5
    
    # Return the proportion of times option A was chosen on this trial.
    # TTB will have a value close to 1, while Tallying will have a value close to 0.
    return float((data.loc[is_reversal_trial, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4000 (var=0.2400)
**Predicted under pi_3:** 0.3467 (var=0.0398)
**Predicted under pi_4:** 0.5433 (var=0.0453)

### Experiment 4
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_trial_8 = (a_str == '0101') & (b_str == '1100')
    is_trial_11 = (a_str == '1101') & (b_str == '1011')
    
    ttb_consistent_8 = data[is_trial_8]['response'] == 1
    ttb_consistent_11 = data[is_trial_11]['response'] == 0
    
    total_trials = len(ttb_consistent_8) + len(ttb_consistent_11)
    if total_trials == 0:
        return 0.5
        
    ttb_consistent_count = ttb_consistent_8.sum() + ttb_consistent_11.sum()
    return float(ttb_consistent_count / total_trials)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.5383 (var=0.0262)
**Predicted under pi_4:** 0.5333 (var=0.0169)

### Experiment 5
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['pair'] = data['A_str'] + data['B_str']
    
    tally_0 = {'11010001', '01101000', '11110000', '11111101', '00110001', '11100101', '10111001', '01100100'}
    tally_1 = {'10101011', '00010101', '10100111', '11001101', '11001110'}
    
    is_0 = data['pair'].isin(tally_0)
    is_1 = data['pair'].isin(tally_1)
    
    matches = (is_0 & (data['response'] == 0)) | (is_1 & (data['response'] == 1))
    valid = is_0 | is_1
    
    return float(matches.sum() / valid.sum()) if valid.sum() > 0 else 0.5
```

**Observed (real) value:** 0.5046 (var=0.0133)
**Predicted under pi_3:** 0.6451 (var=0.0098)
**Predicted under pi_4:** 0.5444 (var=0.0027)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    favored_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 10: A=[0, 0, 1, 1], B=[0, 1, 0, 0]
        # Tallying difference = 1 (A wins 2 features, B wins 1)
        # WADD difference = 0.27 (A=1.21, B=0.94)
        if a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            favored_chosen.append(1 if resp == 0 else 0)
            
        # Trial 14: A=[1, 1, 0, 0], B=[1, 0, 1, 1]
        # Tallying difference = 1 (B wins 2 features, A wins 1)
        # WADD difference = 0.27 (B=2.16, A=1.89)
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            favored_chosen.append(1 if resp == 1 else 0)
            
    if not favored_chosen:
        return 0.5
    return float(np.mean(favored_chosen))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.5483 (var=0.0225)
**Predicted under pi_4:** 0.4033 (var=0.0218)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed data consistently points to aggregate behavior that is remarkably close to random guessing (metrics hover around 0.5 or 0 difference). Theory 2 (Noisy TTB) captures this high stochasticity better than Theory 1 (WADD), which consistently overpredicts the extent to which subjects use all validities to make deterministic choices. However, in Experiment 3, where TTB and Tallying diverge, the observed metric falls to 0.40, leaning away from TTB (which predicts >0.5) and towards Tallying (which predicts <0.5). Similarly, in Experiment 6, TTB-based predictions miss the 0.5 mark more than WADD. This suggests that subjects are not using validities as weights (WADD) nor are they using lexicographic rules (TTB). Instead, they are likely either guessing randomly or using a very simple Equal-Weights (Tallying) heuristic, counting the number of positive features and ignoring the given validities entirely, combined with a high lapse rate.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Noisy Tallying' (Equal Weights) theory. The theory should posit that subjects ignore the specific cue validities provided in the instructions, as integrating them is too cognitively demanding. Instead, they simply count the number of positive features (1s) for each option and choose the one with the higher total. To account for the observed metrics being very close to 0.5 across all experiments, the model must include a very high lapse rate (epsilon parameter centered around 0.8 to 0.95), reflecting that subjects frequently guess randomly."
}
```

## Usage

```json
{
  "prompt_token_count": 7657,
  "candidates_token_count": 379,
  "total_token_count": 9041
}
```
