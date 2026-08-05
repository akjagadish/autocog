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
People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
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
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Heuristic Mixture (Strategy Selection) Theory: Decision-makers do not rely on a single strategy; instead, they probabilistically select between different heuristics on each trial. Specifically, they mix between a non-compensatory, lexicographic strategy (Take-The-Best) and a non-weighted, compensatory strategy (Tallying). A mixture parameter governs the probability of using TTB versus Tallying, allowing the model to capture intermediate choice behaviors by blending one-reason decision making with simple feature-counting.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        ttb_scores = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        ttb_scores = np.array([0.0, 1.0])
    else:
        ttb_scores = np.array([0.0, 0.0])
        
    beta = float(parameters["beta"])
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy Mixture
    p_mix = float(parameters["p_mix"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate WADD from a Heuristic Mixture of Take-The-Best (TTB) and Tallying, we exploit the non-linear compensatory nature of WADD. The Heuristic Mixture model can only produce choice probabilities that are linear interpolations between TTB (which strictly chooses based on the highest validity cue) and Tallying (which strictly counts the number of winning cues). We construct 'Tie-Breaker' trials where Tallying predicts a tie (equal number of winning cues for both options) and TTB strongly favors Option B. Consequently, the Mixture model MUST favor Option B (or predict a tie if p_mix = 0). However, WADD with a low gamma parameter (which flattens validities) will actually favor Option A, because the sum of multiple moderately-high validities for Option A exceeds the sum of the highest and lowest validities for Option B. This creates a stark qualitative and quantitative divergence where WADD predicts a preference reversal that the Mixture model mathematically cannot capture.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on 'Tie-Breaker' trials (Trials 1 and 6) where Option A has cues 2 and 3, and Option B has cue 1 and either 4 or 5. For the Heuristic Mixture model, Tallying predicts a tie (2 vs 2 cues) while Take-The-Best strictly favors Option B (which has the highest-validity cue). Therefore, the Mixture model will mathematically cap the probability of choosing Option A at 0.5. In contrast, the Weighted Additive (WADD) model, by non-linearly scaling and summing validities, will often find that the combined weight of Option A's moderately high validities exceeds Option B's extreme validities, predicting a preference reversal (P(Choose A) > 0.5). This stark divergence in choice proportions strongly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify the tie-breaker trials (Trials 1 and 6) where Option A is [0, 1, 1, 0, 0]
    is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 1, 0, 0))
    if not is_target.any():
        return 0.5
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_target, 'response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.5975 (var=0.0257)
- Predicted under pi_4 (simulated): 0.2908 (var=0.0232)
- Observed on real data: 0.6217 (var=0.0568)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.96, 0.92, 0.88, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Heuristic Mixture model from WADD, we exploit WADD's non-linear compensatory nature. The Mixture model is strictly a linear interpolation between Take-The-Best (TTB) and Tallying. If both TTB and Tallying favor Option A, the Mixture model MUST predict Option A. However, by carefully spacing the validities (e.g., one very high validity, two moderately high validities, and two low validities), we can create trials where Option A possesses the single best cue AND a greater number of total winning cues, but Option B possesses multiple moderately high cues. Under these conditions, an intermediate gamma in WADD will cause the sum of Option B's exponentiated weights to exceed Option A's, leading WADD to predict Option B. This creates a stark qualitative divergence where WADD predicts a preference reversal that the Mixture model mathematically cannot capture.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option that is favored by both Take-The-Best (it has the highest validity cue) and Tallying (it has 3 positive cues compared to 2) on the specifically designed conflict trials (Trials 1 and 2). The Heuristic Mixture model must predict this option with high probability. However, the WADD model, via its gamma exponentiation parameter, can assign greater combined weight to the second and third most valid cues, leading it to frequently predict the opposite option. Thus, the Mixture model will yield a metric near 1.0, while WADD will yield a substantially lower value.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10011') & (b_str == '01100')
    t2_mask = (a_str == '01100') & (b_str == '10011')
    
    t1_choices = data.loc[t1_mask, 'response'] == 0
    t2_choices = data.loc[t2_mask, 'response'] == 1
    
    total_target_choices = t1_choices.sum() + t2_choices.sum()
    total_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_trials == 0:
        return 0.5
    return float(total_target_choices / total_trials)

### RESULTS
- Predicted under pi_3 (simulated): 0.4706 (var=0.0862)
- Predicted under pi_4 (simulated): 0.9294 (var=0.0055)
- Observed on real data: 0.6025 (var=0.0898)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent / total)

```

**Observed (real) value:** 0.3520 (var=0.0355)
**Predicted under pi_3:** 0.4170 (var=0.0576)
**Predicted under pi_4:** 0.4920 (var=0.0537)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Fixed validities from the experimental design
    val = np.array([0.65, 0.95, 0.55, 0.75, 0.85])
    cue_order = np.argsort(-val, kind='stable').tolist()
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie, skip
            
        # TTB prediction
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        # Only consider trials where the two heuristics make STRICTLY OPPOSITE predictions
        if ttb_pred is not None and tally_pred != ttb_pred:
            matches.append(1.0 if row['response'] == tally_pred else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.6600 (var=0.0377)
**Predicted under pi_3:** 0.6763 (var=0.0203)
**Predicted under pi_4:** 0.5563 (var=0.0670)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_f0 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    chose_f0 = (a_has_f0 & chose_a) | (b_has_f0 & chose_b)
    return float(chose_f0.mean())
```

**Observed (real) value:** 0.3862 (var=0.0372)
**Predicted under pi_3:** 0.4656 (var=0.0213)
**Predicted under pi_4:** 0.5700 (var=0.0501)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3 where A=[1, 0, 1, 0, 0] and B=[0, 1, 0, 1, 0]
    # This is the only trial where the sum of features is 2 for both options.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = (a_sums == 2) & (b_sums == 2)
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.6225 (var=0.0889)
**Predicted under pi_3:** 0.8100 (var=0.0256)
**Predicted under pi_4:** 0.7338 (var=0.0340)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple experiments, Theory 1 (WADD with exponentiated validities, pi_3) consistently outperforms Theory 2 (Heuristic Mixture, pi_4). In Experiment 1, pi_3 accurately captures the preference reversal (0.59 vs observed 0.62), while pi_4 completely fails (0.29). In Experiment 2, pi_4 predicts a near-ceiling effect (0.93) whereas the observed data is 0.60, which pi_3 approximates much better (0.47). Similarly, in Experiments 3, 4, and 5, pi_3's predictions are systematically closer to the observed metrics. The Heuristic Mixture model is too restrictive, as linear interpolations between TTB and Tallying mathematically prevent it from capturing the nuanced, compensatory cue integration subjects actually perform. However, while pi_3 performs well, its reliance on an arbitrary exponentiation parameter (gamma) lacks a strong mechanistic or normative justification.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Bayesian Cue Integration' (or Naive Bayes) theory. In this theory, instead of using an arbitrary exponent to scale weights (like WADD) or mixing rigid heuristics (like the Mixture model), decision-makers translate the provided cue validities into log-odds weights: w_i = log(v_i / (1 - v_i)). The evidence for each option is the sum of the log-odds for the cues it possesses. This provides a principled, normatively grounded mechanism for how people naturally scale probabilities into decision weights, naturally steepening high validities and flattening low ones without requiring ad-hoc non-linear transformations."
}
```

## Usage

```json
{
  "prompt_token_count": 6689,
  "candidates_token_count": 382,
  "total_token_count": 7951
}
```
