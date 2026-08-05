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
Tallying (Equal-Weight) Theory: People compare two options by simply counting the number of positive cues (features) for each option and choosing the one with the higher count. This theory posits that decision-makers ignore the varying validities of the cues, treating all features as equally important. It is a compensatory strategy because multiple cues can outweigh a single cue, but it is more frugal than a Weighted Additive (WADD) rule because it avoids multiplying by or storing cardinal validities. Response noise is modeled via a softmax function over the tally scores, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: count the number of positive cues for each option
    # Since cues are binary (0 or 1), we can just sum them.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Validity-Biased Tallying: Decision-makers evaluate options using a compensatory strategy where they compute a weighted sum of the features. However, instead of using the objective cue validities as weights (as in pure WADD) or completely ignoring them (as in pure Tallying), they use subjective weights that are heavily compressed toward equality. This means the weight of each cue is a mixture of a uniform value and its actual validity, capturing the dominant tallying behavior while allowing for a slight, noisy pull from the objective cue validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Interpolate between uniform weights (Tallying) and objective validities (WADD)
    w = (1.0 - alpha) * 1.0 + alpha * validities
    
    # Calculate weighted sum of features for each option
    scores = stim @ w
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To dissociate Tallying from Validity-Biased Tallying, we rely on the fact that Tallying completely ignores cue validities and relies solely on the sum of positive features. In contrast, Validity-Biased Tallying blends uniform weighting with the objective validities. We design critical trials where the number of positive features is tied (making Tallying indifferent and predicting a 50/50 choice), but the objective validities strongly favor one option over the other. In these trials, Validity-Biased Tallying will predict a systematic preference for the option with higher-validity cues. We also include trials where Tallying slightly favors one option (more features) but the validities strongly favor the other option, which will produce a graded shift in choice probabilities under the validity-biased model compared to pure Tallying.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on trials where the simple sum of positive features is tied between options A and B, but the sum of objective validities differs. Pure Tallying ignores validities and predicts a 50/50 choice on these trials, making the proportion of choices for the higher-validity option approximately 0.5. Validity-Biased Tallying mixes uniform weighting with objective validities, so it will systematically prefer the higher-validity option, leading to a proportion significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    A_mat = np.stack(data['option_a_ratings'].values)
    B_mat = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_mat.sum(axis=1)
    sum_B = B_mat.sum(axis=1)
    
    val_A = A_mat.dot(v)
    val_B = B_mat.dot(v)
    
    tied_mask = (sum_A == sum_B) & (np.abs(val_A - val_B) > 1e-4)
    
    if not np.any(tied_mask):
        return 0.5
        
    responses = data['response'].values[tied_mask]
    val_A_tied = val_A[tied_mask]
    val_B_tied = val_B[tied_mask]
    
    chose_higher_val = ((val_A_tied > val_B_tied) & (responses == 0)) | ((val_B_tied > val_A_tied) & (responses == 1))
    
    return float(np.mean(chose_higher_val))

### RESULTS
- Predicted under pi_3 (simulated): 0.4833 (var=0.0032)
- Predicted under pi_5 (simulated): 0.6103 (var=0.0108)
- Observed on real data: 0.5207 (var=0.0044)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** To cleanly dissociate Validity-Biased Tallying from pure Tallying, we exploit Tallying's complete insensitivity to cue validities. Tallying relies solely on the sum of positive features, predicting indifference (50/50 choice) whenever the feature counts are tied, and a strict preference for the option with more features otherwise. Validity-Biased Tallying, by contrast, blends uniform weights with objective validities. We design critical trials where options have an equal number of positive features but differ drastically in the validities of those features (e.g., two high-validity vs. two low-validity cues); here, Validity-Biased Tallying predicts a systematic preference while Tallying predicts guessing. We also include trials where Tallying favors one option due to a higher feature count, but the validities strongly favor the other option, leading Validity-Biased Tallying to predict either a reversal or a heavily attenuated preference.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In trials where both options have an equal number of positive features, pure Tallying predicts indifference (a 50/50 choice probability) because it ignores cue validities. Validity-Biased Tallying, however, still incorporates cue validities to some extent, leading to a systematic preference for the option with higher-validity features. In the chosen experimental design, all trials with tied feature counts (Trials 1, 3, 5, and 8) are constructed such that Option A has a higher sum of objective validities. Therefore, calculating the proportion of times Option A is chosen (response == 0) on these tied trials will strongly discriminate the two theories: Tallying predicts a value of ~0.5, whereas Validity-Biased Tallying predicts a value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    tied_trials = data[a_sums == b_sums]
    if len(tied_trials) == 0:
        return 0.5
    return float((tied_trials['response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4858 (var=0.0042)
- Predicted under pi_5 (simulated): 0.5913 (var=0.0130)
- Observed on real data: 0.4733 (var=0.0049)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_cues = data['option_b_ratings'].apply(sum)
    a_cues = data['option_a_ratings'].apply(sum)
    mask = b_cues > a_cues
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6919 (var=0.0034)
**Predicted under pi_3:** 0.7638 (var=0.0082)
**Predicted under pi_5:** 0.7967 (var=0.0132)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3892 (var=0.0034)
**Predicted under pi_3:** 0.3602 (var=0.0033)
**Predicted under pi_5:** 0.3488 (var=0.0047)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.55, 0.5, 0.5])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    wadd_a = np.dot(a_ratings, validities)
    wadd_b = np.dot(b_ratings, validities)
    
    tally_pred = (sum_b > sum_a).astype(int)
    wadd_pred = (wadd_b > wadd_a).astype(int)
    
    critical_mask = (sum_a != sum_b) & (wadd_a != wadd_b) & (tally_pred != wadd_pred)
    
    if not np.any(critical_mask):
        return 0.5
        
    responses = data['response'].values[critical_mask]
    tally_preds = tally_pred[critical_mask]
    
    return float(np.mean(responses == tally_preds))

```

**Observed (real) value:** 0.6783 (var=0.0060)
**Predicted under pi_3:** 0.7333 (var=0.0207)
**Predicted under pi_5:** 0.6767 (var=0.0302)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_consistent += (1 if resp == 0 else 0)
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_consistent += (1 if resp == 1 else 0)
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_consistent / total
```

**Observed (real) value:** 0.3467 (var=0.0161)
**Predicted under pi_3:** 0.2508 (var=0.0216)
**Predicted under pi_5:** 0.2983 (var=0.0343)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    valid_trials = sum_a != sum_b
    if not valid_trials.any():
        return 0.5
        
    tally_choice = (sum_b > sum_a).astype(int)
    match = (data['response'] == tally_choice)
    
    return float(match[valid_trials].mean())
```

**Observed (real) value:** 0.7113 (var=0.0051)
**Predicted under pi_3:** 0.7943 (var=0.0137)
**Predicted under pi_5:** 0.8490 (var=0.0087)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_pred = np.where(sum_a > sum_b, 0, np.where(sum_b > sum_a, 1, -1))
    ttb_pred = np.where(a_ratings[:, 0] > b_ratings[:, 0], 0, np.where(b_ratings[:, 0] > a_ratings[:, 0], 1, -1))
    discordant = (tally_pred != -1) & (ttb_pred != -1) & (tally_pred != ttb_pred)
    if not np.any(discordant):
        return 0.5
    responses = data['response'].values
    ttb_matches = (responses[discordant] == ttb_pred[discordant]).sum()
    return float(ttb_matches / np.sum(discordant))
```

**Observed (real) value:** 0.2792 (var=0.0051)
**Predicted under pi_3:** 0.2604 (var=0.0207)
**Predicted under pi_5:** 0.1821 (var=0.0162)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, which specifically isolated tied-tally trials to test for validity bias, the observed data showed choice probabilities very close to 50% (0.52 and 0.47). Pure Tallying (Theory 1) successfully predicted this indifference, while Validity-Biased Tallying (Theory 2) predicted a systematic preference for higher-validity options that did not materialize. However, pure Tallying slightly overpredicts the consistency of tally-based choices in other experiments (e.g., Exp 7 and Exp 8). In Experiment 8, about 28% of choices aligned with Take-The-Best (TTB) when TTB and Tallying conflicted. This suggests that while compensatory validity-weighting (Theory 2) is incorrect, pure Tallying (Theory 1) lacks a mechanism to explain the minority of choices that follow validities. A mixture of distinct heuristics rather than a blending of weights seems more appropriate.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Strategy Mixture: Tallying and Take-The-Best' theory. Instead of a compensatory model that compresses validities toward uniform weights (which fails to predict the pure guessing behavior on tied tally trials), this theory should posit that decision-makers probabilistically sample between two distinct, non-compensatory heuristics: pure Tallying and pure Take-The-Best (TTB). On any given trial, the subject either counts cues (ignoring validities) or strictly follows the single most valid discriminating cue. This can capture the dominant Tallying behavior while accounting for the ~25-30% of choices that align with TTB in conflict cases, without predicting a blended validity bias on tied-tally trials."
}
```

## Usage

```json
{
  "prompt_token_count": 7425,
  "candidates_token_count": 399,
  "total_token_count": 9142
}
```
