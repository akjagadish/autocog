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
Weighted Additive (WADD) Model: Decision-makers evaluate options by computing a compensatory score for each, weighting every feature by its subjective importance (the log-odds of its validity). To balance extreme validities, decision-makers apply a smoothing or regularization process (Laplace smoothing) to the validities before computing their log-odds. This allows multiple weaker cues to perfectly balance a single strong cue, naturally yielding near-tied evaluations for adversarial choice pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    # Laplace smoothing to regularize extreme validities
    val_smoothed = (val + kappa) / (1.0 + 2.0 * kappa)
    
    # Transform smoothed validities to log-odds for normative compensatory weighting
    # Clip to avoid log(0) or division by zero
    val_clipped = np.clip(val_smoothed, 0.001, 0.999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute WADD scores as the dot product of features and weights
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
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

## THEORY 2 — pi_5
Task Disengagement / Position Bias Theory: Subjects find the fictitious product choice task unengaging or too cognitively demanding to evaluate systematically. Instead of using the expert ratings, they adopt a completely deterministic, zero-effort heuristic of exclusively selecting the same option position (e.g., always choosing Option A or always Option B) on every single trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    pref = int(parameters["preferred_option"])
    n_opts = np.asarray(state).shape[0]
    
    probs = np.zeros(n_opts)
    if pref < n_opts:
        probs[pref] = 1.0
    else:
        probs[0] = 1.0
        
    return probs

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** To conclusively dissociate the Weighted Additive (WADD) model from the Task Disengagement / Position Bias theory, this design employs strong stimulus-driven reversals. Position Bias theory posits that subjects ignore the stimuli and deterministically choose the same position (e.g., always Option A or always Option B) across all trials. WADD, however, integrates the feature values weighted by their smoothed validities. By including trials where Option A strongly dominates Option B, and symmetrically matched trials where Option B strongly dominates Option A, WADD predicts clear switching behavior corresponding to the superior option. Position Bias cannot account for any variance in choices driven by the changing feature values, thus failing completely on trials where the non-preferred position is objectively superior.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Position Bias theory posits that subjects deterministically choose the same option (either A or B) on every trial, regardless of the stimuli. Therefore, the within-subject variance of their binary responses will be exactly 0. In contrast, the Weighted Additive (WADD) model predicts that subjects evaluate the features and will choose Option A when it is superior and Option B when it is superior. Because the experimental design includes symmetric pairs of trials (e.g., A dominating B, and B dominating A), WADD predicts subjects will switch their responses, resulting in a within-subject response variance of approximately 0.25. Taking the mean of this within-subject variance perfectly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2471 (var=0.0000)
- Predicted under pi_5 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To conclusively dissociate the Task Disengagement / Position Bias theory from the Weighted Additive (WADD) model, this design features pairs with extreme, unambiguous stimulus-driven advantages for Option A, and symmetrically matched trials with extreme advantages for Option B. The Position Bias theory predicts that subjects completely ignore the ratings and deterministically choose the same position (e.g., always Option A) across all trials. WADD, as a compensatory model driven by feature values, must predict that choices shift to track the superior option. WADD cannot mathematically predict a 100% preference for Option A on trials where Option B has a strictly dominant feature profile. Thus, a subject exhibiting pure position bias will drastically penalize the WADD model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Task Disengagement / Position Bias theory, subjects deterministically choose the exact same option (e.g., always A or always B) across all trials. Therefore, the within-subject variance of their responses will be exactly 0. Under the WADD model, subjects' choices are driven by the features, which vary across trials to favor A and B interchangeably. Thus, their responses will fluctuate between 0 and 1, yielding a within-subject variance close to 0.25. By computing the mean within-subject variance, we get ~0 for Position Bias and ~0.25 for WADD, cleanly discriminating the two.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2478 (var=0.0000)
- Predicted under pi_5 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues ordered 0 to 4 based on validities)
        ttb_winner = None
        for j in range(5):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for i in range(5) if a[i] > b[i])
        b_wins = sum(1 for i in range(5) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only evaluate on trials where the two heuristics strictly oppose
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_match += 1
            total += 1
            
    return float(ttb_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.4317 (var=0.0054)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    matches = (data['response'] == tally_pred)
    return float(matches.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.5667 (var=0.0047)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_has_f1 = data['option_b_ratings'].apply(lambda x: x[0] == 1).astype(int)
    chose_f1 = (data['response'] == b_has_f1)
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.6669 (var=0.0149)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    chose_f1 = ((a_f1 == 1) & (data['response'] == 0)) | ((b_f1 == 1) & (data['response'] == 1))
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.6677 (var=0.0158)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(tuple)
    trial_means = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    subj_devs = (trial_means - 0.5).abs().groupby('subject_id').mean()
    return float(subj_devs.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.1671 (var=0.0056)
**Predicted under pi_5:** 0.5000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    dom_a = (a_sum > 0) & (b_sum == 0)
    dom_b = (a_sum == 0) & (b_sum > 0)
    subset = data[dom_a | dom_b]
    if len(subset) == 0:
        return 0.5
    correct = ((dom_a) & (data['response'] == 0)) | ((dom_b) & (data['response'] == 1))
    return float(correct[dom_a | dom_b].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.6669 (var=0.0240)
**Predicted under pi_5:** 0.5000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data perfectly matches the predictions of Theory 2 (Task Disengagement / Position Bias). The within-subject variance of responses is exactly 0.0000 (Experiments 1 and 2), meaning subjects are deterministically choosing the exact same option on every single trial. Consequently, any metric evaluating feature-based choices (such as choosing the dominant option or matching a heuristic) yields exactly 0.5000 across symmetric trial sets, as subjects simply spam one position regardless of the stimuli. Theory 1 (WADD) assumes compensatory feature integration and predicts variance and choice shifts that are completely absent in the real data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Theory 1 is thoroughly falsified by the zero within-subject variance. Replace it with a new theory that explores a different facet of task disengagement or non-compensatory heuristics. For example, propose a 'Random Responding' theory (where subjects choose randomly on every trial, yielding high within-subject variance) or a 'First Impression Anchoring' theory (where subjects genuinely evaluate the first trial, then deterministically repeat that choice). This will help contrast against the pure, a priori Position Bias of Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 6614,
  "candidates_token_count": 279,
  "total_token_count": 7933
}
```
