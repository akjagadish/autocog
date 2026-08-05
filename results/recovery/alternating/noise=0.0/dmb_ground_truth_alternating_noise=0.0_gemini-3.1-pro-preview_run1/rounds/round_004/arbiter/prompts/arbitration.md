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
Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
Position Bias / Option A Default Theory: Participants do not systematically integrate feature validities or use complex heuristics. Instead, due to reading order (e.g., left-to-right or top-to-bottom) or motor ease, they exhibit a slight systematic bias toward choosing the first option presented (Option A). This structural bias is mixed with random guessing, leading to a state-independent choice probability where Option A is chosen slightly more often than Option B, regardless of the features of the options.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the state and history, relying solely on a structural bias toward Option A.
    p_A = float(parameters["p_A"])
    return np.array([p_A, 1.0 - p_A])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** This design aims to dissociate the Zero-Intelligence theory from the Position Bias theory. Because both theories assume participants ignore the feature values entirely, the specific feature patterns do not change their predictions. However, by including trials where Option B overwhelmingly dominates Option A (and vice versa), we can verify that participants are indeed ignoring the features. The Zero-Intelligence theory predicts exactly 50/50 guessing across all trials, whereas the Position Bias theory predicts a systematic, state-independent preference for Option A (up to 60%). If participants choose Option A exactly 50% of the time even when Option B is objectively superior on all features, it will support Zero-Intelligence and rule out Position Bias.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Zero-Intelligence theory predicts that participants will choose Option A exactly 50% of the time, as they are guessing uniformly at random. In contrast, the Position Bias theory predicts a systematic preference for Option A (up to 60%). By calculating the overall proportion of trials where Option A is chosen across all experimental conditions, we can directly measure this baseline bias. Because both theories assume choices are independent of the actual feature values, pooling across all trials minimizes trial-level noise and provides the tightest possible estimate of the base rate of choosing Option A, maximizing the statistical power to distinguish the 50% vs. >50% predictions.

Source:
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.4873 (var=0.0023)
- Predicted under pi_6 (simulated): 0.5379 (var=0.0027)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** Both the Position Bias and Zero-Intelligence theories assume that participants completely ignore the feature values and validities. To dissociate them, we must look at the baseline choice probability for Option A across trials that vary drastically in objective quality. We include trials where Option B overwhelmingly dominates Option A (e.g., B has all positive ratings, A has none), trials where Option A dominates, and mixed trials. The Competing Theory (Zero-Intelligence) predicts exactly 50% choice of Option A across all trial types. The Advocated Theory (Position Bias) predicts a flat but elevated choice probability for Option A (e.g., 55-60%) regardless of whether Option A is objectively terrible or perfect. By showing that p(A) remains stably above 0.5 even when B is clearly superior, we can quantitatively support Position Bias over pure Random Choice while simultaneously ruling out feature-integrating heuristics.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Since both theories assume features are ignored, the sequence of choices is fully summarized by the total number of times Option A was chosen. However, simple linear transformations like the raw proportion or empirical logit failed to reach statistical significance because they do not optimally separate the specific distributions of the two theories (a point mass at p=0.5 vs a uniform distribution of p in [0.5, 0.6]). By computing the exact Log-Likelihood Ratio (LLR) of the data under the two theories' structural assumptions, we apply the Neyman-Pearson lemma to construct the most statistically powerful metric possible for this exact contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Count choices for Option A
    k = np.sum(data['response'] == 0)
    n = len(data)
    
    # The Advocated Theory (Position Bias) posits p_A is uniformly distributed between 0.5 and 0.6.
    # The Competing Theory (Zero-Intelligence) posits p_A is exactly 0.5.
    # We construct the log-likelihood ratio (LLR) of the observed choices under PB vs ZI.
    # This is the theoretically optimal test statistic (Neyman-Pearson lemma) for discriminating them.
    
    ps = np.linspace(0.5, 0.6, 51)
    
    # For each candidate p in the PB theory, calculate the log likelihood ratio compared to p=0.5
    # LLR_p = k * log(p / 0.5) + (n - k) * log((1 - p) / 0.5)
    log_terms = k * np.log(2 * ps) + (n - k) * np.log(2 * (1 - ps))
    
    # Integrate out the uniform prior over p using the log-sum-exp trick for numerical stability
    max_log = np.max(log_terms)
    llr = max_log + np.log(np.mean(np.exp(log_terms - max_log)))
    
    return float(llr)

### RESULTS
- Predicted under pi_4 (simulated): 0.1791 (var=0.5921)
- Predicted under pi_6 (simulated): 14.6204 (var=1.1961)
- Observed on real data: -2.0012 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice: first cue that discriminates determines choice
        ttb_choice = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice is None:
            ttb_choice = 0.5
            
        if ttb_choice == row['response']:
            matches.append(1.0)
        elif ttb_choice == 0.5:
            matches.append(0.5)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5046 (var=0.0028)
**Predicted under pi_6:** 0.5115 (var=0.0030)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    valid_trials = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        if a_wins > b_wins:
            if r == 0:
                matches += 1
            valid_trials += 1
        elif b_wins > a_wins:
            if r == 1:
                matches += 1
            valid_trials += 1
    return float(matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.5042 (var=0.0108)
**Predicted under pi_4:** 0.4954 (var=0.0020)
**Predicted under pi_6:** 0.5031 (var=0.0017)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5029 (var=0.0019)
**Predicted under pi_6:** 0.4567 (var=0.0045)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4985 (var=0.0000)
**Predicted under pi_4:** 0.5069 (var=0.0030)
**Predicted under pi_6:** 0.4543 (var=0.0032)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    diffs = data.apply(get_wadd_diff, axis=1)
    
    # WADD predicts choice A (0) when diffs > 0, and choice B (1) when diffs < 0
    is_correct = ((diffs > 0) & (data['response'] == 0)) | ((diffs < 0) & (data['response'] == 1))
    
    return float(is_correct.mean())
```

**Observed (real) value:** 0.5008 (var=0.0004)
**Predicted under pi_4:** 0.4856 (var=0.0028)
**Predicted under pi_6:** 0.5100 (var=0.0025)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    val_a = A.dot(val)
    val_b = B.dot(val)
    
    # The expected choice under deterministic WADD
    expected = (val_a < val_b).astype(int)
    correct = (data['response'].values == expected).astype(float)
    
    # Weight the accuracy by the absolute difference in weighted sums
    diff = np.abs(val_a - val_b)
    
    # Return the weighted accuracy
    return float(np.sum(correct * diff) / np.sum(diff))
```

**Observed (real) value:** 0.4990 (var=0.0006)
**Predicted under pi_4:** 0.4894 (var=0.0064)
**Predicted under pi_6:** 0.4887 (var=0.0054)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Feature 0 is the single most valid cue (validity 0.95)
    a_cue_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue_0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Identify trials where the subject chose the option endorsed by the best cue
    aligned = (chose_a & (a_cue_0 > b_cue_0)) | (chose_b & (b_cue_0 > a_cue_0))
    
    # The Single-Cue model with very high lapse predicts a slight bias (mean p~0.525)
    # toward the best cue, whereas Zero-Intelligence predicts exactly p=0.5.
    # We compute the total log-likelihood ratio (LLR) of the choices under the 
    # Single-Cue hypothesis (p=0.525) versus the Zero-Intelligence hypothesis (p=0.5).
    # This is the optimal test statistic (Neyman-Pearson) for discriminating the two.
    llr = aligned * np.log(0.525 / 0.5) + (~aligned) * np.log(0.475 / 0.5)
    
    return float(llr.sum())
```

**Observed (real) value:** -3.6043 (var=0.0395)
**Predicted under pi_4:** 0.6981 (var=0.2895)
**Predicted under pi_6:** -13.1134 (var=0.2220)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract the first cue (the single most valid cue) for both options
    cue0_a = data['option_a_ratings'].apply(lambda x: x[0])
    cue0_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Filter to trials where cue 0 discriminates between the two options
    valid_trials = cue0_a != cue0_b
    if not valid_trials.any():
        return 0.0
        
    subset = data[valid_trials]
    cue0_a_sub = cue0_a[valid_trials]
    cue0_b_sub = cue0_b[valid_trials]
    
    # Response == 0 means A, Response == 1 means B
    # Check if the subject's choice aligns with the option favored by cue 0
    aligned = (subset['response'] == (cue0_b_sub > cue0_a_sub).astype(int))
    
    # Return the TOTAL sum of aligned choices instead of the mean.
    # This aggregates the small residual signal into a large cumulative effect.
    return float(aligned.sum())
```

**Observed (real) value:** 891.0000 (var=80.8704)
**Predicted under pi_4:** 1743.0000 (var=12.9604)
**Predicted under pi_6:** 1817.0000 (var=14.6644)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data across all experiments strongly supports the Zero-Intelligence (Random Choice) theory (pi_4). In Experiment 1, the choice proportion for Option A is exactly 0.5000, and in Experiment 2, the log-likelihood ratio strongly rejects the Position Bias theory (pi_6). Across all other experiments, accuracy and choice proportions consistently hover exactly around 0.50. Theory 1 (pi_4) captures this perfectly, while Theory 2 (pi_6) predicts a systematic baseline bias toward Option A that is completely absent in the real data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Position Bias theory with a 'Sequential Pattern' theory, such as 'Response Alternation' or 'Response Persistence'. Since the overall choice proportion is exactly 50%, participants might not be guessing independently on each trial but instead following a sequential heuristic (e.g., alternating between A and B, or repeating the previous choice). This would maintain the 50% overall choice probability for Option A, but introduce a distinct trial-by-trial autocorrelation that can be tested against independent random guessing."
}
```

## Usage

```json
{
  "prompt_token_count": 7807,
  "candidates_token_count": 263,
  "total_token_count": 9757
}
```
