# metric_exp01_attempt_03

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=7):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Rank-Weighted Capacity-Bounded Integration' (Competing) by exploiting how each theory penalizes excess features. The Competing theory assumes a capacity limit (K = 1, 2, or 3) and penalizes options based strictly on the RAW COUNT of active features beyond this limit, completely ignoring the validities of those excess features. By keeping the top 3 features of Option A constant across trials and adding exactly two extra features, the Competing theory predicts the exact same subjective value for Option A regardless of whether those two extra features are moderately strong (e.g., 0.80, 0.75) or very weak (e.g., 0.55, 0.50). In contrast, the Advocated theory dilutes evidence by the SUM OF VALIDITIES of all active features. Therefore, adding moderately strong features inflates the dilution denominator significantly more than adding weak features, predicting that Option A will be valued much lower (and chosen less often) when its extra features are stronger. This produces a strict invariance prediction for the Competing theory against a graded 'less-is-more' prediction for the Advocated theory.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Rank-Weighted Capacity-Bounded Integration with Bounded Non-linear Penalty: Decision-makers integrate cues based on their validity, but cognitive capacity limits the number of features that can be positively evaluated. The top K valid active features for an option are summed to form its base value. Any additional active features beyond this capacity limit act as a cognitive complexity penalty. This penalty scales non-linearly with the number of excess features and subtracts from the base value, but the overall subjective value is bounded at zero to prevent extreme negative evaluations. This explains why adding many weak features penalizes an option heavily without causing unrealistic certainty in choice probabilities.

**Parameters:**
- lambda_val: [0.1, 10.0]
- beta: [0.1, 20.0]
- penalty: [0.0, 5.0]
- K: {1, 2, 3}
- gamma: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    penalty = float(parameters["penalty"])
    K = int(parameters["K"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    w = val ** lambda_val
    
    # Get validities of active features
    w_a = w[a == 1]
    w_b = w[b == 1]
    
    # Sort descending
    w_a = np.sort(w_a)[::-1]
    w_b = np.sort(w_b)[::-1]
    
    # Sum top K and subtract non-linear penalty for the rest
    n_excess_a = len(w_a[K:])
    n_excess_b = len(w_b[K:])
    
    v_a = max(0.0, np.sum(w_a[:K]) - penalty * (n_excess_a ** gamma))
    v_b = max(0.0, np.sum(w_b[:K]) - penalty * (n_excess_b ** gamma))
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
[0] rationale: The metric computes the difference in the choice rate of Option B when Option A is supplemented with strong vs. weak extra features. The Competing Theory assumes that capacity-bounded integration either favors strong extra features (if they fit in capacity K) or treats strong and weak extra features equally (if they exceed capacity and incur the same count-based penalty). Thus, it predicts the metric will be positive or zero. Conversely, the Advocated Theory posits a dilution effect proportional to the sum of validities. Stronger extra features increase the dilution denominator much more than weaker ones, often leading to a paradoxical penalty that makes Option A less preferred when it has strong (but not top) features compared to weak features. Thus, the Advocated Theory predicts a negative metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t2 = (1, 0, 1, 0, 0, 0)
    t3 = (1, 0, 0, 0, 1, 0)
    t4 = (1, 0, 1, 1, 0, 0)
    t5 = (1, 0, 0, 0, 1, 1)
    
    resp_t2 = data.loc[data['A_tuple'] == t2, 'response'].mean()
    resp_t3 = data.loc[data['A_tuple'] == t3, 'response'].mean()
    resp_t4 = data.loc[data['A_tuple'] == t4, 'response'].mean()
    resp_t5 = data.loc[data['A_tuple'] == t5, 'response'].mean()
    
    # Metric: (Response T3 - Response T2) + (Response T5 - Response T4)
    # Response is 0 for A, 1 for B. 
    # A lower response means higher preference for A.
    return float((resp_t3 - resp_t2) + (resp_t5 - resp_t4))
outcome: self_sim=0.0232 (var=0.0348) adversary_sim=0.0695 (var=0.0320) welch_t=-0.896 p=0.3746 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates a strict mathematical bound in the Competing theory that is frequently violated by the Advocated theory. We compare Trial 4 (A adds two strong features: 0.80, 0.75) and Trial 5 (A adds two weak features: 0.50, 0.50). In the Competing theory, the penalty is strictly based on the count of excess features. Since both T4 and T5 have the same total number of features (3), they incur the exact same complexity penalty regardless of capacity limit K. Because T4 has higher validity features, its base value is always >= T5. Thus, the Competing theory guarantees that preference for A in T4 is >= T5, meaning the probability of choosing B (response=1) is strictly P(B|T4) <= P(B|T5). In contrast, the Advocated theory dilutes by the sum of validities. T4 has a much larger sum (2.50) than T5 (1.95). For most values of the dilution parameter gamma (e.g., gamma > 1), T4 is penalized so heavily that its subjective value drops BELOW T5, causing a paradoxical reversal where P(B|T4) > P(B|T5). By computing `max(0, resp_t4 - resp_t5)` per subject, the Competing theory expects 0 (plus minor binomial noise), while the Advocated theory expects a robust positive value for the vast majority of the parameter space.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t4 = (1, 0, 1, 1, 0, 0)
    t5 = (1, 0, 0, 0, 1, 1)
    
    subj_metrics = []
    for subj, df in data.groupby('subject_id'):
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        r5 = df.loc[df['A_tuple'] == t5, 'response'].mean()
        
        # Difference in probability of choosing B
        val = r4 - r5
        
        # We only accumulate positive differences.
        subj_metrics.append(max(0.0, float(val)))
        
    return float(np.mean(subj_metrics))
outcome: self_sim=0.0347 (var=0.0047) adversary_sim=0.0358 (var=0.0030) welch_t=-0.060 p=0.9526 (N=25, alpha=0.01) -> reject

[2] rationale: This metric uses a difference-in-differences approach to contrast how each theory handles the addition of a moderately strong vs. a weak feature. We compare the shift in choice probability when adding a 0.75 validity feature (Trial 2 -> Trial 4) versus adding a 0.50 validity feature (Trial 3 -> Trial 5). 

In the Competing theory, the complexity penalty is based solely on the count of excess features. Since both additions increase the feature count by 1, the penalty increase is identical. However, if capacity K=3, the 0.75 feature adds more to the base value than the 0.50 feature, making Option A relatively MORE attractive in T4 than T5. Thus, the Competing theory predicts this double difference will be zero or negative. 

In the Advocated theory, evidence dilution is proportional to the sum of validities. Adding a 0.75 feature increases the dilution denominator significantly more than adding a 0.50 feature (sum goes from 1.75 to 2.50 vs. 1.45 to 1.95). Because the non-linear numerator may largely ignore the 0.75 feature, this massive denominator inflation causes a paradoxical heavy penalty. Thus, the Advocated theory predicts a robust positive double difference, as adding the stronger feature damages the option more than adding the weaker one.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t2 = (1, 0, 1, 0, 0, 0)
    t3 = (1, 0, 0, 0, 1, 0)
    t4 = (1, 0, 1, 1, 0, 0)
    t5 = (1, 0, 0, 0, 1, 1)
    
    subj_vals = []
    for subj, df in data.groupby('subject_id'):
        r2 = df.loc[df['A_tuple'] == t2, 'response'].mean()
        r3 = df.loc[df['A_tuple'] == t3, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        r5 = df.loc[df['A_tuple'] == t5, 'response'].mean()
        
        # Difference-in-differences: 
        # (Drop in preference for A when adding a 0.75 feature) - 
        # (Drop in preference for A when adding a 0.50 feature)
        val = (r4 - r2) - (r5 - r3)
        subj_vals.append(float(val))
        
    return float(np.mean(subj_vals))
outcome: self_sim=-0.0274 (var=0.0452) adversary_sim=-0.0084 (var=0.0411) welch_t=-0.322 p=0.7486 (N=25, alpha=0.01) -> reject

[3] rationale: This metric evaluates the shift in choice probability when two relatively strong features (0.80 and 0.75) are added to the best feature (0.95) from Trial 1 to Trial 4. In the Competing theory, adding these features often triggers a strict complexity penalty (if capacity K=1 or 2) that subtracts from the base value, pushing the preference heavily towards Option B. In the Advocated theory, adding these features increases the dilution denominator to 2.5^gamma. Because the non-linear numerator also increases, the shift in preference is heavily dependent on the balance of these effects but generally avoids the hard zero-bounding seen in the Competing theory's penalty. This simple difference in response rates between T4 and T1 robustly separates the theories' overall penalty magnitudes.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t1 = (1, 0, 0, 0, 0, 0)
    t4 = (1, 0, 1, 1, 0, 0)
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        r1 = df.loc[df['A_tuple'] == t1, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        subj_diffs.append(r4 - r1)
        
    return float(np.mean(subj_diffs))
outcome: self_sim=0.3989 (var=0.0758) adversary_sim=0.2779 (var=0.1706) welch_t=+1.219 p=0.2296 (N=25, alpha=0.01) -> reject

[4] rationale: The Competing theory assumes a capacity limit (K=1, 2, or 3) and penalizes options based strictly on the raw count of active features beyond this limit. Since Option A has exactly 5 active features in both Trial 1 and Trial 4, and its top 3 features are identical, the Competing theory predicts the exact same subjective value for Option A in both trials, meaning the difference in response rates will be zero in expectation. In contrast, the Advocated theory dilutes evidence by the sum of validities. Trial 1 has much stronger extra features than Trial 4, resulting in a significantly larger dilution denominator. For most parameter values, this causes a paradoxical penalty where Option A is valued less in Trial 1 than Trial 4, leading to a robust positive difference in the probability of choosing Option B.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t1 = (1, 1, 1, 1, 1, 0, 0)
    t4 = (1, 1, 1, 0, 0, 1, 1)
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        r1 = df.loc[df['A_tuple'] == t1, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        subj_diffs.append(r1 - r4)
        
    return float(np.mean(subj_diffs))
outcome: self_sim=0.0067 (var=0.0327) adversary_sim=-0.0075 (var=0.0128) welch_t=+0.332 p=0.7415 (N=25, alpha=0.01) -> reject

[5] rationale: The Competing theory posits a capacity limit (K=1, 2, or 3) and penalizes based on the RAW COUNT of excess features. Across all 4 trial types, Option A always has exactly 5 active features, and its top 3 features are always identical. Thus, the Competing theory predicts the exact same subjective value for Option A (and B) across all 4 trials. Any variation in response rates is purely due to binomial noise. In contrast, the Advocated theory dilutes evidence by the SUM of validities, which varies significantly across the 4 trials (from 4.25 in T1 down to 3.75 in T4). This causes systematic shifts in choice probabilities depending on the trial type. Because the directional effect of this shift depends heavily on the gamma parameter (sometimes penalizing T1 more, sometimes T4 more), simple differences cancel out when averaged across the parameter space. By computing the standard deviation of response rates across the 4 trial types per subject, we capture the systematic variance predicted by the Advocated theory without it canceling out, while the Competing theory yields only a tight distribution of binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t1 = (1, 1, 1, 1, 1, 0, 0)
    t2 = (1, 1, 1, 1, 0, 1, 0)
    t3 = (1, 1, 1, 0, 1, 0, 1)
    t4 = (1, 1, 1, 0, 0, 1, 1)
    
    subj_stds = []
    for subj, df in data.groupby('subject_id'):
        r1 = df.loc[df['A_tuple'] == t1, 'response'].mean()
        r2 = df.loc[df['A_tuple'] == t2, 'response'].mean()
        r3 = df.loc[df['A_tuple'] == t3, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        
        # Compute the standard deviation of response rates across the 4 trial types
        subj_stds.append(np.std([r1, r2, r3, r4]))
        
    return float(np.mean(subj_stds))
outcome: self_sim=0.0831 (var=0.0013) adversary_sim=0.0701 (var=0.0011) welch_t=+1.307 p=0.1974 (N=25, alpha=0.01) -> reject

[6] rationale: The Competing theory posits a capacity limit (K=1, 2, or 3) and penalizes based strictly on the RAW COUNT of excess features. Across all 4 trial types, Option A always has exactly 5 active features, and its top 3 features are identical. Thus, the Competing theory predicts the exact same subjective value for Option A across all 4 trials. Any variation in response rates is purely due to binomial noise. The Advocated theory dilutes evidence by the SUM of validities, which varies significantly across the 4 trials. Because the directional effect of this shift depends heavily on the parameters (sometimes T1 is penalized more, sometimes T4), simple differences cancel out. This metric isolates the systematic variance predicted by the Advocated theory by computing the sample variance of the 4 response rates and explicitly subtracting the mathematically expected binomial variance. For the Competing theory, this 'excess variance' is exactly zero in expectation. For the Advocated theory, it is robustly positive, cleanly separating the theories while removing binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t1 = (1, 1, 1, 1, 1, 0, 0)
    t2 = (1, 1, 1, 1, 0, 1, 0)
    t3 = (1, 1, 1, 0, 1, 0, 1)
    t4 = (1, 1, 1, 0, 0, 1, 1)
    
    subj_excess_vars = []
    for subj, df in data.groupby('subject_id'):
        r1 = df.loc[df['A_tuple'] == t1, 'response'].mean()
        r2 = df.loc[df['A_tuple'] == t2, 'response'].mean()
        r3 = df.loc[df['A_tuple'] == t3, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        
        rates = [r1, r2, r3, r4]
        if any(np.isnan(x) for x in rates):
            continue
            
        S2 = np.var(rates, ddof=1)
        r_mean = np.mean(rates)
        
        # Expected binomial variance of the 4 proportions (n=24 each)
        # Unbiased estimator of p(1-p)/24 using r_mean (which has n=96)
        expected_S2 = (4.0 / 95.0) * r_mean * (1.0 - r_mean)
        
        excess_var = S2 - expected_S2
        subj_excess_vars.append(excess_var)
        
    if not subj_excess_vars:
        return 0.0
    return float(np.mean(subj_excess_vars))
outcome: self_sim=0.0010 (var=0.0001) adversary_sim=-0.0002 (var=0.0000) welch_t=+0.559 p=0.5794 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    return float(data['response'].mean())",
  "rationale": "The experimental design pits Option A (5 features) against Option B (3 features) where the top 3 features are identical. The Competing theory applies a count-based penalty to excess features. Since Option A always has 2 more excess features than Option B, Option A is consistently and heavily penalized across all trials, leading to a strong overall preference for Option B (response -> 1). In contrast, the Advocated theory dilutes value by the sum of validities. For most of its parameter space (especially when the dilution parameter gamma is large), the subjective values of both options are driven close to zero, resulting in choice probabilities near 0.5. Thus, the overall mean response cleanly discriminates the strong 'less-is-more' penalty of the Competing theory from the heavily diluted, near-random guessing predicted by the Advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 7739,
  "candidates_token_count": 215,
  "total_token_count": 21660
}
```
