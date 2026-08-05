# metric_exp01_attempt_02

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
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Absolute Evidence with Own-Cue Dilution (Advocated) from Evidence Dilution and Non-linear Weighting (Competing) by contrasting their dilution denominators. The Advocated theory dilutes evidence based purely on the raw count of present cues, meaning that representing a given amount of validity with multiple weak cues instead of one strong cue heavily inflates the denominator. The Competing theory dilutes based on the sum of the validities of the present cues, meaning the denominator is identical for any two options with the same total validity, regardless of how many cues make up that total. By matching or closely constraining the sum of validities across options while varying the cue counts (e.g., Trial 1 pits a single 1.0 validity cue against two 0.50 validity cues), the Advocated theory predicts a massive dilution penalty for the option with more cues, whereas the Competing theory treats their denominators identically.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Absolute Evidence with Own-Cue Dilution Theory: Decision-makers evaluate options by integrating the non-linearly weighted validities of all present features. However, they dilute this accumulated evidence by the sheer number of features the option possesses. By dividing absolute evidence by the option's total cue count raised to a parameter gamma, the model effectively computes a weighted average of feature validities. This strongly penalizes options that pad their profile with numerous weak features, naturally capturing the 'less is more' effect without distortions from filtering out shared or opponent cues.

**Parameters:**
- lambda_val: [0.1, 10.0]
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
    
    w = val ** lambda_val
    
    ev_a = np.sum(w * a)
    ev_b = np.sum(w * b)
    
    own_cues_a = np.sum(a)
    own_cues_b = np.sum(b)
    
    v_a = ev_a / (np.maximum(1.0, own_cues_a) ** gamma)
    v_b = ev_b / (np.maximum(1.0, own_cues_b) ** gamma)
    
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
[0] rationale: This metric calculates the drop in preference when an option is supplemented by two weak cues (Cues 4 and 5) versus one strong cue (Cue 3) of equal total validity. The Advocated theory dilutes evidence by the raw count of features, penalizing the two-cue supplement significantly more than the one-cue supplement, leading to a large positive value. The Competing theory dilutes by the sum of validities, which is identical for both supplements, predicting a value near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x[0], bool) else ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x[0], bool) else ''.join(map(str, x)))
    
    # Trial 2: A=[1, 0, 0, 1, 1], B=[0, 1, 0, 0, 0]
    t2 = (a_str == '10011') & (b_str == '01000')
    # Trial 3: A=[1, 0, 1, 0, 0], B=[0, 1, 0, 0, 0]
    t3 = (a_str == '10100') & (b_str == '01000')
    # Trial 4: A=[1, 0, 0, 0, 0], B=[0, 1, 0, 1, 1]
    t4 = (a_str == '10000') & (b_str == '01011')
    # Trial 5: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 0, 0]
    t5 = (a_str == '10000') & (b_str == '01100')
    
    p_a_t2 = (data.loc[t2, 'response'] == 0).mean() if t2.any() else 0.5
    p_a_t3 = (data.loc[t3, 'response'] == 0).mean() if t3.any() else 0.5
    p_b_t4 = (data.loc[t4, 'response'] == 1).mean() if t4.any() else 0.5
    p_b_t5 = (data.loc[t5, 'response'] == 1).mean() if t5.any() else 0.5
    
    return float((p_a_t3 - p_a_t2) + (p_b_t5 - p_b_t4))
outcome: evaluation failed (SyntaxError: Generator expression must be parenthesized (<string>, line 2))

[1] rationale: This metric measures the difference in choice probability when an option is supplemented by one strong cue (Cue 3) versus two weak cues (Cues 4 and 5) that sum to the exact same validity. The Advocated theory dilutes by raw cue count, penalizing the two-cue supplement more severely and predicting a positive difference. The Competing theory dilutes by the sum of validities, which is identical for both supplements, predicting a difference near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def to_str(lst):
        return ''.join(str(int(x)) for x in lst)
    
    a_str = data['option_a_ratings'].apply(to_str)
    b_str = data['option_b_ratings'].apply(to_str)
    
    t2 = (a_str == '10011') & (b_str == '01000')
    t3 = (a_str == '10100') & (b_str == '01000')
    t4 = (a_str == '10000') & (b_str == '01011')
    t5 = (a_str == '10000') & (b_str == '01100')
    
    p_a_t2 = (data.loc[t2, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3, 'response'] == 0).mean()
    p_b_t4 = (data.loc[t4, 'response'] == 1).mean()
    p_b_t5 = (data.loc[t5, 'response'] == 1).mean()
    
    diff = (p_a_t3 - p_a_t2) + (p_b_t5 - p_b_t4)
    
    if pd.isna(diff):
        return 0.0
    return float(diff)
outcome: self_sim=0.2779 (var=0.2402) adversary_sim=0.1032 (var=0.1439) welch_t=+1.410 p=0.1655 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the average choice probability of the option padded with two weak cues (i.e., the 3-cue option) in Trials 2 and 4. The Advocated theory dilutes absolute evidence by the raw cue count, applying a massive penalty (denominator of 3^gamma) to the 3-cue option, often driving its choice share very low. The Competing theory dilutes by the sum of validities, treating the additional two weak cues identically to one strong cue of equal sum, and therefore evaluates the 3-cue option much more favorably.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def to_str(lst):
        return ''.join(str(int(x)) for x in lst)
    
    a_str = data['option_a_ratings'].apply(to_str)
    b_str = data['option_b_ratings'].apply(to_str)
    
    # Trial 2: A has 3 cues (1,4,5), B has 1 cue (2)
    t2 = (a_str == '10011') & (b_str == '01000')
    # Trial 4: A has 1 cue (1), B has 3 cues (2,4,5)
    t4 = (a_str == '10000') & (b_str == '01011')
    
    p_a_t2 = (data.loc[t2, 'response'] == 0).mean()
    p_b_t4 = (data.loc[t4, 'response'] == 1).mean()
    
    val2 = p_a_t2 if not pd.isna(p_a_t2) else 0.5
    val4 = p_b_t4 if not pd.isna(p_b_t4) else 0.5
    
    # Return the average choice share of the 3-cue option
    return float((val2 + val4) / 2.0)
outcome: self_sim=0.1821 (var=0.0205) adversary_sim=0.2163 (var=0.0131) welch_t=-0.933 p=0.3559 (N=25, alpha=0.01) -> reject

[3] rationale: Measures the baseline preference for Option A over Option B on Trial 1, where both options have exactly one cue (validities 0.9 vs 0.85). Under the Advocated theory, cue count is 1 for both, so gamma has no diluting effect; Option A is strictly preferred because 0.9^L > 0.85^L. Under the Competing theory, dilution is based on the sum of validities. The subjective values are 0.9^(L-gamma) and 0.85^(L-gamma). If gamma > L, the exponent becomes negative, causing the lower validity cue to paradoxically yield a higher subjective value, leading to frequent preference reversals (choosing B). This creates a massive divergence in the expected choice share of A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    t1 = (a_str == '10000') & (b_str == '01000')
    p_a_t1 = (data.loc[t1, 'response'] == 0).mean()
    return float(p_a_t1) if not pd.isna(p_a_t1) else 0.5
outcome: self_sim=0.7200 (var=0.0261) adversary_sim=0.5632 (var=0.0605) welch_t=+2.665 p=0.01093 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the overall probability of choosing Option A in Trials 1, 3, and 4. In all these trials, Option A has fewer cues than Option B, but the sum of validities is comparable. The Advocated theory dilutes evidence by the raw number of cues, heavily penalizing Option B and leading to a strong preference for Option A. The Competing theory dilutes by the sum of validities, which is similar or higher for Option B, leading to a much lower preference for Option A. Averaging across three trials provides a stable estimate with low between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    t1 = (a_str == '10000') & (b_str == '00011')
    t3 = (a_str == '10100') & (b_str == '01011')
    t4 = (a_str == '01000') & (b_str == '00110')
    
    target_trials = t1 | t3 | t4
    if not target_trials.any():
        return 0.5
        
    return float((data.loc[target_trials, 'response'] == 0).mean())
outcome: self_sim=0.7514 (var=0.0118) adversary_sim=0.7653 (var=0.0090) welch_t=-0.481 p=0.6325 (N=25, alpha=0.01) -> reject

[5] rationale: This metric calculates the choice probability of Option A in Trial 3. In this trial, Option A has fewer cues (2) but slightly lower total validity (1.6) compared to Option B, which has more cues (3) and higher total validity (1.9). The Advocated theory dilutes evidence by the raw number of cues, penalizing Option B heavily (denominator of 3^gamma vs 2^gamma), leading to a strong preference for Option A. The Competing theory dilutes by the sum of validities (1.9^gamma vs 1.6^gamma), which is a much smaller relative penalty. Moreover, because Option B has a higher sum of validities, the Competing theory will often prefer Option B or be indifferent, whereas the Advocated theory will reliably yield a high choice share for Option A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    # Trial 3: A=[1, 0, 1, 0, 0] (2 cues, val sum=1.6), B=[0, 1, 0, 1, 1] (3 cues, val sum=1.9)
    t3 = (a_str == '10100') & (b_str == '01011')
    
    if not t3.any():
        return 0.5
        
    return float((data.loc[t3, 'response'] == 0).mean())
outcome: self_sim=0.6050 (var=0.0258) adversary_sim=0.6417 (var=0.0267) welch_t=-0.799 p=0.428 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Return the overall probability of choosing Option B across all trials\n    return float((data['response'] == 1).mean())",
  "rationale": "In this experimental design, Option B systematically has an equal or greater number of cues compared to Option A across all trials (e.g., 2 vs 1 in T1 and T4, 3 vs 2 in T3, 2 vs 2 in T2). However, Option B's cues are chosen such that their sum of validities is always equal to or strictly greater than Option A's. The Advocated theory dilutes absolute evidence purely by the raw count of present cues, meaning Option B will suffer massive penalties (e.g., divided by 2^gamma or 3^gamma) and be chosen very rarely. The Competing theory dilutes by the sum of validities; since Option B matches or exceeds Option A on this sum, it does not suffer a disproportionate penalty and will be chosen much more frequently. Averaging the choice share of Option B across all 96 trials provides a highly stable, low-variance scalar that cleanly discriminates the core mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 6361,
  "candidates_token_count": 265,
  "total_token_count": 15509
}
```
