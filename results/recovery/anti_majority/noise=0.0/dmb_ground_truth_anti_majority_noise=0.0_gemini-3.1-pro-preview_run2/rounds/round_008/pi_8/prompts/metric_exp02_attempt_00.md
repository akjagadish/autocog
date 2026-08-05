# metric_exp02_attempt_00

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Evidence Dilution Theory (Advocated) from Absolute Evidence with Own-Cue Dilution Theory (Competing) by contrasting the proportional growth of the dilution denominators. The Competing theory dilutes by cue count, meaning adding any cue (even a weak one) increases the denominator by a full integer step (e.g., from 2 to 3), heavily penalizing options with many weak cues. The Advocated theory dilutes by the sum of validities, so adding a weak cue (e.g., 0.55) increases the denominator by only 0.55, resulting in a proportionally milder penalty. By comparing trials where the ratio of cue counts diverges significantly from the ratio of validity sums (e.g., 3:1 count ratio vs ~2:1 validity sum ratio), the models predict distinctly different magnitudes of preference shifts.

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
[0] rationale: Focuses on Trial 3, where Option A has a higher sum of validities (1.7 vs 1.6) but a lower cue count (2 vs 3) than Option B. The Advocated theory dilutes by the sum of validities, meaning A is penalized more than B, reducing the probability of choosing A. The Competing theory dilutes by cue count, meaning B is penalized heavily (3 vs 2), strongly increasing the probability of choosing A. The proportion of times A is chosen on this trial will clearly distinguish the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_trial_3(row):
        return tuple(row['option_a_ratings']) == (1, 1, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 0, 1, 1, 1)
    mask = data.apply(is_trial_3, axis=1)
    if mask.sum() == 0:
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.5942 (var=0.0267) adversary_sim=0.6125 (var=0.0277) welch_t=-0.393 p=0.696 (N=25, alpha=0.01) -> reject

[1] rationale: Focuses on Trial 2, where Option A has a single high-validity cue (0.9) and Option B has two lower-validity cues (0.5, 0.5). The Advocated theory dilutes by the sum of validities, which are very similar for A (0.9) and B (1.0), so dilution does not strongly penalize B relative to A. The Competing theory dilutes by cue count, heavily penalizing B (2 cues) compared to A (1 cue). Thus, the Competing theory should predict a much higher probability of choosing A on this trial than the Advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    mask = data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)
    if mask.sum() == 0:
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.8275 (var=0.0127) adversary_sim=0.8400 (var=0.0207) welch_t=-0.342 p=0.734 (N=25, alpha=0.01) -> reject

[2] rationale: Trial 1 equates the cue count (2 vs 2) but differs in the sum of validities (1.7 for A vs 1.1 for B). The Competing theory dilutes by cue count, so both options are penalized equally, allowing Option A's inherently stronger validities to dominate and resulting in a very high probability of choosing A. The Advocated theory, however, dilutes by the sum of validities, meaning Option A is penalized substantially more than Option B. This asymmetric penalty under the Advocated theory can suppress the preference for A, making the choice proportion of A in Trial 1 a strong differentiator between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Focus on Trial 1 where cue count is equal but validity sums differ.
    mask = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 0))
    if mask.sum() == 0:
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.5750 (var=0.0399) adversary_sim=0.5842 (var=0.0268) welch_t=-0.177 p=0.86 (N=25, alpha=0.01) -> reject

[3] rationale: This metric computes the difference in the probability of choosing Option A between Trial 2 and Trial 4. In both trials, Option A has 1 cue and Option B has 2 cues. However, in Trial 4, Option A's evidence is strictly weaker (0.8 vs 0.9) and Option B's evidence is strictly stronger (0.6+0.5 vs 0.5+0.5) compared to Trial 2. 

Under the Competing theory, dilution depends only on cue counts, which are identical (1 vs 2) across both trials. Therefore, the Competing theory strictly predicts P(A | Trial 2) > P(A | Trial 4) across all parameters because the evidence unequivocally shifts in favor of B in Trial 4. 

Under the Advocated theory, dilution depends on the sum of validities. In Trial 4, Option A's dilution penalty drops (0.8 vs 0.9) while Option B's dilution penalty rises (1.1 vs 1.0). For a large dilution parameter (gamma), this shift in penalties massively benefits Option A in Trial 4, overriding the weaker evidence and causing the Advocated theory to predict P(A | Trial 4) > P(A | Trial 2). This fundamental divergence in the sign of the difference makes it a highly discriminating metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1))
    t4_mask = (data['option_a_ratings'].apply(tuple) == (0, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 0))
    
    if t2_mask.sum() == 0 or t4_mask.sum() == 0:
        return 0.0
        
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t4)
outcome: self_sim=0.0692 (var=0.0194) adversary_sim=0.0517 (var=0.0139) welch_t=+0.480 p=0.6337 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the difference in preference for Option A between Trial 1 and Trial 4. In both trials, both options have exactly 2 cues, so the Competing theory (which dilutes by cue count) applies equal dilution to A and B in both trials. Because Option A's validities in Trial 1 (1.0, 0.55) vs B (0.95, 0.5) provide a strictly larger evidence gap than in Trial 4 (1.0, 0.5 vs 0.95, 0.55), the Competing theory predicts a stronger preference for A in Trial 1: P(A|T1) > P(A|T4). However, the Advocated theory dilutes by the sum of validities. In Trial 4, the validity sums are equal (1.5 vs 1.5). In Trial 1, Option A has a higher validity sum (1.55) than B (1.45), subjecting A to a stronger dilution penalty than B. This penalty suppresses P(A) in Trial 1, leading the Advocated theory to predict P(A|T1) < P(A|T4). The metric P(A|T1) - P(A|T4) cleanly separates the theories by capturing this ordinal reversal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 0, 1, 0))
    t4_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0))
    
    if t1_mask.sum() == 0 or t4_mask.sum() == 0:
        return 0.0
        
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t1 - p_a_t4)
outcome: self_sim=-0.0558 (var=0.0166) adversary_sim=-0.0042 (var=0.0220) welch_t=-1.315 p=0.1949 (N=25, alpha=0.01) -> reject

[5] rationale: This metric calculates the difference in preference for Option A between Trial 2 and Trial 3. In Trial 2, Option A has a single strong cue and B has two weaker cues; the Competing theory heavily penalizes B due to its higher cue count, driving a strong preference for A. In Trial 3, Option A has 3 cues and B has 2 cues; the Competing theory heavily penalizes A, driving a strong preference for B (low P(A)). Thus, the Competing theory predicts a large positive difference P(A|T2) - P(A|T3). The Advocated theory, however, dilutes by the sum of validities. In Trial 3, Option B has a much higher validity sum (1.95) than A (1.55), subjecting B to a severe dilution penalty. This can drastically reduce the preference for B in Trial 3, raising P(A|T3) and thereby shrinking or even reversing the difference P(A|T2) - P(A|T3). This divergence creates a robust contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1))
    t3_mask = (data['option_a_ratings'].apply(tuple) == (0, 0, 1, 1, 1)) & (data['option_b_ratings'].apply(tuple) == (1, 1, 0, 0, 0))
    
    if t2_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t3)
outcome: self_sim=0.5175 (var=0.0547) adversary_sim=0.5200 (var=0.0604) welch_t=-0.037 p=0.9708 (N=25, alpha=0.01) -> reject

[6] rationale: This metric computes the difference in preference for Option A between Trial 1 and Trial 4, but only for subjects who demonstrate low noise by strongly preferring A in Trial 2 (where A is unequivocally superior under both theories). By zeroing out noisy subjects, we drastically reduce the variance driven by the random guessing parameter (epsilon). Under the Competing theory, dilution depends only on cue count, which is equal (2 vs 2) in both T1 and T4. Since A's validities provide a slightly larger evidence gap in T1 than T4, the Competing theory predicts P(A|T1) >= P(A|T4). However, the Advocated theory dilutes by the sum of validities. In T1, Option A has a higher validity sum than Option B, subjecting it to a stronger dilution penalty, whereas in T4 the sums are equal. This asymmetric penalty suppresses P(A) in T1, leading the Advocated theory to predict P(A|T1) < P(A|T4). The noise-filtered difference cleanly separates these opposing predictions with high statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['is_t1'] = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 0, 1, 0))
    data['is_t2'] = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1))
    data['is_t4'] = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0))
    
    def subj_metric(df):
        t1 = df[df['is_t1']]
        t2 = df[df['is_t2']]
        t4 = df[df['is_t4']]
        if len(t1) == 0 or len(t2) == 0 or len(t4) == 0:
            return 0.0
        p_a_t1 = (t1['response'] == 0).mean()
        p_a_t2 = (t2['response'] == 0).mean()
        p_a_t4 = (t4['response'] == 0).mean()
        
        if p_a_t2 >= 0.75:
            return float(p_a_t1 - p_a_t4)
        return 0.0
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=-0.0442 (var=0.0154) adversary_sim=0.0108 (var=0.0192) welch_t=-1.480 p=0.1455 (N=25, alpha=0.01) -> reject

[7] rationale: This metric calculates the difference in the subject's propensity to choose the option with MORE cues between Trial 3 and Trial 2. In Trial 2, the higher-cue option (0,0,0,1,1) has the same validity sum as the single-cue option (1,0,0,0,0), so the Advocated theory applies equal dilution, whereas the Competing theory applies a harsh cue-count penalty. In Trial 3, the higher-cue option (0,0,1,1,1) has a LOWER validity sum than its competitor (1,1,0,0,0), so the Advocated theory actually penalizes it LESS, whereas the Competing theory again penalizes it heavily based on count. By tracking the shift in preference for the higher-cue option across these two conceptually distinct penalty structures, we isolate the unique divergence between validity-sum dilution and cue-count dilution.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 3: Options are { (1,1,0,0,0), (0,0,1,1,1) }
    # We want the probability of choosing the option with 3 cues: (0,0,1,1,1)
    def is_3cue_chosen(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if set([a, b]) == {(1,1,0,0,0), (0,0,1,1,1)}:
            chosen = a if row['response'] == 0 else b
            return 1.0 if chosen == (0,0,1,1,1) else 0.0
        return np.nan
    
    # Trial 2: Options are { (1,0,0,0,0), (0,0,0,1,1) }
    # We want the probability of choosing the option with 2 cues: (0,0,0,1,1)
    def is_2cue_chosen(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if set([a, b]) == {(1,0,0,0,0), (0,0,0,1,1)}:
            chosen = a if row['response'] == 0 else b
            return 1.0 if chosen == (0,0,0,1,1) else 0.0
        return np.nan

    data_copy = data.copy()
    data_copy['t3_val'] = data_copy.apply(is_3cue_chosen, axis=1)
    data_copy['t2_val'] = data_copy.apply(is_2cue_chosen, axis=1)
    
    def subj_metric(df):
        t3_prob = df['t3_val'].dropna().mean()
        t2_prob = df['t2_val'].dropna().mean()
        if pd.isna(t3_prob) or pd.isna(t2_prob):
            return 0.0
        # Difference in preference for the higher-cue option between T3 and T2
        return float(t3_prob - t2_prob)
        
    return float(data_copy.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.1825 (var=0.0394) adversary_sim=0.1650 (var=0.0322) welch_t=+0.327 p=0.745 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 1 and Trial 2 using string conversion for hashability\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t1_mask = (data['a_str'] == '10100') & (data['b_str'] == '01010')\n    t2_mask = (data['a_str'] == '10001') & (data['b_str'] == '01010')\n    \n    if t1_mask.sum() == 0 or t2_mask.sum() == 0:\n        return 0.0\n        \n    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()\n    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()\n    \n    return float(p_a_t1 - p_a_t2)",
  "rationale": "This metric isolates the fundamental difference in how the two theories penalize evidence by comparing Trial 1 and Trial 2. In both trials, Option B is identical (0,1,0,1,0) and both options always have exactly 2 cues. Option A in Trial 1 (1,0,1,0,0) provides strictly stronger objective evidence than Option A in Trial 2 (1,0,0,0,1). \n\nUnder the Competing theory, dilution depends only on cue count. Since all options across both trials have exactly 2 cues, dilution is held constant. Therefore, the Competing theory evaluates Trial 1 as strictly easier than Trial 2, unambiguously predicting P(A|T1) > P(A|T2).\n\nUnder the Advocated theory, dilution depends on the sum of validities. In Trial 2, Option A's validity sum (1.50) equals B's (1.50), leading to equal dilution. In Trial 1, Option A's validity sum (1.70) is higher than B's (1.50), subjecting A to a distinctly harsher dilution penalty. For sufficiently strong dilution parameters, this asymmetric penalty in Trial 1 suppresses the choice of A, leading the Advocated theory to predict P(A|T1) < P(A|T2). The difference P(A|T1) - P(A|T2) thus provides a clean directional contrast between the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 7653,
  "candidates_token_count": 626,
  "total_token_count": 11096
}
```
