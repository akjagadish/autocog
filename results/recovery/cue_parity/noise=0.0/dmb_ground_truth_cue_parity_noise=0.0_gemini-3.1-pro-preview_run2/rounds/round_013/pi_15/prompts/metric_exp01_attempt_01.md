# metric_exp01_attempt_01

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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 11: A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 12: A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the advocated 'Environment-Contingent Strategy Selection' theory from the competing 'Sequential Evidence Accumulation' (SEA) theory by probing for validity-insensitive compensatory integration. The advocated theory posits a probability mixture of Take-The-Best (TTB) and unweighted Tallying. Because TTB only evaluates the top discriminating cue and Tallying only counts the sheer number of cues, the advocated model is completely blind to the specific validities of secondary cues. It strongly predicts that as long as the TTB winner is unchanged and the raw tally difference is constant, choice probabilities will be strictly identical (flat lines within groups), but will step up or down categorically when the tally difference changes (steps between groups). SEA, conversely, evaluates cues sequentially and accumulates evidence weighted by a non-linear transformation of validity. If SEA's threshold is low, it stops at the first cue and predicts flat lines everywhere (failing to predict the steps between tally groups). If its threshold is high, it integrates secondary cues, predicting systematic slopes WITHIN each tally group as the validities of the secondary cues shift. SEA mathematically cannot produce stepped flat lines.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Environment-Contingent Strategy Selection with Mean-Relative Dominant Cue Sensitivity: Decision-makers select between non-compensatory (Take-The-Best) and compensatory (Tallying) heuristics based on the structural properties of the environment. Specifically, the probability of deploying Take-The-Best increases as a logistic function of the difference between the top cue's validity and the average validity of all cues. In environments where the top cue strongly stands out from the overall cue distribution, individuals rely on TTB; when validities are relatively flat, they fall back to Tallying.

**Parameters:**
- gamma: [0.1, 50.0]
- threshold: [0.0, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Structural property: top cue vs mean validity
    sorted_val = np.sort(val)[::-1]
    dispersion = sorted_val[0] - np.mean(val)
    
    # Probability of using TTB based on environment structure
    # Using logistic function to map dispersion to probability
    p_ttb_strategy = 1.0 / (1.0 + np.exp(-gamma * (dispersion - threshold)))
    
    # Take-The-Best (TTB) Strategy
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    z_ttb = beta_ttb * scores_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    p_ttb_choice = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    scores_tally = np.array([tally_a, tally_b])
    
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally_choice = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_mix = p_ttb_strategy * p_ttb_choice + (1.0 - p_ttb_strategy) * p_tally_choice
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
[0] rationale: The advocated theory (Environment-Contingent Strategy Selection) posits that decision-makers rely on a mixture of Take-The-Best and Tallying. Neither of these strategies is sensitive to the sequential ordering of secondary cue validities. Therefore, the advocated theory predicts identical choice probabilities across trials where the TTB winner and Tally difference are constant (e.g., T1 vs T3, and T6 vs T8). In contrast, the competing theory (Sequential Evidence Accumulation) weights cues non-linearly by their validity. As a result, shifting the opposing evidence to less valid cues (as in T1 vs T3, and T8 vs T6) systematically increases the probability of choosing Option A. The metric computes the sum of these probability differences: (P(A|T1) - P(A|T3)) + (P(A|T8) - P(A|T6)). This value will be approximately 0 for the advocated theory, but significantly positive for the competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to string representations for easy querying
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate probability of choosing Option A
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # Trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 0]
    t1 = data[(data['A_str'] == '11000') & (data['B_str'] == '00110')]['chose_A'].mean()
    # Trial 3: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 0]
    t3 = data[(data['A_str'] == '10010') & (data['B_str'] == '01100')]['chose_A'].mean()
    # Trial 6: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 0, 0]
    t6 = data[(data['A_str'] == '10000') & (data['B_str'] == '01100')]['chose_A'].mean()
    # Trial 8: A=[1, 0, 0, 0, 0], B=[0, 0, 0, 1, 1]
    t8 = data[(data['A_str'] == '10000') & (data['B_str'] == '00011')]['chose_A'].mean()
    
    # Fallbacks in case of missing data for a subject
    t1 = t1 if pd.notna(t1) else 0.5
    t3 = t3 if pd.notna(t3) else 0.5
    t6 = t6 if pd.notna(t6) else 0.5
    t8 = t8 if pd.notna(t8) else 0.5
    
    # The metric contrasts choice probabilities that are identical under the 
    # advocated theory but systematically different under SEA.
    return (t1 - t3) + (t8 - t6)
outcome: self_sim=0.0000 (var=0.1136) adversary_sim=0.2400 (var=0.2007) welch_t=-2.141 p=0.03781 (N=25, alpha=0.01) -> reject

[1] rationale: By aggregating the probability differences across four distinct pairs of trials that share the same TTB and Tally properties but differ in secondary cue validities, we can significantly increase the signal-to-noise ratio. The advocated theory predicts an expected sum of 0, as it ignores the specific validities of secondary cues. The SEA theory predicts a strongly positive sum, as it systematically accumulates evidence based on the exact validities present. Aggregating across four pairs instead of two doubles the expected mean difference for SEA, improving statistical power against per-subject binomial variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to string representations for easy querying
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate probability of choosing Option A
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    def get_p(a_str, b_str):
        m = data[(data['A_str'] == a_str) & (data['B_str'] == b_str)]['chose_A'].mean()
        return m if pd.notna(m) else 0.5

    # Group 1: TTB winner A, Tally diff 0
    t1 = get_p('11000', '00110')
    t3 = get_p('10010', '01100')
    
    # Group 2: TTB winner B, Tally diff 0
    t4 = get_p('01100', '00011')
    t5 = get_p('01010', '00101')
    
    # Group 3: TTB winner A, Tally diff -1
    t8 = get_p('10000', '00011')
    t6 = get_p('10000', '01100')
    
    # Group 4: TTB winner A, Tally diff +2
    t9 = get_p('11100', '00010')
    t10 = get_p('10011', '01000')
    
    # The metric sums the differences across four independent pairs of trials.
    # For each pair, the advocated theory predicts identical choice probabilities 
    # (difference = 0) because TTB and Tally predictions are constant within the pair.
    # The competing SEA theory predicts a positive difference for each pair due to 
    # the non-linear weighting of specific cue validities.
    return (t1 - t3) + (t4 - t5) + (t8 - t6) + (t9 - t10)
outcome: self_sim=0.0644 (var=0.1319) adversary_sim=0.3244 (var=0.3051) welch_t=-1.967 p=0.05595 (N=25, alpha=0.01) -> reject

[2] rationale: To discriminate between the Environment-Contingent Strategy Selection (advocated) and Sequential Evidence Accumulation (competing) theories, we group trials into sets where the advocated theory predicts strictly identical choice probabilities (because the Take-The-Best winner and Tally difference are constant within each group). Within these matched groups, the SEA theory predicts varying choice probabilities because it accumulates evidence sequentially based on the exact validities of secondary cues. By taking the difference between the high-evidence and low-evidence trials within each group, the advocated theory expects a sum of exactly 0, while the SEA theory expects a significantly positive sum. By averaging T1 and T4 rather than strictly summing all pairwise differences, we reduce the per-subject variance of the metric, increasing the statistical power to detect the SEA signature at N=25.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to string representations for easy querying
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate probability of choosing Option A
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    def get_p(a_str, b_str):
        m = data[(data['A_str'] == a_str) & (data['B_str'] == b_str)]['chose_A'].mean()
        return m if pd.notna(m) else 0.5

    # Group 1: TTB winner A, Tally diff 0
    # T1 and T4 have high evidence difference for SEA (0.4)
    t1 = get_p('11000', '00110')
    t4 = get_p('01100', '00011')
    # T3 has zero evidence difference for SEA (0.0)
    t3 = get_p('10010', '01100')
    diff_0 = (t1 + t4) / 2.0 - t3
    
    # Group 2: TTB winner A, Tally diff -1
    # T8 has higher evidence difference (-0.25) than T6 (-0.65)
    t8 = get_p('10000', '00011')
    t6 = get_p('10000', '01100')
    diff_minus1 = t8 - t6
    
    # Group 3: TTB winner A, Tally diff +2
    # T9 has higher evidence difference (1.9) than T10 (1.3)
    t9 = get_p('11100', '00010')
    t10 = get_p('10011', '01000')
    diff_plus2 = t9 - t10
    
    # The metric sums the differences across three independent groups of trials.
    return diff_0 + diff_minus1 + diff_plus2
outcome: self_sim=0.0300 (var=0.0790) adversary_sim=0.2844 (var=0.2245) welch_t=-2.309 p=0.02632 (N=25, alpha=0.01) -> reject

[3] rationale: Previous metrics failed because they summed the differences across pairs of trials, which linearly increased the signal but additively compounded the per-subject binomial variance, resulting in high noise and insignificant p-values at N=25. This metric groups trials into two large pools: 'High' and 'Low' evidence for Option A under the Sequential Evidence Accumulation theory. Under the advocated Environment-Contingent Strategy Selection theory, the aggregate Take-The-Best and Tallying properties are perfectly balanced between the two pools, predicting an expected difference of exactly 0. By taking the difference of the *means* of these large pools (each containing 36 trials per subject) rather than summing individual pairwise differences, we drastically reduce the per-subject variance while isolating the SEA theory's sensitivity to secondary cue validities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # High evidence for Option A under SEA
    # T1, T2, T8, T9
    high_mask = (
        ((data['A_str'] == '11000') & (data['B_str'] == '00110')) |
        ((data['A_str'] == '10100') & (data['B_str'] == '01010')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '00011')) |
        ((data['A_str'] == '11100') & (data['B_str'] == '00010'))
    )
    
    # Low evidence for Option A under SEA
    # T3, T6, T7, T10
    low_mask = (
        ((data['A_str'] == '10010') & (data['B_str'] == '01100')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '01100')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '00110')) |
        ((data['A_str'] == '10011') & (data['B_str'] == '01000'))
    )
    
    high_mean = data[high_mask]['chose_A'].mean()
    low_mean = data[low_mask]['chose_A'].mean()
    
    if pd.isna(high_mean):
        high_mean = 0.5
    if pd.isna(low_mean):
        low_mean = 0.5
        
    return float(high_mean - low_mean)
outcome: self_sim=0.0522 (var=0.0085) adversary_sim=0.0789 (var=0.0183) welch_t=-0.814 p=0.4204 (N=25, alpha=0.01) -> reject

[4] rationale: This metric perfectly balances the Tally differences between two pools of trials while maximizing the difference in Sequential Evidence Accumulation (SEA) predictions. The 'High' pool consists of trials T4, T7, and T10, which have Tally differences of 0, -1, and +1 respectively. The 'Low' pool consists of trials T1, T5, and T12, which also have Tally differences of 0, -1, and +1 respectively. Because the advocated Environment-Contingent Strategy Selection theory uses a mixture of Take-The-Best (which always favors A here) and Tallying (which is identical on average between the two pools), it predicts an expected difference of exactly 0. In contrast, the SEA theory predicts a significantly higher probability of choosing A in the 'High' pool because the secondary cues present strongly favor A or weakly favor B, relative to the 'Low' pool where secondary cues strongly favor B or weakly favor A. By comparing the means of these two balanced 3-trial pools, we drastically reduce the per-subject variance while maintaining a strong theoretical contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # High P(A) under SEA: Weak opposing evidence from B or strong secondary evidence for A
    # T4 (Tally Diff 0): A=[1,0,0,0,0], B=[0,0,0,0,1]
    # T7 (Tally Diff -1): A=[1,0,0,0,0], B=[0,0,0,1,1]
    # T10 (Tally Diff +1): A=[1,1,0,0,0], B=[0,0,0,0,1]
    high_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '00001')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '00011')) |
        ((data['A_str'] == '11000') & (data['B_str'] == '00001'))
    )
    
    # Low P(A) under SEA: Strong opposing evidence from B or weak secondary evidence for A
    # T1 (Tally Diff 0): A=[1,0,0,0,0], B=[0,1,0,0,0]
    # T5 (Tally Diff -1): A=[1,0,0,0,0], B=[0,1,1,0,0]
    # T12 (Tally Diff +1): A=[1,0,0,1,0], B=[0,0,0,0,1]
    low_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01000')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '01100')) |
        ((data['A_str'] == '10010') & (data['B_str'] == '00001'))
    )
    
    high_mean = data[high_mask]['chose_A'].mean()
    low_mean = data[low_mask]['chose_A'].mean()
    
    if pd.isna(high_mean):
        high_mean = 0.5
    if pd.isna(low_mean):
        low_mean = 0.5
        
    return float(high_mean - low_mean)
outcome: self_sim=-0.0217 (var=0.0112) adversary_sim=0.0258 (var=0.0196) welch_t=-1.354 p=0.1826 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    # High Tally pool: Tally differences of 0 and +1\n    high_mask = (\n        ((data['A_str'] == '10000') & (data['B_str'] == '01000')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00100')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00010')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00001')) | \n        ((data['A_str'] == '11000') & (data['B_str'] == '00001')) | \n        ((data['A_str'] == '10100') & (data['B_str'] == '00001')) | \n        ((data['A_str'] == '10010') & (data['B_str'] == '00001'))   \n    )\n    \n    # Low Tally pool: Tally differences of -1 and -2\n    low_mask = (\n        ((data['A_str'] == '10000') & (data['B_str'] == '01100')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00110')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00011')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '01110')) | \n        ((data['A_str'] == '10000') & (data['B_str'] == '00111'))   \n    )\n    \n    high_mean = data[high_mask]['chose_A'].mean()\n    low_mean = data[low_mask]['chose_A'].mean()\n    \n    if pd.isna(high_mean): high_mean = 0.5\n    if pd.isna(low_mean): low_mean = 0.5\n        \n    return float(high_mean - low_mean)",
  "rationale": "This metric contrasts the macro-level sensitivity to the sheer number of opposing cues (Tally difference) across all trials. The advocated Environment-Contingent Strategy Selection theory explicitly incorporates a compensatory Tallying heuristic, which strongly penalizes Option A when the tally difference is negative. Consequently, ECSS predicts a large positive difference in choice probabilities between the High Tally pool (diffs 0, +1) and Low Tally pool (diffs -1, -2). In contrast, the competing Sequential Evidence Accumulation theory evaluates cues sequentially and stops when a threshold is met. Because Option A exclusively possesses the most valid cue across all trials, SEA frequently terminates search immediately at cue 1, rendering it largely blind to the secondary cues that define the Tally differences. Thus, SEA predicts a near-zero difference between these two large pools. By partitioning all 96 trials into two halves, this metric maximizes the theoretical gap while minimizing per-subject binomial variance."
}
```

## Usage

```json
{
  "prompt_token_count": 8227,
  "candidates_token_count": 893,
  "total_token_count": 20274
}
```
