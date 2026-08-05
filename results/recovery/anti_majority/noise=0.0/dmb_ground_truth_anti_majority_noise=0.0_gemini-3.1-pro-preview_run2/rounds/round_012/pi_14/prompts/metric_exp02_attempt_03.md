# metric_exp02_attempt_03

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
**Validities (n_features=10):** [0.95, 0.95, 0.95, 0.9, 0.5, 0.8, 0.6, 0.7, 0.7, 0.7]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 1, 0]

**Rationale:** This design quantitatively dissociates 'Rank-Weighted Capacity-Bounded Integration' (Advocated) from 'Evidence Dilution and Non-linear Weighting' (Competing) by exploiting the variance of excess cue validities while holding their sum and count constant. Under the Advocated theory, options sharing the exact same top K cues (for K in {1, 2, 3}) have identical base values. Furthermore, because its penalty depends solely on the raw count of excess features, two options with the same number of excess features will incur the exact same penalty, leading to a strict prediction of 50% choice probability (complete indifference). In contrast, the Competing theory calculates evidence using a convex non-linear weighting function (lambda >= 1) and dilutes by the sum of validities. By pairing options that have identical validity sums but different validity variances (e.g., {0.90, 0.50} vs {0.70, 0.70}), the Competing theory's convex numerator strictly prefers the option with higher variance, predicting a strong preference where the Advocated theory predicts a perfect tie.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
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
[0] rationale: Under the Advocated theory, the top K cues (for K=1, 2, or 3) are identical for options A and B across all trials, and the number of excess active cues is also identical. Because the penalty function in the Advocated theory is completely blind to the validities of the excess cues (it only counts them), the subjective values for A and B are mathematically identical, leading to a rigid prediction of 50% choice probability (indifference) regardless of parameter values. The Competing theory, however, integrates the validities of all present cues into both its non-linear evidence numerator and its dilution denominator. Because Option B has much stronger excess cues than Option A, the Competing theory will predict a systematic preference (either strongly favoring B due to the numerator, or strongly favoring A due to dilution). By calculating the absolute deviation of each subject's mean response from 0.5, we perfectly capture this divergence: the Advocated theory will yield values near 0 (only reflecting binomial noise), while the Competing theory will yield large deviations.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    subj_means = data.groupby('subject_id')['response'].mean()
    return float(np.mean(np.abs(subj_means - 0.5)))
outcome: self_sim=0.0619 (var=0.0034) adversary_sim=0.0627 (var=0.0048) welch_t=-0.046 p=0.9633 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Advocated theory, the penalty function is completely blind to the validities of the excess cues. Because we matched the top K cues and the number of excess cues across Options A and B in all 4 trial types, the Advocated theory predicts a rigid 50% choice probability for all trials, regardless of its parameters. Thus, the variance of the choice proportions across the 4 trial types for any given subject will solely reflect binomial noise (expected to be around 0.25 / 24 ≈ 0.0104). In contrast, the Competing theory integrates all validities into both its non-linear numerator and its dilution denominator. Because the sums of validities and the specific active cues vary substantially across the 4 trial types, the Competing theory predicts choice probabilities that fluctuate across the trial types. By calculating the within-subject variance of the mean responses across the 4 trial types, we isolate this fluctuation: the Advocated theory will yield a tight, low value (pure noise), while the Competing theory will yield a systematically higher value reflecting true changes in underlying choice probabilities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Create a unique identifier for each of the 4 trial types based on Option A's ratings
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    # Compute the mean response (choice probability for B) for each trial type per subject
    trial_means = data.groupby(['subject_id', 'trial_id'])['response'].mean()
    # Compute the variance of these 4 trial means for each subject, then average across subjects
    return float(trial_means.groupby('subject_id').var().mean())
outcome: self_sim=0.0241 (var=0.0006) adversary_sim=0.0109 (var=0.0001) welch_t=+2.597 p=0.01454 (N=25, alpha=0.01) -> reject

[2] rationale: Under the Advocated theory, Trial 1 features identical top K cues and excess counts for both options, resulting in a strict 50% choice probability. In Trial 4, Option B has strictly superior top K validities (for K=2,3), leading to a higher choice probability for B. Thus, the difference in B choices between Trial 4 and Trial 1 is positive. Conversely, under the Competing theory, the difference in the non-linear evidence numerator between B and A is identical in both trials. However, the dilution denominator ratio (B's sum of validities / A's sum of validities) is larger in Trial 4 than in Trial 1. This means Option B suffers a greater relative dilution penalty in Trial 4, predicting fewer B choices in Trial 4 compared to Trial 1. Thus, the metric (Trial 4 mean - Trial 1 mean) strongly discriminates the theories by predicting opposite signs.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 4: Option A has 3 active cues
    mask_t4 = data['option_a_ratings'].apply(lambda x: sum(x)) == 3
    # Trial 1: Option A has 5 active cues
    mask_t1 = data['option_a_ratings'].apply(lambda x: sum(x)) == 5
    
    t4_means = data[mask_t4].groupby('subject_id')['response'].mean()
    t1_means = data[mask_t1].groupby('subject_id')['response'].mean()
    
    return float((t4_means - t1_means).mean())
outcome: self_sim=0.1408 (var=0.0519) adversary_sim=-0.0017 (var=0.0176) welch_t=+2.703 p=0.01017 (N=25, alpha=0.01) -> reject

[3] rationale: Under the Advocated theory, Trials 1 and 2 feature perfectly matched top K validities (for K=1, 2, 3) and identical excess cue counts, resulting in a strict 50% choice probability. In Trials 3 and 4, Option B has superior top K validities (for K=2, 3), leading to a higher choice probability for B. Thus, the difference (T3 + T4) - (T1 + T2) is systematically positive. The Competing theory applies dilution based on the sum of validities, which affects all trial types differently and does not predict this specific directional divergence, keeping its expected difference near zero. By using all four trial types, we significantly reduce within-subject noise compared to comparing only two trials, tightening the variance and allowing the mean difference to robustly discriminate the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    m1 = a_sums == 5
    m2 = (a_sums == 4) & data['option_a_ratings'].apply(lambda x: x[2] == 1)
    m3 = (a_sums == 4) & data['option_a_ratings'].apply(lambda x: x[2] == 0)
    m4 = a_sums == 3
    
    t1 = data[m1].groupby('subject_id')['response'].mean()
    t2 = data[m2].groupby('subject_id')['response'].mean()
    t3 = data[m3].groupby('subject_id')['response'].mean()
    t4 = data[m4].groupby('subject_id')['response'].mean()
    
    return float(((t3 + t4) - (t1 + t2)).mean())
outcome: self_sim=0.2108 (var=0.0912) adversary_sim=0.0258 (var=0.0444) welch_t=+2.512 p=0.01584 (N=25, alpha=0.01) -> reject

[4] rationale: In all four trial types, the original Option A and Option B have perfectly matched top K validities and identical numbers of excess features. Thus, the Advocated theory (which penalizes based purely on the count of excess features) predicts an exact 50% choice probability for all trials, regardless of its parameters. Previous attempts failed because the options are randomly swapped (left/right) during the experiment, meaning the raw 'response' mean averages to 0.5 for BOTH theories. By exploiting the design fact that original Option B ALWAYS has a 1 in the 8th feature and original Option A ALWAYS has a 0, we can perfectly align the choices to the underlying options. The Advocated theory will show an absolute deviation from 0.5 that is purely due to binomial noise (~0.04), while the Competing theory (which integrates the differing sums of validities into its dilution denominator) will systematically pull preferences away from 0.5, yielding a significantly larger absolute deviation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The experimental design guarantees that the original Option B ALWAYS has a 1 in the 8th feature (index 7),
    # while the original Option A ALWAYS has a 0 in the 8th feature.
    # Because the options are randomly assigned to left/right (A/B) on each trial, we must align the responses.
    a_last = data['option_a_ratings'].apply(lambda x: x[7])
    b_last = data['option_b_ratings'].apply(lambda x: x[7])
    
    # chose_target is 1 if the subject chose the original Option B, and 0 if they chose original Option A.
    chose_target = np.where(data['response'] == 0, a_last, b_last)
    
    # Calculate each subject's mean probability of choosing the original Option B
    subj_means = data.assign(chose_target=chose_target).groupby('subject_id')['chose_target'].mean()
    
    # The Advocated theory predicts exactly 0.5 for all trials, so the deviation will be near 0 (only binomial noise).
    # The Competing theory predicts systematic deviations from 0.5 depending on the subject's parameters.
    return float(np.mean(np.abs(subj_means - 0.5)))

outcome: self_sim=0.0488 (var=0.0010) adversary_sim=0.0542 (var=0.0018) welch_t=-0.507 p=0.6144 (N=25, alpha=0.01) -> reject

[5] rationale: Under the Advocated theory, the penalty function strictly counts excess cues without considering their validities. Because we matched the top K cues and the number of excess cues across the original Options A and B in all 4 trial types, the Advocated theory predicts a rigid 50% choice probability for every trial, regardless of parameter values. Thus, the variance of the aligned choice proportions across the 4 trial types for any given subject will solely reflect binomial noise. In contrast, the Competing theory integrates all validities into both its non-linear numerator and its dilution denominator. Because the total sum of validities and the specific active cues vary substantially across the 4 trial types, the Competing theory predicts choice probabilities that systematically fluctuate across the trial types. By aligning the choices to the original Option B and calculating the within-subject variance of the mean responses across the 4 trial types, we perfectly isolate this fluctuation: the Advocated theory will yield a tight, low value (pure noise), while the Competing theory will yield a systematically higher value reflecting true shifts in underlying choice probabilities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # The experimental design guarantees that original Option B ALWAYS has a 1 in the 8th feature (index 7),
    # while original Option A ALWAYS has a 0 in the 8th feature.
    a_last = data['option_a_ratings'].apply(lambda x: x[-1])
    b_last = data['option_b_ratings'].apply(lambda x: x[-1])
    
    # Align responses: 1 if subject chose original Option B, 0 if original Option A
    chose_B = np.where(data['response'] == 0, a_last, b_last)
    
    # Identify trial type uniquely by the sum of cues of the original Option A (5, 6, 4, or 7)
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    orig_A_sum = np.where(a_last == 0, a_sum, b_sum)
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_type': orig_A_sum,
        'chose_B': chose_B
    })
    
    # Calculate the mean probability of choosing B for each of the 4 trial types per subject
    subj_trial_means = df.groupby(['subject_id', 'trial_type'])['chose_B'].mean()
    
    # Calculate the variance of these 4 means for each subject, then average across subjects
    subj_vars = subj_trial_means.groupby('subject_id').var()
    
    return float(subj_vars.mean())
outcome: self_sim=0.0090 (var=0.0000) adversary_sim=0.0116 (var=0.0002) welch_t=-0.849 p=0.4014 (N=25, alpha=0.01) -> reject

[6] rationale: Under the Advocated theory, the penalty function strictly counts excess cues without considering their validities. Because we perfectly matched the top K cues and the number of excess cues across the original Options A and B in all 4 trial types, the Advocated theory predicts a rigid 50% choice probability for every trial, regardless of its parameter values. Thus, the difference in choice probabilities between any subsets of trials will be exactly 0 (subject only to binomial noise). In contrast, the Competing theory integrates all validities into both its non-linear numerator and its dilution denominator. Option A always has a higher sum of validities than Option B, meaning it suffers more dilution, which pushes the preference towards Option B. This dilution discrepancy is large in Trials 1 and 3, but much smaller in Trials 2 and 4. Therefore, the Competing theory predicts a systematically higher preference for Option B in Trials 1 & 3 compared to Trials 2 & 4. By calculating the difference in the rate of choosing B between these two trial clusters, the metric yields ~0 for the Advocated theory and a systematically positive value for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Original Option B always has a 1 in the 8th feature (index 7),
    # while original Option A always has a 0 in the 8th feature.
    # We align the responses to see if the subject chose original Option B.
    a_last = data['option_a_ratings'].apply(lambda x: x[7])
    b_last = data['option_b_ratings'].apply(lambda x: x[7])
    chose_B = np.where(data['response'] == 0, a_last, b_last)
    
    # The 4 trial types can be uniquely identified by the number of active features.
    # T1: 5 active, T2: 6 active, T3: 4 active, T4: 7 active.
    # Both original A and B have the same number of active features in any given trial.
    a_sum = data['option_a_ratings'].apply(sum)
    
    df = data.assign(chose_B=chose_B)
    
    # Trials 1 and 3 (sums 5 and 4) have a large difference in the sum of validities
    # between Option A and Option B (0.70 and 0.40, respectively).
    # Trials 2 and 4 (sums 6 and 7) have a small difference in the sum of validities
    # between Option A and Option B (0.05 and 0.35, respectively).
    m_high_diff = df[a_sum.isin([4, 5])].groupby('subject_id')['chose_B'].mean()
    m_low_diff  = df[a_sum.isin([6, 7])].groupby('subject_id')['chose_B'].mean()
    
    return float((m_high_diff - m_low_diff).mean())
outcome: self_sim=0.0225 (var=0.0082) adversary_sim=-0.0008 (var=0.0137) welch_t=+0.789 p=0.4345 (N=25, alpha=0.01) -> reject

[7] rationale: Under the Advocated theory, the penalty function is completely blind to the validities of the excess cues. Because the top K cues and the number of excess cues are perfectly matched across the original Options A and B in all trials, the Advocated theory predicts a rigid 50% choice probability for every trial, regardless of its parameter values. Thus, the overall proportion of choosing Option B (p_hat) will simply be 0.5 plus binomial noise, making the squared deviation (p_hat - 0.5)^2 extremely small and tightly clustered around its expected noise variance (~0.0026). In contrast, the Competing theory integrates all validities into its dilution denominator. Because Option A always has a systematically higher sum of validities than Option B, subjects with low dilution parameters (gamma < 2) will show a strong, consistent preference across all trials (either strongly favoring A or strongly favoring B depending on lambda_val). This systematic deviation pushes their overall p_hat far away from 0.5, resulting in a much larger squared deviation. By squaring the deviation, we non-linearly amplify the signal from these subjects while suppressing the uniform binomial noise, powerfully discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The experimental design guarantees that original Option B ALWAYS has a 1 in the 8th feature (index 7),
    # while original Option A ALWAYS has a 0 in the 8th feature.
    a_last = data['option_a_ratings'].apply(lambda x: x[-1])
    b_last = data['option_b_ratings'].apply(lambda x: x[-1])
    
    # Align responses: 1 if subject chose original Option B, 0 if original Option A
    chose_B = np.where(data['response'] == 0, a_last, b_last)
    
    # Calculate the overall proportion of choosing Option B across all 96 trials
    p_hat = float(np.mean(chose_B))
    
    # Return the squared deviation from 0.5
    return (p_hat - 0.5) ** 2

outcome: self_sim=0.0000 (var=0.0000) adversary_sim=0.0000 (var=0.0000) welch_t=-0.000 p=1 (N=25, alpha=0.01) -> reject

[8] rationale: By design, every trial pairs two options that have identical top K validities (for K=1,2,3), identical counts of excess features, and an identical sum of validities. Because the Advocated theory bases its penalty solely on the raw count of excess features, it evaluates both options as mathematically identical, predicting exactly 50% choice probability for every trial. In contrast, the Competing theory evaluates options using a convex non-linear weighting function (lambda >= 1). Given a constant sum, the convex numerator strictly prefers the option with more extreme validities (higher variance). By identifying the option with the higher sum of squared validities ('Original Option A'), we perfectly align the choices. The Advocated theory will yield a mean choice proportion of ~0.5 (driven purely by binomial noise), whereas the Competing theory will yield a systematically higher mean (>0.5), robustly discriminating the two.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.95, 0.95, 0.9, 0.5, 0.8, 0.6, 0.7, 0.7, 0.7])
    val_sq = validities ** 2
    
    # Calculate sum of squared validities for the active features of A and B
    a_sq = data['option_a_ratings'].apply(lambda x: np.sum(np.array(x) * val_sq))
    b_sq = data['option_b_ratings'].apply(lambda x: np.sum(np.array(x) * val_sq))
    
    # In this design, the 'Original Option A' always has a higher variance in its excess validities,
    # which corresponds to a strictly higher sum of squared validities.
    # response == 0 means left option (option_a) was chosen, response == 1 means right (option_b).
    chose_orig_A = np.where(a_sq > b_sq, data['response'] == 0, data['response'] == 1)
    
    # Calculate the proportion of times each subject chose the Original Option A
    subj_means = data.assign(chose_orig_A=chose_orig_A).groupby('subject_id')['chose_orig_A'].mean()
    
    # Return the average proportion across all subjects
    return float(subj_means.mean())
outcome: self_sim=0.4992 (var=0.0018) adversary_sim=0.5352 (var=0.0058) welch_t=-2.063 p=0.04609 (N=25, alpha=0.01) -> reject

[9] rationale: By design, the options in every trial have identical top K validities, identical counts of excess features, and an identical sum of validities. The Advocated theory evaluates these options as mathematically identical, predicting exactly 50% choice probability for every trial. The Competing theory evaluates options using a convex non-linear weighting function (lambda >= 1), meaning it strictly prefers the option with more extreme validities (higher variance) when the sums are equal. Trial 2 has a very small variance difference and dilutes the effect. By isolating Trials 1, 3, and 4—uniquely identifiable because exactly one option possesses feature 3 (validity 0.9)—we focus on the trials with the largest variance disparity. The Advocated theory will yield a mean choice proportion of ~0.5 (driven purely by binomial noise), whereas the Competing theory will yield a systematically higher mean, robustly discriminating the two.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Feature 3 (index 3) has validity 0.9.
    # In this design, Original Option A has it in Trials 1, 3, and 4. Original Option B never has it.
    # Trial 2 features neither option having feature 3.
    a_has_3 = data['option_a_ratings'].apply(lambda x: x[3] == 1)
    b_has_3 = data['option_b_ratings'].apply(lambda x: x[3] == 1)
    
    # We isolate the trials where exactly one option has feature 3 (Trials 1, 3, 4)
    mask = a_has_3 != b_has_3
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    a_has_3_sub = a_has_3[mask]
    b_has_3_sub = b_has_3[mask]
    
    # 1 if the subject chose the option with feature 3, 0 otherwise
    chose_3 = np.where(subset['response'] == 0, a_has_3_sub, b_has_3_sub)
    
    # Calculate the mean proportion of choosing the option with feature 3 per subject
    subj_means = subset.assign(chose_3=chose_3).groupby('subject_id')['chose_3'].mean()
    
    return float(subj_means.mean())
outcome: self_sim=0.5014 (var=0.0030) adversary_sim=0.5394 (var=0.0071) welch_t=-1.884 p=0.06658 (N=25, alpha=0.01) -> reject

[10] rationale: Under the Advocated theory, the choice probability is exactly 50% for all trials because the penalty function relies solely on excess feature counts. Thus, the proportion of choosing Original Option A in Trials 1 and 3 follows a binomial distribution around 0.5. Taking the maximum of these two proportions yields an expected value of ~0.56 due strictly to binomial noise. In contrast, the Competing theory integrates validities non-linearly, meaning it strictly prefers the option with higher validity variance (Original Option A). This preference is mathematically strongest in Trials 1 and 3. By taking the maximum of the choice proportions for Original Option A across these two specific trial types, we isolate the strongest possible signal of the Competing theory, pushing its expected metric significantly higher (>0.65). This max-pooling robustly discriminates the theories by amplifying the Competing theory's signal while keeping the Advocated theory's variance tightly bounded.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.95, 0.95, 0.9, 0.5, 0.8, 0.6, 0.7, 0.7, 0.7])
    val_sq = validities ** 2
    
    # Calculate sum of squared validities to perfectly identify Original Option A
    # Original Option A always has the higher variance in its excess validities
    a_sq = data['option_a_ratings'].apply(lambda x: np.sum(np.array(x) * val_sq))
    b_sq = data['option_b_ratings'].apply(lambda x: np.sum(np.array(x) * val_sq))
    
    # 1 if subject chose Original Option A, 0 otherwise
    chose_orig_A = np.where(a_sq > b_sq, data['response'] == 0, data['response'] == 1)
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'chose_orig_A': chose_orig_A,
        'a_sq': a_sq,
        'b_sq': b_sq,
        'option_a_ratings': data['option_a_ratings'],
        'option_b_ratings': data['option_b_ratings']
    })
    
    def get_orig_A(row):
        if row['a_sq'] > row['b_sq']:
            return tuple(row['option_a_ratings'])
        else:
            return tuple(row['option_b_ratings'])
            
    df['orig_A'] = df.apply(get_orig_A, axis=1)
    
    # Trials 1 and 3 are exactly those where Original Option A is [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    target_A = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    df_t1_t3 = df[df['orig_A'] == target_A].copy()
    
    if df_t1_t3.empty:
        return 0.5
        
    # Uniquely identify T1 vs T3 by Original Option B
    def get_orig_B(row):
        if row['a_sq'] > row['b_sq']:
            return tuple(row['option_b_ratings'])
        else:
            return tuple(row['option_a_ratings'])
            
    df_t1_t3['orig_B'] = df_t1_t3.apply(get_orig_B, axis=1)
    
    # For each subject, calculate the mean choice for T1 and T3, then take the max
    def subject_max(sub_df):
        return sub_df.groupby('orig_B')['chose_orig_A'].mean().max()
        
    subj_maxes = df_t1_t3.groupby('subject_id').apply(subject_max)
    
    return float(subj_maxes.mean())

outcome: self_sim=0.5475 (var=0.0069) adversary_sim=0.6017 (var=0.0092) welch_t=-2.138 p=0.03778 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    validities = np.array([0.95, 0.95, 0.95, 0.9, 0.5, 0.8, 0.6, 0.7, 0.7, 0.7])\n    # Use a high power to strongly emphasize the difference in extreme validities\n    # which the Competing theory's convex weighting function strictly prefers.\n    weights = validities ** 6\n    \n    # Calculate weighted sum for A and B\n    a_w = data['option_a_ratings'].apply(lambda x: np.sum(np.array(x) * weights))\n    b_w = data['option_b_ratings'].apply(lambda x: np.sum(np.array(x) * weights))\n    \n    # chosen option's weighted sum minus unchosen option's weighted sum\n    chosen_w = np.where(data['response'] == 0, a_w, b_w)\n    unchosen_w = np.where(data['response'] == 0, b_w, a_w)\n    \n    diff = chosen_w - unchosen_w\n    \n    # Calculate the mean difference for each subject\n    subj_means = data.assign(diff=diff).groupby('subject_id')['diff'].mean()\n    \n    # Return the average across subjects\n    return float(subj_means.mean())",
  "rationale": "Under the Advocated theory, options in all trials have identical top K validities and identical counts of excess features, resulting in a strict prediction of 50% choice probability for every trial. Therefore, the expected difference in any weighted feature sum between the chosen and unchosen option is exactly 0, subject only to tight binomial noise. In contrast, the Competing theory evaluates options using a convex non-linear weighting function (lambda >= 1), meaning it systematically prefers the option with more extreme validities (higher variance) when the sum of validities is matched. By raising the validities to the 6th power, we create a 'weight' that perfectly tracks the Competing theory's non-linear numerator. The metric computes the difference in this weighted sum between the chosen and unchosen option. The Advocated theory will yield a tight distribution around 0, while the Competing theory will yield a robust, systematically positive value, powerfully discriminating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 10365,
  "candidates_token_count": 570,
  "total_token_count": 16246
}
```
