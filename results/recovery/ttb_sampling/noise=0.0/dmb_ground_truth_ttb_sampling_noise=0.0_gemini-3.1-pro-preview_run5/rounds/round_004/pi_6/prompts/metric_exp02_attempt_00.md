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
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate pure TTB from Probabilistic Cue Retrieval TTB, we use a 6-feature environment with linearly decreasing validities from 1.0 to 0.5. We manipulate the validity of the first discriminating cue by systematically tying the most valid cues. In the 'oppose' condition, the first discriminating cue favors Option A, but all subsequent cues favor Option B. Pure TTB strictly stops at the first discriminating cue, predicting a constant, high probability of choosing Option A across all these trials regardless of which cue discriminates. In contrast, Probabilistic Cue Retrieval TTB predicts that as the validity of the first discriminating cue decreases, the probability of skipping it increases. Consequently, for the 'oppose' trials, Probabilistic TTB predicts a monotonically decreasing probability of choosing Option A. We also include 'agree' trials where subsequent cues also favor Option A, serving as a control where both models predict high choice probabilities, confirming the divergence is driven by cue retrieval failures falling back on opposing cues.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Retrieval TTB: Decision-makers follow the non-compensatory, sequential search architecture of Take The Best (TTB), but execution is noisy at the cue level. When a discriminating cue is evaluated, there is a probability of skipping it or failing to retrieve its value. This skip probability is a function of a base skip rate and is inversely proportional to the cue's validity (operationalized as proportional to 1 - validity). If a cue is skipped, the decision-maker continues to the next most valid cue. This produces slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration.

**Parameters:**
- p_skip: [0.0, 0.4]
- epsilon: [0.0, 0.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='stable')
    
    p_skip_base = float(parameters["p_skip"])
    
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    # Sequential search with probabilistic cue retrieval
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of skipping this cue is proportional to its invalidity
            p_skip_cue = p_skip_base * (1.0 - validities[idx])
            p_eval = 1.0 - p_skip_cue
            
            if a[idx] > b[idx]:
                p_A += p_continue * p_eval
            else:
                p_B += p_continue * p_eval
                
            p_continue *= p_skip_cue
            
    # If all discriminating cues are skipped, or no cues discriminate, guess randomly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    # Global response noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: This metric calculates the difference in choice probability for Option A between trials where the downstream cues agree with the first discriminating cue versus trials where they oppose it. Pure TTB strictly stops at the first discriminating cue, so it predicts identical choice probabilities (and a metric value of 0). Probabilistic Cue Retrieval TTB posits that cues can be skipped, so it predicts a lower probability of choosing A when downstream cues oppose it, leading to a positive metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Create string representation of A and B
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # We want to measure the rate of choosing A (response == 0)
    data['choose_A'] = (data['response'] == 0).astype(float)
    
    # Trial 3: First disc cue is 1. Opposing downstream cues.
    t3 = data[(data['A_str'] == '11000') & (data['B_str'] == '10111')]['choose_A'].mean()
    # Trial 4: First disc cue is 1. Agreeing downstream cues.
    t4 = data[(data['A_str'] == '11110') & (data['B_str'] == '10001')]['choose_A'].mean()
    # Trial 5: First disc cue is 2. Opposing downstream cues.
    t5 = data[(data['A_str'] == '11100') & (data['B_str'] == '11011')]['choose_A'].mean()
    # Trial 6: First disc cue is 2. Agreeing downstream cues.
    t6 = data[(data['A_str'] == '11110') & (data['B_str'] == '11001')]['choose_A'].mean()
    
    # Handle NaNs just in case
    t3 = t3 if pd.notna(t3) else 0.5
    t4 = t4 if pd.notna(t4) else 0.5
    t5 = t5 if pd.notna(t5) else 0.5
    t6 = t6 if pd.notna(t6) else 0.5
    
    return (t4 - t3) + (t6 - t5)

outcome: self_sim=0.0433 (var=0.0256) adversary_sim=-0.0200 (var=0.0238) welch_t=+1.425 p=0.1607 (N=25, alpha=0.01) -> reject

[1] rationale: By aggregating multiple trials into two sets, we reduce the per-subject variance. The 'agree' set consists of trials where the downstream cues agree with the first discriminating cue. The 'oppose' set consists of trials where the first discriminating cue has a lower validity (higher skip probability) and the downstream cues oppose it. For pure TTB, the choice is solely determined by the first discriminating cue, so the TTB adherence rate should be identical in both sets (metric = 0). For Probabilistic Cue Retrieval TTB, the skip rate is higher in the 'oppose' set and skipping leads to a different choice, so adherence to the TTB prediction drops significantly. Thus, the metric will be positive for PCR-TTB and near zero for pure TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trials where downstream cues AGREE with the first discriminating cue
    agree_mask = (
        ((data['A_str'] == '11110') & (data['B_str'] == '00001')) | # Trial 2
        ((data['A_str'] == '11110') & (data['B_str'] == '10001')) | # Trial 4
        ((data['A_str'] == '11110') & (data['B_str'] == '11001'))   # Trial 6
    )
    
    # Trials where downstream cues OPPOSE the first discriminating cue (and validity is lower)
    oppose_mask_A = (
        ((data['A_str'] == '11100') & (data['B_str'] == '11011')) | # Trial 5
        ((data['A_str'] == '11110') & (data['B_str'] == '11101'))   # Trial 7
    )
    oppose_mask_B = (
        ((data['A_str'] == '11101') & (data['B_str'] == '11110'))   # Trial 8
    )
    
    # TTB prediction is A (0) for agree trials and oppose_mask_A, and B (1) for oppose_mask_B
    agree_ttb = (data[agree_mask]['response'] == 0).mean()
    
    oppose_ttb_A = (data[oppose_mask_A]['response'] == 0).sum()
    oppose_ttb_B = (data[oppose_mask_B]['response'] == 1).sum()
    oppose_total = oppose_mask_A.sum() + oppose_mask_B.sum()
    
    if oppose_total == 0 or agree_mask.sum() == 0:
        return 0.0
        
    oppose_ttb = (oppose_ttb_A + oppose_ttb_B) / oppose_total
    
    return float(agree_ttb - oppose_ttb)
outcome: self_sim=0.0539 (var=0.0075) adversary_sim=-0.0033 (var=0.0062) welch_t=+2.441 p=0.01841 (N=25, alpha=0.01) -> reject

[2] rationale: We maximize the contrast between the two models by comparing TTB adherence on the two most extreme sets of trials. 'Early Agree' trials have highly valid first cues and agreeing downstream cues, so both pure TTB and PCR-TTB predict near-perfect adherence (modulo base noise). 'Late Oppose' trials rely on much less valid first cues (which PCR-TTB skips with high probability) and have opposing downstream cues (which flip the choice when the first cue is skipped). Pure TTB predicts identical adherence across both sets (metric = 0), whereas PCR-TTB predicts a massive drop in adherence for the Late Oppose trials, yielding a large positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Early Agree: Highest validity discriminating cues, downstream cues agree.
    # TTB adherence will be extremely high for both pure TTB and PCR-TTB.
    mask_early_agree = (
        ((data['A_str'] == '11110') & (data['B_str'] == '00001')) | # Trial 2
        ((data['A_str'] == '11110') & (data['B_str'] == '10001'))   # Trial 4
    )
    
    # Late Oppose: Lowest validity discriminating cues, downstream cues oppose.
    # Pure TTB still strictly adheres, but PCR-TTB skips these often and flips choice.
    mask_late_oppose_A = ((data['A_str'] == '11110') & (data['B_str'] == '11101')) # Trial 7
    mask_late_oppose_B = ((data['A_str'] == '11101') & (data['B_str'] == '11110')) # Trial 8
    
    adh_early = (data[mask_early_agree]['response'] == 0).mean()
    
    late_A_correct = (data[mask_late_oppose_A]['response'] == 0).sum()
    late_B_correct = (data[mask_late_oppose_B]['response'] == 1).sum()
    total_late = mask_late_oppose_A.sum() + mask_late_oppose_B.sum()
    
    adh_late = (late_A_correct + late_B_correct) / total_late if total_late > 0 else 0.5
    
    # Return the contrast in TTB adherence between the most robust and most fragile trials
    return float(adh_early - adh_late)
outcome: self_sim=0.0583 (var=0.0093) adversary_sim=0.0008 (var=0.0075) welch_t=+2.216 p=0.03156 (N=25, alpha=0.01) -> reject

[3] rationale: By deriving a direct estimator for the 'p_skip' parameter in PCR-TTB using all 96 trials, we maximize the mean difference while minimizing variance. Pure TTB strictly follows the first discriminating cue, so the adherence difference between 'agree' and 'oppose' trials is zero, yielding an estimate of 0. PCR-TTB predicts that adherence drops on 'oppose' trials proportionally to the invalidity of the first discriminating cue. This metric calculates the weighted least squares estimate of the skip rate based on this theoretical relationship, yielding a strong positive value for PCR-TTB and near-zero for pure TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Agree trials: TTB adherence is highly robust for both theories
    mask_agree = (
        ((data['A_str'] == '11110') & (data['B_str'] == '00001')) |
        ((data['A_str'] == '11110') & (data['B_str'] == '10001')) |
        ((data['A_str'] == '11110') & (data['B_str'] == '11001'))
    )
    adh_agree = (data[mask_agree]['response'] == 0).mean()
    if pd.isna(adh_agree): adh_agree = 0.5
    
    # Oppose trials: PCR-TTB predicts adherence drops proportionally to cue invalidity
    mask_t1 = ((data['A_str'] == '10000') & (data['B_str'] == '01111'))
    mask_t3 = ((data['A_str'] == '11000') & (data['B_str'] == '10111'))
    mask_t5 = ((data['A_str'] == '11100') & (data['B_str'] == '11011'))
    mask_t7 = ((data['A_str'] == '11110') & (data['B_str'] == '11101'))
    mask_t8 = ((data['A_str'] == '11101') & (data['B_str'] == '11110'))
    
    adh_t1 = (data[mask_t1]['response'] == 0).mean()
    adh_t3 = (data[mask_t3]['response'] == 0).mean()
    adh_t5 = (data[mask_t5]['response'] == 0).mean()
    adh_t7 = (data[mask_t7]['response'] == 0).mean()
    adh_t8 = (data[mask_t8]['response'] == 1).mean()
    
    def safe_mean(val): return val if pd.notna(val) else 0.5
    adh_t1, adh_t3, adh_t5, adh_t7, adh_t8 = map(safe_mean, [adh_t1, adh_t3, adh_t5, adh_t7, adh_t8])
    
    # PCR-TTB predicts: adh_agree - adh_oppose_i = invalidity_i * p_skip
    # We estimate p_skip using weighted least squares (sum(inv * diff) / sum(inv^2))
    # Invalidities: t1=0.05, t3=0.15, t5=0.25, t7=0.35, t8=0.35. Sum = 1.15. Sum of squares = 0.3325.
    numerator = 1.15 * adh_agree - (0.05 * adh_t1 + 0.15 * adh_t3 + 0.25 * adh_t5 + 0.35 * adh_t7 + 0.35 * adh_t8)
    p_skip_est = numerator / 0.3325
    
    return float(p_skip_est)
outcome: self_sim=0.1643 (var=0.0739) adversary_sim=-0.0226 (var=0.0555) welch_t=+2.597 p=0.01252 (N=25, alpha=0.01) -> reject

[4] rationale: In this design, Option A is the pure TTB choice across all 8 trials. In 'Agree' trials, Option A has all positive features, meaning that even if PCR-TTB skips the first discriminating cue, the subsequent cues still favor A. In 'Late Oppose' trials (Trials 5 and 7), the first discriminating cue has low validity (high skip probability), and the subsequent cues strongly favor B. Pure TTB predicts identical adherence to Option A across both sets. PCR-TTB predicts a significant drop in choosing A for the Late Oppose trials. By using half the trials for the baseline and a quarter for the contrast, we maximize the mean difference while keeping per-subject variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of features present in option A
    data['A_sum'] = data['option_a_ratings'].apply(sum)
    
    # Agree trials (Trials 2, 4, 6, 8): Option A has all 5 features = 1.
    # Downstream cues agree with the first discriminating cue, meaning that even if
    # a cue is skipped under PCR-TTB, subsequent cues still strongly favor Option A.
    agree_mask = data['A_sum'] == 5
    
    # Late Oppose trials (Trials 5, 7): Option A has 3 or 4 features = 1.
    # The first discriminating cue has lower validity (higher skip probability),
    # and downstream cues strongly OPPOSE it (favoring Option B).
    oppose_mask = data['A_sum'].isin([3, 4])
    
    # Calculate the probability of choosing Option A (response == 0)
    p_a_agree = (data[agree_mask]['response'] == 0).mean()
    p_a_oppose = (data[oppose_mask]['response'] == 0).mean()
    
    if pd.isna(p_a_agree): p_a_agree = 0.5
    if pd.isna(p_a_oppose): p_a_oppose = 0.5
        
    # Pure TTB strictly stops at the first discriminating cue (which always favors A here),
    # so it predicts identical choice probabilities (metric ~ 0).
    # PCR-TTB predicts a massive drop in choosing A on the late oppose trials due to cue skipping,
    # yielding a large positive metric.
    return float(p_a_agree - p_a_oppose)
outcome: self_sim=0.0442 (var=0.0054) adversary_sim=0.0246 (var=0.0061) welch_t=+0.913 p=0.3658 (N=25, alpha=0.01) -> reject

[5] rationale: By taking advantage of the specific experimental design where the sum of Option A's features perfectly identifies the trial type, we can aggregate all 96 trials per subject to minimize variance. Pure TTB strictly stops at the first discriminating cue (which always favors A), predicting identical choice probabilities across all trials (metric ≈ 0). PCR-TTB predicts that cue skipping increases as validity decreases, causing choices to flip to Option B specifically on the 'Oppose' trials. By weighting the Oppose trials proportionally to their cue invalidity, we maximize the theoretical contrast while keeping per-subject variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # In this design, Option A is the pure TTB choice for ALL 8 trials.
    # We can uniquely identify the trials by the sum of Option A's features.
    # Agree trials (2, 4, 6, 8) all have A_sum == 5.
    # Oppose trials (1, 3, 5, 7) have A_sum == 1, 2, 3, 4 respectively.
    data['A_sum'] = data['option_a_ratings'].apply(sum)
    
    # Baseline: P(A) on Agree trials. Both models predict high adherence.
    baseline = (data[data['A_sum'] == 5]['response'] == 0).mean()
    if pd.isna(baseline):
        baseline = 0.5
        
    # For Oppose trials, PCR-TTB predicts that the probability of choosing A 
    # drops proportionally to the invalidity of the first discriminating cue.
    # The invalidities are 0.05, 0.15, 0.25, 0.35 for A_sum = 1, 2, 3, 4.
    # We construct a weighted average of P(A) on Oppose trials, weighting 
    # heavily the trials where PCR-TTB predicts the largest drop.
    oppose_weighted = 0.0
    total_weight = 0.0
    
    for a_sum_val in [1, 2, 3, 4]:
        mask = data['A_sum'] == a_sum_val
        if mask.sum() > 0:
            p_a = (data[mask]['response'] == 0).mean()
            # Weights proportional to cue invalidity: 1, 3, 5, 7
            weight = 2 * a_sum_val - 1
            oppose_weighted += p_a * weight
            total_weight += weight
            
    if total_weight > 0:
        oppose_weighted /= total_weight
    else:
        oppose_weighted = baseline
        
    # Pure TTB predicts identical P(A) across all trials, so the difference is ~0.
    # PCR-TTB predicts a large drop in the weighted Oppose trials, yielding a strong positive metric.
    return float(baseline - oppose_weighted)
outcome: self_sim=0.0356 (var=0.0038) adversary_sim=0.0165 (var=0.0053) welch_t=+1.003 p=0.321 (N=25, alpha=0.01) -> reject

[6] rationale: The difference in TTB adherence between Agree and Oppose trials is theoretically 0 for Pure TTB, but positive for PCR-TTB due to cue skipping. However, the raw probability difference is small (~4%) and gets buried by binomial noise at N=25. To cross the significance threshold, we perfectly split the 96 trials into equal halves (48 Agree vs 48 Oppose) to minimize variance, and apply a logit transformation. The logit transformation heavily amplifies probability differences near 1.0. This means subjects with low global noise (epsilon) who exhibit even a few cue skips will produce a massive positive spike in the metric, pulling the PCR-TTB mean far away from Pure TTB while keeping the adversary's expected value strictly at zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of features present in Option B
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    # In this design, Option A is the pure TTB choice for all 8 trial types.
    # Agree trials (2, 4, 6, 8) all have B_sum < 4.
    is_agree = data['B_sum'] < 4
    
    # Oppose trials (1, 3, 5, 7) all have B_sum == 4.
    is_oppose = data['B_sum'] == 4
    
    n_agree = is_agree.sum()
    n_oppose = is_oppose.sum()
    
    if n_agree == 0 or n_oppose == 0:
        return 0.0
        
    # Count how many times Option A was chosen (response == 0)
    a_agree = (data[is_agree]['response'] == 0).sum()
    a_oppose = (data[is_oppose]['response'] == 0).sum()
    
    # Laplace smoothing to avoid log(0) and stabilize variance.
    # Because n_agree == n_oppose == 48, this smoothing introduces ZERO bias
    # if a subject is perfectly consistent (e.g., chooses A every single time).
    p_agree = (a_agree + 0.5) / (n_agree + 1.0)
    p_oppose = (a_oppose + 0.5) / (n_oppose + 1.0)
    
    # Apply logit transformation to amplify differences near P(A) = 1.0
    logit_agree = np.log(p_agree / (1.0 - p_agree))
    logit_oppose = np.log(p_oppose / (1.0 - p_oppose))
    
    return float(logit_agree - logit_oppose)
outcome: self_sim=0.2690 (var=0.3929) adversary_sim=0.1072 (var=0.8339) welch_t=+0.731 p=0.469 (N=25, alpha=0.01) -> reject

[7] rationale: Previous attempts failed either due to buggy trial classification (e.g., misclassifying Trial 8 as an 'Oppose' trial, or using non-existent feature strings) or by diluting the effect with Early trials (Trials 1-4) where the cue skip probability is too low to produce a measurable difference. By strictly isolating the 'Late' trials (5, 6, 7, 8) where the first discriminating cue has the lowest validities (0.75 and 0.65) and thus the highest skip probabilities, we maximize the theoretical mean difference between Pure TTB and PCR-TTB. Matching exactly on the sum of features ensures we cleanly separate the 'Late Agree' baseline from the 'Late Oppose' contrast without any string-parsing errors, keeping per-subject variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of features to uniquely identify trial types without string matching
    data['A_sum'] = data['option_a_ratings'].apply(sum)
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    # Late Agree trials (Trial 6 and 8): 
    # The first discriminating cue has lower validity (high skip probability).
    # However, if skipped, the downstream cues STILL strongly favor Option A.
    # Both Pure TTB and PCR-TTB predict high adherence to Option A.
    mask_agree = (data['A_sum'] == 5) & (data['B_sum'].isin([2, 3]))
    
    # Late Oppose trials (Trial 5 and 7):
    # The first discriminating cue has lower validity (high skip probability).
    # If skipped, the downstream cues strongly OPPOSE it (favoring Option B).
    # Pure TTB predicts identical adherence to Option A as in the Agree trials.
    # PCR-TTB predicts a significant drop in choosing Option A.
    mask_oppose = (data['B_sum'] == 4) & (data['A_sum'].isin([3, 4]))
    
    # Calculate the probability of choosing Option A (response == 0)
    p_agree = (data[mask_agree]['response'] == 0).mean()
    p_oppose = (data[mask_oppose]['response'] == 0).mean()
    
    if pd.isna(p_agree): p_agree = 0.5
    if pd.isna(p_oppose): p_oppose = 0.5
        
    # The difference will be ~0 for Pure TTB, and strongly positive for PCR-TTB.
    return float(p_agree - p_oppose)
outcome: self_sim=0.0442 (var=0.0064) adversary_sim=0.0250 (var=0.0072) welch_t=+0.821 p=0.416 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Calculate the sum of features to cleanly identify trial types\n    data['A_sum'] = data['option_a_ratings'].apply(sum)\n    data['B_sum'] = data['option_b_ratings'].apply(sum)\n    \n    # 'Agree' trials where the first discriminating cue has high skip probability (Trials 9 and 10)\n    # A dominates B on all remaining cues. Both theories predict high adherence to A.\n    mask_agree = (data['A_sum'] == 6) & (data['B_sum'] >= 3)\n    \n    # 'Oppose' trials where the first discriminating cue has high skip probability (Trials 4 and 5)\n    # A is favored by the first discriminating cue, but B dominates on all remaining cues.\n    # Pure TTB predicts the same high adherence to A as in the Agree trials.\n    # PCR-TTB predicts a significant drop in choosing A because the cue is skipped frequently.\n    mask_oppose = (data['B_sum'] == 5) & (data['A_sum'] >= 4)\n    \n    p_agree = (data[mask_agree]['response'] == 0).mean()\n    p_oppose = (data[mask_oppose]['response'] == 0).mean()\n    \n    if pd.isna(p_agree): p_agree = 0.5\n    if pd.isna(p_oppose): p_oppose = 0.5\n        \n    # The difference will be ~0 for Pure TTB, and strongly positive for PCR-TTB.\n    return float(p_agree - p_oppose)",
  "rationale": "Previous metrics failed because they misidentified the trial types due to confusing string matching, mixing up Agree and Oppose trials. This metric cleanly identifies the exact trials using the sum of features. Trials 1-5 are purely 'Oppose' trials (where downstream cues favor B), uniquely identified by B_sum == 5. Trials 6-10 are purely 'Agree' trials (where downstream cues favor A), uniquely identified by A_sum == 6. By contrasting only the trials with the lowest validity discriminating cues (Trials 4, 5 vs 9, 10), we isolate the scenarios where PCR-TTB predicts the highest skip rates, maximizing the mean difference while keeping the trial sets perfectly symmetric to minimize variance."
}
```

## Usage

```json
{
  "prompt_token_count": 9746,
  "candidates_token_count": 571,
  "total_token_count": 14573
}
```
