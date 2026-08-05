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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Probabilistic Search Take-The-Best (PS-TTB) from Tallying (Equal Weighting), we employ a 5-feature design with a steep gradient in cue validities. Tallying predicts choices based entirely on the simple count of positive features, ignoring their validities. PS-TTB, in contrast, probabilistically searches cues weighted by their validities and stops at the first discriminating cue. The design includes 'conflict' trials where one option has more positive features overall (favored by Tallying) but the other option is favored by the highest-validity cue (favored by PS-TTB). It also features 'tie' trials where both options have an equal number of positive features (Tallying predicts guessing) but PS-TTB strongly favors the option with higher-validity features. This setup ensures maximal divergence in the predicted choice probabilities between the two models.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Parameters:**
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
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
**Description:** Tallying (Equal Weighting) posits that decision-makers simply count the total number of positive features for each option, ignoring the differential validities of the cues. The option with the higher count of positive features is preferred. This represents a compensatory but highly frugal heuristic, where evidence is accumulated equally across all available cues. If the counts are equal, the decision-maker guesses. Response noise is modeled via a softmax over these counts with an independent lapse rate. To account for empirical choices that often deviate from pure tallying on conflict trials, the decision process incorporates substantial choice noise.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: This metric isolates 'conflict' trials where the option favored by the single most valid cue (cue 0) has fewer total positive features than the competing option. PS-TTB will predominantly choose the option with cue 0, leading to a high value on this metric. Conversely, Tallying ignores validities and counts features, so it will predominantly choose the competing option, leading to a low value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue0_A = data['option_a_ratings'].apply(lambda x: x[0])
    cue0_B = data['option_b_ratings'].apply(lambda x: x[0])
    
    sum_A = data['option_a_ratings'].apply(sum)
    sum_B = data['option_b_ratings'].apply(sum)
    
    conflict_mask = ((cue0_A == 1) & (sum_A < sum_B)) | ((cue0_B == 1) & (sum_B < sum_A))
    
    conflict_data = data[conflict_mask]
    if len(conflict_data) == 0:
        return 0.5
        
    chose_cue0 = ((conflict_data['response'] == 0) & (conflict_data['option_a_ratings'].apply(lambda x: x[0]) == 1)) | \
                 ((conflict_data['response'] == 1) & (conflict_data['option_b_ratings'].apply(lambda x: x[0]) == 1))
                 
    return float(chose_cue0.mean())
outcome: self_sim=0.4067 (var=0.0098) adversary_sim=0.3742 (var=0.0181) welch_t=+0.973 p=0.3358 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates 'tie' trials where both options have an equal number of positive features. On these trials, Tallying (Equal Weighting) perceives no difference in overall value and will guess randomly (predicting exactly 0.5). However, on all tie trials in this design, the highest validity cue (cue 0) strictly favors one of the options. Probabilistic Search Take-The-Best (PS-TTB) will heavily favor the option with the highest validity cue. Thus, Tallying predicts a tight distribution around 0.5, while PS-TTB predicts a significantly higher proportion of choices aligned with cue 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    tie_mask = (sum_a == sum_b)
    tie_data = data[tie_mask]
    
    if len(tie_data) == 0:
        return 0.5
        
    cue0_a = tie_data['option_a_ratings'].apply(lambda x: x[0])
    cue0_b = tie_data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_cue0 = ((tie_data['response'] == 0) & (cue0_a == 1)) | \
                 ((tie_data['response'] == 1) & (cue0_b == 1))
                 
    return float(chose_cue0.mean())
outcome: self_sim=0.5178 (var=0.0058) adversary_sim=0.5217 (var=0.0076) welch_t=-0.168 p=0.8672 (N=25, alpha=0.01) -> reject

[2] rationale: This metric computes the difference in the probability of choosing Option A between Trial 7 (where A has 3 positive features and B has 1, but B has the highest-validity cue) and Trial 1 (where A has 1 positive feature and B has 4, but A has the highest-validity cue). Tallying strongly prefers A on Trial 7 and B on Trial 1, leading to a large positive difference. PS-TTB's preferences are heavily influenced by the highest-validity cue, pulling the difference towards zero or even negative values (depending on the search temperature).
metric_source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    trial_7_mask = (sum_a == 3) & (sum_b == 1)
    trial_1_mask = (sum_a == 1) & (sum_b == 4)
    
    p_a_7 = 1.0 - data[trial_7_mask]['response'].mean()
    p_a_1 = 1.0 - data[trial_1_mask]['response'].mean()
    
    if pd.isna(p_a_7): p_a_7 = 0.5
    if pd.isna(p_a_1): p_a_1 = 0.5
    
    return float(p_a_7 - p_a_1)
outcome: self_sim=0.2750 (var=0.0951) adversary_sim=0.3167 (var=0.1164) welch_t=-0.453 p=0.6526 (N=25, alpha=0.01) -> reject

[3] rationale: This metric measures the difference in the probability of choosing Option A between Trial 3 and Trial 7. On Trial 3, Option A has fewer positive features (2 vs 3) but is favored by the two most valid cues; thus, Tallying will prefer B while PS-TTB will strongly prefer A. On Trial 7, Option A has more positive features (3 vs 1) but Option B is favored by the single most valid cue; thus, Tallying will strongly prefer A while PS-TTB will prefer B. Consequently, Tallying predicts a negative difference (low P(A) on T3 minus high P(A) on T7), whereas PS-TTB predicts a positive difference (high P(A) on T3 minus low P(A) on T7).
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t7_mask = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 0))
    
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    p_a_t7 = 1.0 - data[t7_mask]['response'].mean()
    
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    if pd.isna(p_a_t7): p_a_t7 = 0.5
    
    return float(p_a_t3 - p_a_t7)

outcome: self_sim=-0.1900 (var=0.0614) adversary_sim=-0.2700 (var=0.0949) welch_t=+1.012 p=0.3169 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 10. On Trial 1 (A=[1,1,0,0,0], B=[0,0,1,1,1]), Tallying prefers B due to a higher feature sum (3 vs 2), whereas PS-TTB prefers A because the highest-validity cues favor A. Conversely, on Trial 10 (A=[0,1,1,1,1], B=[1,0,0,0,0]), Tallying strongly prefers A (4 vs 1), whereas PS-TTB prefers B because the highest-validity cue favors B. Thus, Tallying predicts a strongly negative difference (low minus high), while PS-TTB predicts a strongly positive difference (high minus low), providing a highly robust double dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t10_mask = (a_tuples == (0, 1, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t10 = 1.0 - data[t10_mask]['response'].mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t10): p_a_t10 = 0.5
    
    return float(p_a_t1 - p_a_t10)
outcome: self_sim=-0.1422 (var=0.0711) adversary_sim=-0.2356 (var=0.1080) welch_t=+1.103 p=0.2759 (N=25, alpha=0.01) -> reject

[5] rationale: Previous metrics failed because they relied on a single trial or a pair of trials, leading to excessively high between-subject variance. This metric instead identifies ALL 'conflict' trials in the dataset—trials where the option favored by the highest-validity discriminating cue (predicted by PS-TTB) has fewer total positive features than the competing option (predicted by Tallying). In this 10-trial design, exactly 8 out of 10 trials are conflict trials. By calculating the proportion of choices that align with the highest-validity cue across all 8 conflict trials (amounting to ~72 trials per subject), we drastically reduce trial-level noise. PS-TTB will consistently yield values > 0.5 (close to 1), whereas Tallying will yield values < 0.5 (close to 0), providing a massive mean gap with extremely tight per-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    # Identify the first discriminating cue for each trial
    first_nonzero_idx = np.argmax(diff != 0, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nonzero_idx]
    
    # fda: first diff favors Option A; fdb: first diff favors Option B
    fda = first_diff > 0
    fdb = first_diff < 0
    
    # Identify options favored by the simple sum of features (Tallying)
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    sfa = sum_a > sum_b
    sfb = sum_a < sum_b
    
    # Isolate conflict trials where the highest valid cue and the feature sum point in opposite directions
    conflict_mask = (fda & sfb) | (fdb & sfa)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values[conflict_mask]
    fda_conflict = fda[conflict_mask]
    fdb_conflict = fdb[conflict_mask]
    
    # Did the subject choose the option favored by the first discriminating cue?
    chose_first_diff = (fda_conflict & (responses == 0)) | (fdb_conflict & (responses == 1))
    
    return float(np.mean(chose_first_diff))
outcome: self_sim=0.4422 (var=0.0041) adversary_sim=0.3994 (var=0.0143) welch_t=+1.573 p=0.1243 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    # Calculate the sum of positive features for each option\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Exclude trials where both options have the same number of positive features\n    mask = sum_a != sum_b\n    valid_data = data[mask]\n    \n    if len(valid_data) == 0:\n        return 0.0\n        \n    # Determine the 'majority' option (0 for A, 1 for B)\n    maj_choice = np.where(sum_a[mask] > sum_b[mask], 0, 1)\n    \n    # Check if the subject chose the majority option\n    is_maj = (valid_data['response'] == maj_choice).astype(float)\n    \n    # Assign back to a copy to avoid SettingWithCopyWarning\n    valid_data = valid_data.copy()\n    valid_data['is_maj'] = is_maj\n    \n    # Calculate the proportion of majority choices per subject\n    subj_acc = valid_data.groupby('subject_id')['is_maj'].mean()\n    \n    # Return the mean of the 4th power across subjects to amplify the upper tail\n    return float(np.mean(subj_acc ** 4))",
  "rationale": "Tallying makes decisions based on the difference in the number of positive features. Subjects using Tallying will consistently choose the option with more positive features (the 'majority' option), and their accuracy with respect to the feature sum will be high (and can reach very high levels for high beta). Probabilistic Search Take-The-Best (PS-TTB), however, searches cues. Because its temperature tau is sampled from a wide range [0.01, 100], it often acts like random search, which is sensitive to the ratio rather than the difference of features, mathematically capping its maximum majority-choice probability at a much lower level than Tallying. Furthermore, when tau is small, PS-TTB strictly follows the highest-validity cue, which was deliberately designed to OPPOSE the feature sum on 80% of the trials. Thus, PS-TTB subjects will either have a mediocre agreement with the feature sum (large tau) or a very low agreement (small tau). By computing the subject-level proportion of choices aligned with the feature sum and taking it to the 4th power, we heavily amplify the high-accuracy subjects uniquely produced by Tallying while suppressing the uniformly lower accuracies produced by PS-TTB, yielding a highly discriminative and low-variance metric."
}
```

## Usage

```json
{
  "prompt_token_count": 6397,
  "candidates_token_count": 639,
  "total_token_count": 35616
}
```
