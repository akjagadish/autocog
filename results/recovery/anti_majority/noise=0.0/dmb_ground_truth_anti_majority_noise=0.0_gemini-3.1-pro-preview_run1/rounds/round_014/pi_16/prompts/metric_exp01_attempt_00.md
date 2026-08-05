# metric_exp01_attempt_00

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
**Validities (n_features=8):** [0.95, 0.95, 0.95, 0.75, 0.75, 0.55, 0.55, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Variance Aversion) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Competing Theory penalizes by the range (max - min) of the active validities, whereas the Advocated Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have similar sums of validities and the exact same range, but different standard deviations (e.g., Option A has features clustered at the mean, while Option B has features clustered at the extremes). The Competing Theory applies the identical spread penalty to both options, whereas the Advocated Theory assigns a larger variance penalty to Option B (the extreme cluster) and thus strongly prefers Option A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Variance Aversion and Feature Consistency

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_var: [0.0, 50.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        base_score = np.sum(active_w)
        
        # Apply variance penalty for multiple unique features
        if len(active_w) > 1:
            # Use standard deviation scaled by the number of active features
            # so the penalty competes symmetrically with the additive base score.
            std_w = np.std(active_w)
            return base_score - lambda_var * std_w * len(active_w)
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## COMPETING THEORY
**Description:** Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
[0] rationale: In all trials, one option consists of unique features clustered at the mean (0.7, 0.7 or just 0.7), while the other consists of extreme unique features (0.9, 0.5). For exactly two unique features, the standard deviation penalty and the max-min spread penalty are mathematically proportional. However, the Advocated Theory's parameter range for the variance penalty (lambda_var up to 50) is much larger than the Competing Theory's spread penalty (lambda_penalty up to 10). Thus, the Advocated Theory will simulate much stronger, more deterministic preferences for the variance-averse option. By calculating the proportion of variance-averse choices, we can discriminate between the two models' predicted magnitudes of the effect.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    is_trial_4 = (a_str == '1010010')
    var_averse_choice = is_trial_4.astype(int)
    return float((data['response'] == var_averse_choice).mean())

outcome: self_sim=0.8617 (var=0.0083) adversary_sim=0.8187 (var=0.0185) welch_t=+1.310 p=0.1974 (N=25, alpha=0.01) -> reject

[1] rationale: Across the experimental design, the options are set up such that one option always has unique features with zero variance (e.g., [0.7, 0.7] or just [0.7]), while the other has high variance (e.g., [0.9, 0.5]). Both theories penalize the high-variance option, but the Advocated Theory's variance penalty (lambda_var) spans a much wider parameter range (up to 50) than the Competing Theory's spread penalty (lambda_penalty up to 10). By aggregating the proportion of variance-averse choices across all trials (rather than just one), we pool the signal of this penalty magnitude difference, reducing per-subject noise and robustly discriminating the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify the variance-averse choice for each trial
    # Trials 1, 2, 3, 5: Option A has zero variance in unique features, Option B has high variance.
    # Trial 4: Option A has high variance, Option B has zero variance.
    # We can identify Trial 4 by Option B having exactly two 1s.
    
    b_ratings = data['option_b_ratings'].apply(lambda x: sum(x))
    is_trial_4 = (b_ratings == 2)
    
    var_averse_choice = np.where(is_trial_4, 1, 0)
    
    # Calculate the overall proportion of variance-averse choices
    return float((data['response'] == var_averse_choice).mean())

outcome: self_sim=0.7093 (var=0.0046) adversary_sim=0.6882 (var=0.0062) welch_t=+1.009 p=0.3182 (N=25, alpha=0.01) -> reject

[2] rationale: By properly identifying the variance-averse option across all 5 trial types, we pool the choice data over all 95 trials per subject. In every trial, one option has unique features with zero variance (0.7, 0.7) while the other has high variance (0.9, 0.5). Because the Advocated Theory's variance penalty parameter spans a much wider effective range than the Competing Theory's spread penalty, it predicts a much stronger, more deterministic preference for the variance-averse option. Pooling across all 95 trials drastically reduces the per-subject variance of this proportion compared to evaluating a single trial type, thereby increasing the statistical power to discriminate the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    var_averse_choices = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        shared = (a == 1) & (b == 1)
        unique_a = a & ~shared
        
        # The variance-averse option always has its unique features drawn from indices 2, 3, 4 (validity 0.7)
        # The high-variance option has its unique features drawn from indices 0, 1 (0.9) and 5, 6 (0.5)
        if unique_a[2] == 1 or unique_a[3] == 1 or unique_a[4] == 1:
            var_averse_choice = 0
        else:
            var_averse_choice = 1
            
        var_averse_choices.append(var_averse_choice)
        
    return float(np.mean(data['response'] == var_averse_choices))
outcome: self_sim=0.8617 (var=0.0083) adversary_sim=0.8187 (var=0.0185) welch_t=+1.310 p=0.1974 (N=25, alpha=0.01) -> reject

[3] rationale: In this specific experimental design, every trial features at most 2 unique features per option. For exactly two features, the standard deviation scaled by the number of features (Advocated Theory) is mathematically identical to the max-min range (Competing Theory). Thus, the models are structurally equivalent here, differing only in their parameter priors: the Advocated Theory's variance penalty (lambda_var) spans up to 50, whereas the Competing Theory's spread penalty spans up to 10. This allows the Advocated Theory to produce much more extreme aversion to the high-variance options. By calculating the proportion of subjects who exhibit a highly deterministic preference (>80%) for the variance-averse options, we can robustly discriminate the heavy tail of extreme variance aversion predicted exclusively by the Advocated Theory's wider parameter space.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    b_ratings_sum = data['option_b_ratings'].apply(sum)
    is_trial_4 = (b_ratings_sum == 2)
    
    safe_choice = np.where(is_trial_4, 1, 0)
    is_safe = (data['response'] == safe_choice).astype(float)
    
    subj_means = is_safe.groupby(data['subject_id']).mean()
    return float((subj_means > 0.8).mean())
outcome: self_sim=0.0200 (var=0.0196) adversary_sim=0.0000 (var=0.0000) welch_t=+0.714 p=0.4819 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Identify Trial 1 where Option A and Option B have the exact same range of validities\n    # but different standard deviations.\n    # Trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0], B=[0, 1, 1, 0, 0, 0, 1, 1]\n    is_trial_1 = data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 1, 1, 0, 0)\n    \n    # Calculate the proportion of times Option A was chosen in Trial 1\n    trial_1_data = data[is_trial_1]\n    if len(trial_1_data) == 0:\n        return 0.5\n        \n    return float((trial_1_data['response'] == 0).mean())",
  "rationale": "In Trial 1, Option A and Option B have the exact same number of unique features (4) and the exact same max-min spread of validities (0.95 to 0.55). Consequently, the Competing Theory applies the identical spread penalty to both options, leaving its preference to be determined solely by small differences in the base score (depending on gamma). In contrast, the Advocated Theory penalizes by the standard deviation of validities. Option B's validities are clustered at the extremes (two 0.95s, two 0.55s), resulting in a higher standard deviation than Option A (which has validities of 0.95, 0.75, 0.75, 0.55). Thus, the Advocated Theory uniquely predicts a strong preference for Option A due to its lower variance penalty. Isolating this specific trial provides a clean, highly discriminative signal."
}
```

## Usage

```json
{
  "prompt_token_count": 5427,
  "candidates_token_count": 449,
  "total_token_count": 8976
}
```
