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
**Validities (n_features=10):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Max-Distance Variance Penalty) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their divergent penalties for feature dispersion. Both models integrate the thresholded validities of unique features additively, but they apply different penalties. The Competing Theory penalizes by the range (max - min) of the active validities, whereas the Advocated Theory penalizes by the sum of squared differences from the maximum active feature (max-distance penalty). We design 'Indifference vs Preference' trials where Options A and B have the exact same sum of validities and the exact same range, causing the Competing Theory to predict indifference. However, Option A has features clustered near its minimum while Option B has features clustered near its maximum, resulting in a much larger max-distance penalty for Option A and leading the Advocated Theory to strictly prefer Option B. We also design 'Full Reversal' trials where Option A has a smaller range but a larger max-distance penalty than Option B. In these trials, the Competing Theory strictly prefers Option A (due to the smaller range) while the Advocated Theory strictly prefers Option B (due to the smaller max-distance penalty).

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Max-Distance Variance Penalty: Decision-makers simplify choices by cancelling out shared features and evaluating the unique features relative to a subjective threshold. However, rather than a simple spread (max - min) or standard deviation penalty, they apply a max-distance penalty: options with isolated high-validity features accompanied by low-validity features are heavily penalized, whereas dense clusters of moderate-validity features suffer very little penalty. This is modeled by subtracting the sum of squared differences between an option's maximum active feature and its other active features from its base additive score.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 20.0]
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
        
        # Max-distance penalty applied if there are multiple unique features
        if len(active_w) > 1:
            # Penalize features based on their squared distance from the maximum active feature
            # This heavily discounts isolated high-validity features vs dense clusters
            conflict_penalty = lambda_penalty * np.sum((np.max(active_w) - active_w) ** 2)
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
[0] rationale: This metric calculates the proportion of times the subject chooses Option B on 'Full Reversal' trials (Trials 3 and 4). On these specific trials, Option A has a smaller range of validities than Option B but a larger max-distance penalty (due to a single high validity and multiple low validities). The Competing Theory, which penalizes by range, strictly prefers Option A. The Advocated Theory, which penalizes by max-distance variance, strictly prefers Option B. Thus, the metric should be near 0 for the Competing Theory and near 1 for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        t3_a = (0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0)
        t4_a = (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1)
        return a == t3_a or a == t4_a

    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    return float(target_data['response'].mean())
outcome: self_sim=0.3113 (var=0.1033) adversary_sim=0.3919 (var=0.1148) welch_t=-0.863 p=0.3923 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates trials where the unique features of Option A and Option B have the exact same range (max - min) in validities. On these trials, the Competing Theory applies identical conflict penalties to both options, meaning its choice is driven purely by the base additive scores. However, in these specific trials, Option A features are clustered near the minimum while Option B features are clustered near the maximum, causing the Advocated Theory to apply a strictly larger max-distance penalty to Option A. As a result, the Advocated Theory will systematically shift its preferences towards Option B (higher mean response) compared to the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([1.0, 0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5])
    
    def is_equal_range(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        shared = (a > 0) & (b > 0)
        unique_a = a.copy()
        unique_a[shared] = 0
        unique_b = b.copy()
        unique_b[shared] = 0
        
        val_a = validities[unique_a > 0]
        val_b = validities[unique_b > 0]
        
        if len(val_a) > 1 and len(val_b) > 1:
            range_a = np.max(val_a) - np.min(val_a)
            range_b = np.max(val_b) - np.min(val_b)
            return abs(range_a - range_b) < 1e-5
        return False

    mask = data.apply(is_equal_range, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    return float(target_data['response'].mean())
outcome: self_sim=0.8472 (var=0.0113) adversary_sim=0.7878 (var=0.0217) welch_t=+1.634 p=0.1095 (N=25, alpha=0.01) -> reject

[2] rationale: This metric isolates Trial 5. In this trial, Option A and Option B have the exact same base sum of validities and the exact same range (max - min). However, Option A has 3 features at its minimum validity while Option B has only 2 features at its minimum. The Competing Theory (Spread Penalty) penalizes both options similarly because their ranges are identical. The Advocated Theory (Max-Distance Penalty) heavily penalizes Option A because the penalty scales with the sum of squared differences, and Option A has more features clustered far from its maximum. This leads the Advocated Theory to strongly prefer Option B (response closer to 1), whereas the Competing Theory remains relatively indifferent (response closer to 0.5), creating a large, stable gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5 where Option A has 3 low features and Option B has 2 low features
    # This maximizes the difference between the Max-Distance Penalty (which scales with the NUMBER of discordant features)
    # and the Spread Penalty (which only cares about the max and min, ignoring density).
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    t5_a = '100000111000'
    
    t5_data = data[data['a_str'] == t5_a]
    if len(t5_data) == 0:
        return 0.5
        
    return float(t5_data['response'].mean())
outcome: self_sim=0.8500 (var=0.0195) adversary_sim=0.8050 (var=0.0238) welch_t=+1.080 p=0.2855 (N=25, alpha=0.01) -> reject

[3] rationale: This metric computes the difference in the probability of choosing Option B between two sets of trials: {Trial 2, Trial 5} and {Trial 1, Trial 6}. Across all four of these trials, the Competing Theory applies the exact same relative spread penalty (Range A - Range B is exactly 0), and the base additive scores are perfectly balanced regardless of the non-linear utility parameter (gamma). Consequently, the Competing Theory predicts identical choice probabilities across all four trials, yielding a metric value consistently near 0. In contrast, the Advocated Theory's max-distance penalty is sensitive to the density of features. Option A has more features clustered far from its maximum validity in Trials 2 and 5 compared to Trials 1 and 6, resulting in a much larger relative penalty for Option A in the first set. Therefore, the Advocated Theory predicts a significantly higher probability of choosing Option B in Trials 2 and 5 than in Trials 1 and 6, yielding a robustly positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create string representation of option A to safely identify trials
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Option A configurations for the two sets of trials
    t1_a = '100010100000'
    t6_a = '100011000000'
    
    t2_a = '100100110000'
    t5_a = '100000111000'
    
    mask_low = a_str.isin([t1_a, t6_a])
    mask_high = a_str.isin([t2_a, t5_a])
    
    mean_low = data[mask_low]['response'].mean()
    mean_high = data[mask_high]['response'].mean()
    
    if pd.isna(mean_low) or pd.isna(mean_high):
        return 0.0
        
    return float(mean_high - mean_low)

outcome: self_sim=-0.0056 (var=0.0092) adversary_sim=0.0206 (var=0.0087) welch_t=-0.981 p=0.3315 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    t1_t2_mask = a_str.isin(['0100010100', '1000101000'])\n    t3_t4_mask = a_str.isin(['1000011000', '0100001100'])\n    mean_t1_t2 = data[t1_t2_mask]['response'].mean()\n    mean_t3_t4 = data[t3_t4_mask]['response'].mean()\n    if pd.isna(mean_t1_t2) or pd.isna(mean_t3_t4):\n        return 0.0\n    return float(mean_t1_t2 - mean_t3_t4)",
  "rationale": "This metric calculates the difference in the probability of choosing Option B between the 'Indifference' trials (Trials 1 and 2) and the 'Full Reversal' trials (Trials 3 and 4). Under the Competing Theory, the spread penalty is perfectly balanced in Trials 1 and 2, leading to indifference (response ~ 0.5), while in Trials 3 and 4, Option A has a strictly smaller range, leading to a preference for Option A (response < 0.5). Therefore, the Competing Theory predicts a positive difference. Conversely, the Advocated Theory heavily penalizes Option A across all four trials due to its max-distance variance (isolated high validities), resulting in a strong preference for Option B (response > 0.5) in both sets of trials, yielding a difference near zero."
}
```

## Usage

```json
{
  "prompt_token_count": 5875,
  "candidates_token_count": 420,
  "total_token_count": 10183
}
```
