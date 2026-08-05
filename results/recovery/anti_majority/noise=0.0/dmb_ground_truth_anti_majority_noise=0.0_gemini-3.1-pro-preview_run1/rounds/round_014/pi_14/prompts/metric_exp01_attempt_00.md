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
**Validities (n_features=12):** [0.9, 0.9, 0.9, 0.85, 0.85, 0.7, 0.7, 0.55, 0.55, 0.5, 0.5, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Variance Aversion and Feature Consistency), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Advocated Theory penalizes by the range (max - min) of the active validities, whereas the Competing Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have the exact same sum of validities and the exact same range, but different standard deviations (e.g., Option A has validities clustered at the mean, while Option B has validities clustered at the extremes). We also design full reversal trials where Option A has a larger range but a smaller standard deviation than Option B. The Advocated Theory will strictly prefer the option with the smaller range, while the Competing Theory will strictly prefer the option with the smaller standard deviation.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: The experimental design pits options with extreme validities (e.g., 0.9 and 0.5) against options with moderate validities (e.g., 0.7 and 0.7) while keeping the sum of validities constant. The Advocated Theory uses a spread penalty based on the range, while the Competing Theory uses a variance penalty based on standard deviation. These two functional forms scale differently across the parameter space, leading to systematically different overall probabilities of choosing Option B (the moderate option). The mean response (proportion of B choices) directly captures this divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

outcome: self_sim=0.6892 (var=0.0119) adversary_sim=0.7327 (var=0.0061) welch_t=-1.623 p=0.1119 (N=25, alpha=0.01) -> reject

[1] rationale: In Trials 1 through 4, Option A has unique features with validities 0.9 and 0.5, while Option B has unique features 0.7 and 0.7. Both theories penalize Option A for its feature variance/spread, driving choices toward Option B. However, the Competing Theory's prior allows for substantially larger variance penalties (lambda_var up to 50) compared to the Advocated Theory's spread penalty (lambda_penalty up to 10), leading to a higher B-choice rate in these trials for the Competing Theory. To dramatically reduce between-subject variance caused by individual differences in base feature evaluation (gamma, rho), we subtract the response rate from Trial 6, where Option A has unique features 0.9 and 0.9 and Option B has 0.7 and 0.7. Since neither option in Trial 6 triggers a spread/variance penalty, this trial acts as a subject-specific baseline for relative feature valuation. The difference precisely isolates the penalty magnitude.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    
    t1_4_mask = a_keys.isin([
        (1, 1, 0, 0, 1, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 0, 0, 1, 0),
        (1, 0, 0, 0, 1, 1)
    ])
    t6_mask = a_keys == (1, 1, 0, 0, 0, 0)
    
    t1_4_mean = data[t1_4_mask]['response'].mean()
    t6_mean = data[t6_mask]['response'].mean()
    
    return float(t1_4_mean - t6_mean)
outcome: self_sim=0.6579 (var=0.0528) adversary_sim=0.6875 (var=0.0511) welch_t=-0.459 p=0.6484 (N=25, alpha=0.01) -> reject

[2] rationale: The Competing Theory allows for a much larger penalty parameter (lambda_var up to 50) than the Advocated Theory (lambda_penalty up to 10). Because of this, the Competing Theory will produce much more extreme choice probabilities across the different trial types (pushing probabilities closer to 0 or 1). By calculating the range (max - min) of the average choice rates across the 8 unique trial types, we capture this difference in extremeness. This metric should be highly stable within subjects (low between-subject variance) while showing a clear mean difference between the models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['trial_key'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    means = data.groupby('trial_key')['response'].mean()
    return float(means.max() - means.min())
outcome: self_sim=0.6683 (var=0.0236) adversary_sim=0.6967 (var=0.0284) welch_t=-0.621 p=0.5372 (N=25, alpha=0.01) -> reject

[3] rationale: By contrasting the response rates of Trials 1-4 (where Option A has a large spread/variance and Option B has zero) against Trial 5 (where both options have some spread/variance but differing means), we isolate the specific scaling behavior of the penalty functions. The Competing Theory's larger parameter range for variance penalty will disproportionately affect Trials 1-4 relative to Trial 5, creating a more reliable within-subject divergence between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['trial_key'] = data['option_a_ratings'].apply(lambda x: tuple(x))
    t1_4 = [(1, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 0), (1, 1, 0, 0, 1, 0), (1, 0, 0, 0, 1, 1)]
    t5 = [(0, 1, 0, 0, 0, 1)]
    m1 = data[data['trial_key'].isin(t1_4)]['response'].mean()
    m5 = data[data['trial_key'].isin(t5)]['response'].mean()
    return float(m1 - m5)
outcome: self_sim=0.0246 (var=0.0129) adversary_sim=0.0108 (var=0.0114) welch_t=+0.441 p=0.6613 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_keys = data['option_a_ratings'].apply(tuple)\n    b_keys = data['option_b_ratings'].apply(tuple)\n    \n    t2_a = (1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0)\n    t2_b = (0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0)\n    \n    t4_a = (1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0)\n    t4_b = (0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0)\n    \n    mask = ((a_keys == t2_a) & (b_keys == t2_b)) | ((a_keys == t4_a) & (b_keys == t4_b))\n    return float(data[mask]['response'].mean())",
  "rationale": "Trials 2 and 4 act as strict reversal trials between the two theories. In these trials, Option A's unique features are [0.9, 0.7, 0.7, 0.5] and Option B's unique features are [0.85, 0.85, 0.55, 0.55]. The Advocated Theory measures spread using the range: A's range (0.4) is larger than B's range (0.3), so it penalizes A more and strictly prefers B (response -> 1). The Competing Theory measures spread using standard deviation: A's standard deviation (~0.141) is strictly smaller than B's standard deviation (0.15), so it penalizes B more and strictly prefers A (response -> 0). Because these inequalities hold across all possible values of the non-linear utility parameter (gamma), the mean response on these trials provides a highly robust, parameter-independent discriminator."
}
```

## Usage

```json
{
  "prompt_token_count": 5225,
  "candidates_token_count": 519,
  "total_token_count": 10525
}
```
