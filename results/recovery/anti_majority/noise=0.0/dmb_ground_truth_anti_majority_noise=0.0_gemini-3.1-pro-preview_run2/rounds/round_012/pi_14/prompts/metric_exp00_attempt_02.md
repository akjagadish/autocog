# metric_exp00_attempt_02

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
**Validities (n_features=7):** [0.95, 0.85, 0.8, 0.75, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates 'Rank-Weighted Capacity-Bounded Integration' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting how they penalize options with multiple features. The Advocated theory penalizes options based strictly on the count of excess features beyond a capacity limit K. Therefore, if two options have the same number of active features and share the same top feature(s), they suffer the exact same penalty, leaving the option with the stronger secondary features preferred or tied. In contrast, the Competing theory dilutes evidence based on the sum of the validities of all present features. By pairing options that share a top cue and have the same total count of active features, but differ in the validities of their secondary features (e.g., strong secondary cues vs. weak secondary cues), the Advocated theory predicts the option with strong secondary cues will always be preferred or tied. The Competing theory, however, predicts that the strong secondary cues will heavily inflate the dilution denominator, severely penalizing the option and causing a paradoxical preference for the option with the weaker secondary cues.

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
[0] rationale: The experimental design pairs options that have the same number of active features and share the top feature(s), but option B always has stronger secondary features than option A. Under the Advocated theory (Capacity-Bounded Integration), the penalty depends only on the count of excess features, which is identical for both options. Thus, option B is consistently preferred or tied, leading to a high average response rate. Under the Competing theory (Evidence Dilution), the dilution penalty is proportional to the sum of the validities of present features. Because option B has stronger secondary features, its sum of validities is much higher, leading to a massive dilution penalty that paradoxically causes option A to be preferred in many parameter regimes. The overall proportion of choosing option B (response == 1) directly captures this divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.5835 (var=0.0176) adversary_sim=0.5048 (var=0.0072) welch_t=+2.497 p=0.01666 (N=25, alpha=0.01) -> reject

[1] rationale: Calculates the difference in preference for option B between Trial 2 (fewer active features) and Trial 4 (more active features). Under the Advocated theory, the relative advantage of option B's top K features is larger in Trial 2 than Trial 4, leading to a positive difference. Under the Competing theory, the dilution effect scales non-linearly with the total sum of validities, which creates a very different pattern of preference across these two trials, thus separating the theories better than the overall mean.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['n_features_a'] = data['option_a_ratings'].apply(lambda x: sum(x))
    t2_resp = data[data['n_features_a'] == 3]['response'].mean()
    t4_resp = data[data['n_features_a'] == 5]['response'].mean()
    return float(t2_resp - t4_resp)
outcome: self_sim=0.0933 (var=0.0588) adversary_sim=0.0067 (var=0.0219) welch_t=+1.526 p=0.1351 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1, 1, 1))\n    return float(data[t1_mask]['response'].mean())",
  "rationale": "Focuses specifically on Trial 1, where the difference in the sum of validities between Option B (3.35) and Option A (2.6) is the largest. Under the Competing theory, this large difference creates a massive dilution penalty for Option B, causing a paradoxical preference for Option A (response < 0.5). Under the Advocated theory, because both options have 4 features and share the top feature, the complexity penalty is identical, and Option B's superior secondary features make it equally or more preferred (response >= 0.5). Isolating this specific trial maximizes the divergence between the two predictions while maintaining low per-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 4018,
  "candidates_token_count": 229,
  "total_token_count": 6199
}
```
