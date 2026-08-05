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
**Validities (n_features=5):** [0.95, 0.9, 0.8, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Top-Cue Lexicographic with Signed Tallying Fallback) from the Competing Theory (Evidence Dilution) by exploiting their diametrically opposed handling of shared cues and weak cues. The Advocated Theory bases its strategy selection strictly on the difference between the most valid *unique* cues of each option, and its tallying fallback uses the simple difference in total cue counts. Consequently, adding a shared cue to both options leaves the Advocated Theory's predicted choice probabilities perfectly invariant. In contrast, the Competing Theory integrates all present cues into its dilution denominator; adding shared cues non-linearly dilutes both options, shifting the choice probabilities. By including trials with and without shared weak cues, as well as trials that manipulate the difference in top unique validities to trigger the tallying fallback, we can robustly separate the thresholded unique-cue mechanism from global continuous dilution.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Top-Cue Lexicographic with Signed Tallying Fallback Theory: Decision-makers generally rely on the single most valid unique cue (lexicographic strategy). However, when the top unique cues of the two options are insufficiently discriminative (close in validity), they fall back to a tallying strategy. Crucially, the tallying weight can be negative, allowing the model to capture 'less is more' effects where decision-makers penalize options that pad their profile with numerous weak cues.

**Parameters:**
- beta_lex: [1.0, 30.0]
- beta_tal: [-10.0, 10.0]
- tau: [5.0, 50.0]
- theta: [0.0, 0.05]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    beta_lex = float(parameters["beta_lex"])
    beta_tal = float(parameters["beta_tal"])
    tau = float(parameters["tau"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify unique cues for each option
    a_uniq = a * (1.0 - b)
    b_uniq = b * (1.0 - a)
    
    v_a = val * a_uniq
    v_b = val * b_uniq
    
    # Find the validity of the best unique cue for each option
    v_max_a = np.max(v_a) if np.sum(a_uniq) > 0 else 0.0
    v_max_b = np.max(v_b) if np.sum(b_uniq) > 0 else 0.0
    
    # Difference in top unique cue validities
    diff = abs(v_max_a - v_max_b)
    
    # Probability of using the Lexicographic strategy
    p_lex = 1.0 / (1.0 + np.exp(-tau * (diff - theta)))
    
    # Lexicographic prediction
    z_lex = beta_lex * (v_max_a - v_max_b)
    z_lex = np.clip(z_lex, -20.0, 20.0)
    p_a_lex = 1.0 / (1.0 + np.exp(-z_lex))
    
    # Tallying prediction (using total cue counts)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    z_tal = beta_tal * (tally_a - tally_b)
    z_tal = np.clip(z_tal, -20.0, 20.0)
    p_a_tal = 1.0 / (1.0 + np.exp(-z_tal))
    
    # Mixture of the two strategies
    p_a = p_lex * p_a_lex + (1.0 - p_lex) * p_a_tal
    p_b = 1.0 - p_a
    
    p = np.array([p_a, p_b])
    
    # Apply lapse rate
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
[0] rationale: This metric computes the difference in the probability of choosing Option A between Trial 1 and Trial 2. In both trials, the unique cues for A and B are identical (the first cue for A, the second for B). Trial 2 simply adds two shared weak cues to both options. Under the Advocated Theory (Top-Cue Lexicographic with Signed Tallying Fallback), shared cues are ignored by both the unique-cue comparison and the tallying difference, resulting in perfectly invariant choice probabilities between the two trials (difference = 0). Under the Competing Theory (Evidence Dilution), adding shared weak cues dilutes the subjective values of both options, pulling the choice probability closer to 0.5. Thus, the Competing Theory predicts a positive difference (P(A) drops in Trial 2), while the Advocated Theory predicts exactly zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create string representations to identify the trials
    trial_A = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    trial_B = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 0, 0, 0]
    mask1 = (trial_A == '10000') & (trial_B == '01000')
    
    # Trial 2: A=[1, 0, 0, 1, 1], B=[0, 1, 0, 1, 1]
    mask2 = (trial_A == '10011') & (trial_B == '01011')
    
    # Response is 0 for A and 1 for B. P(A) = 1 - mean(response)
    p_a_1 = 1.0 - data[mask1]['response'].mean() if mask1.sum() > 0 else 0.5
    p_a_2 = 1.0 - data[mask2]['response'].mean() if mask2.sum() > 0 else 0.5
    
    # Return the difference in probability of choosing A
    return float(p_a_1 - p_a_2)

outcome: self_sim=-0.0179 (var=0.0227) adversary_sim=0.0705 (var=0.0544) welch_t=-1.593 p=0.1189 (N=25, alpha=0.01) -> reject

[1] rationale: This metric exploits the 'pull to 0.5' effect uniquely predicted by the Evidence Dilution (Competing) theory. Trial 1 and Trial 2 share the exact same unique cues and tally difference, meaning the Advocated Theory predicts identical choice probabilities for both (P(A|T1) = P(A|T2)). Thus, for the Advocated theory, the expected value of `(P(A|T1) - 0.5) * (P(A|T1) - P(A|T2))` is simply the binomial variance of the sample proportion, which is extremely small and approaches 0 on the pooled data. In contrast, the Competing Theory integrates all cues into its dilution denominator. Adding shared weak cues in Trial 2 increases the denominator, shrinking the absolute difference in subjective values between the two options. This mathematically forces the choice probability in Trial 2 to be closer to 0.5 than in Trial 1. As a result, `(P(A|T1) - P(A|T2))` always shares the same sign as `(P(A|T1) - 0.5)`, making their product strictly positive and substantially larger than the binomial noise. This provides a robust, single-sided test that cleanly separates the thresholded unique-cue mechanism from global continuous dilution.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert sequence columns to strings for safe comparison
    trial_A = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    trial_B = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 0, 0, 0]
    mask1 = (trial_A == '10000') & (trial_B == '01000')
    # Trial 2: A=[1, 0, 0, 1, 1], B=[0, 1, 0, 1, 1]
    mask2 = (trial_A == '10011') & (trial_B == '01011')
    
    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0
        
    # Response is 0 for A, 1 for B. P(A) = 1 - mean(response)
    p_a_1 = 1.0 - data.loc[mask1, 'response'].mean()
    p_a_2 = 1.0 - data.loc[mask2, 'response'].mean()
    
    # The metric computes the directional shift towards 0.5
    return float((p_a_1 - 0.5) * (p_a_1 - p_a_2))
outcome: self_sim=-0.0015 (var=0.0004) adversary_sim=0.0062 (var=0.0031) welch_t=-0.661 p=0.5136 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Convert sequence columns to strings for safe comparison\n    trial_A = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    trial_B = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Trial 2: A=[1, 0, 0, 1, 1], B=[0, 1, 0, 1, 1]\n    mask2 = (trial_A == '10011') & (trial_B == '01011')\n    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 1, 0, 1, 1]\n    mask3 = (trial_A == '10000') & (trial_B == '01011')\n    \n    if mask2.sum() == 0 or mask3.sum() == 0:\n        return 0.0\n        \n    # Response is 0 for A, 1 for B. We calculate P(B) = mean(response)\n    p_b_2 = data.loc[mask2, 'response'].mean()\n    p_b_3 = data.loc[mask3, 'response'].mean()\n    \n    # Return the difference in probability of choosing B\n    return float(p_b_2 - p_b_3)",
  "rationale": "This metric compares the probability of choosing Option B between Trial 2 and Trial 3. In both trials, Option B is identical and burdened with multiple weak cues. However, Option A is burdened with the same weak cues in Trial 2, but is free of them in Trial 3. \n\nUnder the Competing Theory (Evidence Dilution), Option A is heavily diluted in Trial 2 but not in Trial 3. Consequently, the subjective value of A drops significantly in Trial 2, making Option B much more competitive. Thus, Evidence Dilution predicts a robustly positive difference (P(B|T2) - P(B|T3) > 0).\n\nUnder the Advocated Theory (Top-Cue Lexicographic), the unique cues for A and B are identical across both trials. The only difference is the tallying fallback: Trial 2 has a tally difference of 0, while Trial 3 has a tally difference of -2. Because the tally weight parameter (beta_tal) is symmetrically distributed around 0 across the population, the expected difference in choice probabilities at the population level is zero. This creates a strong mean separation between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4960,
  "candidates_token_count": 630,
  "total_token_count": 12999
}
```
