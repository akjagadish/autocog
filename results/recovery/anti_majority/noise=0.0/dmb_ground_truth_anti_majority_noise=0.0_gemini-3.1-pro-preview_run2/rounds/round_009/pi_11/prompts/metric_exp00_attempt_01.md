# metric_exp00_attempt_01

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
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.6, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Top-Cue Lexicographic with Signed Tallying Fallback) from the Competing Theory (Evidence Dilution) by exploiting their mathematical treatment of shared features. Under the Advocated Theory, both the lexicographic mechanism (which relies solely on the difference between the best unique cues) and the tallying fallback (which relies on the difference in total cue counts) perfectly cancel out any features shared by both options. Thus, adding a shared high-validity cue to both options leaves the predicted choice probabilities mathematically identical. In contrast, the Competing Theory calculates a dilution penalty based on the sum of validities of all present cues. Adding a shared high-validity cue significantly alters the dilution denominators and the ratio of overall subjective values, shifting the predicted choice probabilities. By pairing baseline trials with trials containing added shared cues, the Advocated Theory predicts strictly identical choices across the pairs, whereas the Competing Theory predicts a divergence.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



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
[0] rationale: This metric calculates the difference in choice probabilities between trials where a high-validity shared cue is present (Trials 2 and 4) versus absent (Trials 1 and 3). Under the Advocated Theory (Top-Cue Lexicographic with Signed Tallying), shared cues cancel out perfectly in both the lexicographic and tallying phases, predicting a difference of exactly zero. Under the Competing Theory (Evidence Dilution), adding a shared high-validity cue significantly increases the dilution denominator for both options, shrinking their subjective values and pushing choice probabilities toward 0.5. Thus, the metric should be near zero for the Advocated Theory and non-zero for the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    mean_shared = data[a_first == 1]['response'].mean()
    mean_not_shared = data[a_first == 0]['response'].mean()
    return float(mean_shared - mean_not_shared)
outcome: self_sim=0.0062 (var=0.0103) adversary_sim=0.0929 (var=0.0173) welch_t=-2.609 p=0.01227 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify trials based on the presence of features 0 and 3 in Option A\n    # Trial 1: A=[0, 1, 0, 0, 0]\n    # Trial 2: A=[1, 1, 0, 0, 0]\n    a_f0 = data['option_a_ratings'].apply(lambda x: x[0])\n    a_f3 = data['option_a_ratings'].apply(lambda x: x[3])\n    \n    t1_mask = (a_f0 == 0) & (a_f3 == 0)\n    t2_mask = (a_f0 == 1) & (a_f3 == 0)\n    \n    mean_t1 = data[t1_mask]['response'].mean()\n    mean_t2 = data[t2_mask]['response'].mean()\n    \n    return float(mean_t2 - mean_t1)",
  "rationale": "By isolating Trial 1 and Trial 2, we directly test the effect of adding a shared high-validity cue (feature 0) without any other confounds. Under the Advocated Theory (Top-Cue Lexicographic with Signed Tallying), both the lexicographic and tallying components perfectly cancel out the shared cue, predicting exactly zero difference in choice probability between Trial 1 and Trial 2. Under the Competing Theory (Evidence Dilution), adding a shared cue drastically changes the dilution denominators, systematically shifting the preference. Focusing solely on this pair minimizes variance compared to aggregating over all trials."
}
```

## Usage

```json
{
  "prompt_token_count": 3864,
  "candidates_token_count": 379,
  "total_token_count": 6419
}
```
