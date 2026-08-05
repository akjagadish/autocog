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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Weak Tallying with Position Bias from the Random Choice / Minimal Effort theory by manipulating the spatial position of positive features while controlling for their sum. The competing theory relies purely on simple tallying, predicting perfect 50/50 ties whenever both options have the same number of positive features, regardless of where they appear. In contrast, the advocated theory incorporates a position bias (e.g., favoring left-most features due to reading order), predicting systematic deviations from 50/50 in these 'tied' tallying trials. Furthermore, we include a trial where one option has fewer positive features but located on the extreme left, testing if position bias can override a simple tallying advantage.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Weak Tallying with Position Bias: In the absence of trial-by-trial feedback, participants abandon complex, validity-based compensatory or non-compensatory strategies. Instead, their choices are heavily dominated by random guessing (a very high lapse rate). The tiny fraction of systematic variance that remains is driven by minimal-effort heuristics: a weak preference for the option with a higher total number of positive features (tallying), combined with a slight positional bias that might favor or disfavor features based on reading order.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- position_bias: [-1.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    n_features = stim.shape[1]
    position_bias = float(parameters["position_bias"])
    
    # Features on the left (index 0) get higher weights if position_bias > 0
    # Weights range from 1.0 (rightmost) to 1.0 + position_bias * (n_features - 1) (leftmost)
    w = 1.0 + position_bias * np.arange(n_features - 1, -1, -1)
    
    # Calculate score for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Random Choice / Minimal Effort: In the absence of trial-by-trial feedback and when faced with complex multi-attribute binary arrays, participants largely abandon systematic cue-based strategies. Instead, they resort to minimal effort processing, which manifests as random guessing or behavior dominated by an extremely high lapse rate. Any residual systematicity is extremely weak, resulting in choice probabilities that are consistently very close to 0.5 across all experimental conditions.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.9, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Minimal effort evaluation (e.g., simple tallying of 1s)
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: The Competing theory (Random Choice / Minimal Effort) relies on simple tallying. Because trials 1-4 are perfectly matched in their total number of positive features between Option A and Option B, the Competing theory predicts a choice probability of exactly 0.5 for these trials, with only a tiny deviation in trial 5. In contrast, the Advocated theory (Weak Tallying with Position Bias) applies spatially-dependent weights. This means that in trials 1-4, the weighted scores for A and B will systematically differ depending on the subject's 'position_bias' parameter. Consequently, the Advocated theory predicts that subjects will exhibit systematic choice probabilities that deviate from 0.5 across the unique trial types. By calculating the mean absolute deviation of the per-trial choice proportions from 0.5, we capture this systematic variance. The metric will be consistently close to the expected binomial noise floor for the Competing theory, but systematically higher for the Advocated theory due to the added variance from the position bias.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Create a hashable string representation of each unique trial type
    data['trial_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + \
                        data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Calculate the choice proportion for each unique trial type per subject
    p_b = data.groupby(['subject_id', 'trial_str'])['response'].mean()
    
    # Calculate the mean absolute deviation from random choice (0.5)
    return float(np.abs(p_b - 0.5).mean())
outcome: self_sim=0.0971 (var=0.0009) adversary_sim=0.0891 (var=0.0012) welch_t=+0.873 p=0.3873 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data = data.copy()\n    # Identify trials where the simple sum of features is equal (Trials 1-4)\n    data['sum_a'] = data['option_a_ratings'].apply(sum)\n    data['sum_b'] = data['option_b_ratings'].apply(sum)\n    equal_sum_mask = data['sum_a'] == data['sum_b']\n    filtered_data = data[equal_sum_mask]\n    \n    if len(filtered_data) == 0:\n        return 0.0\n        \n    # For each subject, compute the unbiased estimate of (P(A) - 0.5)^2\n    # This removes the binomial variance (noise floor) from the expected value.\n    def unbiased_sq_dev(df):\n        n = len(df)\n        if n < 2:\n            return 0.0\n        x = (df['response'] == 0).sum()\n        p_hat = x / n\n        # Unbiased estimator for (p - 0.5)^2\n        return (p_hat - 0.5)**2 - (x * (n - x)) / (n**2 * (n - 1))\n        \n    return float(filtered_data.groupby('subject_id').apply(unbiased_sq_dev).mean())",
  "rationale": "In Trials 1-4, the total number of positive features is perfectly matched between Option A and Option B. The Competing theory (Random Choice / Minimal Effort) relies purely on tallying, so it predicts a choice probability of exactly 0.5 for Option A on these trials for all subjects. Any deviation from 0.5 in the empirical data under this theory is purely due to binomial noise. \n\nIn contrast, the Advocated theory (Weak Tallying with Position Bias) applies a spatially-dependent weight (position bias). Since Options A and B have different spatial distributions of features in Trials 1-4, the subject's idiosyncratic position bias will cause their true underlying choice probability to systematically deviate from 0.5. \n\nBy computing the *unbiased estimator* of the squared deviation from 0.5 (which analytically subtracts the expected binomial noise variance), the expected value of this metric is exactly 0.0 under the Competing theory. Under the Advocated theory, the expected value will be strictly positive, driven by the variance of the position bias across subjects. This isolates the theoretical signal from the binomial noise floor."
}
```

## Usage

```json
{
  "prompt_token_count": 3483,
  "candidates_token_count": 602,
  "total_token_count": 12534
}
```
