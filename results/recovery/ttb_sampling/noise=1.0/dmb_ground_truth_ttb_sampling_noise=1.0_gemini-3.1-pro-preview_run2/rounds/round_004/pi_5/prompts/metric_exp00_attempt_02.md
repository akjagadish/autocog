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
**Validities (n_features=6):** [0.9, 0.85, 0.75, 0.65, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Random Choice / Minimal Effort theory (advocated) from the Weak Tallying with Position Bias theory (competing), this design manipulates the spatial position of positive features while controlling for their sum. The advocated theory posits that any residual systematicity is purely based on simple tallying, predicting perfect 50/50 ties whenever both options have the same number of positive features, regardless of where they appear. In contrast, the competing theory incorporates a position bias (e.g., favoring left-most features), predicting systematic deviations from 50/50 in these 'tied' tallying trials. Furthermore, we include trials where one option has fewer positive features but located on the extreme left, testing if position bias can override a simple tallying advantage.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: In the Advocated theory (Random Choice / Minimal Effort), choices on perfectly tied trials (like Trial 1 and Trial 2) are driven strictly by random guessing, meaning the true probability of choosing Option A is exactly 0.5. Because the trials are independent, the expected value of the product of their empirical deviations from 0.5 is exactly zero. In the Competing theory (Weak Tallying with Position Bias), a subject's unique position bias pushes their preference toward the left or right. Since Option A is left-skewed in both Trial 1 and Trial 2, a given subject's choice probabilities will deviate from 0.5 in the SAME direction for both trials. This positive covariance means the expected value of the product of deviations will be strictly positive for the Competing theory. This approach isolates the variance caused by position bias while completely canceling out binomial noise, providing a highly discriminative and stable signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert option A ratings to a string to reliably identify trial types
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A has 1s on the left half, B on the right half (both sum to 3)
    t1 = data[data['a_str'] == '111000']
    # Trial 2: A has 1s on alternating left-skewed positions, B on right-skewed (both sum to 3)
    t2 = data[data['a_str'] == '101010']
    
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
        
    # Calculate the proportion of times the subject chose Option A (response == 0)
    p_a_t1 = (t1['response'] == 0).mean()
    p_a_t2 = (t2['response'] == 0).mean()
    
    # Compute the cross-trial product of deviations from 0.5
    return float((p_a_t1 - 0.5) * (p_a_t2 - 0.5))
outcome: self_sim=0.0002 (var=0.0001) adversary_sim=0.0003 (var=0.0001) welch_t=-0.035 p=0.9725 (N=25, alpha=0.01) -> reject

[1] rationale: In the Advocated theory (Random Choice / Minimal Effort), choices are driven almost entirely by random guessing, meaning the choice probability for Option A is very close to 0.5 for all trial types. Consequently, the variance of the observed choice fractions across the different trial types within a subject is almost purely due to binomial noise. In the Competing theory (Weak Tallying with Position Bias), each subject has a consistent position bias that systematically pushes their choice probabilities away from 0.5 by different amounts depending on the spatial distribution of features in each trial type. Because the trial types vary significantly in their left/right feature skew, this position bias induces an additional source of variance in the true choice probabilities across trial types. Thus, the across-trial-type variance of choice fractions will be systematically higher in the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['trial_type'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    rates = data.groupby('trial_type')['response'].apply(lambda x: (x == 0).mean())
    if len(rates) < 2:
        return 0.0
    return float(rates.var())
outcome: self_sim=0.0003 (var=0.0000) adversary_sim=0.0004 (var=0.0001) welch_t=-0.067 p=0.9467 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    # Calculate the proportion of times each subject chose Option A\n    subject_means = data.groupby('subject_id')['response'].apply(lambda x: (x == 0).mean())\n    # Return the mean squared deviation from 0.5\n    return float(((subject_means - 0.5) ** 2).mean())",
  "rationale": "In the experimental design, Option A is consistently 'left-heavy' (features shifted to the left) and Option B is 'right-heavy' across all 5 trial types. In the Advocated theory (Random Choice / Minimal Effort), choices are driven by random guessing and weak tallying. Because tallying is perfectly balanced across the 5 trial types (Option A has 7 positive features total, Option B has 7 total), the expected overall probability of choosing Option A is exactly 0.5 for every subject. Thus, the squared deviation of a subject's Option A choice fraction from 0.5 is solely driven by binomial noise (which is very small over 95 trials). In the Competing theory (Weak Tallying with Position Bias), each subject has an idiosyncratic position bias. Because Option A is always left-heavy, a left-bias will consistently push the subject to choose A, and a right-bias will consistently push them to choose B. This consistent push across all 95 trials substantially inflates the between-subject variance of the overall A-choice fraction. By computing the mean squared deviation of each subject's A-choice fraction from 0.5, we capture a massive, stable signal of position bias in the Competing theory that is virtually absent in the Advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4029,
  "candidates_token_count": 385,
  "total_token_count": 8161
}
```
