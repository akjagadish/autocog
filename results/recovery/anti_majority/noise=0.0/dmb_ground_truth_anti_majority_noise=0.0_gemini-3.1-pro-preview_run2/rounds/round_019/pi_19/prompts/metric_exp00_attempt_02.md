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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Evidence Averaging (Advocated) from Shared-Feature Cancellation (Competing). The Competing theory assumes that decision-makers first cancel out any features shared by both options before averaging the unique remaining features. Therefore, it mathematically guarantees that adding identical shared features to both options will result in exactly the same choice probabilities as the baseline trial without those features. In contrast, the Advocated theory averages all active features. Adding a shared feature to both options alters both their numerators and denominators, inherently changing their averages and typically compressing the difference between them. By comparing a baseline trial to trials where weak, strong, or multiple shared features are appended, the Competing theory predicts strict invariance in choice probabilities, whereas the Advocated theory predicts a systematic regression toward chance.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.01, 10.0]
- prior_sum: [0.0, 10.0]
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
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
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
**Description:** Shared-Feature Cancellation and Evidence Averaging: Decision-makers evaluate options by first identifying and canceling out features that are shared (active) in both options. After removing these shared elements to simplify the task, they evaluate each option by averaging the validities of its remaining unique active cues. A base prior is included in the average to establish a baseline expectation, but its lower bounds are extended to exactly zero to allow subjects to completely turn off the prior if they rely purely on the raw average of unique cues.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.0, 10.0]
- prior_sum: [0.0, 10.0]
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
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Shared-Feature Cancellation
    shared = a * b
    a_unique = a - shared
    b_unique = b - shared
    
    n_a = np.sum(a_unique)
    n_b = np.sum(b_unique)
    
    # Calculate average validity of unique features with a base prior (safeguarded against div-by-zero)
    v_a = (np.sum(w * a_unique) + prior_sum) / max(n_a + prior_count, 1e-6)
    v_b = (np.sum(w * b_unique) + prior_sum) / max(n_b + prior_count, 1e-6)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
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
[0] rationale: The Competing theory (Shared-Feature Cancellation) predicts that choices in Trial 1 and Trial 4 should be identical, because Trial 4 simply adds shared features to both options which are immediately canceled out, leaving the exact same unique features as Trial 1. Therefore, the difference in choice proportions between these trials should be zero. In contrast, the Advocated theory (Evidence Averaging) predicts that adding shared features inherently changes the averages of both options, typically compressing the difference between them. Thus, the choice probability for the preferred option should regress toward 0.5 in Trial 4 compared to Trial 1, leading to a strictly positive difference in the mean response.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Count the number of active features in option A to identify the trial type
    a_sums = data['option_a_ratings'].apply(sum)
    
    # Trial 1 has 1 active feature in A; Trial 4 has 3 active features in A
    t1_mask = a_sums == 1
    t4_mask = a_sums == 3
    
    # Calculate the difference in the proportion of choosing B (response == 1)
    # between Trial 4 and Trial 1
    return float(data.loc[t4_mask, 'response'].mean() - data.loc[t1_mask, 'response'].mean())
outcome: self_sim=0.0617 (var=0.0136) adversary_sim=0.0350 (var=0.0144) welch_t=+0.796 p=0.4297 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Competing theory (Shared-Feature Cancellation), when Feature 5 is shared by both options, it is immediately canceled out. Therefore, the choice probabilities should be identical whether Feature 5 is present (Trials 2 and 4) or absent (Trials 1 and 3). The expected difference is exactly 0. Under the Advocated theory (Evidence Averaging), adding a weak shared feature dilutes the average validity of the options. Because Option A generally relies on fewer but stronger features, adding a weak feature dilutes Option A's evidence more severely than Option B's, reducing Option A's advantage and increasing the probability of choosing Option B. By comparing the main effect of this shared feature across all trials, we maximize statistical power (using 96 trials per subject) while isolating the exact mechanism of divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Feature 5 (index 4) is the weakest feature.
    # In Trials 2 and 4, it is present in both options (shared).
    # In Trials 1 and 3, it is absent from both options.
    has_f5 = data['option_a_ratings'].apply(lambda x: x[4] == 1)
    
    # Calculate the proportion of times Option B was chosen when F5 is present vs absent.
    p_b_with_f5 = data.loc[has_f5, 'response'].mean()
    p_b_without_f5 = data.loc[~has_f5, 'response'].mean()
    
    return float(p_b_with_f5 - p_b_without_f5)
outcome: self_sim=0.0262 (var=0.0084) adversary_sim=0.0075 (var=0.0082) welch_t=+0.727 p=0.4707 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    def calc_excess_var(subj_df):\n        # Create a unique string key for each trial type based on Option A's features\n        keys = subj_df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n        \n        # Calculate the mean response for each of the 4 trial types\n        trial_means = subj_df.groupby(keys)['response'].mean()\n        \n        # Overall proportion of choosing B for this subject\n        p = subj_df['response'].mean()\n        \n        # Observed variance of the 4 trial means (ddof=0)\n        obs_var = np.var(trial_means, ddof=0)\n        \n        # Expected variance purely due to binomial noise.\n        # With 96 trials total and 4 groups of 24, the exact expected sample variance \n        # of the 4 means under the null hypothesis (constant p) is exactly:\n        exp_var = p * (1.0 - p) * 3.0 / 95.0\n        \n        # Return the excess variance above binomial noise\n        return float(obs_var - exp_var)\n\n    # Average the excess variance across all subjects\n    return float(data.groupby('subject_id').apply(calc_excess_var).mean())",
  "rationale": "Under the Competing theory (Shared-Feature Cancellation), the unique features evaluated by the subject are completely identical across all 4 trial types. Consequently, the true probability of choosing Option B is exactly the same across all 96 trials for a given subject. Any variation in the observed proportion of B choices across the 4 trial types is purely due to binomial sampling noise. Under the Advocated theory (Evidence Averaging), however, the true choice probabilities systematically vary across the 4 trial types because adding shared features alters the average evidence of each option differently.\n\nWe can elegantly distinguish these theories by computing the variance of the 4 trial-type means for each subject, and subtracting the analytically expected variance due to binomial noise (`p * (1 - p) * 3 / 95`). Under the Competing theory, this 'excess variance' has an expected value of exactly 0, and its variance across subjects is extremely tight. Under the Advocated theory, the excess variance will be systematically positive due to the true underlying variation in choice probabilities. This metric perfectly isolates the theoretical divergence while structurally eliminating baseline preference noise."
}
```

## Usage

```json
{
  "prompt_token_count": 4075,
  "candidates_token_count": 592,
  "total_token_count": 14492
}
```
