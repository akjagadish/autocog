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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Thresholded Tallying from standard Tallying (Equal Weight), we use a 5-feature design with a clear gap in validities: three high-validity features and two low-validity features. Standard Tallying ignores validities entirely and counts all positive features equally. Thresholded Tallying, however, applies a soft threshold that down-weights or eliminates the low-validity features, effectively tallying only the high-validity ones. By pitting options with fewer high-validity features against options with a greater total number of features (which are mostly low-validity), we create clear dissociation trials. For instance, an option with 2 high-validity and 0 low-validity features will be preferred by Thresholded Tallying over an option with 1 high-validity and 2 low-validity features, whereas standard Tallying will prefer the latter due to its higher unweighted sum.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [-1.0, 1.0]
- k: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric isolates trials where standard Tallying and Thresholded Tallying make opposing predictions. It computes the proportion of decisions on these conflict trials that align with Thresholded Tallying. For subjects using standard Tallying, this metric will be significantly below 0.5, whereas for subjects using Thresholded Tallying, it will be significantly above 0.5. This provides maximum discrimination between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Thresholded Tallying focuses heavily on the first 3 (high-validity) features
    tt_a = a_ratings[:, :3].sum(axis=1)
    tt_b = b_ratings[:, :3].sum(axis=1)
    diff_tt = tt_a - tt_b
    
    # Standard Tallying equally weights all 5 features
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    diff_tally = tally_a - tally_b
    
    # Identify conflict trials where the two heuristics prefer different options
    conflict_mask = (diff_tt * diff_tally) < 0
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Check if the subject's response aligns with the Thresholded Tallying prediction
    tt_predicts_a = diff_tt > 0
    tt_predicts_b = diff_tt < 0
    
    aligned = (tt_predicts_a & (responses == 0)) | (tt_predicts_b & (responses == 1))
    
    return float(np.mean(aligned[conflict_mask]))
outcome: self_sim=0.2693 (var=0.0706) adversary_sim=0.1623 (var=0.0065) welch_t=+1.927 p=0.06409 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    sum_a = a_ratings.sum(axis=1)\n    sum_b = b_ratings.sum(axis=1)\n    \n    # Identify trials where one option has sum=3 and the other has sum=2\n    mask_3_2 = (sum_a == 3) & (sum_b == 2)\n    mask_2_3 = (sum_a == 2) & (sum_b == 3)\n    valid_trials = mask_3_2 | mask_2_3\n    \n    # For these trials, did the subject choose the option with sum=3?\n    responses = data['response'].values\n    chose_3 = np.zeros(len(data), dtype=bool)\n    chose_3[mask_3_2 & (responses == 0)] = True\n    chose_3[mask_2_3 & (responses == 1)] = True\n    \n    # Differentiate between trials based on the number of high-validity features \n    # (the first 3 features) in the option with sum=3.\n    high_a = a_ratings[:, :3].sum(axis=1)\n    high_b = b_ratings[:, :3].sum(axis=1)\n    \n    high_of_3 = np.zeros(len(data))\n    high_of_3[mask_3_2] = high_a[mask_3_2]\n    high_of_3[mask_2_3] = high_b[mask_2_3]\n    \n    # Trial 7/8: the sum=3 option has 3 high-validity features\n    mask_78 = valid_trials & (high_of_3 == 3)\n    \n    # Conflict trials (1, 2, 3, 5, 6): the sum=3 option has only 1 high-validity feature\n    mask_conflict = valid_trials & (high_of_3 == 1)\n    \n    if np.sum(mask_78) == 0 or np.sum(mask_conflict) == 0:\n        return 0.0\n        \n    p_78 = np.mean(chose_3[mask_78])\n    p_conflict = np.mean(chose_3[mask_conflict])\n    \n    # Difference in probability of choosing the sum=3 option\n    return float(p_78 - p_conflict)",
  "rationale": "This metric exploits a structural symmetry in the experimental design to perfectly control for the Tallying model's predictions. In both the congruent trials (7 and 8) and the conflict trials (1, 2, 3, 5, 6), one option has exactly 3 positive features and the other has exactly 2. Because pure Tallying relies exclusively on the unweighted sum of features, its preference for the sum=3 option is mathematically identical across both sets of trials, regardless of the subject's noise (beta) or lapse (epsilon) parameters. Consequently, for subjects using pure Tallying, the difference in the proportion of times they choose the sum=3 option between these two sets of trials will be zero in expectation, with very low variance. Conversely, Thresholded Tallying filters out low-validity features. In trials 7 and 8, the sum=3 option contains mostly high-validity features and is strongly preferred. In the conflict trials, the sum=3 option is dominated by low-validity features, causing Thresholded Tallying to prefer the sum=2 option instead. Thus, for Thresholded Tallying, the difference in choice proportions will be strongly positive, cleanly dissociating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3729,
  "candidates_token_count": 930,
  "total_token_count": 9254
}
```
