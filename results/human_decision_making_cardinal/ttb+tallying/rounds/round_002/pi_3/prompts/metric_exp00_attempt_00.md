# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Heuristic Decision Making domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.05). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[7, 7, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 2: A=[6, 6, 6, 6, 6]  B=[7, 7, 5, 5, 5]
  trial 3: A=[8, 8, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 4: A=[6, 6, 6, 6, 6]  B=[8, 8, 5, 5, 5]
  trial 5: A=[10, 10, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 6: A=[6, 6, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  trial 7: A=[10, 10, 2, 2, 2]  B=[2, 2, 3, 3, 3]
  trial 8: A=[2, 2, 3, 3, 3]  B=[10, 10, 2, 2, 2]
  trial 9: A=[9, 5, 5, 5, 5]  B=[1, 6, 6, 6, 6]
  trial 10: A=[1, 6, 6, 6, 6]  B=[9, 5, 5, 5, 5]

**Rationale:** To dissociate the purely compensatory Weighted Additive (WADD) theory from the Dynamic Strategy Selection (Threshold) model, we manipulate the maximum cardinal difference ('max_diff') between options across features. The Threshold model predicts that subjects will use Tallying when the maximum feature difference is small (falling below their threshold) and switch to WADD only when a feature difference is large enough to cross the threshold. In contrast, the pure WADD model integrates validity-weighted cardinal differences on every trial, regardless of the maximum difference. We construct a 5-feature design where WADD and Tallying consistently make opposing predictions: one option wins on the two most valid cues, while the other option wins on the three less valid cues. By parametrically varying the magnitude of the differences (max_diff = 1, 2, 4, and 8), we can observe whether subjects' choices flip from Tallying to WADD as the difference grows (supporting the Threshold model) or remain consistent with WADD throughout (supporting the advocated pure WADD theory).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

**Parameters:**
- beta: [0.01, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Dynamic Strategy Selection (Threshold Model): Decision-makers dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy on a trial-by-trial basis. The choice of strategy depends on the maximum cardinal difference between the options across all features. If this difference exceeds a certain threshold, the decision-maker is more likely to use WADD to account for the large magnitude; otherwise, they default to the simpler Tallying heuristic. This is modeled as a probabilistic mixture where the weight of WADD is a soft step function (sigmoid) of the maximum feature difference.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 10.0]
- steepness: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD score computation
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a_wadd = np.sum(a * val) / sum_val
        score_b_wadd = np.sum(b * val) / sum_val
    else:
        score_a_wadd = np.sum(a)
        score_b_wadd = np.sum(b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # Tallying score computation
    score_a_tally = float(np.sum(a > b))
    score_b_tally = float(np.sum(b > a))
    scores_tally = np.array([score_a_tally, score_b_tally])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Dynamic strategy selection based on max cardinal difference
    max_diff = np.max(np.abs(a - b))
    w_wadd = 1.0 / (1.0 + np.exp(-steepness * (max_diff - threshold)))
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
- option_b_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    # WADD consistently predicts the option with the higher rating on the first feature\n    wadd_predicts_b = (a_ratings[:, 0] < b_ratings[:, 0]).astype(int)\n    wadd_consistent = (data['response'].values == wadd_predicts_b).astype(float)\n    \n    # Calculate the maximum cardinal difference across features for each trial\n    max_diff = np.max(np.abs(a_ratings - b_ratings), axis=1)\n    \n    # Define large and small max_diff trials based on the experimental design\n    large_diff_mask = max_diff >= 4\n    small_diff_mask = max_diff <= 2\n    \n    if np.sum(large_diff_mask) == 0 or np.sum(small_diff_mask) == 0:\n        return 0.0\n        \n    large_diff_wadd_prop = np.mean(wadd_consistent[large_diff_mask])\n    small_diff_wadd_prop = np.mean(wadd_consistent[small_diff_mask])\n    \n    return float(large_diff_wadd_prop - small_diff_wadd_prop)",
  "rationale": "The experimental design contrasts WADD and Tallying by having them make opposing predictions on every trial. The Threshold model predicts that subjects will use Tallying (WADD-inconsistent) on trials with small maximum feature differences (max_diff <= 2) and switch to WADD (WADD-consistent) on trials with large maximum feature differences (max_diff >= 4). The pure WADD theory predicts WADD-consistent choices across all trials regardless of max_diff. Therefore, the difference in the proportion of WADD-consistent choices between large and small max_diff trials will be near zero for the pure WADD model, but significantly positive for the Threshold model."
}
```

## Usage

```json
{
  "prompt_token_count": 3807,
  "candidates_token_count": 512,
  "total_token_count": 6743
}
```
