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
**Validities (n_features=5):** [0.95, 0.65, 0.6, 0.55, 0.5]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[6, 2, 2, 2, 2]  B=[3, 3, 3, 3, 3]
  trial 2: A=[8, 4, 4, 4, 4]  B=[5, 5, 5, 5, 5]
  trial 3: A=[10, 6, 6, 6, 6]  B=[7, 7, 7, 7, 7]
  trial 4: A=[10, 8, 8, 8, 8]  B=[9, 9, 8, 8, 8]
  trial 5: A=[3, 3, 3, 3, 3]  B=[6, 2, 2, 2, 2]
  trial 6: A=[5, 5, 5, 5, 5]  B=[8, 4, 4, 4, 4]
  trial 7: A=[7, 7, 7, 7, 7]  B=[10, 6, 6, 6, 6]
  trial 8: A=[9, 9, 8, 8, 8]  B=[10, 8, 8, 8, 8]
  trial 9: A=[9, 3, 3, 3, 3]  B=[6, 5, 5, 5, 5]
  trial 10: A=[4, 4, 4, 4, 4]  B=[7, 2, 2, 2, 2]

**Rationale:** To quantitatively dissociate the Threshold-based Binarization (Satisficing WADD) theory from the pure Weighted Additive (WADD) theory, we exploit their divergent treatment of cardinal rating magnitudes. Pure WADD integrates all cardinal differences linearly, weighted by validity. Satisficing WADD, however, binarizes ratings based on an internal satisficing threshold, ignoring magnitudes once ratings are on the same side of the threshold. We construct a 5-feature design with a rating scale up to 10 and systematically vary the 'plateau' of ratings across trials to probe different potential threshold values (e.g., 3, 5, 7, 9). On these conflict trials, Option A has a massive advantage on the most valid feature but falls just below the probed threshold on all other features. Option B has a moderate value on all features, falling exactly on or just above the probed threshold. Pure WADD consistently prefers Option A due to its overwhelming cardinal advantage on the most important feature. However, Satisficing WADD prefers Option B because Option B crosses the satisficing threshold on all features (or more features), while Option A only crosses it on one. By including trials that target various threshold levels, we can identify whether subjects linearly integrate magnitudes (supporting pure WADD) or exhibit threshold-based preference reversals (supporting Satisficing WADD).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Threshold-based Binarization (Satisficing WADD): Decision-makers simplify complex cardinal information by converting continuous or multi-level ratings into binary cues based on a satisficing threshold. A feature is considered satisfactory (1) if its rating meets or exceeds the threshold, and unsatisfactory (0) otherwise. The options are then evaluated by computing the validity-weighted sum of these binarized features (WADD on binary cues). This mechanism naturally explains why extreme cardinal advantages (e.g., 10 vs 5) might be ignored if both options exceed the satisficing threshold, allowing an option with distributed moderate advantages to win against an option with a single extreme advantage.

**Parameters:**
- threshold: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
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
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize validities
    sum_val = np.sum(val)
    if sum_val > 0:
        val = val / sum_val
        
    # Binarize features based on the threshold
    bin_a = (a >= threshold).astype(float)
    bin_b = (b >= threshold).astype(float)
    
    # Compute validity-weighted sum of binarized features
    score_a = np.sum(bin_a * val)
    score_b = np.sum(bin_b * val)
    
    scores = np.array([score_a, score_b])
    
    # Convert to probabilities via softmax with max-subtraction
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])\n    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])\n    \n    # Identify the High Validity Option (HVO)\n    hvo_is_a = a_f1 > b_f1\n    hvo_is_b = b_f1 > a_f1\n    \n    hvo_chosen = np.where(hvo_is_a, 1 - data['response'], \n                          np.where(hvo_is_b, data['response'], np.nan))\n    \n    # Target trials have a difference of exactly 3 on the most valid feature\n    # and vary their maximum value (baseline shifts: 6, 8, or 10)\n    max_f1 = np.maximum(a_f1, b_f1)\n    mask = max_f1.isin([6, 8, 10]) & (np.abs(a_f1 - b_f1) == 3)\n    \n    if not mask.any():\n        return 0.0\n        \n    df_filtered = data[mask].copy()\n    df_filtered['hvo_chosen'] = hvo_chosen[mask]\n    df_filtered['level'] = max_f1[mask]\n    \n    # Calculate HVO choice rate per baseline level\n    level_rates = df_filtered.groupby('level')['hvo_chosen'].mean()\n    \n    # Return the variance of these rates across levels\n    if len(level_rates) > 1:\n        return float(np.var(level_rates, ddof=0))\n    else:\n        return 0.0",
  "rationale": "The metric computes the variance in the choice rate of the High Validity Option (HVO) across three different cardinal 'baseline' levels (where the maximum rating on the most valid feature is 6, 8, or 10). Because the cardinal difference between the options is perfectly identical across these trials ([3, -1, -1, -1, -1]), Pure WADD predicts a constant choice probability for the HVO across all three levels, resulting in a variance near 0. Satisficing WADD, however, evaluates options by binarizing against a threshold. As the baseline shifts across this threshold, the binarized feature vectors change drastically, leading to sharp preference reversals and a high variance in the HVO choice rate."
}
```

## Usage

```json
{
  "prompt_token_count": 3606,
  "candidates_token_count": 618,
  "total_token_count": 8662
}
```
