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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[10, 10, 2, 2, 2]  B=[5, 5, 5, 5, 5]
  trial 2: A=[10, 10, 4, 4, 4]  B=[7, 7, 7, 7, 7]
  trial 3: A=[10, 10, 7, 7, 7]  B=[9, 9, 9, 9, 9]
  trial 4: A=[8, 8, 8, 8, 8]  B=[10, 10, 7, 7, 7]
  trial 5: A=[6, 6, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  trial 6: A=[4, 4, 4, 4, 4]  B=[10, 10, 3, 3, 3]

**Rationale:** To quantitatively dissociate the pure Weighted Additive (WADD) theory from the Threshold-based Binarization (Satisficing WADD) theory, we exploit their divergent treatment of cardinal rating magnitudes. Pure WADD integrates all cardinal differences linearly, weighted by validity. Satisficing WADD, however, binarizes ratings based on an internal threshold, ignoring magnitudes once ratings are on the same side of the threshold. We construct a 5-feature design with a rating scale up to 10 and systematically vary the 'plateau' of ratings across trials. In one set of trials (1-3), Option A has a massive advantage on the two most valid features but a slight disadvantage on the remaining three, such that pure WADD strongly prefers A. However, if a subject's satisficing threshold falls between the lower and higher ratings, Satisficing WADD will prefer Option B because B crosses the threshold on all 5 features while A only crosses it on 2. In the reverse set of trials (4-6), the pattern is flipped to catch other potential threshold values. By probing multiple potential threshold boundaries (e.g., 4, 6, 8), we can determine whether subjects continuously integrate magnitudes (supporting pure WADD) or exhibit step-function preference reversals indicative of binarization (supporting Satisficing WADD).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])\n    \n    a_mat = np.vstack(data['option_a_ratings'].values)\n    b_mat = np.vstack(data['option_b_ratings'].values)\n    \n    score_diff = np.sum((a_mat - b_mat) * val, axis=1)\n    \n    choice_a = (data['response'] == 0).astype(float)\n    \n    if len(np.unique(score_diff)) <= 1 or len(np.unique(choice_a)) <= 1:\n        return 0.0\n        \n    return float(np.corrcoef(score_diff, choice_a)[0, 1])",
  "rationale": "This metric calculates the Pearson correlation between the subject's probability of choosing Option A and the continuous Weighted Additive (WADD) score difference between Option A and Option B. Under the pure WADD theory, choices are directly generated by a softmax over these exact continuous score differences, leading to a high positive correlation. Under the Satisficing WADD theory, the cardinal feature values are binarized based on a threshold before being integrated, which destroys the linear relationship with the continuous score differences and significantly lowers the correlation. This provides a robust, parameter-independent discriminant."
}
```

## Usage

```json
{
  "prompt_token_count": 3451,
  "candidates_token_count": 346,
  "total_token_count": 8232
}
```
