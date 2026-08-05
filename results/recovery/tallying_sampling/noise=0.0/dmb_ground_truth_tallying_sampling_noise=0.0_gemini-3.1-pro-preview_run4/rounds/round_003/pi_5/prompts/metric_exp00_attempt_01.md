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
**Validities (n_features=6):** [0.98, 0.92, 0.6, 0.55, 0.52, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate WADD with Linearly Compressed Weights from Tallying, we use a 6-feature design with two highly valid cues and four low-validity cues. Tallying assumes all features are weighted equally (effectively alpha=0), predicting choices based purely on the count of positive features. Compressed WADD posits that individuals interpolate between equal weights and objective validities (alpha > 0). By including trials where Tallying predicts a tie (equal feature counts) but Compressed WADD predicts a distinct preference based on validity, and trials where an option with fewer but highly valid features overcomes an option with more but lower-validity features, we can cleanly separate the two models. If subjects show sensitivity to validities (preferring the higher-weighted option despite equal or slightly lower feature counts), it supports Compressed WADD over pure Tallying.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive Strategy (WADD) with Linearly Compressed Weights: Individuals compute a compensatory weighted sum of positive features for each option, but their subjective weights are compressed relative to objective validities. People interpolate between raw validities and equal weights (Tallying), flattening the weight distribution. This allows a large number of less valid cues to compensate for a highly valid cue, matching human behavior in compensatory environments.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- alpha: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Interpolate between raw validities and equal weights
    weights = alpha * validities + (1.0 - alpha) * 1.0
    
    # Weighted Additive Strategy
    scores = stim @ weights
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: This metric isolates trials where Option A has fewer positive features than Option B. In the experimental design, Option A in these trials always contains the highly valid cues, while Option B contains a larger number of low-validity cues. The Tallying model (Equal Weights) strictly prefers Option B (response = 1) because it simply counts features. The WADD model (with compressed weights) will often prefer Option A (response = 0) because the high validities outweigh the numerical disadvantage. Thus, the proportion of times Option A is chosen on these trials will be near 0 for Tallying and significantly higher for WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_a < sum_b
    if not mask.any():
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.2079 (var=0.0191) adversary_sim=0.1625 (var=0.0149) welch_t=+1.232 p=0.224 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Define the specific trials where WADD and Tallying diverge maximally\n    # Trial 1: Tallying strongly prefers B (sum 2 vs 3), WADD often prefers A (validity 1.9 vs 1.67)\n    t1 = ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0))\n    # Trial 4: Tallying is indifferent (sum 1 vs 1), WADD prefers A (validity 0.98 vs 0.6)\n    t4 = ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))\n    # Trial 6: Tallying is indifferent (sum 2 vs 2), WADD prefers A (validity 1.90 vs 1.15)\n    t6 = ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))\n    \n    a_tup = data['option_a_ratings'].apply(tuple)\n    b_tup = data['option_b_ratings'].apply(tuple)\n    \n    mask = ((a_tup == t1[0]) & (b_tup == t1[1])) | \\\n           ((a_tup == t4[0]) & (b_tup == t4[1])) | \\\n           ((a_tup == t6[0]) & (b_tup == t6[1]))\n    \n    if not mask.any():\n        return 0.0\n        \n    # Return the proportion of times Option A was chosen on these diagnostic trials\n    return float((data.loc[mask, 'response'] == 0).mean())",
  "rationale": "Instead of aggregating over all trials where Option A has fewer features (which diluted the effect in previous attempts), this metric isolates the three exact trials where the models' predictions are most distinct. In Trial 1, Tallying strictly prefers B while WADD prefers A. In Trials 4 and 6, Tallying predicts a tie (50% chance of choosing A), whereas WADD strongly prefers A due to higher cue validities. The mean proportion of choosing Option A on these specific trials should be around 33% for Tallying, but significantly higher for WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 3313,
  "candidates_token_count": 592,
  "total_token_count": 6065
}
```
