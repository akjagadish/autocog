# metric_exp00_attempt_00

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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Rationale:** To robustly dissociate the Mixture of Constant Choice and Take-The-Best (TTB) theory from the pure Strong Position Bias theory, this design introduces trials where TTB strictly opposes feature-summing strategies (Tallying/WADD), alongside trials where TTB predicts a tie (identical options). The pure Position Bias model predicts rigidly constant choices (100% A or 100% B) across all trials, completely ignoring the stimulus. In contrast, the Mixture theory predicts that choice probabilities will systematically deviate from the preferred option in the exact direction of the TTB prediction. By heavily pitting TTB against Tallying, we ensure that any deviations from pure constant choice are specifically driven by single-cue TTB lapses rather than general evidence-summing lapses.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Mixture of Constant Choice and Take-The-Best: Subjects primarily exhibit a degenerate strategy of relying on a fixed position preference (always choosing Option A or always Option B) due to low engagement or lack of trial-by-trial feedback. However, on a small fraction of trials, they lapse into using a single-cue heuristic (Take-The-Best), where they compare the options on the most valid cue. This mixture maintains the near-zero variance in choice proportions across most experiments while capturing the slight preference for TTB over Tallying in disagreement trials.

**Parameters:**
- preferred_option: {0, 1}
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    pref = int(parameters["preferred_option"])
    epsilon = float(parameters["epsilon"])
    
    # Constant choice probabilities
    p_const = np.array([1.0, 0.0]) if pref == 0 else np.array([0.0, 1.0])
    
    # Take-The-Best (TTB) prediction
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    ttb_pred = -1
    for idx in order:
        if a[idx] > b[idx]:
            ttb_pred = 0
            break
        elif b[idx] > a[idx]:
            ttb_pred = 1
            break
            
    if ttb_pred == 0:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_pred == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_const + epsilon * p_ttb
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
**Description:** Strong Position Bias / Constant Choice: Due to the lack of trial-by-trial feedback and low engagement, subjects adopt a degenerate strategy of always choosing the same option (e.g., always Option A or always Option B) regardless of the cues. This leads to choice probabilities of 1.0 or 0.0 for a given subject across all trials, perfectly explaining the near-zero within-subject variance across trial types and the extreme choice probabilities observed.

**Parameters:**
- preferred_option: {0, 1}

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # The subject has a strict preference for either Option A (0) or Option B (1)
    pref = int(parameters["preferred_option"])
    
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subject_score(df):\n        mode = df['response'].mode().iloc[0]\n        score = 0\n        for _, row in df.iterrows():\n            if row['response'] != mode:\n                a = row['option_a_ratings']\n                b = row['option_b_ratings']\n                ttb_pred = -1\n                for i in range(len(a)):\n                    if a[i] > b[i]:\n                        ttb_pred = 0\n                        break\n                    elif b[i] > a[i]:\n                        ttb_pred = 1\n                        break\n                if ttb_pred == row['response']:\n                    score += 1\n        return score / len(df)\n    return float(data.groupby('subject_id').apply(subject_score).mean())",
  "rationale": "To discriminate between the Mixture of Constant Choice + Take-The-Best (TTB) and the pure Strong Position Bias theory, this metric specifically isolates trials where a subject deviates from their modal (preferred) choice. Under Strong Position Bias, subjects never deviate from their preferred option, so this metric will be exactly 0.0 with zero variance. Under the Mixture theory, subjects occasionally lapse from their preferred option, and these deviations are systematically driven by the TTB heuristic. The metric counts the proportion of trials where a deviation from the subject's mode perfectly aligns with the TTB prediction, yielding a strictly positive value exclusively for the Mixture theory."
}
```

## Usage

```json
{
  "prompt_token_count": 2969,
  "candidates_token_count": 362,
  "total_token_count": 6044
}
```
