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
**Validities (n_features=4):** [0.95, 0.72, 0.67, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 15: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 16: A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take-Two with Conditional Fallback: Decision-makers evaluate the top two most valid features. If these two features agree or one favors an option while the other ties, that option is chosen. If they conflict (each option wins one), the decision-maker probabilistically mixes between reverting to the 1st feature and the 3rd feature. If the top two features tie (neither option wins on either), the decision-maker falls back to a simple tally of all features to break the tie.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-Two expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Evaluate the top two features
    f1, f2 = order[0], order[1]
    
    wins_a = (a[f1] > b[f1]) + (a[f2] > b[f2])
    wins_b = (b[f1] > a[f1]) + (b[f2] > a[f2])
    
    if wins_a > wins_b:
        score_a, score_b = 1.0, 0.0
    elif wins_b > wins_a:
        score_a, score_b = 0.0, 1.0
    else:
        if wins_a == 1 and wins_b == 1:
            # Conflict in top 2 features
            gamma = float(parameters["gamma"])
            
            # F1 preference (revert to most valid feature)
            score_a_f1, score_b_f1 = 0.5, 0.5
            if a[f1] > b[f1]:
                score_a_f1, score_b_f1 = 1.0, 0.0
            elif b[f1] > a[f1]:
                score_a_f1, score_b_f1 = 0.0, 1.0
                
            # F3 preference
            score_a_f3, score_b_f3 = 0.5, 0.5
            if len(order) > 2:
                f3 = order[2]
                if a[f3] > b[f3]:
                    score_a_f3, score_b_f3 = 1.0, 0.0
                elif b[f3] > a[f3]:
                    score_a_f3, score_b_f3 = 0.0, 1.0
                    
            score_a = gamma * score_a_f1 + (1.0 - gamma) * score_a_f3
            score_b = gamma * score_b_f1 + (1.0 - gamma) * score_b_f3
        else:
            # Tie in top 2 features (0 wins each)
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            score_a, score_b = 0.5, 0.5
            if tally_a > tally_b:
                score_a, score_b = 1.0, 0.0
            elif tally_b > tally_a:
                score_a, score_b = 0.0, 1.0
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a0 = data['option_a_ratings'].apply(lambda x: x[0])\n    b0 = data['option_b_ratings'].apply(lambda x: x[0])\n    a1 = data['option_a_ratings'].apply(lambda x: x[1])\n    b1 = data['option_b_ratings'].apply(lambda x: x[1])\n    \n    mask_A = (a0 > b0) & (a1 < b1)\n    mask_B = (a0 < b0) & (a1 > b1)\n    \n    subset_A = data[mask_A]\n    subset_B = data[mask_B]\n    \n    total_conflict = len(subset_A) + len(subset_B)\n    if total_conflict == 0:\n        return 0.5\n        \n    chose_f0_A = (subset_A['response'] == 0).sum()\n    chose_f0_B = (subset_B['response'] == 1).sum()\n    \n    return float((chose_f0_A + chose_f0_B) / total_conflict)",
  "rationale": "This metric calculates the proportion of times the decision-maker chooses the option favored by the most valid feature (f0) when the top two features conflict (f0 favors one option, f1 favors the other). Under the 'Take The Best' theory, the choice is entirely dictated by f0, resulting in a proportion close to 1.0. Under the 'Take-Two with Conditional Fallback' theory, a conflict triggers a probabilistic mix with the third feature (f3), which often ties in these specific experimental trials, diluting the adherence to f0 and resulting in a significantly lower proportion."
}
```

## Usage

```json
{
  "prompt_token_count": 3805,
  "candidates_token_count": 442,
  "total_token_count": 8215
}
```
