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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 1]  B=[0, 1, 1, 0]

**Rationale:** To cleanly dissociate the First-Feature Heuristic (with high lapse) from a Random Guessing baseline, we must manipulate the value of the first (leftmost) feature while varying the other features. The First-Feature Heuristic predicts a higher probability of choosing Option A when Option A has a 1 and Option B has a 0 on the first feature, and a lower probability when the reverse is true. When the first feature is tied, it predicts equal preference (50/50). In contrast, the Random Guessing model predicts a completely flat choice probability across all trials, driven only by an idiosyncratic side bias. By including trials where the first feature strongly favors A, strongly favors B, or is tied, we can detect whether the choice probabilities systematically shift with the first feature or remain invariant.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** First-Feature Heuristic (Left-to-Right Reading Bias) with High Lapse: Subjects minimize cognitive effort by inspecting only the first (leftmost) feature in the array. However, because this task is demanding and feedback is absent, subjects rely heavily on random guessing on the vast majority of trials, only occasionally applying the first-feature rule. This results in behavior that is near-random but retains a faint trace of the leftmost feature's influence.

**Parameters:**
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a_first = stim[0, 0]
    b_first = stim[1, 0]
    
    epsilon = float(parameters["epsilon"])
    
    if a_first > b_first:
        p_core = np.array([1.0, 0.0])
    elif b_first > a_first:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend deterministic first-feature rule with a random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

**Parameters:**
- bias: [0.4, 0.6]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_first = data['option_a_ratings'].apply(lambda x: x[0])\n    b_first = data['option_b_ratings'].apply(lambda x: x[0])\n    choose_a = 1 - data['response']\n    \n    mask_a_better = (a_first == 1) & (b_first == 0)\n    mask_b_better = (a_first == 0) & (b_first == 1)\n    \n    p_a_given_a_better = choose_a[mask_a_better].mean() if mask_a_better.sum() > 0 else 0.5\n    p_a_given_b_better = choose_a[mask_b_better].mean() if mask_b_better.sum() > 0 else 0.5\n    \n    return float(p_a_given_a_better - p_a_given_b_better)",
  "rationale": "This metric isolates the effect of the first feature by comparing the probability of choosing Option A when the first feature uniquely favors A versus when it uniquely favors B. Under the First-Feature Heuristic, this difference will be positive (even with high lapse rates, a faint trace remains). Under the Random Guessing model, this difference will be zero on average, as the first feature is ignored."
}
```

## Usage

```json
{
  "prompt_token_count": 2867,
  "candidates_token_count": 336,
  "total_token_count": 5029
}
```
