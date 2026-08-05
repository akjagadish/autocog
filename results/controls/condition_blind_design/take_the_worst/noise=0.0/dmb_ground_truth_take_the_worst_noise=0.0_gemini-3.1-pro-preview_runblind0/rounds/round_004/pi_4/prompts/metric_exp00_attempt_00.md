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
**Validities (n_features=4):** [0.95, 0.63, 0.6, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People process feature arrays from right to left, exhibiting a recency or layout bias. Instead of consulting cues in order of their objective validities, subjects use a Right-to-Left Take-The-Best (R2L-TTB) heuristic. They evaluate features starting from the last index down to the first, choosing the option that wins on the first discriminating feature encountered in this reversed order. If no feature discriminates, they guess. Response noise is incorporated via a softmax over the binary winner score and a constant lapse rate.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("R2L-TTB expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    # Right-to-left cue order
    cue_order = list(range(n_features - 1, -1, -1))
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for choice probability
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People integrate all available features but exhibit a strong spatial attention bias, weighting features more heavily the further to the right they appear. This Right-biased Compensatory Model evaluates options using a weighted additive sum where the subjective weight of each feature grows exponentially from left to right. This allows the model to capture behavior that appears strictly non-compensatory (like Right-to-Left Take-The-Best) when the growth factor is large, while still accommodating subjects who might integrate multiple cues in a compensatory manner with a heavy rightward bias. Decisions are made by applying a softmax choice rule over these weighted sums, alongside a baseline rate of random guessing.

**Parameters:**
- gamma: [1.0, 10.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Right-biased WADD expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Weights increase exponentially from left to right
    weights = np.array([gamma ** j for j in range(n_features)])
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    # Softmax for choice probability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Trial 1: A=[0,0,0,0], B=[1,1,1,1] -> B dominates\n    is_t1 = (data['option_a_ratings'].apply(tuple) == (0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (1, 1, 1, 1))\n    p_b_t1 = (data[is_t1]['response'] == 1).mean()\n    \n    # Trial 11: A=[1,0,1,0], B=[0,0,1,0] -> A wins on the leftmost feature only\n    is_t11 = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 0))\n    p_a_t11 = (data[is_t11]['response'] == 0).mean()\n    \n    if pd.isna(p_b_t1) or pd.isna(p_a_t11):\n        return 0.0\n        \n    return float(p_b_t1 - p_a_t11)",
  "rationale": "In R2L-TTB, the choice probability depends solely on a fixed score difference (1 vs 0) for the winner, regardless of how many features favor it. Thus, R2L-TTB predicts that the probability of choosing B in Trial 1 (where B wins on all features) should be identical to the probability of choosing A in Trial 11 (where A wins on only the leftmost feature, and all other features are tied). In contrast, the Right-biased WADD model produces a massive score difference for Trial 1 (sum of all weights) but a very small score difference for Trial 11 (only the weight of the leftmost feature, gamma^0 = 1). Therefore, WADD predicts a much higher accuracy for Trial 1 than Trial 11, while R2L-TTB predicts a difference of zero."
}
```

## Usage

```json
{
  "prompt_token_count": 3331,
  "candidates_token_count": 511,
  "total_token_count": 8907
}
```
