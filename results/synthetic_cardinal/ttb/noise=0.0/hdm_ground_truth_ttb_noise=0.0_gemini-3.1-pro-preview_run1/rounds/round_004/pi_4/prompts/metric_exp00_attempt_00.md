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

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 50). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.6, 0.55, 0.5]

**rating_max:** 1

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate Weighted Tallying from Pure Tallying, we use 6 binary features (rating_max=1) to eliminate any magnitude-based effects and focus entirely on how feature wins are counted. Pure Tallying counts all winning features equally, whereas Weighted Tallying weights each win by its validity. We select a set of validities [0.90, 0.85, 0.80, 0.60, 0.55, 0.50] that creates a clear separation between high-validity and low-validity features. The trials are designed to pit these mechanisms against each other: in some trials, one option wins on two high-validity features while the other wins on three low-validity features (Pure Tallying prefers the latter, Weighted Tallying prefers the former). We also include trials where both options have an equal number of winning features (a tie for Pure Tallying) but differ in the validities of those features, leading to a strong preference in Weighted Tallying.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

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
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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
**Description:** Pure Tallying Heuristic: Decision-makers evaluate options by completely ignoring the provided validities or weights of the cues. Instead, they simply count the number of features on which one option is strictly better than the other. The option with the higher total count of winning features is chosen. This unweighted counting mechanism explains the robust preference for options with more winning features, even if those features are of lower validity.

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins; completely ignore validities and magnitudes
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    scores = np.array([a_wins, b_wins], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify trials where one of the options has the two highest-validity features (index 0 and 1) \n    # and nothing else. This corresponds to Trial 1: [1, 1, 0, 0, 0, 0] vs [0, 0, 0, 1, 1, 1].\n    a_target = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)\n    b_target = data['option_b_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)\n    \n    chose_a = (data['response'] == 0)\n    chose_b = (data['response'] == 1)\n    \n    # Check if the subject chose the target option\n    target_chosen = (a_target & chose_a) | (b_target & chose_b)\n    valid_trials = a_target | b_target\n    \n    if valid_trials.sum() == 0:\n        return 0.5\n        \n    return float(target_chosen.sum() / valid_trials.sum())",
  "rationale": "This metric calculates the proportion of times the subject chooses the option with exactly two winning features of highest validity (0.90 and 0.85) over an option with three winning features of lower validity (0.60, 0.55, and 0.50). Under Pure Tallying, subjects simply count the winning features (2 vs 3) and thus strongly prefer the latter option, yielding a metric close to 0. Under Weighted Tallying, subjects weight the wins by validity (1.75 vs 1.65), reversing the preference and yielding a metric close to 1. This provides a massive, stable discrimination between the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3233,
  "candidates_token_count": 455,
  "total_token_count": 7074
}
```
