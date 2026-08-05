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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** To dissociate Weighted Additive (WADD) with Extreme Noise from Pure Random Guessing, we must look for any systematic deviation from 0.50 that correlates with the weighted sum of features. Pure Random Guessing predicts exactly 50/50 choices across all trials, regardless of the stimulus. WADD with Extreme Noise predicts that choices will be heavily regressed toward 0.50, but will still retain a small, monotonic signal favoring the option with the higher weighted sum. By including trials with extreme differences (e.g., all positive vs. all negative cues) as well as moderate and reversed differences, we can test whether the small deviations from chance are systematic (supporting WADD) or non-existent (supporting Pure Guessing).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) with Extreme Noise Theory: Subjects attempt to integrate all available cues by weighting them according to their provided validities. However, the cognitive demand of integrating multiple conflicting fictitious validities is overwhelming. This results in extreme decision conflict and distraction, leading to a near-total reliance on guessing (a very high lapse rate). Consequently, choice probabilities are pulled almost entirely toward chance (0.50), masking the underlying compensatory process in the aggregate behavioral data.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.95, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Weighted sum per option (dot product with per-feature weights)
    scores = stim @ (validities * w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with extreme uniform lapse (guessing)
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Pure Random Guessing (Complete Cognitive Overload): Subjects completely ignore the cues and validities due to the high cognitive demand and artificial nature of the fictitious task. This results in choices that are strictly driven by a 50/50 binomial coin flip, without any underlying compensatory or non-compensatory signal.

**Parameters:**
- guessing_rate: {0.5}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    p = float(parameters['guessing_rate'])
    return np.array([p, 1.0 - p])
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
[0] rationale: Calculates the proportion of choices that align with the Weighted Additive (WADD) model using the provided objective validities. While WADD with Extreme Noise predicts this proportion will be heavily regressed toward 0.50, it will still be systematically greater than 0.50. Pure Random Guessing predicts exactly 0.50. By averaging across all trials, this metric maximizes the signal-to-noise ratio to detect the small residual compensatory signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.5])
    
    def get_wadd_choice(row):
        score_a = np.dot(row['option_a_ratings'], validities)
        score_b = np.dot(row['option_b_ratings'], validities)
        return 0 if score_a > score_b else 1
        
    wadd_choices = data.apply(get_wadd_choice, axis=1)
    return float(np.mean(data['response'] == wadd_choices))
outcome: self_sim=0.5058 (var=0.0019) adversary_sim=0.4927 (var=0.0030) welch_t=+0.930 p=0.3571 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert option ratings to their sums to identify trial types\n    sum_a = data['option_a_ratings'].apply(sum)\n    \n    # We isolate the trials with the most extreme differences in WADD scores.\n    # Trial 1: A has all features (sum=6), B has none. WADD strongly favors A (response 0).\n    # Trial 2: A has no features (sum=0), B has all. WADD strongly favors B (response 1).\n    # Trial 5: A has 1 feature (sum=1), B has 5. WADD strongly favors B (response 1).\n    # Trial 6: A has 5 features (sum=5), B has 1. WADD strongly favors A (response 0).\n    # Trials 3 and 4 have much smaller score differences, so we exclude them to concentrate the signal.\n    \n    mask_t1 = (sum_a == 6)\n    mask_t2 = (sum_a == 0)\n    mask_t5 = (sum_a == 1)\n    mask_t6 = (sum_a == 5)\n    \n    consistent_t1 = (data.loc[mask_t1, 'response'] == 0).sum()\n    consistent_t2 = (data.loc[mask_t2, 'response'] == 1).sum()\n    consistent_t5 = (data.loc[mask_t5, 'response'] == 1).sum()\n    consistent_t6 = (data.loc[mask_t6, 'response'] == 0).sum()\n    \n    total_consistent = consistent_t1 + consistent_t2 + consistent_t5 + consistent_t6\n    total_trials = mask_t1.sum() + mask_t2.sum() + mask_t5.sum() + mask_t6.sum()\n    \n    if total_trials == 0:\n        return 0.5\n        \n    return float(total_consistent / total_trials)",
  "rationale": "Due to the extreme noise (lapse rate between 95% and 100%), the WADD compensatory signal is heavily diluted, causing the overall choice probabilities to regress almost entirely to 0.50. A metric that averages accuracy across all trials includes trials where the WADD score difference is too weak to overcome the binomial noise, masking the effect. By restricting the analysis strictly to the trials with the most extreme objective score differences (Trials 1, 2, 5, and 6), we maximize the expected deviation from 0.50 under the WADD theory. Pure Random Guessing will still yield exactly 0.50 on this subset. This subsetting strategy provides the highest possible signal-to-noise ratio for distinguishing the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3102,
  "candidates_token_count": 668,
  "total_token_count": 25368
}
```
