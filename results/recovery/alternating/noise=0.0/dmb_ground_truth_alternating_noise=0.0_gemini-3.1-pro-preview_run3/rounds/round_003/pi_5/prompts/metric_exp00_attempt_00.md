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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 8: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Spatial Bias / Response Heuristic model from Random Guessing, we must isolate response-based patterns from stimulus-driven effects. Because both theories assume subjects ignore feature values, the stimulus design must ensure that any observed spatial bias or autocorrelation cannot be attributed to the stimuli themselves. We use a perfectly symmetric set of trial pairs where Option A's rating vectors are exactly mirrored by Option B's vectors across the experiment. This guarantees that any feature-based strategy would result in a 50/50 overall choice rate and zero sequential dependence. Consequently, any significant deviation from 50% (spatial bias) or significant autocorrelation (tendency to repeat/alternate) will unambiguously identify the advocated theory's parameters, whereas Random Guessing strictly predicts zero for both.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Response Heuristic / Spatial Bias: Subjects in these paradigms often ignore feature information entirely and instead rely on presentation-based response heuristics. Decision-making is driven by a persistent spatial bias (e.g., a preference for Option A over Option B) combined with a sequential dependence, such as a tendency to repeat the previous choice or alternate between options. This produces behavior that appears random with respect to cue validities but contains structured autocorrelation and spatial preference.

**Parameters:**
- beta_bias: [-3.0, 3.0]
- beta_repeat: [-3.0, 3.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    beta_bias = float(parameters["beta_bias"])
    beta_repeat = float(parameters["beta_repeat"])
    
    # Determine the indicator for the previous choice
    # 1.0 if Option A was chosen, -1.0 if Option B was chosen, 0.0 if first trial
    if len(history["response"]) == 0:
        last_a = 0.0
    else:
        last_a = 1.0 if history["response"][-1] == 0 else -1.0
        
    # Logit for choosing Option A
    z = beta_bias + beta_repeat * last_a
    
    # Convert to probability using sigmoid
    p_a = 1.0 / (1.0 + np.exp(-z))
    
    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Random Guessing: The data indicates that subjects in these specific experimental paradigms completely ignore all feature information and validities. Likely due to a lack of trial-by-trial feedback or low motivation, subjects simply choose between the two options with equal probability on every trial, resulting in chance-level performance across all metrics.

**Parameters:**
- dummy: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Reference the dummy parameter to satisfy the parameter usage constraint
    _ = float(parameters["dummy"])
    
    # Pure random guessing: uniform 50/50 probability regardless of the stimulus
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    def subject_score(df):\n        resp = df['response'].values\n        if len(resp) < 2:\n            return 0.5\n        p_A = float((resp == 0).mean())\n        p_B = float((resp == 1).mean())\n        p_rep = float((resp[1:] == resp[:-1]).mean())\n        p_alt = float((resp[1:] != resp[:-1]).mean())\n        return max(p_A, p_B, p_rep, p_alt)\n        \n    return float(data.groupby('subject_id').apply(subject_score).mean())",
  "rationale": "The Response Heuristic model predicts that subjects will exhibit either a strong spatial bias (consistently choosing one option) or a strong sequential dependence (consistently repeating or alternating choices). The Random Guessing model predicts independent 50/50 choices, resulting in no such patterns. By computing the maximum of the proportion of choosing A, choosing B, repeating a choice, or alternating a choice for each subject, we capture the strongest behavioral regularity present. Under Random Guessing, this maximum will tightly cluster slightly above 0.5 due to random noise. Under the Response Heuristic model, because the bias and repeat parameters are drawn from [-3, 3], at least one of these proportions will be substantially elevated, resulting in a significantly higher metric."
}
```

## Usage

```json
{
  "prompt_token_count": 2705,
  "candidates_token_count": 342,
  "total_token_count": 6200
}
```
