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

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[6, 2, 2, 2]  B=[5, 10, 10, 10]
  trial 2: A=[3, 3, 3, 0]  B=[2, 2, 2, 10]
  trial 3: A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  trial 4: A=[8, 10, 10, 10]  B=[9, 0, 0, 0]
  trial 5: A=[10, 10, 0, 0]  B=[10, 0, 10, 10]
  trial 6: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 7: A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  trial 8: A=[10, 0, 10, 0]  B=[0, 10, 0, 10]

**Rationale:** To quantitatively dissociate the Random Guessing theory from the Response Inertia theory, we must analyze the sequential dependencies (autocorrelation) of choices across trials. Both theories assume subjects ignore the stimulus features entirely, predicting an overall 50% choice rate for Option A across the experiment. However, Random Guessing posits that each trial is independent (zero memory), predicting that the probability of repeating the previous choice is exactly 0.5. Response Inertia posits that choices depend on the immediately preceding choice, predicting a repeat probability significantly different from 0.5 (either >0.5 for perseveration or <0.5 for alternation). We present a diverse set of trials that strongly dissociate standard feature-based heuristics (TTB, WADD, Tallying, EQW) to ensure that if subjects are actually using feature information, we can capture it, while providing a rigorously randomized sequence to test the trial-to-trial transition probabilities required to separate the two noise-based theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** In complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. Thus, behavior resembles zero-intelligence random guessing.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])
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
**Description:** Response Inertia and Alternation: Subjects in complex, feedback-free multi-attribute choice tasks ignore the stimulus features completely. Instead, their choices are driven by sequential dependencies, where they either exhibit perseveration (a tendency to repeat their previous choice) or alternation (a tendency to switch choices). This results in aggregate choice probabilities that appear as 50/50 random guessing on feature-based metrics, but are mechanistically generated by trial-to-trial memory rather than zero-memory noise.

**Parameters:**
- p_repeat: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    
    if not history or len(history.get("response", [])) == 0:
        return np.array([0.5, 0.5])
        
    last_resp = history["response"][-1]
    
    if last_resp == 0:
        p_A = p_repeat
    else:
        p_A = 1.0 - p_repeat
        
    return np.array([p_A, 1.0 - p_A])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    deviations = []\n    for _, df in data.groupby('subject_id'):\n        resp = df['response'].values\n        if len(resp) > 1:\n            repeats = np.mean(resp[1:] == resp[:-1])\n            deviations.append(abs(repeats - 0.5))\n    return float(np.mean(deviations)) if deviations else 0.0",
  "rationale": "To distinguish Random Guessing from Response Inertia, we look at the sequence of choices. Both theories predict an overall 50% choice rate for each option, but they differ in trial-to-trial dependencies. Random Guessing predicts that the probability of repeating the previous choice is exactly 0.5 (with small binomial variance for finite trials). Response Inertia predicts a repeat probability that systematically deviates from 0.5 (either higher for perseveration or lower for alternation). By calculating the absolute deviation of each subject's repeat rate from 0.5 and averaging across subjects, we get a metric that will be close to 0 under Random Guessing and significantly higher under Response Inertia."
}
```

## Usage

```json
{
  "prompt_token_count": 2682,
  "candidates_token_count": 276,
  "total_token_count": 4239
}
```
