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
**Validities (n_features=3):** [0.95, 0.75, 0.55]

**rating_max:** 100

**Trial pairs (n=6):**
  trial 1: A=[100, 0, 0]  B=[20, 100, 100]
  trial 2: A=[0, 100, 100]  B=[90, 0, 0]
  trial 3: A=[80, 10, 10]  B=[10, 90, 90]
  trial 4: A=[10, 80, 80]  B=[80, 10, 10]
  trial 5: A=[90, 20, 20]  B=[20, 80, 80]
  trial 6: A=[20, 90, 90]  B=[90, 20, 20]

**Rationale:** To quantitatively dissociate Random Guessing from Take The Best (TTB), we exploit TTB's lexicographic stopping rule and its threshold parameter. TTB searches through features by validity and stops at the first feature where the absolute difference exceeds its threshold (which is bounded between 0 and 50). By using a large rating scale (rating_max=100) and ensuring the difference on the highest-validity feature is always greater than 50, we force TTB to deterministically choose the option favored by the most valid cue on every trial, completely ignoring all other features. The Random Guessing model, on the other hand, predicts a flat ~50% choice probability regardless of the feature values. This creates a stark, qualitative and quantitative dissociation: TTB predicts strong alternation of choices (A, B, A, B...) tracking the highest-validity cue, while Random Guessing predicts uniform indifference.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

**Parameters:**
- bias: [0.48, 0.52]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])
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
**Description:** Take The Best (TTB) Heuristic: Decision makers process information sequentially, searching through features in descending order of their subjective validity. They stop at the first feature that discriminates between the options (i.e., the absolute difference in feature values exceeds a certain threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents a classic fast-and-frugal one-reason decision-making benchmark.

**Parameters:**
- validities: validities
- threshold: [0.0, 50.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        diff = a[idx] - b[idx]
        if abs(diff) > threshold:
            if diff > 0:
                p_core = np.array([1.0, 0.0])
            else:
                p_core = np.array([0.0, 1.0])
            break
            
    # Incorporate lapse rate (epsilon)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_cues = np.array(data['option_a_ratings'].tolist())\n    b_cues = np.array(data['option_b_ratings'].tolist())\n    \n    a_favored = a_cues[:, 0] > b_cues[:, 0]\n    b_favored = a_cues[:, 0] < b_cues[:, 0]\n    \n    chose_a = (data['response'] == 0).values\n    \n    rate_a_favored = np.mean(chose_a[a_favored]) if np.any(a_favored) else 0.0\n    rate_b_favored = np.mean(chose_a[b_favored]) if np.any(b_favored) else 0.0\n    \n    return float(rate_a_favored - rate_b_favored)",
  "rationale": "Calculates the difference in the probability of choosing Option A when Option A is favored by the highest-validity cue versus when Option B is favored by it. Random Guessing predicts this difference to be near zero, as choices are independent of cue values. Take The Best strongly relies on the first cue, so it will choose A significantly more often when A is favored by it, leading to a substantial positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 2795,
  "candidates_token_count": 340,
  "total_token_count": 4759
}
```
