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
**Validities (n_features=7):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Random Guessing from Biased Logistic Tallying, this design orthogonally manipulates the Tally difference (from -5 to +5) and the prediction of Take The Best (by flipping the most valid cue). Biased Logistic Tallying predicts a graded choice probability driven by the tally difference, while TTB predicts a step function based on the top cue. The advocated Random Guessing theory predicts that subjects ignore both feature counts and validities, resulting in a completely flat choice probability across all trials regardless of the strong compensatory or lexicographic signals.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** In complex, multi-attribute binary choice tasks without trial-by-trial feedback, subjects largely fail to integrate cue validities or even simple feature counts. Instead, their decisions are dominated by high response noise, effectively reducing their behavior to random guessing with a slight baseline bias for one option over the other. This explains the ~0.5 choice rates, ~0.5 TTB match rates, and ~0.0 differences across experimental conditions.

**Parameters:**
- bias_a: [0.3, 0.7]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
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
**Description:** Biased Logistic Tallying: Subjects make decisions by counting the total number of positive features for each option (Tallying), ignoring cue validities. However, their choices are highly noisy and subject to a baseline spatial/option bias. The decision process is modeled as a logistic function over the difference in feature tallies between the two options, parameterized by a sensitivity weight and a baseline bias. The sensitivity to tally differences is weak, reflecting empirical data where feature counts only slightly sway noisy guessing.

**Parameters:**
- beta: [-0.2, 0.2]
- bias: [-2.0, 2.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    
    # Logistic choice rule based on tally difference and baseline bias
    diff = scores[0] - scores[1]
    logit = beta * diff + bias
    
    # Numerically stable sigmoid
    if logit >= 0:
        p_a = 1.0 / (1.0 + np.exp(-logit))
    else:
        exp_logit = np.exp(logit)
        p_a = exp_logit / (1.0 + exp_logit)
        
    return np.array([p_a, 1.0 - p_a])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    def calc_r2(df):\n        tally_A = df['option_a_ratings'].apply(sum)\n        tally_B = df['option_b_ratings'].apply(sum)\n        tally_diff = tally_A - tally_B\n        choice_A = 1 - df['response']\n        \n        std_t = tally_diff.std()\n        std_c = choice_A.std()\n        if std_t == 0 or std_c == 0 or np.isnan(std_t) or np.isnan(std_c):\n            return 0.0\n        r = np.corrcoef(tally_diff, choice_A)[0, 1]\n        return float(r**2)\n\n    if 'subject_id' in data.columns:\n        return float(data.groupby('subject_id').apply(calc_r2).mean())\n    else:\n        return float(calc_r2(data))\n",
  "rationale": "This metric calculates the squared Pearson correlation between the tally difference (sum of features for A minus sum of features for B) and the choice of option A, averaged across subjects. For the Random Guessing theory, choices are independent of feature tallies, so the expected squared correlation is near zero (only reflecting finite-sample noise). For the Biased Logistic Tallying theory, choice probabilities systematically vary with the tally difference, resulting in a significantly higher squared correlation on average."
}
```

## Usage

```json
{
  "prompt_token_count": 3084,
  "candidates_token_count": 367,
  "total_token_count": 5394
}
```
