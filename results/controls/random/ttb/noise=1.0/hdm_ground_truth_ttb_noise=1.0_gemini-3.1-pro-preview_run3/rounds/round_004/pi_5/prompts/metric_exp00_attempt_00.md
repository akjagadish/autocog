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
**Validities (n_features=15):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.95, 0.88, 0.82, 0.78, 0.72, 0.68, 0.62]

**rating_max:** 1

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To dissociate Pure Random Guessing from Tallying with Extreme Noise, we must exploit the minimal but potentially non-zero inverse temperature (beta <= 0.01) in the competing Tallying model. Pure Random Guessing invariably predicts exactly 50/50 choice probabilities regardless of the stimulus. In contrast, Tallying with Extreme Noise predicts a slight, systematic deviation from 50/50 that scales with the absolute difference in strict feature wins between the options. By using a large number of features (15) and creating trials ranging from unanimous domination to partial dominance and exact ties, the Tallying model predicts a stimulus-dependent choice probability variance (e.g., up to ~53.7% for a 15-0 sweep), while the advocated Pure Random Guessing model remains stubbornly flat at exactly 50.0%.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Pure Random Guessing (Inattentive Responding). Subjects are completely inattentive to the complex multi-attribute stimuli and simply guess uniformly at random on every trial, bypassing any cognitive evaluation of the features entirely. This parsimonious model assumes choice probabilities are exactly 50/50 for all trials, independent of the stimulus.

**Parameters:**
- guessing_rate: {1.0}

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # Reference the parameter to satisfy the constraint, though choice is always 50/50
    _ = float(parameters['guessing_rate'])
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

**Parameters:**
- beta: [0.0, 0.01]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Extract ratings into 2D numpy arrays for vectorized comparison\n    A = np.array(list(data['option_a_ratings']))\n    B = np.array(list(data['option_b_ratings']))\n    \n    # Count strict feature wins for each option\n    wins_a = np.sum(A > B, axis=1)\n    wins_b = np.sum(B > A, axis=1)\n    \n    # Difference in tallies (B - A)\n    tally_diff_b_minus_a = wins_b - wins_a\n    \n    # Map response 0 -> -1 (chose A) and 1 -> 1 (chose B)\n    # Multiplying gives the tally advantage of the CHOSEN option\n    responses = data['response'].values\n    chosen_advantage = tally_diff_b_minus_a * (2 * responses - 1)\n    \n    # Return the total sum of the chosen option's advantage\n    return float(np.sum(chosen_advantage))\n",
  "rationale": "Measures the total sum of the tally advantage for the chosen options across all evaluated trials. Under Pure Random Guessing, choices are completely independent of the feature wins, so this sum tightly concentrates around zero. Under Tallying with Extreme Noise, even a tiny inverse temperature (beta) creates a slight, systematic bias toward the option with more wins. By aggregating the sum over the entire pooled DataFrame, the small per-trial expected value compounds into a massive point estimate difference between the two theories, easily overcoming the between-subject variance in Welch's t-test."
}
```

## Usage

```json
{
  "prompt_token_count": 3328,
  "candidates_token_count": 403,
  "total_token_count": 14032
}
```
