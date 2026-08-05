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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Dual-Process Strategy Selection' theory, we probe the core difference in their threshold mechanisms. The advocated theory asserts that ANY non-zero tally difference strictly determines the choice, with validities only used to break perfect ties. The competing theory posits a variable threshold (1 to 5), predicting that for small tally differences (e.g., 1, 2, or 3), decision-makers might fall back to a validity-sensitive strategy (TTB or WADD). We present a series of trials where Option A has a lower tally but possesses the single highest-validity cue, while Option B has a higher tally built from lower-validity cues. By systematically varying the tally difference (1, 2, 3) against the validities, the advocated theory rigidly predicts the higher-tally option will always be chosen, regardless of the difference magnitude. The competing theory predicts a reversal in choice (favoring the high-validity, low-tally option) whenever the tally difference falls below the subject's threshold.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strict Tallying with Validity Tie-Breaker: Decision-makers primarily rely on the unweighted tally of positive features to evaluate options. They always compute this simple tally first. If one option has a strictly higher tally, they choose it (subject to decision noise), ignoring cue validities entirely. Only when the tallies are exactly equal do they incorporate cue validities as a tie-breaker, utilizing either a lexicographic (Take-The-Best) or Weighted Additive strategy to resolve the decision.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_tie: [0.1, 20.0]
- use_wadd: {0, 1}
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_tie = float(parameters["beta_tie"])
    use_wadd = int(parameters["use_wadd"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if tally_a != tally_b:
        scores = np.array([tally_a, tally_b])
        active_beta = beta_tally
    else:
        active_beta = beta_tie
        if use_wadd == 1:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            scores = np.array([wadd_a, wadd_b])
        else:
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = None
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
                
    z = active_beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Dual-Process Strategy Selection with TTB/WADD Fallback: Decision-makers evaluate the raw tally difference between two options first. If the tally difference is highly discriminative (greater than or equal to a threshold), they rely on the fast, compensatory Tallying heuristic. If the tallies are tied or very close, they switch to a more effortful strategy (WADD or Take-The-Best) and use an independent temperature parameter to scale the distinct evidence magnitudes.

**Parameters:**
- threshold: {1, 2, 3, 4, 5}
- use_ttb: {0, 1}
- beta: [0.1, 20.0]
- beta_fallback: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    use_ttb = int(parameters["use_ttb"])
    beta = float(parameters["beta"])
    beta_fallback = float(parameters["beta_fallback"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if abs(tally_a - tally_b) >= threshold:
        scores = np.array([tally_a, tally_b])
        active_beta = beta
    else:
        active_beta = beta_fallback
        if use_ttb == 1:
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = None
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
        else:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            scores = np.array([wadd_a, wadd_b])
        
    z = active_beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    tally_a = data['option_a_ratings'].apply(sum)\n    tally_b = data['option_b_ratings'].apply(sum)\n    \n    # Focus purely on trials where the tally difference is exactly 1\n    mask = abs(tally_a - tally_b) == 1\n    if not mask.any():\n        return 0.5\n        \n    df = data[mask]\n    tally_a_filtered = tally_a[mask]\n    tally_b_filtered = tally_b[mask]\n    \n    # Check if the subject chose the option with the higher tally\n    chose_a = df['response'] == 0\n    higher_tally_a = tally_a_filtered > tally_b_filtered\n    \n    chose_higher = (chose_a == higher_tally_a)\n    return float(chose_higher.mean())",
  "rationale": "This metric calculates the proportion of times a subject chooses the higher-tally option specifically on trials where the tally difference is exactly 1. The Advocated theory (Strict Tallying) rigidly predicts the subject will choose the higher tally option whenever there is any difference, so this metric should be very high (near 1.0, minus noise). The Competing theory (Dual-Process Strategy Selection) posits a threshold mechanism (from 1 to 5). On trials with a tally difference of 1, 80% of the time a subject's threshold will be strictly greater than 1, causing them to fall back to a validity-based strategy (like TTB). Because the trials are designed so the lower-tally option holds the highest validity cue, TTB users will choose the LOWER tally option. Thus, the Competing theory predicts a drastically lower score on this metric."
}
```

## Usage

```json
{
  "prompt_token_count": 3671,
  "candidates_token_count": 435,
  "total_token_count": 7584
}
```
