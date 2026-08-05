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
**Validities (n_features=4):** [0.95, 0.72, 0.62, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1]  B=[0, 1, 0, 1]
  trial 3: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 10: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 13: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 14: A=[0, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 16: A=[1, 1, 1, 1]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take-The-Best (TTB) assumes that decision-makers do not integrate all information or simply count features. Instead, they rank features by their subjective or instructed validity and compare options lexicographically. They stop at the first feature that discriminates between the two options and choose the one with the higher value on that feature. If all features tie, they guess.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-validities, kind='mergesort')
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    if score_a > score_b:
        p_core = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Adaptive Toolbox with Confidence-Dependent Strategy Selection: Decision-makers predominantly use the Take-The-Best (TTB) heuristic, but probabilistically fall back to Tallying only when their confidence is low—specifically, when the validity of the first discriminating feature falls below a subjective threshold.

**Parameters:**
- epsilon: [0.0, 0.5]
- p_tally: [0.0, 0.5]
- threshold: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    epsilon = float(parameters["epsilon"])
    p_tally = float(parameters["p_tally"])
    threshold = float(parameters["threshold"])
    
    # Take-The-Best (TTB) Mechanism
    order = np.argsort(-validities, kind='mergesort')
    p_ttb = np.array([0.5, 0.5])
    discrim_validity = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_validity = validities[idx]
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_validity = validities[idx]
            break
            
    # Tallying Mechanism (Equal Weights)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Confidence-dependent fallback to Tallying
    if discrim_validity < threshold:
        actual_p_tally = p_tally
    else:
        actual_p_tally = 0.0
        
    # Probabilistic Strategy Selection
    p_mix = (1.0 - actual_p_tally) * p_ttb + actual_p_tally * p_tally_arr
    
    # Incorporate baseline lapse rate (random guessing)
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Convert option ratings to string for easy matching\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trial 13: A=[0, 1, 0, 1], B=[0, 0, 1, 0]\n    # Both TTB and Tallying predict A.\n    t13_mask = (data['A_str'] == '0101') & (data['B_str'] == '0010')\n    \n    # Trial 12: A=[0, 1, 0, 1], B=[0, 0, 1, 1]\n    # TTB predicts A, but Tallying predicts a Tie.\n    t12_mask = (data['A_str'] == '0101') & (data['B_str'] == '0011')\n    \n    # response == 0 means choice A\n    p_a_t13 = (data[t13_mask]['response'] == 0).mean()\n    p_a_t12 = (data[t12_mask]['response'] == 0).mean()\n    \n    if pd.isna(p_a_t13) or pd.isna(p_a_t12):\n        return 0.0\n        \n    return float(p_a_t13 - p_a_t12)",
  "rationale": "By comparing the proportion of 'A' choices on Trial 13 vs Trial 12, we can effectively discriminate the two theories. Both trials share the same discriminating feature (Feature 2, validity 0.72) under TTB, which favors option A. Thus, pure TTB predicts an identical high rate of 'A' choices for both trials, yielding a difference near 0. However, the Competing Theory probabilistically falls back to Tallying on these trials. For Trial 13, Tallying also favors A, so the choice rate remains high. For Trial 12, Tallying results in a tie, which significantly lowers the probability of choosing A. Therefore, the Competing Theory predicts a substantially positive difference between the two trials."
}
```

## Usage

```json
{
  "prompt_token_count": 3464,
  "candidates_token_count": 583,
  "total_token_count": 9333
}
```
