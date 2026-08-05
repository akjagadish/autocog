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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate pure Take The Best (TTB) from TTB with a Downstream Tallying Mixture, we hold the TTB prediction constant while systematically varying the downstream tally (the equal-weight sum of all unexamined cues lower in validity than the discriminating cue). Pure TTB ignores all cues after the first discriminating one, predicting a constant choice probability across trials where the first discriminating cue is the same. In contrast, the Downstream Tallying Mixture model evaluates the remaining cues. By shifting the downstream tally from heavily favoring the unchosen option to heavily favoring the TTB-chosen option, the mixture model predicts a monotonically increasing choice probability, whereas pure TTB predicts a flat line.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB) heuristic posits a lexicographic decision rule where individuals search through features in descending order of validity. They stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. If no feature discriminates, they guess randomly. To account for empirical levels of noise, the choice is mixed with a lapse rate (epsilon) that can span up to 1.0 (pure guessing).

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Take-The-Best with Downstream Tallying Mixture: Decision makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in descending order of validity. However, to account for corroboration and tie-breaking effects, individuals occasionally mix this strategy with a 'downstream tally'—an equal-weight count of only the remaining, unexamined cues that have lower validity than the discriminating cue. This prevents the discriminating cue from double-contributing to the tally, reducing over-prediction of compensatory WADD-like behavior in environments with highly valid cues, while preserving compensatory variance in environments where downstream cues strongly oppose the best cue.

**Parameters:**
- epsilon: [0.0, 0.5]
- w_tally: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    # TTB Prediction
    p_ttb = np.array([0.5, 0.5])
    discrim_idx = len(order)
    for i, idx in enumerate(order):
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_idx = i
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_idx = i
            break
            
    # Downstream Tallying Prediction (only unexamined cues)
    if discrim_idx < len(order) - 1:
        remaining_indices = order[discrim_idx + 1:]
        tally_a = np.sum(a[remaining_indices])
        tally_b = np.sum(b[remaining_indices])
    else:
        tally_a = 0
        tally_b = 0
        
    if tally_a > tally_b:
        p_tally = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Mix TTB and Downstream Tallying
    w_tally = float(parameters["w_tally"])
    p_mix = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_mat = np.stack(data['option_a_ratings'].values)\n    b_mat = np.stack(data['option_b_ratings'].values)\n    resp = data['response'].values\n    \n    # Find the first discriminating cue (TTB stopping point)\n    diff = a_mat != b_mat\n    discrim_idx = np.argmax(diff, axis=1)\n    \n    # TTB choice: 0 if A is better on the discriminating cue, else 1\n    ttb_choice = (a_mat[np.arange(len(a_mat)), discrim_idx] < b_mat[np.arange(len(b_mat)), discrim_idx]).astype(int)\n    \n    # Downstream Tally choice: sum only cues after the discriminating one\n    cols = np.arange(a_mat.shape[1])\n    mask = cols > discrim_idx[:, None]\n    \n    tally_a = np.sum(a_mat * mask, axis=1)\n    tally_b = np.sum(b_mat * mask, axis=1)\n    \n    tally_choice = np.zeros(len(a_mat), dtype=int) - 1\n    tally_choice[tally_a > tally_b] = 0\n    tally_choice[tally_b > tally_a] = 1\n    \n    # Identify trials where Downstream Tally agrees or disagrees with TTB\n    agree_mask = (tally_choice == ttb_choice) & (tally_choice != -1)\n    disagree_mask = (tally_choice != ttb_choice) & (tally_choice != -1)\n    \n    match = (resp == ttb_choice)\n    \n    if np.sum(agree_mask) == 0 or np.sum(disagree_mask) == 0:\n        return 0.0\n        \n    # Return the difference in TTB-adherence between agree and disagree trials\n    return float(np.mean(match[agree_mask]) - np.mean(match[disagree_mask]))",
  "rationale": "By comparing the subject's adherence to the Take-The-Best (TTB) prediction on trials where the downstream tally agrees with TTB versus trials where it disagrees, we can isolate the effect of the downstream tally. Under the pure TTB model, adherence should be equal across both types of trials (difference near zero). Under the Downstream Tallying Mixture model, adherence will be significantly higher when the tally agrees and significantly lower when it disagrees, yielding a positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 3294,
  "candidates_token_count": 645,
  "total_token_count": 7578
}
```
