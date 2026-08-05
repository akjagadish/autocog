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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Feature-Averaging model (Advocated) from the Lateral Inhibition Evidence Accumulator (Competing) by exploiting a strict structural invariance in the Advocated model: cue-wise cancellation. In the Advocated model, features that are active for both options (shared cues) are perfectly cancelled out before any evidence accumulation or soft normalization occurs. Therefore, adding shared cues to both options mathematically guarantees identical choice probabilities. In stark contrast, the Competing model evaluates cues holistically with divisive normalization and lacks cue-wise cancellation. Shared cues contribute to the total active weight for both options, increasing both self-dilution and conflict suppression for the unique cues. By systematically taking base trials and adding shared lower-validity or higher-validity cues, the Advocated model is mathematically forced to predict constant choice probabilities across the sets, while the Competing model predicts systematic shifts as the primary cues are laterally inhibited by the shared 'background' cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Feature-Averaging with Cue-wise Cancellation and Soft Normalization

**Parameters:**
- gamma: [0.1, 10.0]
- alpha: [0.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities to reflect subjective evidence weight, keeping the 0.5 baseline
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Cue-wise cancellation: only consider unique features
    diff = a - b
    unique_a = (diff > 0)
    unique_b = (diff < 0)
    
    # Feature-averaging: soft normalization interpolating between sum (alpha=0) and mean (alpha=1)
    n_a = np.sum(unique_a)
    sum_w_a = np.sum(w[unique_a])
    ev_a = sum_w_a / (1.0 + alpha * (n_a - 1.0)) if n_a > 0 else 0.0
    
    n_b = np.sum(unique_b)
    sum_w_b = np.sum(w[unique_b])
    ev_b = sum_w_b / (1.0 + alpha * (n_b - 1.0)) if n_b > 0 else 0.0
    
    # Convert evidence to probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

**Parameters:**
- alpha: [0.0, 5000.0]
- beta: [0.0, 100.0]
- gamma: [0.1, 5.0]
- theta: [0.1, 500.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    # Create string representation for A and B options to identify trials\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # Map rows to trial types\n    def get_trial(row):\n        a, b = row['A_str'], row['B_str']\n        if a == '10000' and b == '01100': return 1\n        if a == '10010' and b == '01110': return 2\n        if a == '10011' and b == '01111': return 3\n        if a == '01000' and b == '00110': return 4\n        if a == '11000' and b == '10110': return 5\n        if a == '11001' and b == '10111': return 6\n        if a == '00100' and b == '00011': return 7\n        if a == '01100' and b == '01011': return 8\n        return 0\n\n    data['trial'] = data.apply(get_trial, axis=1)\n    \n    # Calculate P(choose A) for each trial. Response 0 means A was chosen.\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    # Group by subject and trial\n    subj_trial = data[data['trial'] > 0].groupby(['subject_id', 'trial'])['chose_A'].mean().unstack(fill_value=np.nan)\n    \n    # For each subject, compute the absolute difference in choice probabilities \n    # between the base trial and the trial with the most added shared features.\n    diffs = []\n    for subj in subj_trial.index:\n        p = subj_trial.loc[subj]\n        d1 = abs(p.get(1, np.nan) - p.get(3, np.nan))\n        d2 = abs(p.get(4, np.nan) - p.get(6, np.nan))\n        d3 = abs(p.get(7, np.nan) - p.get(8, np.nan))\n        \n        valid_d = [d for d in [d1, d2, d3] if pd.notna(d)]\n        if valid_d:\n            diffs.append(np.mean(valid_d))\n            \n    if not diffs:\n        return 0.0\n        \n    return float(np.mean(diffs))",
  "rationale": "This metric evaluates the structural invariance of cue-wise cancellation. According to the Advocated model, shared features are perfectly cancelled before any integration, meaning that adding shared features to both options mathematically guarantees identical choice probabilities. Thus, the absolute difference in choice probability between base trials (e.g., Trial 1) and trials with added shared features (e.g., Trial 3) should be zero, subject only to binomial sampling noise. In contrast, the Competing lateral inhibition model predicts that shared features are evaluated holistically and cause divisive normalization (self-dilution and conflict suppression), leading to systematic shifts in preference when shared features are added. The metric computes the mean absolute difference in P(Choose A) within these trial sets per subject, expecting a near-zero value for the Advocated theory and a significantly positive value for the Competing theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3563,
  "candidates_token_count": 928,
  "total_token_count": 7164
}
```
