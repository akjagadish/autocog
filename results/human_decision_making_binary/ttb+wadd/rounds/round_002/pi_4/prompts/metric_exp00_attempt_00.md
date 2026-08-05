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

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.05). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=8):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.6, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** This design introduces a profound qualitative dissociation by creating 'Double-Agreement Override' trials. In these critical trials, Option A possesses the single highest-validity cue (winning Take The Best) AND a strictly greater total number of positive features (winning Tallying). Because the Probabilistic Heuristic Selection (PHS) model is a mixture of TTB and Tallying, it must overwhelmingly predict Option A on these trials, regardless of the mixing parameter. However, the Non-linear Subjective Weighting (NLSW) model allows for intermediate non-linear scaling of validities (e.g., a gamma exponent between 2 and 5). Under this intermediate non-linear regime, a coalition of high-to-medium validity cues in Option B mathematically outweighs Option A's single highest-validity cue and its numerous low-validity cues. Thus, NLSW uniquely predicts that subjects can prefer Option B despite both TTB and Tallying pointing to Option A, providing a strict test between the models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

**Parameters:**
- gamma: [0.0, 10.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
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


## COMPETING THEORY
**Description:** Probabilistic Heuristic Selection: Subjects do not rigidly apply a single deterministic decision rule across all trials. Instead, they maintain a repertoire of strategies—specifically, a non-compensatory one-reason heuristic (Take The Best) and a compensatory strategy (Tallying/Equal-Weight). On each trial, a subject probabilistically selects which strategy to deploy based on a subject-specific mixing parameter (p_ttb). This trial-by-trial internal blending natively captures intermediate aggregate metric values while preserving low between-subject variance, as each individual exhibits a mixture of both choice patterns.

**Parameters:**
- p_ttb: [0.0, 0.9]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (Non-compensatory)
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner is None:
        p_ttb_arr = np.array([0.5, 0.5])
    else:
        p_ttb_arr = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Strategy 2: Tallying (Compensatory)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_arr + (1.0 - p_ttb) * p_tally_arr
    
    # Late-stage lapse noise
    epsilon = float(parameters["epsilon"])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    overrides = []\n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        \n        ttb_winner = None\n        for i in range(len(a)):\n            if a[i] > b[i]:\n                ttb_winner = 0\n                break\n            elif b[i] > a[i]:\n                ttb_winner = 1\n                break\n                \n        sum_a = sum(a)\n        sum_b = sum(b)\n        tally_winner = 0 if sum_a > sum_b else (1 if sum_b > sum_a else None)\n        \n        if ttb_winner is not None and tally_winner is not None and ttb_winner == tally_winner:\n            chosen = row['response']\n            if chosen != ttb_winner:\n                overrides.append(1)\n            else:\n                overrides.append(0)\n                \n    if not overrides:\n        return 0.0\n    return float(np.mean(overrides))",
  "rationale": "This metric calculates the proportion of 'override' choices on trials where both Take The Best (TTB) and Tallying agree on the same option. The Probabilistic Heuristic Selection (PHS) model, which mixes TTB and Tallying, will overwhelmingly choose the agreed-upon option, resulting in an override rate near zero (driven only by the lapse rate). In contrast, the Non-linear Subjective Weighting (NLSW) model can assign intermediate convex weights to the cues, meaning a coalition of medium-validity cues can mathematically outweigh a single highest-validity cue plus several low-validity cues, leading to a much higher rate of choosing the 'override' option."
}
```

## Usage

```json
{
  "prompt_token_count": 3651,
  "candidates_token_count": 457,
  "total_token_count": 6369
}
```
