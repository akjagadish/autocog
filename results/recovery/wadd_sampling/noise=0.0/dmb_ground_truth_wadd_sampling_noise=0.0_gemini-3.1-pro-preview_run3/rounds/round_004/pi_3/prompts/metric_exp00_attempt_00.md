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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design dissociates the pure Weighted Additive (WADD) theory from the Contingent Heuristic theory by systematically varying the validity gap between the best and second-best discriminating cues. WADD evaluates options via a fully compensatory sum across all trials. In contrast, the Contingent Heuristic falls back to WADD only when the validity gap is smaller than a subjective threshold (delta); when the gap exceeds this threshold, it abruptly switches to a non-compensatory Take-The-Best (TTB) strategy. By including critical trials with large validity gaps where the top cue favors option A but the sum of remaining cues strongly favors option B, we can observe whether choice probabilities reflect a continuous compensatory evaluation (WADD) or a discrete threshold-based switch to a single-cue strategy (Contingent).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory strategy. They multiply each feature's value by its corresponding cue validity and sum these products to form an overall subjective value for each option. The option with the higher weighted sum is chosen. This allows multiple lower-validity cues to collectively outweigh a single high-validity cue, capturing behavior that falls between pure Take The Best and pure Tallying. To account for empirical response noise, the decision process incorporates a moderate degree of stochasticity.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.1, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Contingent Heuristic Theory (Difference-based Compensatory): Decision-makers conditionally apply decision strategies based on the relative strength of the best available evidence compared to the next best piece of evidence. They identify the most valid cue that discriminates between the two options and the second most valid discriminating cue. If the difference in validity between these two cues exceeds a subjective threshold (indicating the top cue is highly salient relative to the rest), they rely entirely on the top cue (Take-The-Best). If the difference is not sufficiently large, they fall back to a fully compensatory Weighted Additive (WADD) strategy.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- delta: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    cue_order = np.argsort(-val, kind='stable')
    
    ttb_winner = None
    v_max = 0.0
    v_next = 0.0
    found_first = False
    
    for cue in cue_order:
        if a[cue] != b[cue]:
            if not found_first:
                v_max = val[cue]
                ttb_winner = 0 if a[cue] > b[cue] else 1
                found_first = True
            else:
                v_next = val[cue]
                break
                
    delta = float(parameters["delta"])
    
    if ttb_winner is not None and (v_max - v_next) >= delta:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
    else:
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        scores = np.array([score_a, score_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_vals = data['option_a_ratings'].apply(tuple)\n    b_vals = data['option_b_ratings'].apply(tuple)\n    \n    # Target trial: A=[1, 0, 0, 0, 0], B=[0, 0, 1, 1, 1]\n    t_a = (1, 0, 0, 0, 0)\n    t_b = (0, 0, 1, 1, 1)\n    \n    mask = (a_vals == t_a) & (b_vals == t_b)\n    \n    if mask.sum() == 0:\n        return 0.5\n        \n    # Return the proportion of times option A was chosen (response == 0)\n    return float((data.loc[mask, 'response'] == 0).mean())",
  "rationale": "This metric isolates a specific trial where Option A is supported only by the most valid cue (0.95), while Option B is supported by three lower-validity cues (0.65, 0.55, 0.50). Under the pure Weighted Additive (WADD) theory, the compensatory sum strongly favors Option B (1.70 vs 0.95), meaning WADD will reliably choose B. Under the Contingent Heuristic theory, the validity gap between the best discriminating cue (0.95) and the next best (0.65) is 0.3. Since this gap is often larger than the subjective threshold delta (drawn from [0.0, 0.5]), the Contingent model frequently switches to Take-The-Best and chooses Option A. Thus, the proportion of Option A choices on this trial will be near 0 for WADD, but substantially higher for the Contingent Heuristic."
}
```

## Usage

```json
{
  "prompt_token_count": 3306,
  "candidates_token_count": 438,
  "total_token_count": 6031
}
```
