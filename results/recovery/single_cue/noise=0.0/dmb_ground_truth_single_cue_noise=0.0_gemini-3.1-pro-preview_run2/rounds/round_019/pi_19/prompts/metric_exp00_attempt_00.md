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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Primacy-Dominant Anchoring (Advocated Theory) from Validity-Weighted Evidence Accumulation with Normalized Attention Decay (Competing Theory), we exploit their structural differences regarding the first and last cues. The Advocated Theory strictly enforces a massive, unnormalized primacy weight (10-30) that dominates all other cues, and an independent recency weight (0-9) that dominates the middle cues. The Competing Theory uses a single decay parameter that normalizes all weights; if decay > 1, the last cue can dominate the first cue, a pattern strictly forbidden by the Advocated Theory. If decay < 1, the Competing Theory mimics primacy but enforces a strict exponential drop-off, meaning the second cue will dominate the last cue if the first is tied. We include trials pitting the first cue against all others, the last cue against all others, and trials where the first cue is tied to expose how the models resolve secondary preferences (independent recency vs. continuous exponential decay).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Primacy-Dominant Anchoring: Decision-makers use the first cue as a powerful anchor that overwhelmingly dominates the evaluation process. While the final cue may receive a secondary recency boost due to short-term memory, the primacy weight is structurally much larger than both the recency weight and the middle cue validities. This explicitly enforces a hierarchy where primacy is the primary driver of choice.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- primacy_weight: [10.0, 30.0]
- recency_weight: [0.0, 9.0]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Enforce Primacy-Dominant Anchoring hierarchy
    w[0] = primacy_weight
    if len(w) > 1:
        w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Validity-Weighted Evidence Accumulation with Normalized Attention Decay (Simplified)

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.1]
- decay: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    decay = float(parameters["decay"])
    
    # Apply exponential attention decay based on cue position (0-indexed)
    positions = np.arange(len(val))
    attention_weights = decay ** positions
    
    # Scale explicitly stated validities and apply attention decay directly
    w = val * attention_weights
    
    # Normalize weights to prevent exponential blowup from dominating the softmax temperature
    sum_w = np.sum(w)
    if sum_w > 0:
        w = w / sum_w
    else:
        w = np.ones_like(w) / len(w)
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Extract the first cue for options A and B\n    cue1_a = data['option_a_ratings'].apply(lambda x: x[0])\n    cue1_b = data['option_b_ratings'].apply(lambda x: x[0])\n    \n    # Filter to trials where the first cue differs between the two options\n    mask = cue1_a != cue1_b\n    if mask.sum() == 0:\n        return 0.5\n        \n    df_diff = data[mask]\n    cue1_a_diff = cue1_a[mask]\n    cue1_b_diff = cue1_b[mask]\n    \n    # Check if the subject's choice aligns with the option favored by the first cue\n    # response == 0 means chose A, response == 1 means chose B\n    chose_a = df_diff['response'] == 0\n    chose_b = df_diff['response'] == 1\n    \n    aligned = ((cue1_a_diff > cue1_b_diff) & chose_a) | ((cue1_b_diff > cue1_a_diff) & chose_b)\n    \n    return float(aligned.mean())",
  "rationale": "The Advocated Theory (Primacy-Dominant Anchoring) enforces a massive structural weight (10-30) on the first cue. Therefore, whenever the first cue differs between options A and B, the model will overwhelmingly choose the option favored by the first cue, leading to an alignment score near 1.0. In contrast, the Competing Theory uses normalized attention decay. Because the decay parameter can be greater than 1 (making later cues dominate) or less than 1 (allowing multiple subsequent cues to collectively outweigh the first cue), its choices will frequently contradict the first cue, resulting in a substantially lower average alignment score."
}
```

## Usage

```json
{
  "prompt_token_count": 3488,
  "candidates_token_count": 455,
  "total_token_count": 6334
}
```
