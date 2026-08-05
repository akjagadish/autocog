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
**Validities (n_features=4):** [0.95, 0.72, 0.86, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 1]  B=[0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 9: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 13: A=[1, 1, 0, 1]  B=[0, 1, 1, 0]
  trial 14: A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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
**Description:** Hybrid TTB-WADD Theory: Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model computes the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blends them using a subject-level mixing parameter.

**Parameters:**
- tau: [0.01, 5.0]
- epsilon: [0.0, 0.05]
- mix_rate: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- TTB Probability ---
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    ttb_p = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_p = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_p = np.array([0.0, 1.0])
            break
            
    # --- WADD Probability ---
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    wadd_scores = np.array([score_a, score_b])
    
    tau = float(parameters["tau"])
    z = (wadd_scores - np.max(wadd_scores)) / tau
    e = np.exp(z)
    wadd_p = e / np.sum(e)
    
    # --- Blend ---
    mix_rate = float(parameters["mix_rate"])
    p_core = mix_rate * ttb_p + (1.0 - mix_rate) * wadd_p
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    def get_acc(a_target, b_target, favored_if_a):\n        mask1 = (a_tuples == a_target) & (b_tuples == b_target)\n        mask2 = (a_tuples == b_target) & (b_tuples == a_target)\n        acc = []\n        if mask1.any():\n            acc.extend((data.loc[mask1, 'response'] == (0 if favored_if_a else 1)).tolist())\n        if mask2.any():\n            acc.extend((data.loc[mask2, 'response'] == (1 if favored_if_a else 0)).tolist())\n        return acc\n\n    acc_small = []\n    acc_small.extend(get_acc((1, 0, 0, 0), (0, 1, 0, 0), True))  # T11\n    acc_small.extend(get_acc((0, 0, 1, 0), (1, 0, 0, 1), False)) # T10\n    acc_small.extend(get_acc((1, 1, 0, 1), (0, 1, 1, 0), True))  # T13\n    \n    acc_large = []\n    acc_large.extend(get_acc((1, 1, 0, 1), (0, 0, 0, 0), True))  # T5\n    acc_large.extend(get_acc((1, 0, 1, 1), (0, 0, 0, 0), True))  # T16\n    acc_large.extend(get_acc((1, 1, 1, 0), (0, 1, 0, 1), True))  # T8\n    \n    if not acc_small or not acc_large:\n        return 0.0\n        \n    return float(np.mean(acc_large) - np.mean(acc_small))",
  "rationale": "Under the Take The Best (TTB) model, the probability of choosing the favored option is determined entirely by the first discriminating cue and is thus constant across all trials where a discriminating cue exists (subject only to a constant lapse rate and softmax temperature on binary scores). Under the Hybrid TTB-WADD model, the choice probability is sensitive to the magnitude of the difference in weighted additive (WADD) scores. By computing the difference in accuracy between trials with a large WADD score difference (T5, T16, T8) and trials with a small WADD score difference (T11, T10, T13), we expect a metric near 0 for pure TTB and significantly greater than 0 for the Hybrid model."
}
```

## Usage

```json
{
  "prompt_token_count": 3353,
  "candidates_token_count": 718,
  "total_token_count": 9822
}
```
