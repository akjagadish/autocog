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
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design dissociates standard Take-The-Best (TTB) from Stochastic Take-The-Best (Search Error). Standard TTB deterministically stops at the first discriminating feature, meaning it will always choose the option favored by the highest-validity discriminating feature (subject only to a uniform lapse rate). Stochastic TTB, however, assumes a probability 'p_skip' of missing a discriminating feature and continuing the search. By pitting a single high-validity feature favoring Option A against multiple lower-validity features favoring Option B, Stochastic TTB predicts a systematic, graded probability of choosing Option B (when the top feature is skipped), whereas standard TTB predicts Option B is only chosen via random lapses. By varying how many top features favor Option A before the lower features favor Option B, we can observe the exponential decay of errors predicted by the Stochastic TTB model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People use a non-compensatory heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of subjective validity, stopping at the first feature that discriminates between the two options, and choose the option with the higher value on that feature. If no features discriminate, they guess randomly.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = 0.0
    b_wins = 0.0
    
    # Search for the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            a_wins = 1.0
            break
        elif b[idx] > a[idx]:
            b_wins = 1.0
            break
            
    # If no feature discriminates, baseline preference is uniform
    if a_wins == 0.0 and b_wins == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([a_wins, b_wins])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Apply lapse rate
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
**Description:** Stochastic Take-The-Best (Search Error)

**Parameters:**
- p_skip: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Search through features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_skip = float(parameters["p_skip"])
    epsilon = float(parameters["epsilon"])
    
    p_A_core = 0.0
    p_B_core = 0.0
    prob_reach = 1.0
    
    # For each feature, there is a chance (1 - p_skip) to evaluate it correctly.
    # If it discriminates, we stop. Otherwise, or if skipped, we continue to the next.
    for idx in order:
        if a[idx] > b[idx]:
            p_A_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
        elif b[idx] > a[idx]:
            p_B_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
            
    # If all features are skipped or none discriminate, guess randomly
    p_A_core += prob_reach * 0.5
    p_B_core += prob_reach * 0.5
    
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply general response lapse
    n_opts = p_core.shape[0]
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Option B is coded as 1, Option A as 0.\n    # In all trials of this design, the first discriminating feature favors Option A.\n    # Standard TTB predicts A is chosen almost deterministically (subject only to random lapses).\n    # Stochastic TTB predicts a higher rate of choosing B because top features can be skipped,\n    # allowing subsequent features (which mostly favor B) to determine the choice.\n    # Thus, the overall proportion of B choices strongly discriminates the two theories.\n    return float(data['response'].mean())",
  "rationale": "In the chosen experimental design, the first discriminating feature always favors Option A. Standard TTB therefore predicts that Option A will be chosen on every trial, with deviations occurring only due to random lapses (epsilon). Stochastic TTB, however, assumes that search can probabilistically skip the first discriminating feature. Since the remaining features in the design overwhelmingly favor Option B, Stochastic TTB predicts a systematically higher rate of choosing Option B. The overall proportion of Option B choices (where response == 1) directly measures this deviation, providing a clean, low-variance metric that sharply discriminates between the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3266,
  "candidates_token_count": 270,
  "total_token_count": 4889
}
```
