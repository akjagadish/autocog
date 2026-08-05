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
**Validities (n_features=4):** [0.95, 0.87, 0.82, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 10: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 13: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Heterogeneous Cue Hierarchies: Individuals all employ a lexicographic, non-compensatory decision strategy (Take-The-Best), but they differ in how they construct their cue hierarchies. Rather than perfectly internalizing the objective cue validities provided in instructions, subjects form subjective cue validities by combining the objective validities with idiosyncratic, subjective weights. The degree to which a subject relies on their idiosyncratic weights versus objective validities varies across the population. This results in some subjects strictly following the objective hierarchy, while others use idiosyncratic hierarchies (e.g., heavily overweighting a specific cue). This heterogeneity explains the high between-subject variance and why objectively dominant options in aggregate only receive partial choice shares.

**Parameters:**
- noise_level: [0.0, 10.0]
- subjective_weights: [(0.0, 1.0)] * n_features
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    subjective_weights = np.asarray(parameters["subjective_weights"], dtype=float)
    noise_level = float(parameters["noise_level"])
    epsilon = float(parameters["epsilon"])
    
    # Construct subjective validities by adding individual-specific noise/preferences to objective validities
    # subjective_weights are in [0, 1], so we center them around 0
    subj_val = validities + noise_level * (subjective_weights - 0.5)
    
    # Lexicographic strategy based on subjective hierarchy
    cue_order = np.argsort(-subj_val, kind="stable").tolist()
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
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
**Description:** Subject-Specific Single-Cue Heuristic and Strong Spatial Bias: Subjects largely ignore the objective validities provided in the instructions. Instead of engaging in complex compensatory or non-compensatory integration of multiple cues, individuals adopt highly simplified, deterministic strategies. The population is split: some subjects rely on a strong spatial bias (e.g., always choosing the left or right option), while others adopt a single-cue heuristic, randomly selecting one feature to follow deterministically and ignoring all others. This extreme simplification explains the ~50% aggregate choice proportions on critical divergence trials, the failure to align with objective-consensus trials, and the extremely high between-subject variance, as choices are driven by idiosyncratic, deterministic biases rather than shared objective validities.

**Parameters:**
- strategy_class: {0, 1}
- spatial_dir: {0, 1}
- cue_weights: [(0.0, 1.0)] * n_features
- epsilon: [0.0, 0.1]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    strategy_class = int(parameters["strategy_class"])
    spatial_dir = int(parameters["spatial_dir"])
    cue_weights = np.asarray(parameters["cue_weights"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    if strategy_class == 0:
        # Spatial Bias: deterministically choose Left (0) or Right (1)
        p_core = np.array([1.0, 0.0]) if spatial_dir == 0 else np.array([0.0, 1.0])
    else:
        # Single-Cue Heuristic: follow a single randomly preferred cue
        k = int(np.argmax(cue_weights))
        if a[k] > b[k]:
            p_core = np.array([1.0, 0.0])
        elif b[k] > a[k]:
            p_core = np.array([0.0, 1.0])
        else:
            # If the chosen cue is tied, guess randomly
            p_core = np.array([0.5, 0.5])
            
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    diff_mask = sum_a != sum_b\n    if not diff_mask.any():\n        return 0.5\n        \n    resp = data.loc[diff_mask, 'response']\n    sum_choice = (sum_b[diff_mask] > sum_a[diff_mask]).astype(int)\n    \n    match = (resp == sum_choice).mean()\n    return float(match)",
  "rationale": "Measures the proportion of choices that align with the simple sum of features (Dawes' rule) on trials where one option has more positive features than the other. Under the Advocated Theory (Take-The-Best), choices will often align with the sum because the hierarchical non-compensatory strategy naturally correlates with the total number of positive cues. Under the Competing Theory, a large portion of subjects use a spatial bias (which matches the sum only 50% of the time) or a single-cue heuristic (which ignores the rest of the cues, reducing the overall alignment with the sum). Thus, the metric should be significantly higher for the Advocated Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3529,
  "candidates_token_count": 323,
  "total_token_count": 6463
}
```
