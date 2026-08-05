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
**Validities (n_features=4):** [0.95, 0.74, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 2: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 16: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture Theory: The population of decision-makers is heterogeneous, consisting of distinct subgroups that rely on fundamentally different decision rules. Rather than every individual using a noisy version of the same heuristic, about half the population employs a non-compensatory, frugal strategy (Take The Best), while the other half employs a compensatory, unweighted strategy (Tallying). This individual difference accounts for both the ~50% aggregate choice proportions on critical divergence trials and the high (~0.25) between-subject variance, which is characteristic of a Bernoulli distribution of highly deterministic strategies.

**Parameters:**
- strategy: {0, 1}
- beta: [5.0, 50.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    strategy = int(parameters["strategy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if strategy == 0:
        # Take The Best (Non-compensatory)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner is None:
            scores = np.array([0.5, 0.5])
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
            
    else:
        # Tallying (Compensatory)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Calculate the sum of positive cues for each option to find Tally-tie trials\n    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))\n    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))\n    tally_tie_mask = (a_sums == b_sums)\n    \n    if not tally_tie_mask.any():\n        return 0.5\n        \n    tie_data = data[tally_tie_mask].copy()\n    \n    # Determine the objective Take-The-Best (TTB) choice\n    # Objective validities are [0.95, 0.74, 0.81, 0.55], so cue order is 0, 2, 1, 3\n    def get_ttb_choice(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        for cue in [0, 2, 1, 3]:\n            if a[cue] > b[cue]:\n                return 0\n            elif b[cue] > a[cue]:\n                return 1\n        return -1\n        \n    tie_data['ttb_choice'] = tie_data.apply(get_ttb_choice, axis=1)\n    \n    valid_mask = tie_data['ttb_choice'] != -1\n    if not valid_mask.any():\n        return 0.5\n        \n    valid_tie_data = tie_data[valid_mask]\n    matches = (valid_tie_data['response'] == valid_tie_data['ttb_choice']).astype(float)\n    \n    return float(matches.mean())",
  "rationale": "We isolate trials where the Tallying strategy predicts a tie (the sum of positive cues is equal for options A and B). On these trials, Tallying users (half the population in SMT) will guess randomly, yielding a ~50% match rate with the objective Take-The-Best (TTB) prediction. TTB users (the other half in SMT) will match the objective TTB prediction ~100% of the time. Thus, SMT predicts an aggregate match rate of ~75% on these trials. In contrast, HCH posits that all users employ TTB but with idiosyncratic, noisy cue hierarchies. Because the Tally-tie trials in this design perfectly balance the number of cues favoring each option, a noisy cue hierarchy will favor the objective TTB choice only ~50% of the time. Thus, HCH predicts an aggregate match rate of ~50%. This metric computes the proportion of choices on Tally-tie trials that align with the objective TTB prediction, creating a robust 75% vs 50% contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3546,
  "candidates_token_count": 690,
  "total_token_count": 17754
}
```
