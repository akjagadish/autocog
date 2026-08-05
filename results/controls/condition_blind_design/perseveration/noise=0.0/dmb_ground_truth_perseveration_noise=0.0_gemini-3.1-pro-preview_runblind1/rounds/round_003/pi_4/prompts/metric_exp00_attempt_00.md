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
**Validities (n_features=4):** [0.95, 0.56, 0.78, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 8: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[1, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 14: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 15: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Take The Best: Decision-makers use a lexicographic heuristic (Take The Best), checking features in descending order of validity and stopping at the first feature that discriminates between options. However, human application of this rule is highly noisy. A heavily restricted inverse temperature parameter accounts for the empirical observation that subjects' choices on these trials hover near chance level (0.50-0.55), rather than the highly deterministic choices (0.85+) predicted by standard TTB.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity descending; stable sort preserves original order on ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Subjects use a Tallying (Equal Weights) heuristic, where they ignore the explicit cue validities and simply count the total number of positive features for each option. They choose the option with the highest total number of positive features, and guess randomly if there is a tie. A highly restricted softmax temperature and lapse rate account for the overall noisy, near-chance behavior observed in the experiments.

**Parameters:**
- beta: [0.0, 0.3]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Count total positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # The validities are [0.95, 0.56, 0.78, 0.55]\n    # The descending order of validities corresponds to indices: 0, 2, 1, 3\n    cue_order = [0, 2, 1, 3]\n    \n    ttb_match_count = 0\n    disagreement_trials_count = 0\n    \n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        \n        # Determine TTB winner\n        ttb_winner = None\n        for j in cue_order:\n            if a[j] > b[j]:\n                ttb_winner = 0\n                break\n            elif b[j] > a[j]:\n                ttb_winner = 1\n                break\n                \n        # Determine Tally winner\n        score_a = sum(a)\n        score_b = sum(b)\n        if score_a > score_b:\n            tally_winner = 0\n        elif score_b > score_a:\n            tally_winner = 1\n        else:\n            tally_winner = None\n            \n        # Focus only on trials where TTB and Tally do not perfectly agree\n        # (e.g. Tally predicts a tie but TTB has a preference, or they strictly disagree)\n        if ttb_winner is not None and ttb_winner != tally_winner:\n            if row['response'] == ttb_winner:\n                ttb_match_count += 1\n            disagreement_trials_count += 1\n            \n    if disagreement_trials_count == 0:\n        return 0.5\n        \n    return float(ttb_match_count) / disagreement_trials_count",
  "rationale": "This metric calculates the proportion of choices that align with the Noisy Take The Best (TTB) heuristic exclusively on trials where TTB and Tallying produce conflicting predictions (either a strict disagreement or where Tallying predicts a tie). On these targeted trials, TTB-simulated data will yield an alignment rate above 0.5, while Tallying-simulated data will yield a rate below 0.5 (due to strict disagreements pulling the average down from chance level). Filtering for these specific trials maximizes the discriminative power between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3085,
  "candidates_token_count": 611,
  "total_token_count": 7357
}
```
