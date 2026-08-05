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
**Validities (n_features=4):** [0.95, 0.55, 0.65, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying with Recency/Right-Most Tie-Breaker: Decision makers primarily evaluate options by tallying the number of features where one option strictly dominates the other, ignoring feature validities (equal weighting). If one option has more winning features, it is chosen. However, if the feature wins are tied, individuals do not simply guess. Instead, they rely on a 'recency' or 'right-most' bias, breaking the tie based solely on the last feature in the array. Responses are subject to softmax noise over the resulting scores and a uniform lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Tie-breaking mechanism: Recency/Right-most feature bias
    if a_wins == b_wins:
        if a[-1] > b[-1]:
            a_wins += 1.0
        elif b[-1] > a[-1]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
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
**Description:** Recency-Weighted Tallying: Decision makers evaluate options by computing a weighted tally of feature wins. Due to recency effects in working memory, features presented later (further to the right) are more salient and receive exponentially increasing weights. Because the growth rate of these weights is bounded, the heuristic strictly preserves standard tallying when one option has more feature wins than the other. However, when the number of feature wins is tied, the exponentially increasing weights naturally break the tie in favor of the option that wins on the most recent (right-most) features, providing a continuous and elegant mechanism for tie-breaking without invoking discontinuous rules.

**Parameters:**
- recency_base: [1.01, 1.3]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    recency_base = float(parameters['recency_base'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Weights exponentially increasing from left to right to model recency in working memory.
    # For recency_base in [1.01, 1.3], the sum of any k weights is strictly less 
    # than the sum of any k+1 weights, preserving the strict tallying property for unequal wins.
    w = recency_base ** np.arange(n_features)
    
    # Weighted tallying of strict feature-wise wins
    a_wins = float(np.sum(w * (a > b)))
    b_wins = float(np.sum(w * (b > a)))
    
    scores = np.array([a_wins, b_wins])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    match_count = 0\n    total_trials = 0\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        resp = row['response']\n        \n        if a == (0, 1, 1, 1) and b == (1, 1, 0, 1):\n            match_count += (1 if resp == 0 else 0)\n            total_trials += 1\n        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 1):\n            match_count += (1 if resp == 0 else 0)\n            total_trials += 1\n        elif a == (0, 1, 0, 1) and b == (0, 0, 1, 1):\n            match_count += (1 if resp == 1 else 0)\n            total_trials += 1\n            \n    if total_trials == 0:\n        return 0.5\n    return match_count / total_trials",
  "rationale": "Focuses on trials where both the overall number of strict feature wins AND the right-most feature are tied between the two options. The Advocated theory predicts a 50-50 guess on these trials because its tie-breaker fails. The Competing theory resolves the tie using the exponentially weighted sum of the remaining features, systematically predicting a specific option. This metric calculates the proportion of choices that align with the Competing theory's secondary tie-breaking predictions."
}
```

## Usage

```json
{
  "prompt_token_count": 3319,
  "candidates_token_count": 402,
  "total_token_count": 6655
}
```
