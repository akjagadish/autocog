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
**Validities (n_features=5):** [0.6, 0.8, 0.7, 0.9, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying with Primacy Tie-Breaking (Advocated Theory) from Primacy-Dominant Anchoring (Competing Theory), we exploit their fundamental difference in priority regarding the first cue. The Advocated Theory uses the first cue strictly as a tie-breaker when the total count of winning features is tied; otherwise, the simple tally dictates the choice. In contrast, the Competing Theory assigns a massive, dominant weight to the first cue, meaning it will dictate the choice even when opposed by a strong tally advantage. We include 'conflict' trials where one option has a clear tally advantage but loses the first cue, leading to opposing predictions. We also include 'tally tie' trials where the first cue is tied, forcing the Advocated Theory to predict a random guess while the Competing Theory predicts a choice based on the remaining middle or recency cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying with Primacy Tie-Breaking: Decision-makers primarily evaluate options by counting the total number of positive features (Tallying), ignoring the stated validities and sequence order. If one option has a strictly greater number of positive features, it is chosen. If the tally scores are tied, decision-makers fall back to a secondary non-compensatory mechanism: using the first cue (primacy) to break the tie.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_primacy: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    beta_tally = float(parameters["beta_tally"])
    beta_primacy = float(parameters["beta_primacy"])
    epsilon = float(parameters["epsilon"])
    
    # Unweighted sum of positive features
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    # If tally scores are different, choice is driven by tally difference
    if tally_a != tally_b:
        scores = np.array([beta_tally * tally_a, beta_tally * tally_b])
    # If tally scores are tied, choice is driven by primacy (first cue)
    else:
        scores = np.array([beta_primacy * a[0], beta_primacy * b[0]])
        
    # Softmax choice rule with max-subtraction for numerical stability
    z = scores - np.max(scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    conflict_trials = []\n    for idx, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        resp = row['response']\n        \n        tally_a = sum(a)\n        tally_b = sum(b)\n        \n        if tally_a == tally_b:\n            continue\n            \n        tally_winner = 0 if tally_a > tally_b else 1\n        \n        primacy_a = a[0]\n        primacy_b = b[0]\n        \n        if primacy_a == primacy_b:\n            continue\n            \n        primacy_winner = 0 if primacy_a > primacy_b else 1\n        \n        if tally_winner != primacy_winner:\n            conflict_trials.append(1 if resp == tally_winner else 0)\n            \n    if len(conflict_trials) == 0:\n        return 0.5\n        \n    return float(np.mean(conflict_trials))",
  "rationale": "This metric calculates the proportion of times a subject chooses the option with the higher tally (total number of positive features) specifically on 'conflict trials'. Conflict trials are defined as those where one option has a strictly higher tally but loses on the first cue (primacy). The Advocated Theory (Tallying with Primacy Tie-Breaking) predicts that subjects will predominantly choose the tally winner on these trials, yielding a value close to 1.0. In contrast, the Competing Theory (Primacy-Dominant Anchoring) predicts that subjects will overwhelmingly choose the primacy winner, yielding a value close to 0.0."
}
```

## Usage

```json
{
  "prompt_token_count": 3348,
  "candidates_token_count": 428,
  "total_token_count": 5398
}
```
