# metric_exp00_attempt_01

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
**Validities (n_features=4):** [0.95, 0.8, 0.74, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Compensatory Attention in Tallying: People primarily use a simple Tallying heuristic (equal weighting) to compare options because it is cognitively economical. However, when Tallying results in a tie, subjects experience uncertainty. To resolve this, they direct their remaining cognitive effort toward the cues they are most uncertain about—the least valid cues. They break the tie using a compensatory strategy where cues are weighted inversely to their validity, effectively approximating a Take-The-Worst heuristic through negative weighting.

**Parameters:**
- beta: [0.1, 100.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        scores = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        scores = np.array([0.0, 1.0])
    else:
        # Tie-breaker: Compensatory Attention (Negative Weighting)
        val = np.asarray(parameters["validities"], dtype=float)
        # Weight cues inversely to their validity (attention to uncertainty)
        uncertainty_weights = 1.0 - val
        
        score_a = np.sum(uncertainty_weights * (a > b))
        score_b = np.sum(uncertainty_weights * (b > a))
        
        scores = np.array([score_a, score_b])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
**Description:** Majority Rule with Reverse Lexicographic Tie-Breaking. People primarily use Tallying (equal weighting) to compare options, counting the number of features where one option strictly dominates the other. If one option wins on more features, it is chosen. However, if there is a tie in the tally, subjects break the tie by consulting cues in ASCENDING order of validity (least valid first). This perfectly captures why subjects follow Tallying in general, but systematically oppose both Weighted Additive (WADD) and Take The Best (TTB) predictions specifically on trials where Tallying results in a tie.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        scores = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        scores = np.array([0.0, 1.0])
    else:
        # Tie-breaker: Reverse Lexicographic (least valid cue first)
        val = np.asarray(parameters["validities"], dtype=float)
        # ASCENDING order of validity
        cue_order = np.argsort(val, kind="stable").tolist()
        
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
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
[0] rationale: Both theories predict that subjects will choose Option B on Trial 13 and Trial 16, as both are tie-breaker trials. However, the Competing theory (Reverse Lexicographic) resolves both ties using the least valid cue (cue 3) as the sole decider, generating an identical, maximal score difference of 1.0 for both trials. In contrast, the Advocated theory (Compensatory Attention) weighs all tied cues inversely to their validity. On Trial 16, the score difference is 0.19, but on Trial 13, opposing cues nearly cancel out, yielding a minimal score difference of 0.04. Consequently, the Advocated theory predicts a significantly higher probability of choosing B on Trial 16 than on Trial 13, whereas the Competing theory predicts no difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
    t13_mask = (data['option_a_ratings'].apply(list) == [0, 1, 1, 0]) & (data['option_b_ratings'].apply(list) == [1, 0, 0, 1])
    
    # Trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
    t16_mask = (data['option_a_ratings'].apply(list) == [0, 0, 1, 0]) & (data['option_b_ratings'].apply(list) == [0, 0, 0, 1])
    
    if t13_mask.sum() == 0 or t16_mask.sum() == 0:
        return 0.0
        
    p_b_16 = data[t16_mask]['response'].mean()
    p_b_13 = data[t13_mask]['response'].mean()
    
    return float(p_b_16 - p_b_13)
outcome: eval error: ValueError: ('Lengths must match to compare', (4800,), (4,))

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]\n    a_t13 = (0, 1, 1, 0)\n    b_t13 = (1, 0, 0, 1)\n    \n    # Trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]\n    a_t16 = (0, 0, 1, 0)\n    b_t16 = (0, 0, 0, 1)\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    t13_mask = (a_tuples == a_t13) & (b_tuples == b_t13)\n    t16_mask = (a_tuples == a_t16) & (b_tuples == b_t16)\n    \n    if t13_mask.sum() == 0 or t16_mask.sum() == 0:\n        return 0.0\n        \n    p_b_16 = data[t16_mask]['response'].mean()\n    p_b_13 = data[t13_mask]['response'].mean()\n    \n    return float(p_b_16 - p_b_13)",
  "rationale": "Both theories predict that subjects will choose Option B on Trial 13 and Trial 16, as both are tie-breaker trials. However, the Competing theory (Reverse Lexicographic) resolves both ties using the least valid cue (cue 3) as the sole decider, generating an identical, maximal score difference of 1.0 for both trials. In contrast, the Advocated theory (Compensatory Attention) weighs all tied cues inversely to their validity. On Trial 16, the score difference is 0.19, but on Trial 13, opposing cues nearly cancel out, yielding a minimal score difference of 0.04. Consequently, the Advocated theory predicts a significantly higher probability of choosing B on Trial 16 than on Trial 13, whereas the Competing theory predicts no difference. (Fixes the previous pandas ValueError by using tuples instead of lists for comparison)."
}
```

## Usage

```json
{
  "prompt_token_count": 4103,
  "candidates_token_count": 579,
  "total_token_count": 7041
}
```
