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
**Validities (n_features=7):** [0.98, 0.88, 0.78, 0.68, 0.58, 0.54, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate Threshold Tallying Theory from Conflict-Induced Guessing Theory, we use a 7-feature design. Conflict-Induced Guessing Theory predicts guessing whenever Take The Best (TTB) and Tallying conflict, and confident choices when they agree. Threshold Tallying Theory ignores validities entirely, predicting uniform guessing when the absolute difference in feature wins (Diff) is below a cognitive threshold (1.5 to 3.5), and confident choices when Diff is above this threshold. We construct critical trials to perfectly cross these predictions. In 'TT-confident / CIG-guess' trials, one option wins on the single highest-validity cue (TTB winner) but loses on all remaining cues (Tallying loser). Here, TTB and Tallying conflict (CIG guesses), but the Diff is 6 (6 wins vs 1 win), which exceeds the maximum threshold of 3.5 (TT is highly confident). In 'TT-guess / CIG-confident' trials, one option wins on the highest-validity cue and a few others, while the alternative wins on just one fewer lower-validity cues. Here, TTB and Tallying agree (CIG is confident), but the Diff is exactly 1, which is strictly below the minimum threshold of 1.5 (TT guesses).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Threshold Tallying Theory posits that decision makers evaluate options by counting the number of features where one option strictly dominates the other (unweighted tallying). However, they only make a confident directional choice if the difference in winning features between the two options meets or exceeds a certain cognitive threshold. If the difference in evidence is below this threshold (e.g., a difference of 0 or 1), the decision maker finds the evidence too ambiguous or weak, leading to cognitive overload or uncertainty, and they resort to uniform guessing. This captures the persistent ~50% choice rates observed across many adversarial trials where the feature counts are closely matched.

**Parameters:**
- threshold: [1.5, 3.5]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins for each option
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    diff = abs(a_wins - b_wins)
    threshold = float(parameters["threshold"])
    
    # If the difference is below the threshold, the evidence is deemed too weak -> guessing
    if diff < threshold:
        p_core = np.array([0.5, 0.5])
    else:
        # Otherwise, make a choice based on the tally scores using softmax
        scores = np.array([float(a_wins), float(b_wins)])
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

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
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 1) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 6)\n    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 2) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 1)\n    t4_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 3) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 2)\n    \n    p_b_t1 = np.mean(data[t1_mask]['response'] == 1) if t1_mask.sum() > 0 else 0.5\n    p_a_t2 = np.mean(data[t2_mask]['response'] == 0) if t2_mask.sum() > 0 else 0.5\n    p_a_t4 = np.mean(data[t4_mask]['response'] == 0) if t4_mask.sum() > 0 else 0.5\n    \n    return float(p_a_t2 + p_a_t4 - p_b_t1)",
  "rationale": "This metric contrasts predictions on trials where the two theories have diametrically opposed confidence levels. Trial 1 is a 'TT-confident / CIG-guess' trial where Threshold Tallying strongly predicts choosing B (due to a tally difference of 5), while Conflict-Induced Guessing predicts ~50% because the best cue favors A but tallying favors B. Trials 2 and 4 are 'TT-guess / CIG-confident' trials where TT predicts ~50% (tally difference of 1 is below threshold) while CIG strongly predicts A (both TTB and tallying favor A). By computing P(A | Trial 2) + P(A | Trial 4) - P(B | Trial 1), we expect a value near 0 for TT (0.5 + 0.5 - 1.0) and near 1.5 for CIG (1.0 + 1.0 - 0.5), yielding a robust and highly discriminative metric."
}
```

## Usage

```json
{
  "prompt_token_count": 3565,
  "candidates_token_count": 609,
  "total_token_count": 6878
}
```
