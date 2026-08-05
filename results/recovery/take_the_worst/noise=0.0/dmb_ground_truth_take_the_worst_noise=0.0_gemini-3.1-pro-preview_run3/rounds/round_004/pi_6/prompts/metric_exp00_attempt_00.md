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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Mixture of Standard Tallying and Reverse WADD) from the Competing Theory (Tallying with Negative WADD tie-breaker) by focusing on non-tie trials. Under the Competing Theory, when options do not tie on the number of winning cues, the decision is driven exclusively by Tallying; the magnitude of the WADD or Reverse WADD difference is completely ignored. Thus, across multiple trials where the Tallying difference is held constant (e.g., Option A always wins exactly one more cue than Option B), the Competing Theory predicts an identical choice probability for Option A. In contrast, the Advocated Theory posits a mixture of Tallying and Reverse WADD across all trials. By holding the Tallying difference constant at +1 for Option A but systematically varying the Reverse WADD difference from strongly favoring B to actually favoring A, the Advocated Theory predicts a steeply varying choice probability across these trials, while the Competing Theory predicts a perfectly flat line. Tie trials are also included to anchor the Negative/Reverse WADD effect.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Misinterpreted Expert Ratings (Mixture of Standard Tallying and Reverse WADD): Decision-makers consist of two sub-populations. One group understands the instructions correctly and uses a standard Tallying heuristic, choosing the option with more positive features. A second group misinterprets the binary expert ratings as 'defects' or 'costs' (1 = bad, 0 = good) and evaluates options using a Reverse WADD strategy, minimizing the validity-weighted sum of these perceived defects. In scenarios where Tallying and Reverse WADD agree, choice consistency is high. When they disagree, they pull the aggregate preference in opposite directions, weakening the apparent Tallying effect. Crucially, when options tie under Tallying, the standard group guesses uniformly, allowing the Reverse WADD group to entirely drive the aggregate behavior, resulting in a strong 'Negative WADD' effect.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_rev: [0.1, 100.0]
- w_tally: [0.0, 1.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # Strategy 1: Standard Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([a_wins, b_wins])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Strategy 2: Reverse WADD (interpreting 1s as costs/defects)
    val = np.asarray(parameters["validities"], dtype=float)
    wadd_a = np.dot(a, val)
    wadd_b = np.dot(b, val)
    
    beta_rev = float(parameters["beta_rev"])
    # Negative WADD scores to penalize options with higher validity-weighted 1s
    z_rev = beta_rev * np.array([-wadd_a, -wadd_b])
    z_rev -= np.max(z_rev)
    p_rev = np.exp(z_rev) / np.sum(np.exp(z_rev))
    
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_rev
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Tallying with a Sub-optimal Tie-Breaker (Negative Cue-Weighting in Ties): Decision-makers primarily compare options using a frugal Tallying heuristic, choosing the option with the greater number of winning cues. However, when options are tied on the number of winning features, individuals do not simply guess or use a rational compensatory strategy. Instead, they exhibit a sub-optimal tie-breaking mechanism where they systematically favor the option with lower-ranked or lower-validity cues (effectively a Negative WADD score). This explains both the strong adherence to Tallying when cue counts differ, and the counter-intuitive preference for lower-validity options when cue counts are tied.

**Parameters:**
- beta: [0.1, 10.0]
- tie_beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    tie_beta = float(parameters["tie_beta"])
    epsilon = float(parameters["epsilon"])
    
    if a_wins != b_wins:
        # Primary strategy: Tallying
        scores = np.array([a_wins, b_wins])
        z = beta * scores
    else:
        # Tie-breaker: Negative WADD (favoring lower validity cues)
        val = np.asarray(parameters["validities"], dtype=float)
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        scores = np.array([-wadd_a, -wadd_b])
        z = tie_beta * scores
        
    z = z - np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    # Project lists to tuples to make them hashable and comparable\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    data['B_tuple'] = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]\n    # Tallying diff: +1 for A. Reverse WADD diff: +0.20 for A.\n    t1_mask = (data['A_tuple'] == (0, 0, 1, 1, 1)) & (data['B_tuple'] == (1, 1, 0, 0, 0))\n    \n    # Trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]\n    # Tallying diff: +1 for A. Reverse WADD diff: -1.40 for A (strongly favors B).\n    t2_mask = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))\n    \n    # Calculate probability of choosing A (response == 0)\n    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()\n    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()\n    \n    # Handle edge case where a subject might not have these specific trials (though the design implies they do)\n    if pd.isna(p_a_t1): p_a_t1 = 0.5\n    if pd.isna(p_a_t2): p_a_t2 = 0.5\n    \n    return float(p_a_t1 - p_a_t2)\n",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 2. In both trials, Option A wins on exactly one more cue than Option B (Tallying difference is +1). According to the Competing Theory, decision-makers rely exclusively on Tallying in non-tie scenarios, so the choice probability for Option A should be identical across both trials, yielding a metric value near 0. In contrast, the Advocated Theory posits a mixture of Tallying and Reverse WADD. While the Tallying component favors A equally in both trials, the Reverse WADD component slightly favors A in Trial 1 but strongly favors B in Trial 2. Therefore, the Advocated Theory predicts a significantly higher probability of choosing A in Trial 1 compared to Trial 2, yielding a large positive metric value."
}
```

## Usage

```json
{
  "prompt_token_count": 3653,
  "candidates_token_count": 699,
  "total_token_count": 6568
}
```
