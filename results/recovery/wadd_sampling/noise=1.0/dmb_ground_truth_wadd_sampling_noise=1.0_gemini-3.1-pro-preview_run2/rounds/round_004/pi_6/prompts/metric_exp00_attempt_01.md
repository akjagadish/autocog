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
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0]  B=[0, 1, 1]
  trial 2: A=[0, 1, 1]  B=[1, 0, 0]
  trial 3: A=[1, 1, 0]  B=[1, 0, 1]
  trial 4: A=[1, 0, 1]  B=[1, 1, 0]
  trial 5: A=[1, 1, 1]  B=[1, 1, 1]

**Rationale:** To quantitatively dissociate Biased Random Guessing from Noisy Take-The-Best (TTB), we manipulate the identity of the TTB winner across trials. Biased Random Guessing predicts a completely flat response profile (a constant probability of choosing A) across all trials, regardless of the stimulus features. In contrast, Noisy TTB predicts that choice probabilities will systematically shift depending on which option is favored by the most valid discriminating cue, even under extreme noise. By including trials where A is the TTB winner, trials where B is the TTB winner, and trials where both options are identical, we can test whether the choice probability for A remains constant or fluctuates based on the TTB winner.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Biased Random Guessing: Due to cognitive overload from processing multiple conflicting binary cues with explicit validities, subjects abandon systematic cue integration and resort to random guessing. However, their guessing is not perfectly uniform; instead, they exhibit a parameterized spatial or positional bias (e.g., a baseline preference for Option A over Option B due to reading order). This allows the model to capture near-random choice behavior while gracefully absorbing slight empirical deviations from exactly 50%.

**Parameters:**
- p_a: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess, but with a spatial/positional bias.
    p_a = float(parameters["p_a"])
    
    # Return the biased probabilities for Option A and Option B.
    return np.array([p_a, 1.0 - p_a])
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
**Description:** Take-The-Best (TTB) heuristic with extreme noise: Decision-makers avoid cognitive overload by not integrating all information. Instead, they search through cues in descending order of their explicitly stated validities. The choice is based entirely on the first cue that discriminates between the two options. However, due to the high cognitive demands of the task or lack of motivation, their behavior is overwhelmingly stochastic, requiring extremely high lapse rates and very low softmax temperatures to capture the near-random empirical choices.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue found
        p_core = np.array([0.5, 0.5])
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * scores
        z -= np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
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
[0] rationale: This metric computes the difference in the probability of choosing Option A between trials where the Take-The-Best (TTB) heuristic favors Option A and trials where it favors Option B. Under the Biased Random Guessing theory, subjects have a constant probability of choosing A across all trials, so this difference should be approximately zero. Under the Noisy TTB theory, even with extreme noise, subjects are systematically more likely to choose A when TTB favors A, resulting in a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def ttb_winner(a, b):
        for i in range(3):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
    
    winners = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        winners.append(ttb_winner(a, b))
    
    df = data.copy()
    df['winner'] = winners
    
    mask_a = df['winner'] == 0
    mask_b = df['winner'] == 1
    
    p_a_given_a = (df.loc[mask_a, 'response'] == 0).mean() if mask_a.sum() > 0 else 0.5
    p_a_given_b = (df.loc[mask_b, 'response'] == 0).mean() if mask_b.sum() > 0 else 0.5
    
    return float(p_a_given_a - p_a_given_b)
outcome: self_sim=0.0011 (var=0.0078) adversary_sim=0.0442 (var=0.0161) welch_t=-1.397 p=0.1696 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    return float(abs(data['response'].mean() - 0.5))",
  "rationale": "The experimental design is perfectly balanced with respect to Take-The-Best (TTB) winners: 2 trials favor Option A, 2 trials favor Option B, and 1 trial is a tie. Consequently, under the Noisy TTB theory, the overall expected probability of choosing Option A across the entire experiment is exactly 0.5 for every subject, regardless of their specific noise parameters. Any deviation from 0.5 in the Noisy TTB simulated data is purely due to finite-trial binomial noise (which is small over 95 trials). In contrast, the Biased Random Guessing theory posits a subject-specific spatial/positional bias (p_a) drawn from [0.0, 1.0]. This bias applies uniformly across all trials, meaning a subject's overall choice proportion will reliably reflect their specific p_a, frequently showing large absolute deviations from 0.5. By simply measuring the absolute deviation of the overall choice proportion from 0.5, we can strongly discriminate the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3185,
  "candidates_token_count": 261,
  "total_token_count": 4974
}
```
