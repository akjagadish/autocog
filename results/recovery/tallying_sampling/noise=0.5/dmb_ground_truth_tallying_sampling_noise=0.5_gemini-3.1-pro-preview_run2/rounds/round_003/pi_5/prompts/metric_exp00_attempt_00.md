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
**Validities (n_features=6):** [0.95, 0.9, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]

**Rationale:** To cleanly dissociate Pure Tallying from Tally-Gated Validity Bias, this design focuses on trials where the tally scores are strictly unequal (e.g., Option A has 3 features, Option B has 2 features) but the validity-weighted sums vary dramatically. Pure Tallying predicts a constant choice probability for the option with the tally advantage, completely ignoring which specific features are active. In contrast, Tally-Gated Validity Bias predicts that validities modulate the strength of preference whenever the tallies are unequal, leading to a graded response curve across these trials. Crucially, we also include 'Tally Tie' trials where both options have an equal number of positive features but vastly different validities; here, the Tally-Gated Validity Bias model predicts exactly 50/50 guessing (as it abruptly aborts validity integration on ties), a unique signature distinguishing it from standard compensatory models.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Tally-Gated Validity Bias: Decision-makers primarily rely on a Tallying heuristic, simply counting the number of positive features for each option. If the tally results in a tie, the decision process abruptly concludes and they guess randomly, without falling back on cue validities. However, if there is a difference in tally scores, the strength of their preference is modulated by the explicit cue validities. This means validities act as a secondary confidence-adjuster rather than a tie-breaker, explaining why validity bias appears in overall choices but is absent when options have an equal number of positive features.

**Parameters:**
- beta: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- w_val: [0.0, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Primary strategy: Tallying
    tally_scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_val = float(parameters["w_val"])
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # If tallying results in a tie, guess randomly (no validity tie-breaking)
    if tally_scores[0] == tally_scores[1]:
        p_core = np.ones(2) / 2.0
    else:
        # If there is a tally difference, validities modulate the response strength
        val_scores = stim @ validities
        scores = (1.0 - w_val) * tally_scores + w_val * val_scores
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Parameters:**
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 1: Option A has +1 tally advantage AND large validity advantage\n    t1_mask = (a_tuples == (1, 1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 0, 0, 1, 1))\n    # Trial 5: Option A has +1 tally advantage BUT validity disadvantage\n    t5_mask = (a_tuples == (0, 0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0, 0))\n    \n    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()\n    p_a_t5 = 1.0 - data.loc[t5_mask, 'response'].mean()\n    \n    if pd.isna(p_a_t1) or pd.isna(p_a_t5):\n        return 0.0\n        \n    return float(p_a_t1 - p_a_t5)",
  "rationale": "Both Trial 1 and Trial 5 present the subject with a choice where Option A has exactly 3 positive features and Option B has exactly 2 positive features. Under the Competing theory (Pure Tallying), subjects should choose Option A with identical probability in both trials because they only care about the tally difference (+1 for A in both cases). Under the Advocated theory (Tally-Gated Validity Bias), the validity of the features modulates the preference: Option A has a massive validity advantage in Trial 1, but a validity disadvantage in Trial 5. Thus, the Advocated theory predicts a large positive difference in the choice proportion of A between these two trials, while the Competing theory predicts exactly zero difference."
}
```

## Usage

```json
{
  "prompt_token_count": 3332,
  "candidates_token_count": 477,
  "total_token_count": 8082
}
```
